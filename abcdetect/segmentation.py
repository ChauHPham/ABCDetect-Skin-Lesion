import random
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from skimage.morphology import remove_small_objects
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .stratification import stratified_sampling, visualize_dx_column_as_histogram


class LesionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self, df: pd.DataFrame, img_dir: Path, mask_dir: Path, transform: A.Compose | None = None
    ) -> None:
        """Initializes the LesionDataset.

        Args:
            df: DataFrame containing image IDs.
            img_dir: Directory containing the images.
            mask_dir: Directory containing the masks.
            transform: Optional transformations to apply to the images and masks.
        """
        super().__init__()
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.images = df["image_id"].astype(str).values
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_name = self.images[index]
        img_path = self.img_dir / f"{image_name}.jpg"
        mask_path = self.mask_dir / f"{image_name}_segmentation.png"

        image_array = np.array(Image.open(img_path).convert("RGB"))
        mask_array = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask_array[mask_array > 0] = 1.0

        # Apply optional transformations
        if self.transform:
            augmented = self.transform(image=image_array, mask=mask_array)
            image: torch.Tensor = augmented["image"]
            mask: torch.Tensor = augmented["mask"]
        else:
            # Convert to tensor without transforms (permute HWC to CHW format and normalize)
            image = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask_array)

        # Ensure mask has channel dimension
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        return image, mask


def show_batch_samples(
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]], *, alpha: float = 0.3, num_samples: int = 3
) -> None:
    """Visualizes a batch of images and their corresponding masks.

    Args:
        loader: DataLoader object containing the dataset.
        alpha: The transparency level for the overlay (0.0 to 1.0).
        num_samples: Number of samples to visualize, should be less than or equal to the batch size.
    """
    print("Visualizing batch samples...\n")
    images, masks = next(iter(loader))
    images = images[:num_samples]
    masks = masks[:num_samples]
    num_samples = min(num_samples, len(images))

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3))

    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).cpu().numpy()  # CHW to HWC
        mask = masks[i].squeeze().cpu().numpy()

        # Create red overlay
        red_overlay = np.zeros_like(img)
        red_overlay[..., 0] = 1.0  # Red channel

        # Blend the original image with red overlay where mask is present
        # Makes the image equal to the second array when mask > 0.5, and the image when mask is 0
        # Stretches the mask shape to 3 dimensions to apply the alpha
        overlay = np.where(mask[..., None] > 0.5, (1 - alpha) * img + alpha * red_overlay, img)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initializes the DoubleConv module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.conv_fn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_fn(x)


class UNET(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] | None = None) -> None:
        """Initializes the UNET model.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            features: List of feature sizes for the encoder and decoder.
        """
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.pool_fn = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling part of UNET
        self.downsampling_fn = nn.ModuleList()
        for feature in features:
            self.downsampling_fn.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Upsampling part of UNET
        self.upsampling_fn = nn.ModuleList()
        for feature in reversed(features):
            self.upsampling_fn.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.upsampling_fn.append(DoubleConv(feature * 2, feature))

        self.bottleneck_fn = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv_fn = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for down_fn in self.downsampling_fn:
            x = down_fn(x)
            skip_connections.append(x)
            x = self.pool_fn(x)  # Reduce the size of the image

        x = self.bottleneck_fn(x)  # The final step between the convolution layer and the dense layer

        # Reverse the skip connections list for use on the encoder side of UNET
        skip_connections.reverse()

        # Iterate through the transconvolution and double conv upsample layers with a step of 2
        for idx in range(0, len(self.upsampling_fn), 2):
            x = self.upsampling_fn[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(
                    x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True
                )

            concat_skip = torch.concat((skip_connection, x), dim=1)  # TODO LEARN
            x = self.upsampling_fn[idx + 1](concat_skip)

        return self.final_conv_fn(x)


class DiceBCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(inputs, targets)

        # Dice loss component
        inputs_sigmoid = torch.sigmoid(inputs)
        intersection = (inputs_sigmoid * targets).sum()
        dice_loss = 1 - (2.0 * intersection) / (inputs_sigmoid.sum() + targets.sum() + 1e-8)

        # Combine losses
        return 0.5 * bce_loss + 0.5 * dice_loss


def apply_morphology(image: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Apply morphological operations to an image.

    Args:
        image: Input image array.
        **kwargs: Additional keyword arguments.

    Returns:
        Processed image array.
    """
    return cv2.morphologyEx(
        image,
        cv2.MORPH_OPEN,  # Use opening operation to remove noise while preserving structure
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )


def get_training_transform(image_height: int = 192, image_width: int = 256) -> A.Compose:
    """Defines the training transformations for the dataset.

    Args:
        image_height: The height of the image after resizing.
        image_width: The width of the image after resizing.

    Returns:
        A Compose object containing the transformations.
    """
    return A.Compose(
        [
            A.Resize(image_height, image_width),
            A.Lambda(image=apply_morphology),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.15),
            A.GaussianBlur(blur_limit=1, p=0.1),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1, p=0.3),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            A.ToTensorV2(),
        ]
    )


def get_evaluation_transform(image_height: int = 192, image_width: int = 256) -> A.Compose:
    """Defines the evaluation transformations for the dataset.

    For best performance, the evaluation transform should use the same resizing as the training transform.

    Args:
        image_height: The height of the image after resizing.
        image_width: The width of the image after resizing.

    Returns:
        A Compose object containing the transformations.
    """
    return A.Compose(
        [
            A.Resize(image_height, image_width),
            A.Lambda(image=apply_morphology),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            A.ToTensorV2(),
        ]
    )


def validate_model(
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: nn.Module,
    loss_fn: nn.Module,
    *,
    device: torch.device,
) -> tuple[float, float]:
    """Validates the model on the validation set.

    Args:
        loader: DataLoader object for the validation set.
        model: The model to validate.
        loss_fn: Loss function to compute the loss.
        device: Device to use for validation (e.g., "cuda" or "cpu").

    Returns:
        A tuple containing the mean loss and the Dice score.
    """
    model.eval()

    num_correct = 0
    num_pixels = 0
    dice_score = 0
    total_loss = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.float().to(device)

            preds = model(data)
            loss = loss_fn(preds, targets)
            total_loss += loss.item()

            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()

            num_correct += (preds == targets).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * targets).sum()) / ((preds + targets).sum() + 1e-8)

    accuracy = num_correct / num_pixels * 100
    mean_loss = total_loss / len(loader)
    dice_score /= len(loader)

    print(f"📊 Val Loss: {mean_loss:.4f}, Accuracy: {accuracy:.2f}%, Dice: {dice_score:.4f}")
    model.train()
    return mean_loss, dice_score.item()


def train_model(
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    *,
    device: torch.device,
) -> None:
    """Trains the model for one epoch.

    Args:
        loader: DataLoader object for the training set.
        model: The model to train.
        loss_fn: Loss function to compute the loss.
        optimizer: Optimizer for updating the model weights.
        epoch: Current epoch number.
        device: Device to use for training (e.g., "cuda" or "cpu").
    """
    model.train()

    scaler = torch.amp.GradScaler(device.type)
    loop = tqdm(loader)

    for data, targets in loop:
        data: torch.Tensor = data.to(device)
        targets: torch.Tensor = targets.to(device)

        # Forward
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backwards
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        loop.set_description(f"Epoch [{epoch}]")
        loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])


def train_segmentation_model(
    ham10k_image_path: Path,
    ham10k_metadata_path: Path,
    ham10k_masks_path: Path,
    output_dir: Path,
    *,
    device: torch.device,
    show_graph: bool = False,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Path | None:
    """Trains a segmentation model using the HAM10000 dataset.

    The trained model is saved to the specified path. The dataset is split into training,
    validation, and test sets using stratified sampling.

    Args:
        ham10k_image_path: Path to the directory containing the images.
        ham10k_metadata_path: Path to the directory containing the metadata.
        ham10k_masks_path: Path to the directory containing the masks.
        output_dir: Path to save the trained model output.
        device: Device to use for training (e.g., "cuda" or "cpu").
        show_graph: Whether to show training graphs.
        batch_size: Batch size for training.
        num_workers: Number of workers for DataLoader.

    Returns:
        Path to the trained model file.
    """
    print("Training segmentation model...\n")
    metadata_df = pd.read_csv(ham10k_metadata_path / "HAM10000_metadata.csv")
    print(metadata_df.head(), "\n")

    # Replace null ages with the mean
    mean_age = round(metadata_df["age"].mean())
    metadata_df.loc[:, "age"] = metadata_df["age"].fillna(mean_age)
    assert not metadata_df.isnull().any().any(), "DataFrame contains null values"

    train_df, validate_df, test_df = stratified_sampling(
        df=metadata_df,
        stratify_by="dx",
        split_ratios=[0.8, 0.1, 0.1],
    )
    print("Train set size:", len(train_df))
    print("Validation set size:", len(validate_df))
    print("Test set size:", len(test_df))

    if show_graph:
        visualize_dx_column_as_histogram(train_df, "Train")
        visualize_dx_column_as_histogram(validate_df, "Validation")
        visualize_dx_column_as_histogram(test_df, "Test")

    transform = get_training_transform()

    training_data = LesionDataset(
        img_dir=ham10k_image_path, mask_dir=ham10k_masks_path, df=train_df, transform=transform
    )
    train_loader = DataLoader(
        dataset=training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    validation_data = LesionDataset(
        img_dir=ham10k_image_path, mask_dir=ham10k_masks_path, df=validate_df, transform=transform
    )
    validate_loader = DataLoader(
        dataset=validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    if show_graph:
        preview_loader = DataLoader(
            dataset=training_data,
            batch_size=batch_size,
            shuffle=True,
        )
        show_batch_samples(preview_loader)

    # Initialize model instance
    model = UNET(in_channels=3, out_channels=1).to(device)

    learning_rate = 1e-4  # Initial learning rate
    max_epochs = 100

    loss_fn = DiceBCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # ReduceLROnPlateau reduces learning rate when dice score plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # higher is better for dice score
        factor=0.5,  # Reduce LR by half when plateau is detected
        patience=3,  # Wait for 3 epochs of no improvement
        min_lr=1e-6,  # Don't reduce LR below this value
    )

    mean_losses = []
    dice_scores = []
    learning_rates = []

    # Early stopping parameters
    patience = 5  # Number of epochs to wait for improvement
    best_dice = 0.0
    patience_counter = 0
    best_model_state: dict | None = None

    try:
        for epoch in range(max_epochs):
            # Store current learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            learning_rates.append(current_lr)

            train_model(train_loader, model, loss_fn, optimizer, epoch, device=device)
            mean_loss, dice_score = validate_model(validate_loader, model, loss_fn, device=device)
            mean_losses.append(mean_loss)
            dice_scores.append(dice_score)

            # Update learning rate scheduler based on dice score
            scheduler.step(dice_score)

            # Check if this is the best model so far
            if dice_score > best_dice:
                best_dice = dice_score
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"✅ New best model saved! Dice score: {best_dice:.4f}, LR: {current_lr:.7f}")
            else:
                patience_counter += 1
                print(
                    f"⏳ No improvement for {patience_counter} epochs. Best dice: {best_dice:.4f}, LR: {current_lr:.7f}"
                )

            # Early stopping check
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch + 1}/{max_epochs}")
                break
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:  # Always save the model state
        if best_model_state is not None:  # Load the best model state
            model.load_state_dict(best_model_state)
        else:
            return None  # No model was trained

        model_save_path = output_dir / f"segmentation_model_{int(pd.Timestamp.now().timestamp())}.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}.")

    print("Training completed.")
    if show_graph:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        fig.suptitle("Segmentation Model Training Progress", fontsize=16)

        # Plot mean loss
        ax1.plot(range(1, len(mean_losses) + 1), mean_losses, "b-o")  # blue line with circle markers
        ax1.set_title("Mean Loss Over Epochs")
        ax1.set_ylabel("Mean Loss")
        ax1.grid(True)

        # Plot dice score
        ax2.plot(range(1, len(dice_scores) + 1), dice_scores, "r-o")  # red line with circle markers
        ax2.set_title("Dice Score Over Epochs")
        ax2.set_ylabel("Dice Score")
        ax2.grid(True)

        # Plot learning rate
        ax3.plot(range(1, len(learning_rates) + 1), learning_rates, "g-o")  # green line with circle markers
        ax3.set_title("Learning Rate Over Epochs")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Learning Rate")
        ax3.set_yscale("log")  # Log scale makes LR changes more visible
        ax3.grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()
    return model_save_path


def enhance_prediction_mask(mask: np.array, min_mask_size: int = 100) -> np.array:
    """Enhances the predicted mask by removing small objects and filling holes.

    Args:
        mask: The predicted binary mask.
        min_mask_size: Minimum size of objects to keep in the mask.

    Returns:
        The enhanced binary mask.
    """
    # Fill black holes inside white regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Dilate and erode mask to maintain its size, but smooth jagged edges
    dilated = cv2.dilate(cleaned_mask, kernel, iterations=1)
    smoothed = cv2.erode(dilated, kernel, iterations=1)

    # Get rid of noise found around and not connected to the mask
    denoised = remove_small_objects(smoothed.astype(bool), min_size=min_mask_size)

    return denoised


def evaluate_segmentation_model(
    ham10k_image_path: Path,
    ham10k_masks_path: Path,
    model_save_path: Path,
    *,
    device: torch.device,
    alpha: float = 0.3,
    num_samples: int = 3,
) -> None:
    """Tests the segmentation model on a few random samples from the dataset.

    Args:
        ham10k_image_path: Path to the directory containing the images.
        ham10k_masks_path: Path to the directory containing the masks.
        model_save_path: Path to the trained model.
        device: Device to use for evaluation (e.g., "cuda" or "cpu").
        alpha: The transparency level for the overlay (0.0 to 1.0).
        num_samples: Number of samples to visualize.
    """
    image_paths = random.sample(list(ham10k_image_path.glob("*")), num_samples)

    fig_size = (12, num_samples * 4)
    fig, axes = plt.subplots(num_samples, 5, figsize=fig_size)

    test_model = UNET(in_channels=3, out_channels=1).to(device)
    test_model.load_state_dict(torch.load(model_save_path, map_location=device))
    test_model.eval()

    for i, image_path in enumerate(image_paths):
        # Fetch the image and the mask of the corresponding file path
        mask_path = ham10k_masks_path / f"{image_path.stem}_segmentation.png"

        # Ensure image is in RGB format, and convert image and mask to np arrays
        image_array = np.array(Image.open(image_path).convert("RGB"))
        mask_array = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask_array[mask_array > 0] = 1.0

        # Apply transformations to the image and the mask
        transform = get_evaluation_transform()
        augmented = transform(image=image_array, mask=mask_array)
        image = augmented["image"]
        mask = augmented["mask"]

        # Apply the model on the input tensor, output a binary mask
        with torch.no_grad():
            output = test_model(image.unsqueeze(0).to(device))
            output = torch.sigmoid(output)  # Sigmoid since it's binary
            predicted_mask_array = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        # Enhance prediction
        enhanced_predicted_mask_array = enhance_prediction_mask(predicted_mask_array)

        # Convert tensor to numpy
        image_array = image.permute(1, 2, 0).cpu().numpy()

        # Create red overlay
        red_overlay = np.zeros_like(image_array)
        red_overlay[..., 0] = 1.0  # Red channel only

        overlay = np.where(
            enhanced_predicted_mask_array[..., None] > 0.5,
            (1 - alpha) * image_array + alpha * red_overlay,
            image_array,
        )

        # Plot: image, raw pred, enhanced mask, ground truth, overlay
        titles = ["Image", "Predicted Mask", "Enhanced Mask", "Correct Mask", "Overlay"]
        visuals = [image_array, predicted_mask_array, enhanced_predicted_mask_array, mask, overlay]

        for j in range(5):
            axes[i, j].imshow(visuals[j], cmap="gray" if j in {1, 2, 3} else None)
            axes[i, j].set_title(titles[j])
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()


def segment_single_image(
    image_path: Path,
    model_save_path: Path,
    output_dir: Path,
    *,
    device: torch.device,
    show_graph: bool = False,
    alpha: float = 0.3,
) -> Path:
    """Segments a single image using the provided model and saves the mask to the output directory.

    Args:
        image_path: Path to the input image.
        model_save_path: Path to the trained model.
        output_dir: Directory to save the generated mask.
        device: Device to use for evaluation (e.g., "cuda" or "cpu").
        show_graph: Whether to show the generated mask and overlay.
        alpha: The transparency level for the overlay (0.0 to 1.0).

    Returns:
        Path to the saved mask file.
    """
    print(f"Segmenting image with model {model_save_path.name}...\n")

    model = UNET(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    orig_image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = orig_image.size
    image_array = np.array(orig_image)

    # Apply transformations to the image
    transform = get_evaluation_transform()
    augmented = transform(image=image_array)
    image = augmented["image"]

    # Generate the mask using the model
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        output = torch.sigmoid(output)  # Apply sigmoid for binary segmentation
        predicted_mask_array = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    # Enhance the predicted mask
    enhanced_mask = enhance_prediction_mask(predicted_mask_array)

    # Create a higher resolution version for smoother resizing
    upscaled_mask = cv2.resize(
        enhanced_mask.astype(np.float32),
        (enhanced_mask.shape[1] * 4, enhanced_mask.shape[0] * 4),
        interpolation=cv2.INTER_LINEAR,
    )
    blurred_mask = cv2.GaussianBlur(upscaled_mask, (9, 9), 2.0)

    # Resize to original image dimensions with cubic interpolation
    smooth_mask = cv2.resize(blurred_mask, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)

    # Threshold but retain some edge smoothness
    final_mask = (smooth_mask > 0.5).astype(np.uint8)

    # Save the mask to the output directory
    mask_output_path = output_dir / f"{image_path.stem}_segmentation.png"
    Image.fromarray((final_mask * 255).astype(np.uint8)).save(mask_output_path)
    print(f"Mask saved to {mask_output_path}.")

    if show_graph:
        # Resize for visualization to improve rendering speed
        visual_image = orig_image.copy()
        visual_mask = Image.fromarray((final_mask * 255).astype(np.uint8))

        # Calculate new dimensions while maintaining aspect ratio
        ratio = min(800 / orig_width, 800 / orig_height)
        new_width = int(orig_width * ratio)
        new_height = int(orig_height * ratio)

        # Resize both image and mask
        visual_image = visual_image.resize((new_width, new_height), Image.LANCZOS)
        visual_mask = visual_mask.resize((new_width, new_height), Image.NEAREST)

        # Convert to arrays for visualization
        visual_image_array = np.array(visual_image) / 255.0
        visual_mask_array = np.array(visual_mask) / 255.0

        # Create red overlay color
        visual_red = np.zeros_like(visual_image_array)
        visual_red[..., 0] = 1.0

        # Display the results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(visual_image_array)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(visual_mask_array, cmap="gray")
        axes[1].set_title("Generated Mask")
        axes[1].axis("off")

        # Create overlay with resized mask
        visual_overlay = np.where(
            visual_mask_array[..., None] > 0.5,
            (1 - alpha) * visual_image_array + alpha * visual_red,
            visual_image_array,
        )
        axes[2].imshow(visual_overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    return mask_output_path
