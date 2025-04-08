import random
from pathlib import Path

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

from .stratification import stratified_sampling

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


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
        self.images = df["image_id"].astype(str).tolist()
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

    images, masks = next(iter(loader))  # grab a batch from the loader
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
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
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
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            A.ToTensorV2(),
        ]
    )


def validate_model(
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]], model: nn.Module, loss_fn: nn.Module
) -> tuple[float, float]:
    """Validates the model on the validation set.

    Args:
        loader: DataLoader object for the validation set.
        model: The model to validate.
        loss_fn: Loss function to compute the loss.

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
            data = data.to(DEVICE)
            targets = targets.float().to(DEVICE)

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
) -> None:
    """Trains the model for one epoch.

    Args:
        loader: DataLoader object for the training set.
        model: The model to train.
        loss_fn: Loss function to compute the loss.
        optimizer: Optimizer for updating the model weights.
        epoch: Current epoch number.
    """
    model.train()

    scaler = torch.amp.GradScaler(DEVICE.type)
    loop = tqdm(loader)

    for data, targets in loop:
        data: torch.Tensor = data.to(DEVICE)
        targets: torch.Tensor = targets.to(DEVICE)

        # Forward
        with torch.amp.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backwards
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        loop.set_description(f"Epoch [{epoch}]")
        loop.set_postfix(loss=loss.item())


def train_segmentation_model(
    ham10k_image_path: Path, ham10k_metadata_path: Path, ham10k_masks_path: Path, output_dir: Path
) -> Path | None:
    """Trains a segmentation model using the HAM10000 dataset.

    The trained model is saved to the specified path. The dataset is split into training,
    validation, and test sets using stratified sampling.

    Args:
        ham10k_image_path: Path to the directory containing the images.
        ham10k_metadata_path: Path to the directory containing the metadata.
        ham10k_masks_path: Path to the directory containing the masks.
        output_dir: Path to save the trained model output.

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

    # Uncomment the following lines to visualize the dx column as a histogram
    # from .stratification import visualize_dx_column_as_histogram
    # visualize_dx_column_as_histogram(train_df)
    # visualize_dx_column_as_histogram(validate_df)
    # visualize_dx_column_as_histogram(test_df)

    batch_size = 16
    learning_rate = 1e-4
    max_epochs = 40

    transform = get_training_transform()

    training_data = LesionDataset(
        img_dir=ham10k_image_path, mask_dir=ham10k_masks_path, df=train_df, transform=transform
    )
    train_loader = DataLoader(
        dataset=training_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    validation_data = LesionDataset(
        img_dir=ham10k_image_path, mask_dir=ham10k_masks_path, df=validate_df, transform=transform
    )
    validate_loader = DataLoader(
        dataset=validation_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    # Uncomment the following line to visualize a batch of training samples
    # show_batch_samples(train_loader)

    # Initialize model instance
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()  # Binary loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mean_losses = []
    dice_scores = []

    # Early stopping parameters
    patience = 5  # Number of epochs to wait for improvement
    best_dice = 0.0
    patience_counter = 0
    best_model_state: dict | None = None

    try:
        for epoch in range(max_epochs):
            train_model(train_loader, model, loss_fn, optimizer, epoch)
            mean_loss, dice_score = validate_model(validate_loader, model, loss_fn)
            mean_losses.append(mean_loss)
            dice_scores.append(dice_score)

            # Check if this is the best model so far
            if dice_score > best_dice:
                best_dice = dice_score
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"✅ New best model saved! Dice score: {best_dice:.4f}")
            else:
                patience_counter += 1
                print(f"⏳ No improvement for {patience_counter} epochs. Best dice: {best_dice:.4f}")

            # Early stopping check
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch + 1}/{max_epochs}")
                break
    except KeyboardInterrupt:
        print("Training interrupted by user.")

    if best_model_state is not None:  # Load the best model state
        model.load_state_dict(best_model_state)
    else:
        return None  # No model was trained

    model_save_path = output_dir / f"segmentation_model_{int(pd.Timestamp.now().timestamp())}.pt"
    torch.save(model.state_dict(), model_save_path)
    print("Training completed.")
    print(f"Model saved to {model_save_path}.")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot mean loss
    ax1.plot(range(1, len(mean_losses) + 1), mean_losses, "b-o")  # blue line with circle markers
    ax1.set_title("Mean Loss Over Epochs")
    ax1.set_ylabel("Mean Loss")
    ax1.grid(True)

    # Plot dice score
    ax2.plot(range(1, len(dice_scores) + 1), dice_scores, "r-o")  # red line with circle markers
    ax2.set_title("Dice Score Over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.grid(True)

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
    alpha: float = 0.3,
    num_samples: int = 3,
) -> None:
    """Tests the segmentation model on a few random samples from the dataset.

    Args:
        ham10k_image_path: Path to the directory containing the images.
        ham10k_masks_path: Path to the directory containing the masks.
        model_save_path: Path to the trained model.
        alpha: The transparency level for the overlay (0.0 to 1.0).
        num_samples: Number of samples to visualize.
    """
    image_paths = random.sample(list(ham10k_image_path.glob("*")), num_samples)

    fig_size = (12, num_samples * 4)
    fig, axes = plt.subplots(num_samples, 5, figsize=fig_size)

    test_model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    test_model.load_state_dict(torch.load(model_save_path))
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
            output = test_model(image.unsqueeze(0).to(DEVICE))
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


def segment_image(image_path: Path, model_save_path: Path, output_dir: Path, *, alpha: float = 0.3) -> Path:
    """Segments a single image using the provided model and saves the mask to the output directory.

    Args:
        image_path: Path to the input image.
        model_save_path: Path to the trained model.
        output_dir: Directory to save the generated mask.
        alpha: The transparency level for the overlay (0.0 to 1.0).

    Returns:
        Path to the saved mask file.
    """
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    image_array = np.array(Image.open(image_path).convert("RGB"))

    # Apply transformations to the image
    transform = get_evaluation_transform()
    augmented = transform(image=image_array)
    image = augmented["image"]

    # Generate the mask using the model
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(DEVICE))
        output = torch.sigmoid(output)  # Apply sigmoid for binary segmentation
        predicted_mask_array = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    # Enhance the predicted mask
    enhanced_mask = enhance_prediction_mask(predicted_mask_array)

    # Save the mask to the output directory
    mask_output_path = output_dir / f"{image_path.stem}_segmentation.png"
    Image.fromarray((enhanced_mask * 255).astype(np.uint8)).save(mask_output_path)

    # Create an overlay for visualization
    image_array = image.permute(1, 2, 0).cpu().numpy()
    red_overlay = np.zeros_like(image_array)
    red_overlay[..., 0] = 1.0  # Red channel only

    overlay = np.where(
        enhanced_mask[..., None] > 0.5,
        (1 - alpha) * image_array + alpha * red_overlay,
        image_array,
    )

    # Display the results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_array)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(enhanced_mask, cmap="gray")
    axes[1].set_title("Generated Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Mask saved to {mask_output_path}.")
    return mask_output_path
