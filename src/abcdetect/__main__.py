import sys
from pathlib import Path

from .download_dataset import download
from .segmentation import train_segmentation_model, evaluate_segmentation_model

# Create directories for datasets and output if they do not exist
datasets_dir = Path("datasets").absolute()
if not datasets_dir.exists():
    print("Creating datasets directory at:", datasets_dir)
    datasets_dir.mkdir(parents=True)
    with (datasets_dir / ".gitignore").open("w") as f:
        f.write("*\n")

output_dir = Path("output").absolute()
if not output_dir.exists():
    print("Creating output directory at:", output_dir)
    output_dir.mkdir(parents=True)
    with (output_dir / ".gitignore").open("w") as f:
        f.write("*\n")

# Download the HAM10K dataset
ham10k_image_path, ham10k_metadata_path, ham10k_masks_path = download(datasets_dir)


def find_segmentation_model() -> Path | None:
    """Finds the latest segmentation model file in the output directory.

    Returns:
        Path to the segmentation model file if found, None otherwise.
    """
    segmentation_model_files = list(output_dir.glob("segmentation_model_*.pt"))
    if segmentation_model_files:
        # Sort files by timestamp (extracted from filename) in descending order
        latest_model = max(
            segmentation_model_files,
            key=lambda p: int(p.stem.split("_")[-1])
        )
        return latest_model

    # Check for a single segmentation model file
    single_segmentation_model = output_dir / "segmentation_model.pt"
    if single_segmentation_model.exists():
        return single_segmentation_model
    return None

# Train the segmentation model, saving it to the output directory
segmentation_model_path = find_segmentation_model()
if segmentation_model_path is None:
    segmentation_model_path = train_segmentation_model(ham10k_image_path, ham10k_metadata_path, ham10k_masks_path, output_dir)

    if segmentation_model_path is None:
        print("No trained segmentation model found, cannot proceed.")
        sys.exit(1)

# Evaluate the segmentation model
evaluate_segmentation_model(ham10k_image_path, ham10k_masks_path, segmentation_model_path)
