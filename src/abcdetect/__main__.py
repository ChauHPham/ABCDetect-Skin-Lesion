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

# Train the segmentation model, saving it to the output directory
segmentation_model_path = output_dir / "segmentation_model.pt"
if not segmentation_model_path.exists():
    train_segmentation_model(ham10k_image_path, ham10k_metadata_path, ham10k_masks_path, segmentation_model_path)

if not segmentation_model_path.exists():
    print("Segmentation model not found. Please train the model first.")
    sys.exit(1)

# Evaluate the segmentation model
evaluate_segmentation_model(ham10k_image_path, ham10k_masks_path, segmentation_model_path)
