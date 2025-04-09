import argparse
import sys
from pathlib import Path

import torch

from .download_dataset import download
from .segmentation import (evaluate_segmentation_model, segment_single_image,
                           train_segmentation_model)


def get_device(device_str: str) -> torch.device:
    """Determine the device based on user input and availability.

    Args:
        device_str: 'auto', 'cuda', or 'cpu'.

    Returns:
        torch.device: The device to use.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU instead.")
        return torch.device("cpu")
    else:
        return torch.device(device_str)


def create_directory(directory_path: Path) -> Path:
    """Create a directory and add .gitignore if it doesn't exist.

    Args:
        directory_path: Path to the directory to create.

    Returns:
        The absolute Path object of the created directory.
    """
    if not directory_path.exists():
        print(f"Creating directory at: {directory_path}")
        directory_path.mkdir(parents=True)
        with (directory_path / ".gitignore").open("w") as f:
            f.write("*\n")
    return directory_path


def is_valid_directory(directory_str: str) -> Path:
    """Validate and convert a directory string to a Path.

    Args:
        directory_str: Directory string to validate.

    Returns:
        The absolute Path object of the directory.

    Raises:
        argparse.ArgumentTypeError: If the directory string is invalid or points to a file.
    """
    path = Path(directory_str).absolute()
    if path.exists() and path.is_file():
        raise argparse.ArgumentTypeError(f"{directory_str} exists but is a file, not a directory")
    return path


def find_segmentation_model(output_dir: Path) -> Path | None:
    """Finds the latest segmentation model file in the output directory.

    Args:
        output_dir: Directory where the segmentation model files are stored.

    Returns:
        Path to the segmentation model file if found, None otherwise.
    """
    segmentation_model_files = list(output_dir.glob("segmentation_model_*.pt"))
    if segmentation_model_files:
        # Sort files by timestamp (extracted from filename) in descending order
        latest_model = max(segmentation_model_files, key=lambda p: int(p.stem.split("_")[-1]))
        return latest_model

    # Check for a single segmentation model file
    single_segmentation_model = output_dir / "segmentation_model.pt"
    if single_segmentation_model.exists():
        return single_segmentation_model
    return None


def download_dataset(datasets_dir: Path, force: bool = False) -> tuple[Path, Path, Path]:
    """Download the HAM10K dataset.

    Args:
        datasets_dir: Directory to store the downloaded dataset.
        force: If True, re-downloads the dataset even if it already exists.

    Returns:
        Tuple of Paths to the image, metadata, and masks files.
    """
    return download(datasets_dir, force=force)


def train_model(
    datasets_dir: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
    show_graph: bool = False,
) -> Path:
    """Download dataset if needed and train the segmentation model.

    Args:
        datasets_dir: Directory to store the dataset.
        output_dir: Directory to save the trained model.
        device: Device to use for training (CPU or GPU).
        batch_size: Batch size for training.
        num_workers: Number of workers for data loading.
        show_graph: If True, display training graphs.

    Returns:
        Path to the trained segmentation model.
    """
    ham10k_image_path, ham10k_metadata_path, ham10k_masks_path = download_dataset(datasets_dir)

    segmentation_model_path = train_segmentation_model(
        ham10k_image_path,
        ham10k_metadata_path,
        ham10k_masks_path,
        output_dir,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        show_graph=show_graph,
    )

    if segmentation_model_path is None:
        print("Training failed. No model produced.")
        sys.exit(1)

    return segmentation_model_path


def segment_image(
    image_path: Path, model_path: Path, output_dir: Path, device: torch.device, show_graph: bool = False
) -> None:
    """Segment a single image using the specified model.

    Args:
        image_path: Path to the image to segment.
        model_path: Path to the trained segmentation model.
        output_dir: Directory to save the segmentation results.
        device: Device to use for segmentation (CPU or GPU).
        show_graph: If True, display segmentation graphs.
    """
    print(f"Segmenting image {image_path} using model {model_path} on {device}")
    segment_single_image(image_path, model_path, output_dir, device=device, show_graph=show_graph)


def main() -> None:
    """Main function that parses arguments and runs the appropriate operation."""
    parser = argparse.ArgumentParser(prog="abcdetect", description="Skin Lesion Detection and Segmentation Tool")

    # Common arguments
    parser.add_argument(
        "--datasets-dir",
        "-d",
        type=is_valid_directory,
        default="datasets",
        help="Directory to store datasets (default: ./datasets)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=is_valid_directory,
        default="output",
        help="Directory to store output files (default: ./output)",
    )
    parser.add_argument(
        "-v", "--show-graph", action="store_true", help="Display matplotlib visualizations during execution"
    )
    parser.add_argument(
        "--device",
        "--dev",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use for model training/inference (default: auto-detect)",
    )

    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")

    # Download-only mode
    download_parser = subparsers.add_parser("download", help="Download the dataset only")
    download_parser.add_argument(
        "--force", "-f", action="store_true", help="Force re-download even if data exists"
    )

    # Train mode
    train_parser = subparsers.add_parser("train", help="Train the segmentation model")
    train_parser.add_argument(
        "--batch-size", "-k", type=int, default=32, help="Batch size for training (default: 32)"
    )
    train_parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=0,
        help="Number of data loader workers (default: 0 for no multiprocessing)",
    )

    # Segment mode
    segment_parser = subparsers.add_parser("segment", help="Segment an image using a trained model")
    segment_parser.add_argument("--image", "-i", type=str, required=True, help="Path to the image to segment")
    segment_parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Path to the model file (uses latest model in output directory if not specified)",
    )

    # Full demo mode
    demo_parser = subparsers.add_parser("demo", help="Run full demonstration (download, train, evaluate)")
    demo_parser.add_argument("--force-download", "-f", action="store_true", help="Force re-download the dataset")
    demo_parser.add_argument(
        "--batch-size", "-k", type=int, default=32, help="Batch size for training (default: 32)"
    )
    demo_parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=0,
        help="Number of data loader workers (default: 0 for no multiprocessing)",
    )

    # Parse arguments
    args = parser.parse_args()

    # If no mode is specified, show help and exit
    if args.mode is None:
        parser.print_help()
        sys.exit(0)

    # Create necessary directories
    datasets_dir = create_directory(args.datasets_dir)
    output_dir = create_directory(args.output_dir)

    # Determine device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Handle the different modes
    if args.mode == "download":
        download_dataset(datasets_dir, force=args.force)
        print("Dataset download completed.")

    elif args.mode == "train":
        segmentation_model_path = train_model(
            datasets_dir,
            output_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            show_graph=args.show_graph,
        )
        print(f"Training completed. Model saved at: {segmentation_model_path}")

    elif args.mode == "segment":
        # Find or use the provided model
        model_path = Path(args.model) if args.model else find_segmentation_model(output_dir)
        if model_path is None or not model_path.is_file():
            print("No segmentation model found. Please train a model first or specify a valid model file.")
            sys.exit(1)

        # Segment the image
        image_path = Path(args.image)
        if not image_path.is_file():
            print(f"Image not found: {image_path}")
            sys.exit(1)

        segment_image(image_path, model_path, output_dir, device=device, show_graph=args.show_graph)

    elif args.mode == "demo":
        # Full demo mode (download, train, evaluate)
        ham10k_image_path, ham10k_metadata_path, ham10k_masks_path = download_dataset(
            datasets_dir, force=getattr(args, "force_download", False)
        )

        segmentation_model_path = find_segmentation_model(output_dir)
        if segmentation_model_path is None:
            segmentation_model_path = train_segmentation_model(
                ham10k_image_path,
                ham10k_metadata_path,
                ham10k_masks_path,
                output_dir,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                show_graph=True,
            )

            if segmentation_model_path is None:
                print("No trained segmentation model found, cannot proceed.")
                sys.exit(1)

        # Evaluate the model
        evaluate_segmentation_model(ham10k_image_path, ham10k_masks_path, segmentation_model_path, device=device)


if __name__ == "__main__":
    main()
