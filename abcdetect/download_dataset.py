import os
import shutil
from pathlib import Path

import kagglehub


def download_kaggle_dataset(identifier: str, output_directory: Path, force_download: bool = False) -> Path:
    """
    Downloads a dataset from Kaggle and stores it in the specified directory.

    This function uses the `kagglehub` library to download the dataset. The dataset is identified by
    its Kaggle identifier (e.g., "tschandl/ham10000-lesion-segmentations").
    The downloaded dataset will be stored in the specified output directory's "/datasets" subdirectory.
    If the output directory does not exist, it will be created.

    Args:
        identifier: The Kaggle dataset identifier.
        output_directory: The directory where the dataset should be stored.
        force_download: If True, re-downloads even if the dataset is already present, otherwise skips the download.

    Returns:
        The local file path where the dataset was downloaded.

    Raises:
        ValueError: If the specified output directory is not a directory.
    """

    if not output_directory.exists():
        output_directory.mkdir(parents=True)
    elif not output_directory.is_dir():
        raise ValueError(f"{output_directory} is not a directory")

    os.environ["KAGGLEHUB_CACHE"] = str(output_directory.absolute())

    print(f"Downloading dataset {identifier} to {output_directory.absolute()}...")
    dataset_path = kagglehub.dataset_download(identifier, force_download=force_download)
    return Path(dataset_path).absolute()


def merge_dirs(src_dir: Path, dst_dir: Path, delete_src_dir: bool = False) -> None:
    """Merges the contents of one directory into another.

    Args:
        src_dir: The source directory containing files to be moved.
        dst_dir: The destination directory where files will be moved.
        delete_src_dir: If True, deletes the source directory after moving its contents.

    Raises:
        ValueError: If the source directory does not exist or is not a directory,
            or if the destination is not a directory.
    """
    if not src_dir.is_dir():
        raise ValueError(f"Source directory {src_dir} does not exist or is not a directory")

    if not dst_dir.exists():
        dst_dir.mkdir(parents=True)
    elif not dst_dir.is_dir():
        raise ValueError(f"Destination {dst_dir} is not a directory")

    count = 0
    for path in src_dir.glob("*"):
        count += 1
        shutil.move(path, dst_dir)

    if delete_src_dir:
        src_dir.rmdir()

    print(f"Successfully moved {count} contents from {src_dir} to {dst_dir}.")


def move_files(src_dir: Path, dst_dir: Path, exclude_subdirectories: bool = False) -> None:
    """Moves files from one directory to another.

    Args:
        src_dir: The source directory containing files to be moved.
        dst_dir: The destination directory where files will be moved.
        exclude_subdirectories: If True, skips moving files from subdirectories.

    Raises:
        ValueError: If the source directory does not exist or is not a directory,
            or if the destination is not a directory.
    """
    if not src_dir.is_dir():
        raise ValueError(f"Source directory {src_dir} does not exist or is not a directory")

    if not dst_dir.exists():
        dst_dir.mkdir(parents=True)
    elif not dst_dir.is_dir():
        raise ValueError(f"Destination {dst_dir} is not a directory")

    for path in src_dir.glob("*"):
        if path.is_dir() and exclude_subdirectories:
            continue  # Skip subdirectories

        shutil.move(path, dst_dir)
        print(f"Successfully moved file {path.name}.")


def download(datasets_dir: Path, force: bool = False) -> tuple[Path, Path, Path]:
    """
    Downloads the HAM10000 dataset and its segmentation masks from Kaggle.

    Args:
        datasets_dir: The base directory where the datasets will be downloaded.
        force: Whether to force a re-download when the dataset is already present locally.

    Returns:
        A tuple containing the paths to the downloaded image dataset, metadata, and segmentation masks.

    Raises:
        ValueError: If the specified datasets directory does not exist or is not a directory.
    """
    if not datasets_dir.exists() or not datasets_dir.is_dir():
        raise ValueError(f"{datasets_dir} does not exist or is not a directory")

    ham10k_dataset_kaggle_identifier = "kmader/skin-cancer-mnist-ham10000"
    ham10k_masks_kaggle_identifier = "tschandl/ham10000-lesion-segmentations"

    ham10k_image_path = datasets_dir / "kmader" / "images"
    ham10k_metadata_path = datasets_dir / "kmader" / "metadata"
    ham10k_masks_path = datasets_dir / "tschandl" / "masks"

    if force:
        print("Removing old HAM10000 dataset files.")
        shutil.rmtree(ham10k_image_path, ignore_errors=True)
        shutil.rmtree(ham10k_metadata_path, ignore_errors=True)
        shutil.rmtree(ham10k_masks_path, ignore_errors=True)

    # Download the HAM10000 dataset if it doesn't exist
    if force or not ham10k_image_path.exists() or not ham10k_metadata_path.exists():
        print("Downloading HAM10000 dataset...")
        ham10k_dataset_location = download_kaggle_dataset(
            ham10k_dataset_kaggle_identifier, datasets_dir, force_download=force
        )

        merge_dirs(ham10k_dataset_location / "HAM10000_images_part_1", ham10k_image_path, delete_src_dir=True)
        merge_dirs(ham10k_dataset_location / "HAM10000_images_part_2", ham10k_image_path, delete_src_dir=True)
        move_files(ham10k_dataset_location, ham10k_metadata_path, exclude_subdirectories=True)

    # Download the segmentation masks dataset if it doesn't exist
    if force or not ham10k_masks_path.exists():
        print("Downloading HAM10000 segmentation masks dataset...")
        ham10k_masks_dataset_location = download_kaggle_dataset(
            ham10k_masks_kaggle_identifier, datasets_dir, force_download=force
        )

        merge_dirs(
            ham10k_masks_dataset_location / "HAM10000_segmentations_lesion_tschandl",
            ham10k_masks_path,
            delete_src_dir=True,
        )

    # Remove the temporary /datasets directory
    if (datasets_dir / "datasets").exists():
        shutil.rmtree(datasets_dir / "datasets", ignore_errors=True)

    print("HAM10000 dataset and segmentation masks downloaded successfully.")
    return ham10k_image_path, ham10k_metadata_path, ham10k_masks_path
