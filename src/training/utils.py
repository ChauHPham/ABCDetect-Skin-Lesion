import kagglehub
import os
import shutil

def download_kaggle_dataset(
        dataset_url:str, 
        desired_output_dir:str,
        force_download=False
    ):
    """
    Downloads a dataset from Kaggle and stores it in the specified directory.

    Args:
        dataset_url (str): The Kaggle dataset identifier (e.g., "tschandl/ham10000-lesion-segmentations").
        desired_output_dir (str): The directory where the dataset should be stored.
        force_download (bool, optional): If True, forces redownload even if the dataset is already present. Defaults to False.

    Returns:
        str: The local file path where the dataset was downloaded.

    Raises:
        Exception: If the specified output directory does not exist.
    """
    
    if not os.path.exists(desired_output_dir):
        raise Exception("Output directory does not exist")

    os.environ["KAGGLEHUB_CACHE"] = desired_output_dir

    dataset_path = kagglehub.dataset_download(dataset_url, force_download=force_download)

    return dataset_path   


def merge_folders(src_folder, dest_folder, remove_src_folder=False):
    if not os.path.exists(src_folder):
        print("Source directory does not exist")
        return ""

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    count = 0
    for filename in os.listdir(src_folder):
        count += 1
        filepath = os.path.join(src_folder, filename)
        shutil.move(filepath, dest_folder)

    if remove_src_folder:
        os.rmdir(src_folder)

    print(f"Successfully moved {count} contents from {src_folder} to {dest_folder}")
    return dest_folder

def split_dataset(directory_path, validation_percentage=0.1):
    val_dir = directory_path + '_validation'
    test_dir = directory_path + '_test'

    if os.path.exists(val_dir) and os.path.exists(test_dir):
        return test_dir, val_dir

    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    all_images = [f for f in os.listdir(directory_path)]

    num_val_images = int(len(all_images) * validation_percentage)

    # Split the images into validation and test sets (no random shuffle)
    val_images = all_images[:num_val_images]
    test_images = all_images[num_val_images:]

    # Move validation images to the validation directory
    for img in val_images:
        img_path = os.path.join(directory_path, img)
        shutil.move(img_path, os.path.join(val_dir, img))

    # Move test images to the test directory
    for img in test_images:
        img_path = os.path.join(directory_path, img)
        shutil.move(img_path, os.path.join(test_dir, img))

    # Delete the original merged images folder (after moving all files)
    shutil.rmtree(directory_path)

    print(f"Dataset split completed. {len(val_images)} images moved to validation, {len(test_images)} to test.")
    print(f"Original folder '{directory_path}' has been deleted.")

    return test_dir, val_dir

def move_all_files(src_folder, dst_folder, exclude_subdirectories=False):
    if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

    for filename in os.listdir(src_folder):
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)

        if os.path.isfile(src_path):  # Skip subdirectories
            shutil.move(src_path, dst_path)
            print(f"Moved {filename} to {dst_path}")
            
        elif not exclude_subdirectories:
            move_all_files(src_path, dst_path)