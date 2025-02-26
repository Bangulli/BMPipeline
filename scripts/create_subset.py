import shutil
import os
from pathlib import Path

def copy_selected_folders(source_dir, destination_dir, folder_names):
    """
    Copies specified folders from source_dir to destination_dir if their names are in folder_names.

    :param source_dir: The directory containing the folders to copy.
    :param destination_dir: The target directory where folders will be copied.
    :param folder_names: A list of folder names to copy.
    """
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)

    if not source_dir.exists():
        print(f"Source directory '{source_dir}' does not exist.")
        return
    
    destination_dir.mkdir(parents=True, exist_ok=True)

    for folder_name in folder_names:
        source_folder = source_dir / folder_name
        destination_folder = destination_dir / folder_name

        if source_folder.exists() and source_folder.is_dir():
            print(f"Copying {source_folder} to {destination_folder}...")
            shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)
        else:
            print(f"Folder '{folder_name}' not found in {source_dir}.")

if __name__ == "__main__":
    # Example usage
    source_directory = "/mnt/nas6/data/Target/mrct1000_nobatch"  # Change this to your source directory
    destination_directory = "/mnt/nas6/data/Target/batch_copy/1k_subset_rtss"  # Change this to your destination directory
    folders_to_copy = ['sub-PAT-'+elem for elem in ['0014', '0154', '0155', '0420', '0305']]

    copy_selected_folders(source_directory, destination_directory, folders_to_copy)

#cp -r /mnt/nas6/data/Target/mrct* /mnt/nas6/data/Target/batch_copy/
