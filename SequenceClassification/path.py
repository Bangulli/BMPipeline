import os
import re

# return the number of directories at a path location
def count_folders(directory):
    # List all entries in the directory
    entries = os.listdir(directory)
    # Filter out directories
    folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
    return len(folders)

# Helper function to extract UID from the filename
def extract_uid_from_filename(filename):
    match = re.search(r'(\d+\.\d+\.\d+\.\d+)', filename)
    if match:
        return match.group(1)
    return None

# Compare UIDs in dir and return list of files that contain the most common UID
def filter_mismatched_files(dcm_files_in_dir):
    if not dcm_files_in_dir:
        return dcm_files_in_dir  # Return the original list if empty
    
    # Extract the UID from all filenames in the directory
    uids = [extract_uid_from_filename(file.name) for file in dcm_files_in_dir]
    
    # Count occurrences of each UID
    uid_counts = {uid: uids.count(uid) for uid in set(uids) if uid}
    if not uid_counts:
        return dcm_files_in_dir
    
    # Find the most common UID
    common_uid = max(uid_counts, key=uid_counts.get)
    
    # Filter out files with a UID that does not match the most common UID
    filtered_files = [file for file, uid in zip(dcm_files_in_dir, uids) if uid == common_uid]
    if len(dcm_files_in_dir) - len(filtered_files)>0:
        mis = [elem for elem in dcm_files_in_dir if elem not in filtered_files]
        print(f"Filtered out {len(dcm_files_in_dir) - len(filtered_files)} files with mismatched UIDs in series {mis}")
    
    return filtered_files

# returns a path with a name pattern, that does not exist in the given directory
def unique_path(directory, name_pattern):
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return path