import os

def count_folders(directory):
    # List all entries in the directory
    entries = os.listdir(directory)
    # Filter out directories
    folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
    return len(folders)