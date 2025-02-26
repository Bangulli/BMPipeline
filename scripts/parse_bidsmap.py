from file_ops.path import *
from file_ops.dicom import *
import pathlib as pl
import subprocess
import timeit
import numpy as np
import yaml, csv

# load all tsvs from all patients put into a csv to compare with the gt excel
# check the existance of all necessary files
def extract_provenance(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    
    orig_names = []
    new_names = []
    
    # Navigate to the DICOM -> anat section
    if 'DICOM' in data and 'anat' in data['DICOM']:
        for item in data['DICOM']['anat']:
            if isinstance(item, dict) and 'provenance' in item:
                name = item['provenance']
                name = name.replace('/mnt/nas4/datasets/ToReadme/Target/MRI_Target/', '')
                orig_names.append(name)
            if isinstance(item, dict) and 'meta' in item:
                meta_data = item['meta']
                if isinstance(meta_data, dict) and 'Provenance' in meta_data:
                    new_names.append(meta_data['Provenance'])
    return orig_names, new_names

def save_to_csv(provenance_data, csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Patient", "Study", "Series"])
        
        for prov in provenance_data:
            parts = prov.split('/')
            parts = parts[:3]
            if len(parts) == 3:
                writer.writerow(parts)

if __name__ == "__main__":
    target_path = pl.Path('/mnt/nas6/data/Target/BIDS/modified_yaml_20250214/code/bidscoin/bidsmap.yaml')
    csv_file = pl.Path('/home/lorenz/data/90 bidsmap results/mapping.csv')
    orig, rename = extract_provenance(target_path)
    save_to_csv(orig, csv_file)
    print(f"CSV saved to {csv_file}")

