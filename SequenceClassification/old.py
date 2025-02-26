### Sequence Classif Imports ###
from file_ops.path import *
from file_ops.dicom import *
import pathlib as pl
import subprocess

### External Imports ###
import numpy as np
import torch as tc
import matplotlib.pyplot as plt
import SimpleITK as sitk

### Internal Imports ###
from registration import input_output as io
from registration import registration as reg
from registration import warping as w
from registration import configs

path_data = pl.Path("/mnt/nas4/datasets/ToReadme/Target/MRI_Target") # path to the dataset, should contain folders of patients/subfolder sutdy/subfolder series/dcm files
path_metadata = pl.Path("/home/lorenz/data/20Metadata") # path to the output, this is where the csv with the metadata is stored
path_classification_results = path_metadata / "classification_results.csv" # path to the result csv of the sequence classifier

# report population count
patient_count = count_folders(path_data) 
print(f"There are {patient_count} patients.")

# extract metadata
extract_metadata(path_data,path_metadata)

# Classify sequences if not done already
if not path_classification_results.is_file():
    command = [
        "docker", "run",
        "-v", f"{path_metadata}:/input/files",
        "-v", f"{path_metadata}:/output",
        "sequence-classification-params-v4",
        "-f", "series-meta-data.csv"
    ]
    # Run the command
    subprocess.run(command)

# Load results
df = pd.read_csv(path_classification_results)
# Remove unwanted column and move PatientID for simplicity of csv reading
df.drop(['Unnamed: 0'], axis=1, inplace=True)
column_to_move = df.pop("PatientID")
df.insert(1, "PatientID", column_to_move)
# Fill the AcquisitionDate for series that don't have it (the subset of Matthieu doesn't have accessionNumber so I can't. I could get the dates from the folders)
if df['AccessionNumber'].isnull().values.any():
    df['AcquisitionDateFill'] = df['AcquisitionDate']
else:
    df['AcquisitionDateFill'] = df.groupby('AccessionNumber')['AcquisitionDate'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
# Reorder columns to place AcquisitionDateFill next to AcquisitionDate
cols = df.columns.tolist()
ad_index = cols.index('AcquisitionDate')
cols.insert(ad_index + 1, cols.pop(cols.index('AcquisitionDateFill')))
df = df[cols]

print(f"{len(pd.unique(df['PatientID']))} patients and {len(pd.unique(df['AcquisitionDateFill']))} dates")
# df = df[['SeriesDescription','AcquisitionDate','MRAcquisitionType','Modality', 'PatientID', 'prediction_class_category','prediction_class']]
print(f"all sequences: {len(df)}")

print(df.columns)