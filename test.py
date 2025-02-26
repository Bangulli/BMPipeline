### Standard Imports ###
import src.log as log
from src.rts2bids import RTS2BIDS
from src.filter_register import PatientPreprocessor

import pathlib as pl
import pandas as pd
import pydicom
import os
import subprocess
from PrettyPrint import *
import platipy
from platipy.dicom.io.rtstruct_to_nifti import convert_rtstruct
from platipy.dicom.io.rtdose_to_nifti import convert_rtdose
from platipy.imaging.tests.data import get_lung_dicom
import SimpleITK as sitk

raw_set = pl.Path('/mnt/nas6/data/Target/batch_copy/1k_subset_rtss')
bids_set = pl.Path('/mnt/nas6/data/Target/batch_copy/BIDS_1k_subset_rtss')
processed_set = pl.Path('/mnt/nas6/data/Target/batch_copy/BIDS_1k_subset_rtss_clean')

##################################### STEP 1: Convert to BIDS -> Move data from original struct to a new dataset folder in the BIDS specification
# on strict bidsmap to make it so only chuv data is converted
if not bids_set.is_dir(): # run the tml_dicom2bids with a template bidsmap, maps data and converts to bids format
    command = [
        "tml_dicom2bids_convert",
        "-i", raw_set,
        "-o", bids_set,
        "-t", "/home/lorenz/BMPipeline/bidsmap_brainmets_modified_no_derived_no_se2d.yaml"
    ]
    # Run the command
    subprocess.run(command)

# converter = RTS2BIDS(
#     raw_set,
#     bids_set
# )
# converter.execute()
os.makedirs(processed_set, exist_ok=True)
register = PatientPreprocessor(bids_set, processed_set)
register.execute()