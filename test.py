### Standard Imports ###
from src.rts2bids import RTS2BIDS
from src.filter_register import PatientPreprocessor
from src.coin_nonchuv import NonCHUVCoiner
from SequenceClassification.dicom import *

import pathlib as pl
import os
import subprocess



if __name__ == '__main__':
    raw_set = pl.Path('/mnt/nas6/data/Target/batch_copy/1k_subset_rtss')
    bids_set = pl.Path('/mnt/nas6/data/Target/batch_copy/BIDS_1k_subset_rtss')
    processed_set = pl.Path('/mnt/nas6/data/Target/batch_copy/BIDS_1k_subset_rtss_clean')
    path_metadata = pl.Path('/mnt/nas6/data/Target/batch_copy/1k_subset_rtss')

    extract_metadata(raw_set, path_metadata)

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

    # coiner = NonCHUVCoiner(raw_set, bids_set, pl.Path('/home/lorenz/data/Other/sequence_selection_nonchuv - Sheet1.csv'), path_metadata/'sliceID_seriesPath_mapping.csv')
    # coiner.execute()

    # converter = RTS2BIDS(raw_set, bids_set)
    # converter.execute()

    os.makedirs(processed_set, exist_ok=True)

    register = PatientPreprocessor(bids_set, processed_set)
    register.execute()

    # /home/lorenz/.venv/bin/python /home/lorenz/BMPipeline/test.py