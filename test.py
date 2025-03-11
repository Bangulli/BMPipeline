### Standard Imports ###
from src.rts2bids import RTS2BIDS
from src.filter_register import PatientPreprocessor
from src.coin_nonchuv import NonCHUVCoiner
from SequenceClassification.dicom import *

import pathlib as pl
import os
import subprocess



if __name__ == '__main__':
    #raw_set = pl.Path("/mnt/nas6/data/Target/mrct1000_nobatch") # must be path to parent folder with patient subfolders/.
    bids_set = pl.Path("/mnt/nas6/data/Target/BIDS_mrct1000") # must be path that doesnt exist, the script creates the target dir itself
    processed_set = pl.Path('/home/lorenz/data/mrct1000_nobatch')

    # coiner = NonCHUVCoiner(raw_set, bids_set, pl.Path('/home/lorenz/data/Other/sequence_selection_nonchuv - Sheet1.csv'), path_metadata/'sliceID_seriesPath_mapping.csv')
    # coiner.execute()

    # converter = RTS2BIDS(raw_set, bids_set)
    # converter.execute()

    # os.makedirs(processed_set, exist_ok=True)

    register = PatientPreprocessor(bids_set, processed_set)
    register.execute()

    # /home/lorenz/.venv/bin/python /home/lorenz/BMPipeline/test.py