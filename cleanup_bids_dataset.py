### Standard Imports ###
import pathlib as pl
import os
from PrettyPrint import *

### converter imports ###
from src.coin_nonchuv import NonCHUVCoiner
from src.filter_register import PatientPreprocessor
from src.rts2bids import RTS2BIDS

raw_set = pl.Path("/mnt/nas6/data/Target/mrct1000_nobatch") # must be path to parent folder with patient subfolders/.
bids_set = pl.Path("/mnt/nas6/data/Target/BIDS_mrct1000_nobatch") # must be path that doesnt exist, the script creates the target dir itself
processed_set = pl.Path('/mnt/nas6/data/Target/PROCESSED_mrct1000_nobatch')
path_metadata = pl.Path('/home/lorenz/data/mrct1000_nobatch')
path_classification_results = path_metadata / "classification_results.csv" # path to the result csv of the sequence classifier

####### AT THIS POINT BIDSCOINER HAS BEEN RUN; THIS SCRIPT IS TO CONVERT NON-CHUV DATA, RTSTRUCTS AND MOVE EVERYTHING TO A NEW DIRECTORY THAT ONLY CONTAINS NECESSARY DATA
if __name__ == '__main__':
    coiner = NonCHUVCoiner(raw_set, bids_set, pl.Path('/home/lorenz/data/Other/sequence_selection_nonchuv - Sheet1.csv'), path_metadata/'sliceID_seriesPath_mapping.csv')
    coiner.execute()

    converter = RTS2BIDS(raw_set, bids_set)
    converter.execute()

    os.makedirs(processed_set, exist_ok=True)

    register = PatientPreprocessor(bids_set, processed_set)
    register.execute()