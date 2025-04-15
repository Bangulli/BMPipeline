### Standard Imports ###
import pathlib as pl
import os
from PrettyPrint import *
import subprocess
### converter imports ###
from src.coin_nonchuv import NonCHUVCoiner
from src.filter_register import PatientPreprocessor
from src.rts2bids import RTS2BIDS
from src.nnUnet_data_preparation import DatasetConverter, DatasetReconverter
from src.nnUnet_predictor import Resegmentor
import logging
import sys

raw_set = pl.Path("/mnt/nas6/data/Target/batch_copy/rerun_test/dicom") # must be path to parent folder with patient subfolders/.
bids_set = pl.Path("/mnt/nas6/data/Target/batch_copy/rerun_test/bids") # must be path that doesnt exist, the script creates the target dir itself
processed_set = pl.Path('/mnt/nas6/data/Target/batch_copy/rerun_test/processed')
path_metadata = pl.Path('/home/lorenz/data/mrct1000_nobatch')
path_classification_results = path_metadata / "classification_results.csv" # path to the result csv of the sequence classifier
multimod_reseg = pl.Path('/mnt/nas6/data/Target/batch_copy/rerun_test/multimod')
reseg = pl.Path('/mnt/nas6/data/Target/batch_copy/rerun_test/singlemod')
nonchuv_data = pl.Path('/home/lorenz/BMPipeline/sequence_selected_nonchuv.xlsx')
bidsmap_path = pl.Path("/home/lorenz/BMPipeline/bidsmap_brainmets_modified_no_derived_no_se2d_excl_angio.yaml")

import logging
import sys

class StreamToLogger:
    """
    Redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = ''

    def write(self, message):
        if message != '\n':
            self.buffer += message
            if message.endswith('\n'):
                self.logger.log(self.level, self.buffer.rstrip())
                self.buffer = ''

    def flush(self):
        if self.buffer:
            self.logger.log(self.level, self.buffer.rstrip())
            self.buffer = ''





####### AT THIS POINT BIDSCOINER HAS BEEN RUN; THIS SCRIPT IS TO CONVERT NON-CHUV DATA, RTSTRUCTS AND MOVE EVERYTHING TO A NEW DIRECTORY THAT ONLY CONTAINS NECESSARY DATA
if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("/home/lorenz/BMPipeline/step0_main_full_pipeline.log"),
            logging.StreamHandler()
        ]
    )

    # Redirect stdout and stderr to logger
    stdout_logger = logging.getLogger('STDOUT')
    stderr_logger = logging.getLogger('STDERR')

    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)

    if not bids_set.is_dir(): # run the tml_dicom2bids with a template bidsmap, maps data and converts to bids format
        command = [
            "tml_dicom2bids_convert",
            "-i", raw_set,
            "-o", bids_set,
            "-t", bidsmap_path
        ]
        # Run the command
        subprocess.run(command)
    else:
        print("BIDS output directory exists, skipped conversion.")

    ## convert data missed by bidscoiner because it is not from CHUV
    coiner = NonCHUVCoiner(raw_set, bids_set, nonchuv_data, path_metadata/'sliceID_seriesPath_mapping.csv')
    coiner.execute(True)
    ## Convert RTstructs to Bids set
    converter = RTS2BIDS(raw_set, bids_set)
    converter.execute()
    os.makedirs(processed_set, exist_ok=True)
    ## Find relevant patients in Bids set and extract relevant dates and structures and then register everything
    register = PatientPreprocessor(bids_set, processed_set)
    register.execute()
    # Assumes dataset is the result of running src.filter_register.PatientPreprocessor
    # Converts every timepoint to a resg-nnUNet prediction case for multimodal and single modal
    DC = DatasetConverter(processed_set, multimod_reseg, reseg)
    DC.execute()
    # Assumes dataset is the result of src.nnUnet_data_preparation.DatasetConverter
    # Runs the prediction 
    RS = Resegmentor(multimod_reseg, reseg)
    RS.execute(task=['504', '524'])
    # Assumes the Resegmentor has been executed before
    # Pipes the result back into the clean set as a new subfolder 'mets'
    DRC = DatasetReconverter(processed_set, multimod_reseg, 'mets_task504-524')
    DRC.execute('524')
    DRC = DatasetReconverter(processed_set, reseg, 'mets_task504-524')
    DRC.execute('504')
