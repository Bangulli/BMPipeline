### Standard Imports ###
import pathlib as pl
import os
from PrettyPrint import *
import subprocess
### converter imports ###
from src.coin_nonchuv import NonCHUVCoiner
from src.filter_register import FilterRegisterMain
from src.rts2bids import RTS2BIDS
from src.nnUnet_data_preparation import DatasetConverter, DatasetReconverter
from src.nnUnet_predictor import Resegmentor
from src.parallel_bidscoiner import run_bidscoiner_multiprocess
import logging
import sys

raw_set = pl.Path("/mnt/nas6/data/Target/symlinked_batches_mrct_1000/known_no_issues") # must be path to parent folder with patient subfolders/.
bids_set = pl.Path("/mnt/nas6/data/Target/batch_copy/Gado_stability_test/bids_mp") # must be path that doesnt exist, the script creates the target dir itself
processed_set = pl.Path('/mnt/nas6/data/Target/batch_copy/Gado_stability_test/processed')
path_metadata = pl.Path('/home/lorenz/data/mrct1000_nobatch')
path_classification_results = path_metadata / "classification_results.csv" # path to the result csv of the sequence classifier
set504 = None
set524 = None
set502 = pl.Path('/mnt/nas6/data/Target/batch_copy/Gado_stability_test/nnUNet_dataset')
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

    ## convert data from raw to bids structure using bidscoiner in multiprocessing
    run_bidscoiner_multiprocess(raw_set, bids_set, bidsmap_path, 5, 1)

    ## convert data missed by bidscoiner because it is not from CHUV
    coiner = NonCHUVCoiner(raw_set, bids_set, nonchuv_data, path_metadata/'sliceID_seriesPath_mapping.csv')
    coiner.execute(True)
    ## Convert RTstructs to Bids set
    converter = RTS2BIDS(raw_set, bids_set)
    converter.execute()
    
    ## Find relevant patients in Bids set and extract relevant dates and structures and then register everything
    os.makedirs(processed_set, exist_ok=True)
    register = FilterRegisterMain(bids_set, processed_set, n_jobs = 5)
    register.execute()
    # Assumes dataset is the result of running src.filter_register.PatientPreprocessor
    # Converts every timepoint to a resg-nnUNet prediction case for multimodal and single modal
    DC = DatasetConverter(processed_set)
    DC.execute(set502, '502')
    # Assumes dataset is the result of src.nnUnet_data_preparation.DatasetConverter
    # Runs the prediction 
    RS = Resegmentor(set524, set504, set502)
    RS.execute(task=['502'])
    
    # Assumes the Resegmentor has been executed before
    # Pipes the result back into the clean set as a new subfolder 'mets'
    # DRC = DatasetReconverter(processed_set, multimod_reseg, 'mets_task504-524')
    # DRC.execute('524')
    DRC = DatasetReconverter(processed_set, set502, 'mets_task_502')
    DRC.execute('502')
