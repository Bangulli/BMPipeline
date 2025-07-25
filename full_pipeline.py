### Standard Imports ###
import pathlib as pl
import os
import time
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

raw_set = pl.Path("/mnt/nas6/data/Target/mrct1000_nobatch") # must be path to parent folder with patient subfolders/.
bids_set = pl.Path("/mnt/nas6/data/Target/Lorenz_testrun/BIDS") # must be path that doesnt exist, the script creates the target dir itself
processed_set = pl.Path('/mnt/nas6/data/Target/Lorenz_testrun/PROCESSED') #  the location of the bids structure output
metadata_map = pl.Path('/home/lorenz/BMPipeline/sliceID_seriesPath_mapping.csv')# path to the secondary result of running sequence classifier: ID to path mapping
set504 = None
set524 = None
set502 = pl.Path('/mnt/nas6/data/Target/Lorenz_testrun/UNet')
nonchuv_data = pl.Path('/home/lorenz/BMPipeline/sequence_selected_nonchuv.xlsx')
bidsmap_path = pl.Path("/home/lorenz/BMPipeline/bidsmap_brainmets_modified_no_derived_no_se2d_excl_angio.yaml")

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
    start = time.time()
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("full_pipeline.log"),
            logging.StreamHandler()
        ]
    )
    # Redirect stdout and stderr to logger
    stdout_logger = logging.getLogger('STDOUT')
    stderr_logger = logging.getLogger('STDERR')
    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)
    ## convert data from raw to bids structure using bidscoiner in multiprocessing
    #run_bidscoiner_multiprocess(raw_set, bids_set, bidsmap_path, 5, 1) some weird interaction between the multiprocessing and the bidscoiner leads to errors in some cases. to be fixed
    ## run single thread bidscoiner. Very slow but stable.
    command = [
                "tml_dicom2bids_convert",
                "-i", raw_set,
                "-o", bids_set,
                "-t", bidsmap_path
                ]
    # Run the command
    subprocess.run(command)
    #raise RuntimeError('stop')
    milestone=time.time()
    ## report milestone runtime
    min, sec = divmod(milestone-start, 60)
    hr, min = divmod(min, 60)
    d, hr = divmod(hr, 24)
    print(f"Milestone: Running bidscoiner took {d}days {hr}h {min}min {round(sec)}s")
    ## convert data missed by bidscoiner because it is not from CHUV
    coiner = NonCHUVCoiner(raw_set, bids_set, nonchuv_data, metadata_map)
    coiner.execute(True)
    ## report milestone runtime
    min, sec = divmod(time.time()-milestone, 60)
    hr, min = divmod(min, 60)
    d, hr = divmod(hr, 24)
    milestone=time.time()
    print(f"Milestone: Running NonCHUVCoiner took {d}days {hr}h {min}min {round(sec)}s")
    ## Convert RTstructs to Bids set
    converter = RTS2BIDS(raw_set, bids_set)
    converter.execute()
    ## report milestone runtime
    min, sec = divmod(time.time()-milestone, 60)
    hr, min = divmod(min, 60)
    d, hr = divmod(hr, 24)
    milestone=time.time()
    print(f"Milestone: Running RTS2BIDS took {d}days {hr}h {min}min {round(sec)}s")
    ## Find relevant patients in Bids set and extract relevant dates and structures and then register everything
    os.makedirs(processed_set, exist_ok=True)
    register = FilterRegisterMain(bids_set, processed_set, n_jobs = 5)
    register.execute()
    ## report milestone runtime
    min, sec = divmod(time.time()-milestone, 60)
    hr, min = divmod(min, 60)
    d, hr = divmod(hr, 24)
    milestone=time.time()
    print(f"Milestone: Running FilterRegister took {d}days {hr}h {min}min {round(sec)}s")
    # Assumes dataset is the result of running src.filter_register.PatientPreprocessor
    # Converts every timepoint to a resg-nnUNet prediction case for multimodal and single modal
    DC = DatasetConverter(processed_set)
    DC.execute(set502, '502')
    # Assumes dataset is the result of src.nnUnet_data_preparation.DatasetConverter
    # Runs the prediction 
    RS = Resegmentor()
    RS.execute(task=['502'], nnUNet_dir=set502)
    # Assumes the Resegmentor has been executed before
    # Pipes the result back into the clean set as a new subfolder 'mets'
    DRC = DatasetReconverter(processed_set, set502, 'mets_task_502')
    DRC.execute('502')
    ## report milestone runtime
    min, sec = divmod(time.time()-milestone, 60)
    hr, min = divmod(min, 60)
    d, hr = divmod(hr, 24)
    milestone=time.time()
    print(f"Milestone: Running Resegmentation took {d}days {hr}h {min}min {round(sec)}s")
    ## report total runtime
    min, sec = divmod(time.time()-start, 60)
    hr, min = divmod(min, 60)
    d, hr = divmod(hr, 24)
    print(f"Running the full pipeline took {d}days {hr}h {min}min {round(sec)}s")
