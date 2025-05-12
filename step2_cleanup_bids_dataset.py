### Standard Imports ###
import pathlib as pl
import os
from PrettyPrint import *
import numpy as np

### converter imports ###
from src.coin_nonchuv import NonCHUVCoiner
from src.filter_register import PatientPreprocessor
from src.rts2bids import RTS2BIDS
from src.nnUnet_data_preparation import DatasetConverter, DatasetReconverter
from src.nnUnet_predictor import Resegmentor

raw_set = pl.Path("/mnt/nas6/data/Target/mrct1000_nobatch") # must be path to parent folder with patient subfolders/.
bids_set = pl.Path("/mnt/nas6/data/Target/BMPipeline_full_rerun/BIDS_mrct1000") # must be path that doesnt exist, the script creates the target dir itself
processed_set = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PROCESSED')
path_metadata = pl.Path('/home/lorenz/data/mrct1000_nobatch')
path_classification_results = path_metadata / "classification_results.csv" # path to the result csv of the sequence classifier
sequence_selection = pl.Path('/home/lorenz/BMPipeline/sequence_selected_nonchuv.xlsx')

####### AT THIS POINT BIDSCOINER HAS BEEN RUN; THIS SCRIPT IS TO CONVERT NON-CHUV DATA, RTSTRUCTS AND MOVE EVERYTHING TO A NEW DIRECTORY THAT ONLY CONTAINS NECESSARY DATA
if __name__ == '__main__':
    ## convert data missed by bidscoiner because it is not from CHUV
    # coiner = NonCHUVCoiner(raw_set, bids_set, sequence_selection, path_metadata/'sliceID_seriesPath_mapping.csv')
    # coiner.execute()
    # ## Convert RTstructs to Bids set
    # converter = RTS2BIDS(raw_set, bids_set)
    # converter.execute()

    # os.makedirs(processed_set, exist_ok=True)
    # # ## Find relevant patients in Bids set and extract relevant dates and structures and then register everything
    # register = PatientPreprocessor(bids_set, processed_set, inclusion_criterion={'studies >=': 4, 'obersvation period >=': 300, 'avg study interval <=': np.inf, 'rtstructs present >=': 1})
    # register.execute()

    clean_set = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PROCESSED')
    multimod_reseg = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/nnUNet_Datasets/task_524')
    reseg = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/nnUNet_Datasets/task_504')
    all_reseg = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/nnUNet_Dataset/task_502')

    # Assumes dataset is the result of running src.filter_register.PatientPreprocessor
    # Converts every timepoint to a resg-nnUNet prediction case for multimodal and single modal
    # DC = DatasetConverter(clean_set, None, all_reseg)
    # DC.execute(task='502')

    # # Assumes dataset is the result of src.nnUnet_data_preparation.DatasetConverter
    # # Runs the prediction 
    # RS = Resegmentor(multimod_reseg, reseg, all_reseg)
    # RS.execute(task=['502'])

    # # Assumes the Resegmentor has been executed before
    # # Pipes the result back into the clean set as a new subfolder 'mets'
    DRC = DatasetReconverter(clean_set, all_reseg, 'mets_task_502')
    DRC.execute('502')