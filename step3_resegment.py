### Standard Imports ###
from src.nnUnet_data_preparation import DatasetConverter, DatasetReconverter
from src.nnUnet_predictor import Resegmentor
import pathlib as pl


clean_set = pl.Path('/mnt/nas6/data/Target/PROCESSED_mrct1000_nobatch')
multimod_reseg = pl.Path('/mnt/nas6/data/Target/nnUNet_Datasets/multimod')
reseg = pl.Path('/mnt/nas6/data/Target/nnUNet_Datasets/singlemod')
all_reseg = pl.Path('/mnt/nas6/data/Target/nnUNet_Datasets/all_singlemod')

# Assumes dataset is the result of running src.filter_register.PatientPreprocessor
# Converts every timepoint to a resg-nnUNet prediction case for multimodal and single modal
DC = DatasetConverter(clean_set, multimod_reseg, reseg)
DC.execute()

# Assumes dataset is the result of src.nnUnet_data_preparation.DatasetConverter
# Runs the prediction 
RS = Resegmentor(None, None, all_reseg)
RS.execute(task=['504', '524'])

# Assumes the Resegmentor has been executed before
# Pipes the result back into the clean set as a new subfolder 'mets'
DRC = DatasetReconverter(clean_set, multimod_reseg, 'mets_task504-524')
DRC.execute('524')

DRC = DatasetReconverter(clean_set, reseg, 'mets_task504-524')
DRC.execute('504')
