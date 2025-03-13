### Standard Imports ###
# from src.rts2bids import RTS2BIDS
# from src.filter_register import PatientPreprocessor
# from src.coin_nonchuv import NonCHUVCoiner
# from SequenceClassification.dicom import *
# from src.nnUnet_data_preparation import DatasetConverter

# import pathlib as pl
# import os
# import subprocess



# if __name__ == '__main__':
#     DC = DatasetConverter(
#         source_set=pl.Path('/mnt/nas6/data/Target/batch_copy/nnunet_prep_experiments/source'),
#         target_set=pl.Path('/mnt/nas6/data/Target/batch_copy/nnunet_prep_experiments/target'), 
#         target_set_multimod=pl.Path('/mnt/nas6/data/Target/batch_copy/nnunet_prep_experiments/target_multimod')
#     )
#     DC.execute()

### Standard Imports ###
import pathlib as pl
import os
from PrettyPrint import *

### converter imports ###
from src.coin_nonchuv import NonCHUVCoiner
from src.filter_register import PatientPreprocessor
from src.rts2bids import RTS2BIDS
bids_set = pl.Path("/mnt/nas6/data/Target/batch_copy/struct filter exp/bids") # must be path that doesnt exist, the script creates the target dir itself
processed_set = pl.Path('/mnt/nas6/data/Target/batch_copy/struct filter exp/clean')

####### AT THIS POINT BIDSCOINER HAS BEEN RUN; THIS SCRIPT IS TO CONVERT NON-CHUV DATA, RTSTRUCTS AND MOVE EVERYTHING TO A NEW DIRECTORY THAT ONLY CONTAINS NECESSARY DATA
if __name__ == '__main__':
    ## convert data missed by bidscoiner because it is not from CHUV
    # coiner = NonCHUVCoiner(raw_set, bids_set, pl.Path('/home/lorenz/data/Other/sequence_selected_nonchuv.csv'), path_metadata/'sliceID_seriesPath_mapping.csv')
    # coiner.execute()
    ## Convert RTstructs to Bids set
    # converter = RTS2BIDS(raw_set, bids_set)
    # converter.execute()

    os.makedirs(processed_set, exist_ok=True)
    ## Find relevant patients in Bids set and extract relevant dates and structures and then register everything
    register = PatientPreprocessor(bids_set, processed_set)
    register.execute()