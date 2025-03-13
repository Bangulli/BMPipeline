### Standard Imports ###
import src.log as log
import pathlib as pl
import subprocess
from PrettyPrint import *

### Setup Logging
LOGGER = Printer()
INFOformat = PPFormat([ColourText('blue'), Effect('bold'), Effect('underlined')])
# LOGGER.tagged_print("INFO", "BIDS output directory exists, skipped conversion.", INFOformat) info log template, cause its not in the pack yet

### Sequence Classification Imports ###
from SequenceClassification.dicom import *
from SequenceClassification.path import *

raw_set = pl.Path("/mnt/nas6/data/Target/symlinked_batches_mrct_1000/unknown") # must be path to parent folder with patient subfolders/.
bids_set = pl.Path("/mnt/nas6/data/Target/BIDS_mrct1000/unknown") # must be path that doesnt exist, the script creates the target dir itself
path_metadata = pl.Path('/home/lorenz/data/mrct1000_nobatch')
path_classification_results = path_metadata / "classification_results.csv" # path to the result csv of the sequence classifier

patient_count = log.count_folders(raw_set) 
LOGGER.tagged_print("INFO", f"There are {patient_count} patients.", INFOformat)



if not bids_set.is_dir(): # run the tml_dicom2bids with a template bidsmap, maps data and converts to bids format
    command = [
        "tml_dicom2bids_convert",
        "-i", raw_set,
        "-o", bids_set,
        "-t", "/home/lorenz/BMPipeline/bidsmap_brainmets_modified_no_derived_no_se2d.yaml"
    ]
    # Run the command
    subprocess.run(command)
else:
    LOGGER.tagged_print("INFO", "BIDS output directory exists, skipped conversion.", INFOformat)
