# BMPipeline

This repository contains the semi-Automated conversion of DICOM image series to a usable dataset for Brain-Metastasis tracking and prediction

## Usage intsructions
tbd: make a list of dependencies for install

### Run all at once
To run the entire pipeline just execute the script step0_main_full_pipeline.py

### Step-by-Step (Recommended)
Running Step-by-Step is recommended because the Pipline is still mostly experimental and may have instabilities that lead to crashes. Most processor objects have internal progress logging so re-running should be smoothe, but some could be irrecoverable if unkown issues occur.

All scripts are in a runable state, just update the paths at the top of the scripts to your structure.

- [Step 1](step1_run_bidscoiner.py)
  - Handles automated sequence selection with BIDSCoiner and a custom template bidsmap
- [Step 2](step2_cleanup_bids_dataset.py)
  - Handles semi automated conversion of Non-CHUV data with manually selected csv
  - Handles conversion of RTStruct and RTDose DICOMs
  - Handles temporal filtering and organization of data
  - Handles automated registration and patient selection
- [Step 3](step3_resegment.py)
  - Handles organization of data in nnUNet compatible format
  - Handles nnUNet based resegmentation