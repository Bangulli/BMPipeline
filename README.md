# BMPipeline

This repository contains the semi-Automated conversion of DICOM image series to a usable dataset for Brain-Metastasis tracking and prediction.
Processes the raw dataset to a filtered and registered set of logitudinal data, with segmentations at every timepoint

Filtration is based on [BIDSCoin](https://github.com/Donders-Institute/bidscoin) with a custom bidsmap in a fully automated workflow, no GUI interaction.
Check their [repository](https://github.com/Donders-Institute/bidscoin) and [documentation](https://bidscoin.readthedocs.io/en/latest/) for details on how to create a bidsmap for your usecase.

## Usage intsructions
Clone this repository and set the location as the working directory.
This repository is run on a __Python 3.10.12 venv environment__.
Create a new environment using:
``` bash
  python3 -m venv YOUR_ENVIRONMENT_NAME
```
Dependencies are found in the [Requirements](requirements.txt)
Install the dependencies using:
``` bash
  source YOUR_ENVIRONMENT_NAME/bin/activate
  pip install -r requirements.txt
```
The exact implementation for BIDSCoiner used is a containerized Docker image from [this repository](https://github.com/TranslationalML/tml_dicom2bids) download and build the container image according to the instructions in the repository after setting up the environment.

## Citation
If you use this repository in your projects, please cit:

```
Kuhn, L., Abler, D., Richiardi, J., Hottinger, A. F., Schiappacasse, L., Dunet, V., Depeursinge, A., Andrearczyk, V. "AI-based response assessment and prediction  in longitudinal imaging for brain metastases treated with stereotactic radiosurgery", in Learning with Longitudinal Medical Images and Data (LMID at MICCAI), 2025 (in press)
```

## Source
The [main](full_pipeline.py) script contains and calls all processor objects in the [src](src) directory. It is ready to run and configured with some basic settings. Just update the paths and you should be good to go
- [parallel_bidscoiner](src/parallel_bidscoiner.py): Runs the [BIDSCoiner](https://github.com/Donders-Institute/bidscoin) for batches of source data in parallel processing.
  - run_bidscoiner_multiprocess (function)
    - Arguments
      - source = pl.Path, the source dataset
      - target = pl.Path, the target dataset folder
      - bidsmap = pl.Path, the template bidsmap
      - n_jobs = int, default 5, how many batches can run in parallel
      - patients_per_batch = int, default 5, how many patients are processed in one Bidscoiner batch. If none will be infered as N_patients/n_jobs
  - BidscoinerJob (object)
    - Parallel processing job for one batch. dont touch, will be infered by run_bidscoiner_multiprocess function
- [coin_nonchuv](src/coin_nonchuv.py): Converts non-CHUV images to Bids
  - NonCHUVCoiner
    - converts manually selected non-CHUV images into a bids-like structure
    - Arguments
      - dicom_set, pl.Path, the path to the source dataset
      - bids_set, pl.Path, the path to the target directory, the bids output
      - ref_csv, pl.Path the csv or xlsx file with the custom selected non-CHUV data
      - map_csv, pl.Path the csv that maps UID to filepath in the dicom_set
    - Functions
      - execute. runs the process
- [rts2bids](src/rts2bids.py): Converts RTStruct, RTDose and corresponding CTs to Bids
  - RTS2BIDS (object)
    - converts RTStruct and to a limited degree RTDose files to a bids like format. RTDose is currently only matched and converted by filename. needs implementation for UID matching if possible
    - Arguments
      - raw_source, pl.Path, the source directory
      - bids_target, pl.Path, the target bids directory
    - Functions
      - execute: runs the process
- [filter_register](src/filter_register.py): Filters longitudinal data, assigns structs to mri, registers struct to mri and all mri to t0
  - FilterRegisterMain (object)
    - Selects patients from the bids directory according to an inclusion criterion and then matches RTs to MRIs by date, filteres the time series and registers everything in parallel processing. First runs all filters and sets up registration jobs for each patient that fullfills inclusion criteria, then runs the registrations in parallel
    - Arguments
      - bids_set, pl.Path, the BIDS dataset
      - clean_set, pl.Path, the output directory for the processed data
      - inclusion_criterion, dict, the filter criterions to include a patient in the clean set, default configuration uses any patient that has at least one RTStruct#
      - n_jobs, int how many patient registration jobs can run in parallel
    - Functions
      - execute: runs the process
  - PatientRegistrationJob (object)
  - executable object that gets setup by FilterRegisterMain and then runs in parallel using multiprocessing. 
  - Arguments
    - bids_set, pl.Path, the bids dataset
    - clean_set, pl.Path, the clean set output
    - pat, str, the patient file name
    - study_dict, a dictionary of strings with anatomical (MRI) study days as keys and RTS studies or None as values, if value for a key is None, only performs MR2MR reg, if not, performs CT2MR using mask registration and then MR2MR for all files.
    - keys, list of keys of the study dict
  - Functions
    - execute: runs the job 
- [nnUnet_data_preparation](src/nnUnet_data_preparation.py): Converts the output of filter register to an nnUNet compatible datastructure for resegmentation. does the reverse for the output of reseg
  - DatasetConverter (object)
    - converts the clean set to a nnUNet dataset to be predicted in reseg
    - Arguments
      - source_set, pl.Path, the clean directory
    - Functions
      - execute: run the process
        - takes task id and outptu dir for task as args
  - DatasetReconverter (object)
    - converts the nnUNet reseg output back into the clean set
    - Arguments
      - target_set, pl.Path, the clean set
      - source_set, pl.Path, the nnUNet output directory
      - met_dir_name, str, the directory name in which to put the reseg masks
    - Functions
      - execute: runs the process
        - takes a task id as variable
- [nnUnet_predictor](src/nnUnet_predictor.py): Runs nnUNet prediction
  - Resegmentor (object)
    - Runs the nnunet resegmentation process on an nnUNet directory.
    - No arguments
    - Functions
      - execute: executes the process
        - takes the dataset source directory and the task as input
        - the task can be list to do multiple or a string to do a single task
- [utils](src/utils.py): Image object type conversions for antsreg
  - used internally in filter_register not really important for anything else

## Modifications
To enable resegmentation pass the path to the nnUNet training directory as a string to the argument "nnUNet_dir" in the processors execute function.
To add more tasks modify the switch case blocks in [Resegmentor](src/nnUnet_predictor.py).
```python
  command_multi = [
                "nnUNet_predict",
                "-i", self.multimodal_set,
                "-o", self.multimodal_set.parent/(self.multimodal_set.name+'_predictions'),
                '-tr', 'nnUNetTrainerV2_Loss_DiceCE_noSmooth',
                '-ctr', 'nnUNetTrainerV2CascadeFullRes',
                '-m', '3d_fullres',
                '-p', 'nnUNetPlansv2.1',
                '-t', 'Task524_BrainMetsResegMultimod1to3'
                ]
```

Update the paths in the [main](full_pipeline.py) script:
```python
  raw_set = pl.Path("/mnt/nas6/data/Target/symlinked_batches_mrct_1000/known_no_issues") # path to the raw dataset
  bids_set = pl.Path("/mnt/nas6/data/Target/BMPipeline_full_rerun/targeted_rerun/bids") # destination path for the fileterd set in NIfTI
  processed_set = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/targeted_rerun/processed') # destination path for the registered set
  path_metadata = pl.Path('/home/lorenz/data/mrct1000_nobatch')
  path_classification_results = path_metadata / "classification_results.csv" # path to the result csv of the sequence classifier, used in nonCHUV2BIDS
  set504 = None # destination path for the nnUNet style dataset used in reseg for task 504
  set524 = None # destination path for the nnUNet style dataset used in reseg for task 524
  set502 = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/targeted_rerun/nnUNet_dataset') # destination path for the nnUNet style dataset used in reseg for task 502
  nonchuv_data = pl.Path('/home/lorenz/BMPipeline/sequence_selected_nonchuv.xlsx') # path to the manually selected conversion file
  bidsmap_path = pl.Path("/home/lorenz/BMPipeline/bidsmap_brainmets_modified_no_derived_no_se2d_excl_angio.yaml") # path to the bidsmap template used in conversion
```

If you want to run different resegmentation networks, update this block:
```python
  # Assumes dataset is the result of running src.filter_register.PatientPreprocessor
  # Converts every timepoint to a resg-nnUNet prediction case for multimodal and single modal
  DC = DatasetConverter(processed_set)
  DC.execute(set502, '502') # destination path and task identifier
  # Assumes dataset is the result of src.nnUnet_data_preparation.DatasetConverter
  # Runs the prediction 
  RS = Resegmentor()
  RS.execute(task=['502'], nnUNet_dir=set502) # task identifier. can be list of tasks to run multiple, e.g. when task 524 is used but not all studies have t2 images
  # Assumes the Resegmentor has been executed before
  # Pipes the result back into the clean set as a new subfolder 'mets'
  DRC = DatasetReconverter(processed_set, set502, 'mets_task_502')
  DRC.execute('502')
```
To enable more tasks expand the switch case inside the executor function.