# BMPipeline

This repository contains the semi-Automated conversion of DICOM image series to a usable dataset for Brain-Metastasis tracking and prediction.

## Usage intsructions
Dependencies are found in the [Requirements](requirements.txt)

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
By either updating the current ones or adding new tasks to the switch case.

Notes: TotalSegmentator can be a bit inconsistent and not immediately usable.

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
  RS = Resegmentor(set524, set504, set502)
  RS.execute(task=['502']) # task identifier. can be list of tasks to run multiple, e.g. when task 524 is used but not all studies have t2 images
  # Assumes the Resegmentor has been executed before
  # Pipes the result back into the clean set as a new subfolder 'mets'
  DRC = DatasetReconverter(processed_set, set502, 'mets_task_502')
  DRC.execute('502')
```

## Config
The [main](full_pipeline.py) script contains and calls all processor objects in the [src](src) directory.
- [parallel_bidscoiner](src/parallel_bidscoiner.py): Runs the bidscoiner for batches of source data in multiprocessing.
- [coin_nonchuv](src/coin_nonchuv.py): Converts non-CHUV images to Bids
- [rts2bids](src/rts2bids.py): Converts RTStruct, RTDose and corresponding CTs to Bids
- [filter_register](src/filter_register.py): Filters longitudinal data, assigns structs to mri, registers struct to mri and all mri to t0
- [nnUnet_data_preparation](src/nnUnet_data_preparation.py): Converts the output of filter register to an nnUNet compatible datastructure for resegmentation. does the reverse for the output of reseg
- [nnUnet_predictor](src/nnUnet_predictor.py): Runs nnUNet prediction
- [utils](src/utils.py): Image object type conversions for antsreg