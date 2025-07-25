import os
import pathlib as pl
import csv
from datetime import datetime
import numpy as np
import SimpleITK as sitk
import re
from PrettyPrint import  *
from PrettyPrint.figures import ProgressBar
import pandas as pd

class TimePoint():
    """
    Data object representing a single timepoint
    Is essentially a dictionary with a T1 and optionally T2 and metastasis 
    """
    def __init__(self, t1: pl.Path, t2: pl.Path | None = None, mets: list[pl.Path] | None = None):
        if t2 is not None:
            if t1.parent != t2.parent:
                raise RuntimeError(f'T1 and T2 images must be from the same study to be considered at the same timepoint')
            self._t2 = t2.name
        else:
            self._t2 = None
            
        self._base = t1.parent.parent # gets rid of the anat dir
        self._t1 = t1.name
            
        self._mets = mets

    ##### Getters
    def get_mets(self) -> sitk.Image | tuple[sitk.Image | None, str]:
        if isinstance(self._mets, pl.Path):
            return self._generate_mets_image()
        elif isinstance(self._mets, list):
            return self._generate_multi_mets_image()
        else:
            return self._mets, '0001.nii.gz'
    
    def get_t1(self) -> tuple[pl.Path, str]:
        return self._base/'anat'/self._t1, '0000.nii.gz'
    
    def get_t2(self) -> tuple[pl.Path, str] | None:
        if self._t2 is not None:
            return self._base/'anat'/self._t2, '0002.nii.gz'
        return self._t2
    
    def get_base(self) -> pl.Path:
        return self._base
    
    ##### Setters
    def set_mets(self, mets: tuple[pl.Path, sitk.Image]):
        self._mets = mets

    ##### Private utils
    def _generate_mets_image(self) -> sitk.Image:
        # filter files in RTS directory for GTV PTV and Struct prefix
        files = []
        for folder in self._mets:
            files += [folder/file for file in os.listdir(folder) if ('GTV' in file or 'PTV' in file) and file.endswith('.nii.gz') and file.startswith('Struct_')]
        
        image = sitk.ReadImage(self._mets/files[0])
        for i in range(1, len(files)):
            image = sitk.Or(image, sitk.ReadImage(files[i]))
        
        return image
    
    def _generate_multi_mets_image(self) -> sitk.Image:
        # filter files in RTS directory for GTV PTV and Struct prefix
        image = None
        for met in self._mets:
            files = [file for file in os.listdir(met) if ('GTV' in file or 'PTV' in file) and file.endswith('.nii.gz') and file.startswith('Struct_')]
            for i in range(len(files)):
                if image is None:
                    image = sitk.ReadImage(met/files[i])
                else:
                    image = sitk.Or(image, sitk.ReadImage(met/files[i]))
        
        return image

class PatientTimeSeries():
    """
    Data Object representing a time series
    consist of a dictionary of timepoint objects with keys 't0', 't1' ... 'tN' that identify the order
    keys are stored internally in a list so the time series can be accessed by simple indexing
    """
    def __init__(self, t0: TimePoint):
        self._time_series={}
        self._time_series['t0'] = t0
        self._keys = []
        self._keys.append('t0')
        self._finalized = False

    ##### Public Methods
    def append(self, tp: TimePoint):
        """
        append a timepoint to the timeseries
        time points cant be removed but it is what it is, not necessary
        """
        if self._finalized:
            raise RuntimeError('Cant append to PatientTimeSeries object, has already been finalized')
        time_key = 't'+str(len(self._keys))
        self._keys.append(time_key)
        self._time_series[time_key] = tp

    def finalize(self):
        """
        to be called after all time points have been added
        runs over timeseries to convert the metastses to an sitk.Image object that for each timepoint to be saved for nnUnet Reseg
        """
        if self._finalized:
            raise RuntimeError('Cant finalize PatientTimeSeries object, has already been finalized')
        
        met_mask = None # variable to store the mask, if new masks are found just update this mask by binary or
        for tp in range(len(self)):
            timepoint = self[tp]
            mets = timepoint.get_mets()

            # switch case for handling mask image
            if met_mask is None and mets is None: # error if both are none, cause that means that there is not struct at t0
                raise RuntimeError('Got no mets at t0')
            elif met_mask is None and isinstance(mets, sitk.Image): # overwrite if met_mask is none and mets is not, cause that can only happen at t0
                met_mask = mets
            elif isinstance(met_mask, sitk.Image) and isinstance(mets, sitk.Image): # update if both are images, that means a follow up treatment has happened
                met_mask = sitk.Or(met_mask, mets)

            timepoint.set_mets(met_mask) # write the mask to the current timepoint

    ##### Python Builtins
    def __len__(self): # get len()
        return len(self._keys)
    
    def __getitem__(self, idx): # access by indexing
        return self._time_series[self._keys[idx]]
    
class DatasetConverter():
    """
    Processed set to nnUNet style set converter
    """
    def __init__(self, source_set: pl.Path):
        """
        source_set =  the source directory
        """
        self.source_set = source_set
        self.log = Printer()

    def execute(self, output, task='502'):
        """
        Runs the Dataset conversion, looks for all image files or RTS folders in the source set
        source set is expected to be Processed by filter_register.PatientProcessor
        output = pl.Path the output directory
        task =  string the task identifier 524 = multiclass seg with t2 (uses 504 with no t2 if t2 not found), 504 = multiclass seg without t2, 502 = single clas seg with t1
        """
        # init variables
        write_header = not (self.source_set/'nnUNet_mapping.csv').is_file()
        header = ['source_study_path', 'nnUNet_UID', 'nnUNet_set_dir']
        with open(self.source_set/f'nnUNet_mapping_task={task}.csv', 'a+') as mapping:
            if isinstance(output, list):
                for op in output:
                    os.makedirs(op, exist_ok=True)
            else: os.makedirs(output, exist_ok=True)
            if isinstance(output, list):
                self.target_timepoint_identifier = 0
                for op in output:
                        self.target_timepoint_identifier += len([file for file in os.listdir(op) if file.endswith('0000.nii.gz')])
            else: self.target_timepoint_identifier = len([file for file in os.listdir(output) if file.endswith('0000.nii.gz')]) ## used to uniquely identify timepoints in the output directory
            ## get processed files
            mapping.seek(0)
            processed_files = [pat.split(',')[0] for pat in mapping.readlines()] # gets the source paths from the mapping file

            ## set up map file, if it doesnt exist yet
            map_writer = csv.DictWriter(mapping, fieldnames=header)
            map_writer.writeheader()

            ## get all studies in set, each study is a unique timepoint for prediction
            patients = [pat for pat in os.listdir(self.source_set) if (self.source_set/pat).is_dir() and pat.startswith('sub-')]

            patients_processed = [pl.Path(pat).parent.name for pat in processed_files]
            patients_processed = [pat for pat in patients_processed if pat != patients_processed[-1]] # dont look at this too much, its just so that if this code fails in the middle of a patient, it doesnt get skipped next run because some studies from the pat were already processed
            
            patients = [pat for pat in patients if pat not in patients_processed] # finally filter out pats that are already done

            ## walk over set and for each patients order studies by time, to avoid mess ups by python directory parsing
            for pat in patients:
                ordered_studies = self._sort_directories(pat) # ensure correct order
                if not (self.source_set/pat/ordered_studies[0]/'rt').is_dir(): # sanity check to ensure t0 has RTSturcts
                    raise RuntimeError(f'Expected t0 to have RTs but could not find any for patient {pat} in study {ordered_studies[0]}')
                # build timeseries
                pat_time_series = PatientTimeSeries(self._get_timepoint(pat, ordered_studies[0]))
                for ses in range(1, len(ordered_studies)):
                    pat_time_series.append(self._get_timepoint(pat, ordered_studies[ses]))
                print("== parsing patient", pat)
                pat_time_series.finalize()

                csv_row = {}

                # iterate timeseries and write in nnUNet format
                for i in range(len(pat_time_series)):
                    nnUNet_UID = self._get_unique_mapped_name()
                    csv_row['nnUNet_UID'] = nnUNet_UID
                    tp = pat_time_series[i]
                    base_path = tp.get_base() # gets the path to the study, nnUNet predicts for each timepoint in the set so we need to organize all images and identify each timepoint uniquely
                    csv_row['source_study_path'] = str(base_path)

                    # extract timepoint data
                    t1 = tp.get_t1()
                    t2 = tp.get_t2()
                    mets = tp.get_mets()

                    if not isinstance(mets, tuple): # sanity check
                        raise RuntimeError(f'Expected a tuple of sitk.Image and str as TimePoint.get_mets() result but got {type(mets)} instead')

                    if task == '524':
                        if not isinstance(output, list): raise RuntimeError("Expected output to be a list of paths for task 524. [pl.Path(set524), pl.Path(set504)] to run pred when t2 is missing")
                        # symlink to destination based on modality presence
                        if t2 is not None:
                            csv_row['nnUNet_set_dir'] = output[0]
                            (output[0]/(nnUNet_UID+t1[1])).symlink_to(t1[0])
                            (output[0]/(nnUNet_UID+t2[1])).symlink_to(t2[0])
                            sitk.WriteImage(mets[0], output[0]/(nnUNet_UID+mets[1]))
                        else:
                            csv_row['nnUNet_set_dir'] = output[1]
                            (output[1]/(nnUNet_UID+t1[1])).symlink_to(t1[0])
                            sitk.WriteImage(mets[0], output[1]/(nnUNet_UID+mets[1]))
                    elif task == '504':
                        if output is None: raise ValueError("Did not receive valid path")
                        csv_row['nnUNet_set_dir'] = output
                        (output/(nnUNet_UID+t1[1])).symlink_to(t1[0])
                        sitk.WriteImage(mets[0], output/(nnUNet_UID+mets[1]))

                    elif task == '502':
                        if output is None: raise ValueError("Did not receive valid path")
                        csv_row['nnUNet_set_dir'] = output
                        (output/(nnUNet_UID+t1[1])).symlink_to(t1[0])
                        sitk.WriteImage(mets[0], output/(nnUNet_UID+mets[1]))

                    map_writer.writerow(csv_row)

    def _get_unique_mapped_name(self):
        name = f"timepoint{self.target_timepoint_identifier:06d}_"
        self.target_timepoint_identifier += 1
        return name
                

    def _get_timepoint(self, pat, ses):
        anat = os.listdir(self.source_set/pat/ses/'anat')
        
        if (self.source_set/pat/ses/'rt').is_dir():
            mets = [file for file in os.listdir(self.source_set/pat/ses/'rt') if (self.source_set/pat/ses/'rt'/file).is_dir()]
        else:
            mets = None

        t1 = [file for file in anat if file.endswith('T1w.nii.gz') and not file.startswith('MASK_')][0]

        t2 = [file for file in anat if file.endswith('T2w.nii.gz') and not file.startswith('MASK_')]
        if t2:
            t2 = t2[0]
        else:
            t2 = None

        t1 = self.source_set/pat/ses/'anat'/t1

        if t2 is not None:
            t2 = self.source_set/pat/ses/'anat'/t2
        
        if mets is not None:
            mets = [self.source_set/pat/ses/'rt'/met for met in mets]

        return TimePoint(t1, t2, mets)

            
    def _sort_directories(self, pat: str): # courtesy of chatgpt
        """
        Finds directories in the specified base_dir that match the pattern
        ses-yyyymmddhhmmss, parses the timestamp, and returns a list of directory
        names sorted in chronological order.
        """
        base_dir = self.source_set/pat
        # List all directories in the base directory
        dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and ((base_dir/d/'anat').is_dir() or (base_dir/d/'rt').is_dir())]
        pattern = r"^ses-(\d{14})$"
        matching_dirs = []
        for d in dirs:
            match = re.match(pattern, d)
            if match:
                timestamp_str = match.group(1)
                # Convert the timestamp string to a datetime object
                dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                matching_dirs.append((d, dt))
    
        # Sort the directories based on the datetime objects
        matching_dirs.sort(key=lambda x: x[1])
        # Return just the sorted directory names
        return [d for d, dt in matching_dirs]
    
class DatasetReconverter():
    def __init__(self, target_set: pl.Path, source_set:pl.Path, met_dir_name:str='mets'):
        self.target_set = target_set
        self.source_set = source_set.parent/(source_set.name+'_predictions')
        self.log = Printer()
        self.met_dir_name = met_dir_name

    def execute(self, mode='502'):
        mapping = pd.read_csv(self.target_set/f'nnUNet_mapping_task={mode}.csv')
        if self.source_set.is_dir():
            self._execute_directory(self.source_set, mapping, mode)
        else:
            self.log.fail(f'Could not find resegmentation results at path {self.source_set}')

    def _execute_directory(self, dir, mapping, mode):
        predictions = [file for file in os.listdir(dir) if file.endswith('.nii.gz')]
        #print(predictions)
        for prediction in predictions:
            encoded = prediction.split('.')[0]+'_'
            path = mapping.loc[mapping['nnUNet_UID'] == encoded, 'source_study_path']
            path = pl.Path(path.iloc[0])
            if '524' in mode:
                if path.parent.parent == self.target_set:
                    mask = sitk.ReadImage(dir/prediction)
                    os.makedirs(path/self.met_dir_name, exist_ok=True)
                    sitk.WriteImage(mask, path/self.met_dir_name/'metastasis_labels_3_class.nii.gz')
                    binary = sitk.GetArrayFromImage(mask)
                    binary[binary == 2] = 0 # ignore edema
                    binary[binary != 0] = 1 # binarize tumor & necrosis
                    binary = sitk.GetImageFromArray(binary)
                    binary.CopyInformation(mask)
                    sitk.WriteImage(binary, path/self.met_dir_name/'metastasis_labels_1_class.nii.gz')    
                else:
                    self.log.error(f'mapped path does not link to the target bids dataset')

            elif '504' in mode: # same as above
                if path.parent.parent == self.target_set:
                    mask = sitk.ReadImage(dir/prediction)
                    os.makedirs(path/self.met_dir_name, exist_ok=True)
                    sitk.WriteImage(mask, path/self.met_dir_name/'metastasis_labels_3_class.nii.gz')
                    binary = sitk.GetArrayFromImage(mask)
                    binary[binary == 2] = 0 # ignore edema
                    binary[binary != 0] = 1 # binarize tumor & necrosis
                    binary = sitk.GetImageFromArray(binary)
                    binary.CopyInformation(mask)
                    sitk.WriteImage(binary, path/self.met_dir_name/'metastasis_labels_1_class.nii.gz')    
                else:
                    self.log.error(f'mapped path does not link to the target bids dataset')


            elif '502' in mode:
                mask = sitk.ReadImage(dir/prediction)
                os.makedirs(path/self.met_dir_name, exist_ok=True)
                sitk.WriteImage(mask, path/self.met_dir_name/'metastasis_labels_1_class.nii.gz') 
        