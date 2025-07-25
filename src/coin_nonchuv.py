import pandas as pd
import os
from PrettyPrint import *
import pathlib as pl
import SimpleITK as sitk

class NonCHUVCoiner():
    """
    Manually selected non-chuv data converter object
    """
    def __init__(self, 
                 dicom_set, 
                 bids_set, 
                 ref_csv, 
                 map_csv, 
                ):
        """
        dicom_set = pl.Path object, the source directory
        bids_set = pl.Path object, the output directory
        ref_csv = string or path object to the selction file. can be a csv or xlsx 
        map_csv = string or path object to the mapping csv, that relates UID and path
        """
        self.dicom_set = dicom_set
        self.bids_set = bids_set
        self.ref = pd.read_csv(ref_csv) if ref_csv.name.endswith('.csv') else pd.read_excel(ref_csv)
        self.map = pd.read_csv(map_csv, sep=';')
        self.log = Printer(log_type='txt', log_prefix='NonCHUVCoiner')

    def execute(self, only_relevant=False):
        """
        Execute the conversion
        only_relevant = bool flag to skip patients that are not in the output directory. saves a lot of log messages
        """
        self.ref = self.ref.dropna(subset=["SelectedSequence"])

        bids_pats = [pat.split('-')[-1] for pat in os.listdir(self.bids_set) if pat.startswith('sub-PAT')]
        # create progress log in case of error during coining
        with open(self.bids_set/'nonchuv_coiner_progress.txt', 'a+') as progfile:
            progfile.seek(0)
            processed = progfile.readlines()
            processed = [uid.replace('\n', '') for uid in processed]
            for idx, row in self.ref.iterrows():
                # get references
                UID = row['SeriesInstanceUID']
                if UID in processed:
                    continue
                
                    
                pat = row['PatientID'].replace('-', '')
                if only_relevant:
                    if pat not in bids_pats:
                        continue
                date = row['AcquisitionDate']
                pred = row['prediction_class_category']
                # init result vars
                map_row_index = None
                mapped_path = None
                for i, r in self.map.iterrows():
                    if r.iloc[1] == UID:
                        map_row_index = i
                        mapped_path = pl.Path(r.iloc[0]).parent
                        break
                # log if matching fails
                if map_row_index is None:
                    self.log.fail(f'Could not find UID mapping for {UID} in Patient {pat} on day {date}')
                # read dicom
                dcm_reader = sitk.ImageSeriesReader()
                dcm_files = dcm_reader.GetGDCMSeriesFileNames(mapped_path)
                dcm_reader.SetFileNames(dcm_files)
                image = dcm_reader.Execute()
                arr = sitk.GetArrayFromImage(image)
                if (arr<0).any():
                    print('found hypointense')
                    arr[arr<0]=0
                    image_new = sitk.GetImageFromArray(arr)
                    image_new.CopyInformation(image)
                    image = image_new
                # write nifti
                output = self._gen_bids_filename(pat, mapped_path, pred)
                sitk.WriteImage(image, output)
                # drop rows that are already matched to speed up as we go along
                self.map.drop(map_row_index) 
                progfile.write((str(UID)+'\n'))
                self.log.success(f'Coined {mapped_path} as {output}')

    def _gen_bids_filename(self, pat, path, pred):
        """
        Generates a bids like filename and path from the metadata in the csv and the population of the target directory
        """
        ses = path.parent.name
        out_path = self.bids_set/('sub-'+pat)/ses/'anat'
        if out_path.is_dir():
            files = [file for file in os.listdir(out_path) if file.endswith(pred+'w.nii.gz')]
            if any(files):
                i = 1
                for file in files:
                    for key_values in file.split('_'):
                        if 'run' in key_values:
                            idx = int(key_values.split('-')[1])
                            if idx >= i:
                                i = idx+1
            else: i = 1
        else: i = 1

        os.makedirs(out_path, exist_ok=True)
        filename = f'sub-{pat}_{ses}_run-{i}_{pred}w.nii.gz'
        return out_path/filename


