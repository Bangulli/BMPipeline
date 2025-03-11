### Standard Imports ###
import src.log as log
from src.filter_register import PatientPreprocessor
import re
import pathlib as pl
import pandas as pd
import pydicom
import os
import csv
import subprocess
from PrettyPrint import *
import platipy
from platipy.dicom.io.rtstruct_to_nifti import convert_rtstruct
from platipy.dicom.io.rtdose_to_nifti import convert_rtdose
from platipy.imaging.tests.data import get_lung_dicom
import SimpleITK as sitk

class RTS2BIDS():
    def __init__(
            self,
            raw_source: pl.Path,
            csv_out: pl.Path,
            raw_patient_pattern: str = r"^sub-PAT-(\d{4})$",
            raw_study_pattern: str = r"ses-\d{14}$"
            ):
        # set path attributes
        self.raw_source = raw_source
        self.csv_out = csv_out
        # set source pattern attributes
        self.raw_patient_pattern = raw_patient_pattern
        self.raw_study_pattern = raw_study_pattern
        # set up logger
        self.log = Printer()
        self.info_format = PPFormat([ColourText('blue'), Effect('bold'), Effect('underlined')]) 

    def execute(self):
        """
        Run the extraction and conversion of rts files
        parse the source directory, look at every session in every patient, if the session has filenames with 'RTSS' in them process them

        make it so this thing parses the entire patient for cts first and then matches the rtss when they are found. 
        sometimes the rts are not in the same dir as the cts
        """
        patients_raw = [pat for pat in os.listdir(self.raw_source) if (self.raw_source/pat).is_dir()]

        patients_bids = pd.read_csv(self.csv_out)
        patients_bids = list(patients_bids['PatientID'])
        patients_raw = [pat for pat in patients_raw if pat not in patients_bids]

        header = ['PatientID', 'Studies', 'CTs', 'RTStudies', 'RTFiles', 'RT_with_matching_CT', 'RTDoseStudies', 'RTDoseFiles', 'CompleteMatch', 'HasGTVPTV', 'AllGTVPTV']

        patients_with_no_ct = 0
        patients_with_no_rt = 0
        patients_with_no_dose = 0
        patient_count = log.count_folders(self.raw_source)
        exists = self.csv_out.is_file()
        with open(self.csv_out, mode='a') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            if not exists:
                writer.writeheader()
            for raw, bids in zip(patients_raw, patients_bids):
                csv_row = {}
                studies = [elem for elem in os.listdir(self.raw_source/raw) if (self.raw_source/raw/elem).is_dir()]
                csv_row['PatientID'] = raw
                csv_row['Studies'] = len(studies)
                ct = []
                for ses in studies: # first extract all cts in case they are in different studies
                    ct += self._get_CT_files(self.raw_source/raw/ses)
                csv_row['CTs'] = len(ct)
                matching = 0
                rtstudies = 0
                rtfiles = 0
                rtdfiles = 0
                doses = 0
                has_gtvptv = 0
                if not any(ct):
                    patients_with_no_ct += 1

                for ses in studies:
                    rts, rtd = self._get_RT_files(self.raw_source/raw/ses)
                    
                    if any(rts):
                        rtstudies += 1
                        for rt in rts:
                            rtfiles += 1
                            rts_path, ct_path = self._match_uids_exhaustive(rt, ct)
                            if rts_path is not None:
                                matching += 1
                                if self._filter_structs(rts_path):
                                    has_gtvptv += 1

                    

                    if any(rtd):
                        doses += 1
                        for i, d in enumerate(rtd):
                            rtdfiles += 1

                csv_row['RTStudies'] = rtstudies
                if (rtstudies) == 0:
                    patients_with_no_rt += 1
                csv_row['RTFiles'] = rtfiles
                csv_row['RT_with_matching_CT'] = matching
                csv_row['RTDoseStudies'] = doses
                if (doses) == 0:
                    patients_with_no_dose += 1
                csv_row['RTDoseFiles'] = rtdfiles
                csv_row['CompleteMatch'] = (rtfiles == matching) and (matching != 0)
                csv_row['HasGTVPTV'] = has_gtvptv
                csv_row['AllGTVPTV'] = (has_gtvptv == matching) and (matching != 0)
                
                writer.writerow(csv_row)
                print(csv_row)
            
            print(f"Found {patient_count} patients, {patients_with_no_ct} have no ct, {patients_with_no_rt} have no rtstruct, {patients_with_no_dose} have no rtdose. See csv at {self.csv_out} for more details.")
            

                    
                
        return

    def _filter_structs(self, rtstruct_path, keep_keywords=(r"GTV", r"PTV")):
        try:
            # Load the RTSTRUCT DICOM file
            ds = pydicom.read_file(rtstruct_path)

            # Get the list of structures
            roi_seq = ds.StructureSetROISequence
            roi_names = {roi.ROINumber: roi.ROIName for roi in roi_seq}

            # Identify ROIs to keep
            keep_rois = {num for num, name in roi_names.items() if any(k in name for k in keep_keywords)}

            return any(keep_rois)
        except:
            return False


    def _convert_patient_ID(self, dir):
        return re.sub(self.raw_patient_pattern, r"sub-PAT\1", dir)
    
    def _get_RT_files(self, ses):
        rts = []
        rtd = []
        for series in os.listdir(ses):
            dcm_files_in_dir = os.listdir(ses/series)

            if any(dcm_files_in_dir):
                while not (ses/series/dcm_files_in_dir[0]).is_file(): # Sometimes the series has a subfolder, this aims to find the deepest folder and then run with that
                    series = os.path.join(series,dcm_files_in_dir[0])
                    dcm_files_in_dir = os.listdir(ses/series)

                rt = pydicom.read_file(ses/series/dcm_files_in_dir[0])

                if rt.Modality == 'RTSTRUCT':
                    rts.append({ses/series/dcm_files_in_dir[0]: rt.ReferencedFrameOfReferenceSequence[0].
                    RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].
                    SeriesInstanceUID})

                elif rt.Modality == 'RTDOSE':
                    rtd.append(ses/series/dcm_files_in_dir[0])

        return rts, rtd
    
    def _get_CT_files(self, ses):
        ct = []
        for series in os.listdir(ses):
            dcm_files_in_dir = os.listdir(ses/series)

            while not (ses/series/dcm_files_in_dir[0]).is_file(): # Sometimes the series has a subfolder, this aims to find the deepest folder and then run with that
                series = os.path.join(series,dcm_files_in_dir[0])
                dcm_files_in_dir = os.listdir(ses/series)

            rt = pydicom.read_file(ses/series/dcm_files_in_dir[0])

            if rt.Modality == 'CT': # or rt.Modality == 'MR':
                ct.append({ses/series: rt.SeriesInstanceUID})

        return ct

    def _match_uids_exhaustive(self, rts, ct, use_fallback=False):
        rts_ref = list(rts.values())[0]
        rts_path = list(rts.keys())[0]
        for ct_series in ct:
            ct_uid = list(ct_series.values())[0]
            ct_path = list(ct_series.keys())[0]
            if rts_ref == ct_uid:
                self.log.success(f'Found matching CT for RTSS {rts_path.parent}')
                return rts_path, ct_path
        if use_fallback:
            for ct_series in ct:
                ct_uid = list(ct_series.values())[0]
                ct_path = list(ct_series.keys())[0] # yields ct session 
                if ct_path.parent == rts_path.parent.parent:
                    self.log.success(f'Found matching CT for RTSS using same study fallback {rts_path.parent}')
                    return rts_path, ct_path
                
        self.log.fail(f'Could not find matching CT for RTSS {rts_path.parent}')
        return None, None

if __name__ == '__main__':
    raw_set = pl.Path('/mnt/nas6/data/Target/mrct1000_nobatch')
    bids_set = pl.Path('/home/lorenz/data/mrct1000_nobatch/dataset_insight.csv')

    converter = RTS2BIDS(
        raw_set,
        bids_set
    )
    converter.execute()