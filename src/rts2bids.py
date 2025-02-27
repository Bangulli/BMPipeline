import pathlib as pl
import os 
import platipy
import pydicom
from platipy.dicom.io.rtstruct_to_nifti import convert_rtstruct
from platipy.dicom.io.rtdose_to_nifti import convert_rtdose
from PrettyPrint import *
import re
import SimpleITK as sitk
import sys


class RTS2BIDS():
    def __init__(
            self,
            raw_source: pl.Path,
            bids_target: pl.Path,
            raw_patient_pattern: str = r"^sub-PAT-(\d{4})$",
            raw_study_pattern: str = r"ses-\d{14}$"
            ):
        # set path attributes
        self.raw_source = raw_source
        self.bids_target = bids_target
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
        patients_bids = [self._convert_patient_ID(pat) for pat in patients_raw]
        for raw, bids in zip(patients_raw, patients_bids):
            if (self.bids_target/bids).is_dir():
                studies = os.listdir(self.raw_source/raw)
                ct = []
                for ses in studies: # first extract all cts in case they are in different studies
                    ct += self._get_CT_files(self.raw_source/raw/ses)
                if any(ct):
                    for ses in studies:
                        rts, rtd = self._get_RT_files(self.raw_source/raw/ses)
                        output = self.bids_target/bids/ses/'rt'
                        
                        if any(rts) and any(ct):
                            for rt in rts:
                                rts_path, ct_path = self._match_uids_exhaustive(rt, ct)
                                
                                if rts_path is not None:
                                    os.makedirs(output/rts_path.parent.name, exist_ok=True)

                                    if rts_path is not None: # do the actual conversion
                                        #continue
                                        try: 
                                            convert_rtstruct(
                                                ct_path,
                                                rts_path,
                                                output_dir=output/rts_path.parent.name
                                            )
                                            reader = sitk.ImageSeriesReader()
                                            dicom_names = reader.GetGDCMSeriesFileNames(ct_path)
                                            reader.SetFileNames(dicom_names)
                                            image = reader.Execute()
                                            sitk.WriteImage(image, output/(rts_path.parent.name+' - CT_reference.nii.gz'))
                                        except:
                                            self.log.fail(f'RTSTRUCT conversion failed for RTSS at path {rts_path.parent}')


                        if any(rtd):
                            os.makedirs(output, exist_ok=True)
                            for i, d in enumerate(rtd):
                                if not d.parent.name.endswith('RTDOSE'):
                                    filename = d.parent.name+' - RTDOSE.nii.gz'
                                else:
                                    filename = d.parent.name+'.nii.gz'
                                #continue
                                rtdose = convert_rtdose(
                                    d,
                                    dose_output_path=output/filename
                                )
                else:
                    self.log.warning(f'Didnt find any CT scans in patient file {self.raw_source/raw}')
                        

            else:
                self.log.fail(f"Cant find patient directory {bids} in bids target")
        return

    def _convert_patient_ID(self, dir):
        return re.sub(self.raw_patient_pattern, r"sub-PAT\1", dir)
    
    def _get_RT_files(self, ses):
        rts = []
        rtd = []
        for series in os.listdir(ses):
            dcm_files_in_dir = os.listdir(ses/series)

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