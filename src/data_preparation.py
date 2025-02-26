import pydicom
import os
import pathlib as pl
import pandas as pd
from pydicom.dataelem import DataElement
import PrettyPrint
from PrettyPrint import *
import re

def inject_sequence_name(df: pd.DataFrame, target_class: list, LOGGER: PrettyPrint.Printer, raw_set: pl.Path, SequenceName_str: str):
    INFOformat = PPFormat([ColourText('blue'), Effect('bold'), Effect('underlined')])
    df = df[df['prediction_class'].isin(target_class)] # filter for class
    df = df[pd.isna(df['SequenceName'])] # filter for sequence name missing
    LOGGER.tagged_print("INFO", f"Found {df.shape[0]} {target_class} images with no SequenceName", INFOformat)

    SequenceName_tag = (0x0018, 0x0024)
    SequenceName = DataElement(SequenceName_tag, "SH", SequenceName_str)

    # iter over filtered results and inject sequence name into the first dicom slice of each series that requires
    for i, row in df.iterrows():
        found = False
        sub = row['PatientID']
        sub = 'sub-'+str(sub)
        ses = row['AcquisitionDate']
        ser = re.sub(r"[\s\-_,.]", '', row['SeriesDescription']).split("(")[0]
        ser = ser.split('/')[-1] # for weird description names like: FL:A/AxT2PROPELLER3mmCraneGado
        ses = 'ses-'+str(int(ses))
        for study in os.listdir(raw_set/sub):
            if study.startswith(ses):
                for series in os.listdir(raw_set/sub/study):
                    if ser in re.sub(r"[\s\-_,.]", "", series):
                        found = True
                        slices = os.listdir(raw_set/sub/study/series)
                        for slice in slices:
                            dcm = pydicom.dcmread(raw_set/sub/study/series/slice)
                            dcm[SequenceName_tag] = SequenceName
                            dcm.save_as(raw_set/sub/study/series/slice)
                        
                        LOGGER.tagged_print("INFO", f"Added SequenceName {SequenceName_str} to DICOM slice at path: {raw_set/sub/study/series}", INFOformat)
                        break
        if not found:
            LOGGER.fail(f"Could not match SeriesDescription {ser} with any series file name")