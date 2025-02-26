import pydicom
import pandas as pd
import pathlib as pl
from .path import filter_mismatched_files, unique_path
import numpy as np
from shutil import copy

# Helper function to check modality
def should_process_dcm(ds):
    if hasattr(ds, 'Modality'):
        modality = ds.Modality
        if modality in ['RTSTRUCT', 'RTDOSE']:  # RS = RTSTRUCT, RD = RTDOSE
            return False
    return True

# Reads a dicom file and returns a pd.Series of metadata info for the relevant tags, specified in dcm_header_names param
def parse_dcm_file(path_to_file, var_name_dcm_path='path_to_dcm_dir', var_name_dcm_filename='dcm_file_name',
                   dcm_header_names=[]):
    ds = pydicom.dcmread(path_to_file, force=True)
    
    # Check if the file should be processed based on its modality
    if not should_process_dcm(ds):
        return None
    
    meta_data = {}
    for dcm_header_name in dcm_header_names:
        if hasattr(ds, dcm_header_name):
            meta_data_attr_value = getattr(ds, dcm_header_name)
            # this removes a comma from the patientname value, idk why probably for the csv
            if dcm_header_name == 'PatientName':
                meta_data_attr_value = meta_data_attr_value.family_comma_given()
                if meta_data_attr_value.endswith(',') or meta_data_attr_value.endswith(', '):
                    meta_data_attr_value = meta_data_attr_value.split(',')[0]
            meta_data[dcm_header_name] = meta_data_attr_value
    meta_data[var_name_dcm_path] = path_to_file.parent
    meta_data[var_name_dcm_filename] = path_to_file.name
    series = pd.Series(meta_data)
    # failsafe for in rare cases there is no acquisitiondate in the dicom tags, so its infered from the filepath
    if not 'AcquisitionDate' in series:
        series['AcquisitionDate'] = path_to_file.parent.parent.name.split('-')[-1][:8]
    return series

# Extract the Metadata of each series in a study
# Returns a Dataframe with the metadata of each eries in a study
def parse_dcm_dir(path_to_dcm_dir, dcm_glob, by_dir=True, dcm_header_names=[]):
    path_to_dcm_dir = pl.Path(path_to_dcm_dir)
    print("== Parsing dicom directory '%s'" % path_to_dcm_dir)
    df_list = []
    map_list = []
    
    #iterate over every folder in the directory
    for dir in path_to_dcm_dir.glob('**'): # get all directories and subdirectories in the dir
        dcm_files_in_dir = [file for file in dir.glob(dcm_glob) if file.is_file()] # extract files in the subdirectory

        # Filter out mismatched UIDs from the files
        dcm_files_in_dir = filter_mismatched_files(dcm_files_in_dir)
        
        if by_dir:
            if len(dcm_files_in_dir) > 0:
                dcm_series = parse_dcm_file(dcm_files_in_dir[0], dcm_header_names=dcm_header_names)
                if dcm_series is not None:
                    df_list.append(dcm_series)
                    if not 'SeriesInstanceUID' in dcm_series.keys(): # it parses a directory that is not dicom, this avoids an error when the key is missing
                        continue
                    map_list.append(pd.Series([dcm_files_in_dir[0], dcm_series['SeriesInstanceUID']]))
        else:
            for file in dcm_files_in_dir:
                dcm_series = parse_dcm_file(file, dcm_header_names=dcm_header_names)
                if dcm_series is not None:
                    df_list.append(dcm_series)
                    map_list.append(pd.Series([file, dcm_series['SeriesInstanceUID']]))
    if df_list:
        df = pd.concat(df_list, axis=1).T
        map_list = pd.concat(map_list, axis=1).T
        return df, map_list
    else:
        return pd.DataFrame(), pd.DataFrame()

# Copies the dicom files of a series to a new location, if their modaility is of interest and returns the path mapping
# Only copies the first dcm file in a series, so the sequence classifier can grab the metadata from there
# def parse_dcm_dir_copy_file(path_to_dcm_dir, dcm_glob, target_path, out_name_template="series-{:04d}.dcm"):
#     path_to_dcm_dir = pl.Path(path_to_dcm_dir)
#     target_path = pl.Path(target_path)
#     print("== Parsing dicom directory '%s'" % path_to_dcm_dir)
#     df_list = []
#     for dir in path_to_dcm_dir.glob('**'):
#         dcm_files_in_dir = [file for file in dir.glob(dcm_glob) if file.is_file()]
#         if len(dcm_files_in_dir) > 0:
#             p_source = dcm_files_in_dir[0]
#             ds = pydicom.dcmread(p_source, force=True)
#             # Check if the file should be processed based on its modality
#             if should_process_dcm(ds):
#                 p_target = unique_path(target_path, out_name_template)
#                 copy(p_source, p_target, follow_symlinks=True)
#                 info = {'p_source': p_source, 'p_target': p_target}
#                 df_list.append(pd.Series(info))
#     if df_list:
#         df = pd.concat(df_list, axis=1).T
#         return df
#     else:
#         return pd.DataFrame()
    
# Extract the metadata each patient, each study and each series. Saves a csv with the metadata. Copies the first dcm file of each series to a directory (path_metadata) for the sequence classifier to process
# removed the copying part to save time, the classifier runs on the csv created anyway
def extract_metadata(path_data, path_metadata):
    if path_metadata.joinpath('series-meta-data.csv').exists():
        print(f"{path_metadata.joinpath('series-meta-data.csv')} exists, skipped extraction")
        return None
    df, path_map = parse_dcm_dir(path_data, dcm_glob='*', by_dir=True, dcm_header_names=dcm_attribute_dict.keys())
    expected_but_not_present = set(dcm_attribute_dict.keys()).difference(df.columns)
    present_but_not_expected = set(df.columns).difference(dcm_attribute_dict.keys())
    for var in expected_but_not_present:
        df[var] = np.nan
    for var in present_but_not_expected:
        df.drop(var, inplace=True, axis=1)
    df.to_csv(path_metadata.joinpath('series-meta-data.csv'), sep=';', index=False)
    path_map.to_csv(path_metadata.joinpath('sliceID_seriesPath_mapping.csv'), sep=';', index=False)
    #df = parse_dcm_dir_copy_file(path_data, dcm_glob='*', target_path=path_metadata)
    #df.to_csv(path_metadata.joinpath('file_mapping.csv'), index=False)
    print(f"Metadata extracted and saved in {path_metadata}")
    
# dict with the dicaom Tags and the column names for the classifier
dcm_attribute_dict = { 'SeriesInstanceUID'          : 'series_instance_uid',
                       'SeriesDescription'          : 'series_description',
                       'SeriesTime'                 : 'series_time',
                       'AccessionNumber'            : 'accession_number',
                       'AcquisitionDate'            : 'acquisition_date',
                       'ContrastBolusAgent'         : 'contrast_bolus_agent',
                       'EchoTime'                   : 'echo_time',
                       'EchoTrainLength'            : 'echo_train_length',
                       'FlipAngle'                  : 'flip_angle',
                       'ImageType'                  : 'image_type',
                       'InversionTime'              : 'inversion_time',
                       'Manufacturer'               : 'manufacturer',
                       'MRAcquisitionType'          : 'mr_acquisition_type',
                       'Modality'                   : 'modality',
                       'PatientID'                  : 'patient_id',
                       'PhotometricInterpretation'  : 'photometric_interpretation',
                       'PixelBandwidth'             : 'pixel_bandwidth',
                       'PixelSpacing'               : 'pixel_spacing',
                       'SliceThickness'             : 'slice_thickness',
                       'ProtocolName'               : 'protocol_name',
                       'RepetitionTime'             : 'repetition_time',
                       'ScanOptions'                : 'scan_options',
                       'ScanningSequence'           : 'scanning_sequence',
                       'SequenceName'               : 'sequence_name',
                       'SequenceVariant'            : 'sequence_variant',
                       'StationName'                : 'station_name'
                       }