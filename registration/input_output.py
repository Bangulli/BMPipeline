### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib

### External Imports ###
import numpy as np
import scipy.ndimage as nd
import torch as tc
import SimpleITK as sitk

### Internal Imports ###

########################

def load_dicom(folder_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(folder_path))
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def load_nifti(file_path):
    image = sitk.ReadImage(file_path)
    return image

def load_volumes(source_path, target_path, normalize=False, type='nifti'):
    if type == 'dicom':
        source_image = load_dicom(source_path)
        target_image = load_dicom(target_path)
    elif type == 'nifti':
        source_image = load_nifti(source_path)
        target_image = load_nifti(target_path)
    if normalize:
        source_image = sitk.Cast(source_image, sitk.sitkFloat32)
        target_image = sitk.Cast(target_image, sitk.sitkFloat32)
        source_image = sitk.RescaleIntensity(source_image, outputMinimum=0, outputMaximum=1)
        target_image = sitk.RescaleIntensity(target_image, outputMinimum=0, outputMaximum=1)
    source_image = sitk.Resample(source_image, referenceImage=target_image)
    return source_image, target_image
