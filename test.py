### Standard Imports ###
import pathlib as pl
import os
from PrettyPrint import *
import numpy as np
import pandas as pd
### converter imports ###
from src.coin_nonchuv import NonCHUVCoiner
from src.filter_register import PatientPreprocessor
from src.rts2bids import RTS2BIDS
from src.nnUnet_data_preparation import DatasetConverter, DatasetReconverter
from src.nnUnet_predictor import Resegmentor
import pydicom

raw_set = pl.Path("/mnt/nas6/data/Target/mrct1000_nobatch") # must be path to parent folder with patient subfolders/.
bids_set = pl.Path("/mnt/nas6/data/Target/BMPipeline_full_rerun/BIDS_mrct1000") # must be path that doesnt exist, the script creates the target dir itself
processed_set = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PROCESSED')
path_metadata = pl.Path('/home/lorenz/data/mrct1000_nobatch')
path_classification_results = path_metadata / "classification_results.csv" # path to the result csv of the sequence classifier
sequence_selection = pl.Path('/home/lorenz/BMPipeline/sequence_selected_nonchuv.xlsx')

####### AT THIS POINT BIDSCOINER HAS BEEN RUN; THIS SCRIPT IS TO CONVERT NON-CHUV DATA, RTSTRUCTS AND MOVE EVERYTHING TO A NEW DIRECTORY THAT ONLY CONTAINS NECESSARY DATA
if __name__ == '__main__':
    # ref = pd.read_csv(sequence_selection) if sequence_selection.name.endswith('.csv') else pd.read_excel(sequence_selection)
    # n_nonchuv = ref["SelectedSequence"].sum()
    # print("amount noncuv", n_nonchuv)

    # n_all = 0
    # for pat in [p for p in os.listdir(bids_set) if p.startswith('sub-PAT')]:
    #     for study in [p for p in os.listdir(bids_set/pat) if p.startswith('ses')]:
    #         if (bids_set/pat/study/'anat').is_dir():
    #             n_all += len([f for f in os.listdir(bids_set/pat/study/'anat') if f.endswith('.nii.gz')])

    # print("amount all", n_all)

    # print(f"Got {n_all} studies in BIDS set, {n_nonchuv} of which are nonchuv = {(n_nonchuv/n_all)*100:.2f}%")
    ref = pd.read_csv(sequence_selection) if sequence_selection.name.endswith('.csv') else pd.read_excel(sequence_selection)
    n_nonchuv = len(ref)
    print("amount noncuv", n_nonchuv)

    n_all = 0
    for pat in [p for p in os.listdir(raw_set) if p.startswith('sub-PAT')]:
        for study in [p for p in os.listdir(raw_set/pat) if p.startswith('ses')]:
            for series in [p for p in os.listdir(raw_set/pat/study) if (raw_set/pat/study/p).is_dir()]:
                file = os.listdir(raw_set/pat/study/series)[0]
                file = pydicom.read_file(raw_set/pat/study/series/file)
                if file['Modality'].value == 'MR': n_all+=1

    print("amount all", n_all)

    print(f"Got {n_all} studies in BIDS set, {n_nonchuv} of which are nonchuv = {(n_nonchuv/n_all)*100:.2f}%")
