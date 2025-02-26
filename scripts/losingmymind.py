import pydicom
import os
import pathlib as pl

target = pl.Path('/mnt/nas6/data/Target/TEMP_Subset_49/sub-PAT042/ses-20131101142235/01401-3DRXTH1mm')
for elem in os.listdir(target):
    ds = pydicom.dcmread(target/elem)
    print(ds.SeriesDescription)