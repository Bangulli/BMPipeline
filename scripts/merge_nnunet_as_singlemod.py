import os
import pathlib as pl


single = pl.Path('/mnt/nas6/data/Target/nnUNet_Datasets/singlemod')
multi = pl.Path('/mnt/nas6/data/Target/nnUNet_Datasets/multimod')
all_data = pl.Path('/mnt/nas6/data/Target/nnUNet_Datasets/all_singlemod')

s_files = [f for f in os.listdir(single) if f.endswith('.nii.gz')]
m_files = [f for f in os.listdir(multi) if f.endswith('.nii.gz')]

for f in s_files:
    (all_data/f).symlink_to(single/f)

for f in m_files:
    (all_data/f).symlink_to(multi/f)

torm = [f for f in os.listdir(all_data) if f.endswith('_0002.nii.gz')]
for f in torm:
    if (all_data/f).is_symlink():
        (all_data/f).unlink()