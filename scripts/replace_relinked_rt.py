import os
import pathlib as pl
import shutil

dataset = pl.Path('/mnt/nas6/data/Target/mrct1000_nobatch')
relinked = pl.Path('/mnt/nas6/data/Target/supplementary_RTCT/dicoms-depersonalized2/velocity_sorted')

dpat = [f for f in os.listdir(dataset) if (dataset/f).is_dir()]
rpat = [f for f in os.listdir(relinked) if (relinked/f).is_dir()]

pats = [p for p in dpat if p in rpat]

for pat in pats:
    ses = [f for f in os.listdir(relinked/pat) if (relinked/pat/f).is_dir()]
    for s in ses:
        files = os.listdir(relinked/pat/s)
        for f in files:
            if (dataset/pat/s/f).is_dir():
                shutil.rmtree(dataset/pat/s/f)
                
            elif (dataset/pat/s/f).is_file():   
                os.remove(dataset/pat/s/f)
            
            # Move the content
            # source to destination
            dest = shutil.move(relinked/pat/s/f, dataset/pat/s/f)
            print(f"Moved relinked file to {dest}")