import pathlib as pl
import os 
import pydicom
from PrettyPrint.figures import ProgressBar

raw = pl.Path('/mnt/nas6/data/Target/mrct1000_nobatch')
bids = pl.Path('/mnt/nas6/data/Target/BIDS_mrct1000')
processed = pl.Path('/mnt/nas6/data/Target/PROCESSED_mrct1000_nobatch')

def count_in_set(path, mode='raw'):
    print(f'= Counting in {mode} set: {path}')
    patients = [pat for pat in os.listdir(path) if pat.startswith('sub-PAT') and (path/pat).is_dir()]
    studies = 0
    anats = 0
    rts = 0
    pats = len(patients)
    for p in ProgressBar(patients, 100):
        studs = [s for s in os.listdir(path/p) if s.startswith('ses-') and (path/p/s).is_dir()]
        studies += len(studs)
        if mode == 'raw':
            for study in studs:
                sers = [s for s in os.listdir(path/p/study) if (path/p/study/s).is_dir()]
                for ser in sers:
                    slices = os.listdir(path/p/study/ser)
                    f = pydicom.read_file(path/p/study/ser/slices[0])
                    if f['Modality'].value == 'MR':
                        anats += 1
                    elif f['Modality'].value == 'RTSTRUCT':
                        rts += 1
        elif mode in ['bids', 'processed']:
            for study in studs:
                if (path/p/study/'anat').is_dir():
                    imgs = [i for i in os.listdir(path/p/study/'anat') if i.endswith('.nii.gz')]
                    anats += len(imgs)
                if (path/p/study/'rt').is_dir():
                    rt = [i for i in os.listdir(path/p/study/'rt') if (path/p/study/'rt'/i).is_dir()]
                    rts += len(rt)
    print(f"== Analysis complete for {mode} set:")
    print(f"=== #Patients: {pats}")
    print(f"=== #Studies: {studies}")
    print(f"=== #MRs: {anats}")
    print(f"=== #RTs: {rts}")

def compute_sample_size(n_patients, Z=1.28, margin_of_error=0.1, standard_deviation=0.5):
    p=standard_deviation
    e=margin_of_error
    ((Z**2*p(1-p))/e**2) / (1+(Z**2*p(1-p))/(e**2*n_patients))


if __name__ == '__main__':
    count_in_set(raw, 'raw')
    count_in_set(bids, 'bids')
    count_in_set(processed, 'processed')