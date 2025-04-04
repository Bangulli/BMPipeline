import shutil
import pathlib as pl
import os
import SimpleITK as sitk
import logging
import sys

class StreamToLogger:
    """
    Redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = ''

    def write(self, message):
        if message != '\n':
            self.buffer += message
            if message.endswith('\n'):
                self.logger.log(self.level, self.buffer.rstrip())
                self.buffer = ''

    def flush(self):
        if self.buffer:
            self.logger.log(self.level, self.buffer.rstrip())
            self.buffer = ''

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("combined_output.log"),
        logging.StreamHandler()
    ]
)

# Redirect stdout and stderr to logger
stdout_logger = logging.getLogger('STDOUT')
stderr_logger = logging.getLogger('STDERR')

sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)

resample =(1,1,1)
img = sitk.ReadImage('/mnt/nas6/data/Target/batch_copy/rerun_test/bids/sub-PAT0686/ses-20110529180153/anat/sub-PAT0686_ses-20110529180153_acq-cor_rec-nd_run-1_T2w.nii.gz')

if resample is not None:
    new_size = [
    int(round(osz * ospc / nspc))
    for osz, ospc, nspc in zip(img.GetSize(), img.GetSpacing(), resample)
    ]
    img = sitk.Resample(img, new_size, outputSpacing=resample, outputDirection=img.GetDirection(), outputOrigin=img.GetOrigin(), outputPixelType=img.GetPixelID())

sitk.WriteImage(img, 'resampled.nii.gz')



# bids_set = pl.Path("/mnt/nas6/data/Target/BIDS_mrct1000")
# pro_set = pl.Path("/mnt/nas6/data/Target/PROCESSED_mrct1000_nobatch")

# missing = {}
# pats = [pat for pat in os.listdir(bids_set) if pat.startswith('sub-PAT')]
# ref = [pat for pat in os.listdir(pro_set) if pat.startswith('sub-PAT')]
# pats = [pat for pat in pats if pat in ref]
# for pat in pats:
#     mis = [ses for ses in os.listdir(bids_set/pat) if not ((bids_set/pat/ses/'anat').is_dir() or (bids_set/pat/ses/'rt').is_dir())]
#     print(f"Patient {pat} is missing series: {mis}")
#     if mis:
#         missing[pat] = mis

# for key, value in missing.items():
#     print(f"Patient {key} is missing data in Studies: {value}")