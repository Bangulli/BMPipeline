import os
import subprocess
import tempfile
import pathlib
import numpy as np
import shutil 
import traceback
from joblib import Parallel, delayed
import time

class BidscoinerJob():
    def __init__(self, source, target, patients, id, bidsmap):
        self.source=source
        self.target=target
        self.patients = patients
        self.id = id
        self.bidsmap=bidsmap
    
    def execute(self, _):
        try: # nested trycatch, first is to report errors without killing all other processes
            try: # second to persist tempdirs in case an error occurs
                work_dir = self.target/self.id
                os.makedirs(work_dir/'symlinks', exist_ok=True) #setup working directory for the job

                ## populate working direcotry source data
                for pat in self.patients:
                    (work_dir/'symlinks'/pat).symlink_to(self.source/pat)

                bids_set = work_dir/'bids_out'
                ## run bidscoiner
                if not bids_set.is_dir(): # run the tml_dicom2bids with a template bidsmap, maps data and converts to bids format
                    command = [
                        "tml_dicom2bids_convert",
                        "-i", work_dir/'symlinks',
                        "-o", bids_set,
                        "-t", self.bidsmap
                    ]
                    # Run the command
                    subprocess.run(command)
                else:
                    print("BIDS output directory exists, skipped conversion.")

                ## cleanup after running
                for pat in self.patients:
                    pat = pat.replace('PAT-', 'PAT')
                    shutil.move(bids_set/pat, self.target/pat)
                shutil.rmtree(work_dir)

            except Exception as e:
                print("Error occured in", self.id, "preventing tempdir deletion")
                print(e)
                raise e

            print('Bidscoincer process id', self.id, 'completed successfully. Exiting process')
            return f"SUCCESS--{self.id}"
        except Exception as e:
            return f"FAIL--{self.id}--{e}\n{traceback.format_exc()}"
        
def run_bidscoiner_multiprocess(source, target, bidsmap, n_jobs=5, patients_per_job=None):
    os.makedirs(target, exist_ok=True)
    patients = [p for p in os.listdir(source) if p.startswith('sub-PAT')]

    
    
    jobs = []
    
    if patients_per_job is None:
        print('Did not get a specification for patient count, infering patient count per job internally')
        patients_per_job = round(len(patients)/n_jobs)
        for j in range(n_jobs):
            upper = (j+1)*patients_per_job if (j+1)*patients_per_job<len(patients) else -1
            sub_list = patients[j*patients_per_job:upper]
            jobs.append(BidscoinerJob(source, target, sub_list, f"Bidscoiner_{j}", bidsmap))

    else:
        print(f"Setting up {round(len(patients)/patients_per_job)} jobs with {patients_per_job} patients per job")
        for j in range(round(len(patients)/patients_per_job)):
            upper = (j+1)*patients_per_job if (j+1)*patients_per_job<len(patients) else -1
            sub_list = patients[j*patients_per_job:upper]
            jobs.append(BidscoinerJob(source, target, sub_list, f"Bidscoiner_{j}", bidsmap))


    start = time.time()
    results = Parallel(n_jobs=n_jobs)(delayed(job.execute)(job) for job in jobs)
    stop = time.time()
    period = stop-start
    mins, secs = divmod(period, 60)
    hrs, mins = divmod(mins, 60)
    print(f'The source set with {n_jobs} took {hrs}h {mins}min {secs}s')
    print('Checking results for errors...')
    for res in results:
        if res.startswith('FAIL--'):
            print("Error occured in", res.split('--')[1])
            print(res)
        else:
            print('No problems in', res.split('--')[1])