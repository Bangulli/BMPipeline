### Registration Imports ###
import pathlib as pl
from PrettyPrint import *
import os
from datetime import datetime, timedelta
import re
import SimpleITK as sitk
from totalsegmentator.python_api import totalsegmentator
import ants
import csv
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


class TimeTester():
    """
    Just a quick method to test the temporal filtering proposed, this is supposed to run on the bids set, before the RT structs are converted
    """
    def __init__(self, bids_set: pl.Path):
        self.bids_set = bids_set

    def execute(self, logfile: pl.Path='/home/lorenz/data/mrct1000_nobatch/temporal_filtering.csv', criterion=[4, 300, 90, 0]):
        header = ['PatientID', 'Studies', 'Redundancies_14d', 'PreTreatmentStudies', 'DetectedTreatmentStart', 'StudiesResult', 'AVGIntervalProcessed', 'TotalStudiesRemoved', 'ObservedTime', 'RTStructsPresent']
        with open(logfile, mode='a') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            patients = [pat for pat in os.listdir(self.bids_set) if (self.bids_set/pat).is_dir() and pat.startswith('sub-')]
            studies_removed = []
            avg_int_processed = []
            days_observed = []
            valid_patients = 0
            print(f'Found {len(patients)} patients in the directory')
            for pat in patients:
                csv_row = {}
                csv_row['PatientID'] = pat
                rt = [elem for elem in os.listdir(self.bids_set/pat) if (self.bids_set/pat/elem/'rt').is_dir()]
                studies = self._sort_directories(pat)
                print(rt, len(os.listdir(self.bids_set/pat)))
                csv_row['RTStructsPresent'] = len(rt)
                #print(studies)
                csv_row['Studies'] = len(studies)
                studies_filtered_red = self._clean_redundancies(studies, pat)
                csv_row['Redundancies_14d'] = len(studies) - len(studies_filtered_red)
                studies_filtered_pre = self._clean_pretreatment(studies_filtered_red, pat)
                csv_row['PreTreatmentStudies'] = len(studies_filtered_red) - len(studies_filtered_pre)
                csv_row['DetectedTreatmentStart'] = studies_filtered_pre[0][0] if any(studies_filtered_pre) else 0
                csv_row['StudiesResult'] = len(studies_filtered_pre)
                avg_int = self._get_avg_int(studies_filtered_pre)
                avg_int_processed.append(avg_int)
                csv_row['AVGIntervalProcessed'] = avg_int
                csv_row['TotalStudiesRemoved'] = len(studies) - len(studies_filtered_pre)
                studies_removed.append(len(studies) - len(studies_filtered_pre))
                csv_row['ObservedTime'] = (studies_filtered_pre[-1][1]-studies_filtered_pre[0][1]).days if any(studies_filtered_pre) else 0
                days_observed.append((studies_filtered_pre[-1][1]-studies_filtered_pre[0][1]).days if any(studies_filtered_pre) else 0)
                

                valid_patients += 1 if (csv_row['StudiesResult'] >= criterion[0] and csv_row['ObservedTime'] >= criterion[1] and csv_row['AVGIntervalProcessed'] <= criterion[2] and csv_row['RTStructsPresent'] > criterion[3]) else 0

                #print(studies_filtered_pre)
                print(csv_row)
                writer.writerow(csv_row)
   

        plt.ion()
        fig, ax = plt.subplots()
        ax.set_ylabel('Days between Studies')
        bplt = ax.boxplot(avg_int_processed)
        plt.savefig("boxplot_avg_interval.png", dpi=300)
        plt.clf()
        fig, ax = plt.subplots()
        ax.set_ylabel('Days Observed')
        bplt = ax.boxplot(days_observed)
        plt.savefig("boxplot_days_observed.png", dpi=300)
        plt.clf()
        fig, ax = plt.subplots()
        ax.set_ylabel('Studies Rejected')
        bplt = ax.boxplot(studies_removed)
        plt.savefig("boxplot_studies_rejected.png", dpi=300)

        print(f'Valid Patients with more than {criterion[0]} anatomical sequences, more than {criterion[1]} days of observation and less than {criterion[2]} days of average time in between sequences and more than {criterion[3]} RTStructs after filtration:', valid_patients)


    def _sort_directories(self, pat): # courtesy of chatgpt
        """
        Finds directories in the specified base_dir that match the pattern
        ses-yyyymmddhhmmss, parses the timestamp, and returns a list of directory
        names sorted in chronological order.
        """
        base_dir = self.bids_set/pat
        # Regular expression pattern to match directory names like "ses-20250224235959"
        pattern = r"^ses-(\d{14})$"
        # List all directories in the base directory
        dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and (base_dir/d/'anat').is_dir()]
        
        matching_dirs = []
        for d in dirs:
            match = re.match(pattern, d)
            if match:
                timestamp_str = match.group(1)
                # Convert the timestamp string to a datetime object
                dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                matching_dirs.append((d, dt))
        
        # Sort the directories based on the datetime objects
        matching_dirs.sort(key=lambda x: x[1])
        # Return just the sorted directory names
        return matching_dirs

    def _clean_redundancies(self, studies: list, pat: str, delta_days: int = 14):
        """
        Filters the anatomical studies, removing any redundancies i.e. when the time between images is too low 
        """
        anat = [elem for elem in studies if (self.bids_set/pat/elem[0]/'anat').is_dir()] # seperate liste by if they have anat or rt

        # Filter out directories that are within 14 days of a more recent one
        filtered_dirs = []
        while anat:
            latest = anat.pop()  # Take the latest directory
            filtered_dirs.append(latest)
            # Remove any directories within 14 days before the latest one
            anat = [(d, dt) for d, dt in anat if dt <= latest[1] - timedelta(days=delta_days)]
        
        return [d for d in sorted(filtered_dirs, key=lambda x: x[1])]
    
    def _clean_pretreatment(self, studies: list, pat: str, delta_months: int = 6, filter_latest = True):
        """
        Filters the anatomical studies, removing any redundancies i.e. when the time between images is too low 
        """
        anat = [elem for elem in studies if (self.bids_set/pat/elem[0]/'anat').is_dir()] # seperate liste by if they have anat or rt
        if not anat or len(anat) == 1:
            return anat
        # Filter out directories that are within 14 days of a more recent one
        filtered_dirs = []
        for i in range(len(anat)-1):
            if anat[i][1]+timedelta(days=delta_months*30) >= anat[i+1][1]:
                filtered_dirs.append(anat[i])
        if filtered_dirs and filter_latest:
            if filtered_dirs[-1][1]+timedelta(days=delta_months*30) >= anat[-1][1]:
                filtered_dirs.append(anat[-1])
        else:
            filtered_dirs.append(anat[-1])
        
        return [d for d in sorted(filtered_dirs, key=lambda x: x[1])]
    
    def _get_avg_int(self, studies:list):
        deltas = []
        for i in range(len(studies)-1):
            deltas.append((studies[i+1][1] - studies[i][1]).days)

        return sum(deltas)/len(deltas) if len(deltas)!= 0 else 0


    
if __name__ == '__main__':
    tt = TimeTester(pl.Path('/mnt/nas6/data/Target/BIDS_mrct1000'))
    tt.execute()