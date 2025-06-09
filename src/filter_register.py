### Registration Imports ###
import pathlib as pl
from PrettyPrint import *
import os
from datetime import datetime, timedelta
import re
import SimpleITK as sitk
from totalsegmentator.python_api import totalsegmentator
import ants
from src.utils import *
import numpy as np
import json
import traceback
import time
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

class PatientRegistrationJob():
    """
    Registration Task class
    Handles the registration of a patient.
    Takes a source and target directory, patient id and the study dict generated int the FilterRegisterMain as constructor args.
    Designed to be shipped off to a background worker for multiprocessing
    Since the totalseg uses nnunet with multiprocessing in the backend it has to be shipped off to a ProcessPoolExecutor because joblib doesnt allow child processes in subprocesses
    """
    def __init__(self, bids_set, clean_set, pat, study_dict, keys):
        """
        bids_set = pl.Path, the bids style source directory
        clean_set = pl.Path, the output directory
        pat = string, patient file name
        study_dict = dict, dictionary of matched study and rts file names
        keys = list, the keys to the above dict
        """
        self.clean_set = clean_set
        self.bids_set = bids_set
        self.pat = pat
        self.keys = keys
        self.study_dict = study_dict
        
    def execute(self):
        """
        exectue the job.
        Register CT to MR if RT is present
        Translate RT to MR
        Register MR to MR t0
        Translate RT to MR t0
        Creates a logfile in each patient directory in the output for debugging
        """
        self.log = Printer('txt', False, 'RegistrationJob', location=self.clean_set/self.pat)
        self.info_format = PPFormat([ColourText('blue'), Effect('bold'), Effect('underlined')]) 
        start = time.time()
        # iterate studies
        try: 
            t0_image = None
            t0_mask = None
            for key in self.keys:
                anat_study = key
                rt_study = self.study_dict[key]
                if not (self.bids_set/self.pat/anat_study/'anat').is_dir():
                    if t0_image is None:
                        self.log.tagged_print('MISSING', f"unexpectedly received an empty directory as anatomical in study {anat_study} for patient {self.pat}", PPFormat([ColourText('red'), Effect('bold'), Effect('underlined')]) )
                        raise RuntimeError(f"unexpectedly received an empty directory as anatomical in study {anat_study} for patient {self.pat}")
                        break
                    else:
                        self.log.tagged_print('MISSING', f"unexpectedly received an empty directory as anatomical in study {anat_study} for patient {self.pat}", PPFormat([ColourText('red'), Effect('bold'), Effect('underlined')]) )
                        self.log.warning(f'Skipping this Series. This needs manual intervention, because it can lead to a lot of missing data')
                        continue
                
                # filter study files
                images = os.listdir(self.bids_set/self.pat/anat_study/'anat')
                t1 = [img for img in images if img.endswith('T1w.nii.gz') or img.endswith('T1wa.nii.gz')]
                t1c = [img for img in images if img.endswith('ContrastT1w.nii.gz')]
                t2 = [img for img in images if img.endswith('T2w.nii.gz') or img.endswith('T2wa.nii.gz')]

                # filter results if multiple t1 and t2 sequences have been coined
                t1, t2 = self._filter_anat(t1, t1c, t2, self.pat, key)

                if t1 is None:
                    if rt_study is None:
                        self.log.fail(f'Could not find T1 in study {anat_study}')
                    else:
                        tp = 'T1' 
                        tp = 'both' if t2 is None and t1 is None else tp
                        self.log.fail(f'Found RTSTRUCT but matched mr study is incomplete, missing {tp} in study {anat_study}')
                    continue

                

                # make output directory
                os.makedirs(self.clean_set/self.pat/anat_study/'anat', exist_ok=True)

                ############## set and copy t0 T1
                if (self.bids_set/self.pat/anat_study/'anat').is_dir() and t0_image is None:
                    self.log.tagged_print('INFO', f'Identified t0 as study day {anat_study} for patient {self.pat}', self.info_format)

                    # read images
                    t0_image = self._load_image(self.bids_set/self.pat/anat_study/'anat'/t1)

                    # save unprocessed fixed
                    sitk.WriteImage(t0_image, self.clean_set/self.pat/anat_study/'anat'/t1)
                    if t2 is not None:
                        sitk.WriteImage(self._load_image(self.bids_set/self.pat/anat_study/'anat'/t2), self.clean_set/self.pat/anat_study/'anat'/t2)
                    # obtain mask
                    if not (self.clean_set/self.pat/anat_study/'anat'/('MASK_'+t1)).is_file():
                        t0_mask = self._segment_image(t0_image, 'mr')
                        sitk.WriteImage(t0_mask, (self.clean_set/self.pat/anat_study/'anat'/('MASK_'+t1)))
                    else:
                        t0_mask = self._load_image((self.clean_set/self.pat/anat_study/'anat'/('MASK_'+t1)))

                ############## if the anatomical image has a corresponding rt image do ct2mr2mr
                if rt_study is not None:
                    os.makedirs(self.clean_set/self.pat/anat_study/'rt', exist_ok =True)
                    # separate ct, rts and dose into lists with filenames
                    rts = os.listdir(self.bids_set/self.pat/rt_study/'rt')
                    ct = [elem for elem in rts if elem.endswith('CT_reference.nii.gz')]
                    rtd = [elem for elem in rts if elem.endswith('RTDOSE.nii.gz')]
                    rts = [elem for elem in rts if (self.bids_set/self.pat/rt_study/'rt'/elem).is_dir()]

                    if not any(ct): # fallback in case the CT is missing
                        self.log.warning(f'Could not find any ct in study {rt_study}, processing anatomical images in {anat_study} anyway')
                        t1_image = self._load_image(self.bids_set/self.pat/anat_study/'anat'/t1)
                        if t2 is not None:
                            t2_image = self._load_image(self.bids_set/self.pat/anat_study/'anat'/t2)
                        else:
                            t2_image = None
                        
                        t1_trans, t2_trans = self._register_mr2mr(t0_image, t0_mask, t1_image, t2_image)
                        sitk.WriteImage(t1_trans, self.clean_set/self.pat/anat_study/'anat'/t1)
                        if t2 is not None:
                            sitk.WriteImage(t2_trans, self.clean_set/self.pat/anat_study/'anat'/t2)
                        self.log.success(f'Anatomical images from study {anat_study} to t0, failed to convert RTs because the reference CT is missing')
                        continue

                        
                
                    RTdict = {} # dict that stores RT files, with filepath as key and sitk.Image as value
                    # read dose images
                    if any(rtd):
                        for d in rtd:
                            RTdict[d] = self._load_image(self.bids_set/self.pat/rt_study/'rt'/d)
                    # read structure images
                    if any(rts):
                        for s in rts:
                            os.makedirs(self.clean_set/self.pat/anat_study/'rt'/s, exist_ok =True)
                            for file in os.listdir(self.bids_set/self.pat/rt_study/'rt'/s):
                                RTdict[s+'/'+file] = self._load_image(self.bids_set/self.pat/rt_study/'rt'/s/file)

                    # set ct as moving image
                    ct_image = self._load_image(self.bids_set/self.pat/rt_study/'rt'/ct[0])

                    # get ct segmentation
                    if (self.bids_set/self.pat/rt_study/'rt'/rts[0]/'Struct_Brain.nii.gz').is_file(): # try loading from RTS
                        ct_mask = self._load_image(self.bids_set/self.pat/rt_study/'rt'/rts[0]/'Struct_Brain.nii.gz')
                    elif (self.bids_set/self.pat/rt_study/'rt'/('MASK_'+ct[0])).is_file(): # try loading from previous segmentation
                        ct_mask = self._load_image((self.bids_set/self.pat/rt_study/'rt'/('MASK_'+ct[0])))
                    else: # use total segmentator to do it
                        ct_mask = self._segment_image(ct_image, 'ct')
                        sitk.WriteImage(ct_mask, (self.bids_set/self.pat/rt_study/'rt'/('MASK_'+ct[0])))
                        
                    t1_image = self._load_image(self.bids_set/self.pat/anat_study/'anat'/t1)

                    # segment t1
                    if not (self.clean_set/self.pat/anat_study/'anat'/('MASK_'+t1)).is_file():
                        t1_mask = self._segment_image(t1_image, 'mr')
                        sitk.WriteImage(t1_mask, (self.clean_set/self.pat/anat_study/'anat'/('MASK_'+t1)))
                    else:
                        t1_mask = self._load_image((self.clean_set/self.pat/anat_study/'anat'/('MASK_'+t1)))

                    if t2 is not None:
                        t2_image = self._load_image(self.bids_set/self.pat/anat_study/'anat'/t2)
                    else:
                        t2_image = None

                    # run registration and transform structures, move ct to corresponding mrt1
                    transformed_structs = self._register_ct2mr(t1_image, t1_mask, ct_image, ct_mask, RTdict)
                    # run registration move all to t0
                    t1_trans, t2_trans, transformed_structs = self._register_mr2mr(t0_image, t0_mask, t1_image, t1_mask, t2_image, transformed_structs)
                    # make output directory and save processed images
                    for key in list(transformed_structs.keys()):
                        struct = transformed_structs[key]
                        sitk.WriteImage(struct, (self.clean_set/self.pat/anat_study/'rt'/key))
                    if (self.clean_set/self.pat/anat_study/'anat'/t1).is_file(): self.log.warning(f'''Overwriting T1 image at location {self.clean_set/self.pat/anat_study/'anat'/t1}''')
                    sitk.WriteImage(t1_trans, self.clean_set/self.pat/anat_study/'anat'/t1)
                    if t2 is not None:
                        if (self.clean_set/self.pat/anat_study/'anat'/t2).is_file(): self.log.warning(f'''Overwriting T2 image at location {self.clean_set/self.pat/anat_study/'anat'/t2}''')
                        sitk.WriteImage(t2_trans, self.clean_set/self.pat/anat_study/'anat'/t2)
                    self.log.success(f'Moved structs, dose and anatomical images from study {anat_study} and struct study {rt_study} to t0')

                ############### if there is no corresponding rt just do mr2mr
                else:
                    t1_image = self._load_image(self.bids_set/self.pat/anat_study/'anat'/t1)
                    if t2 is not None:
                        t2_image = self._load_image(self.bids_set/self.pat/anat_study/'anat'/t2)
                    else:
                        t2_image = None
                    # segment t1
                    if not (self.clean_set/self.pat/anat_study/'anat'/('MASK_'+t1)).is_file():
                        t1_mask = self._segment_image(t1_image, 'mr', fast=False)
                        sitk.WriteImage(t1_mask, (self.clean_set/self.pat/anat_study/'anat'/('MASK_'+t1)))
                    else:
                        t1_mask = self._load_image((self.clean_set/self.pat/anat_study/'anat'/('MASK_'+t1)))

                    t1_trans, t2_trans = self._register_mr2mr(t0_image, t0_mask, t1_image, t1_mask, t2_image)
                    
                    if (self.clean_set/self.pat/anat_study/'anat'/t1).is_file(): self.log.warning(f'''Overwriting T1 image at location {self.clean_set/self.pat/anat_study/'anat'/t1}''')
                    sitk.WriteImage(t1_trans, self.clean_set/self.pat/anat_study/'anat'/t1)
                    if t2 is not None:
                        if (self.clean_set/self.pat/anat_study/'anat'/t2).is_file(): self.log.warning(f'''Overwriting T2 image at location {self.clean_set/self.pat/anat_study/'anat'/t2}''')
                        sitk.WriteImage(t2_trans, self.clean_set/self.pat/anat_study/'anat'/t2)
                    self.log.success(f'Moved anatomical images from study {anat_study} to t0')
            stop = time.time()
            period = stop-start
            mins, secs = divmod(period, 60)
            hrs, mins = divmod(mins, 60)
            self.log.tagged_print('INFO', f'Executing this job took {hrs}h {mins}min {secs}s', self.info_format)
            self.log.success("Job executed with no errors to report")
            return f"{self.pat}--SUCCESS"
        except Exception as e:
            self.log.error(f"{self.pat}--FAILED with exeption {e}\n{traceback.format_exc()}")
            return f"{self.pat}--FAILED with exeption {e}\n{traceback.format_exc()}"


    def _extract_key_value(self, filename:str, key:str):
        match = re.search(rf"_{key}-(\d+)_", filename)
        return int(match.group(1)) if match else -1  # Return -1 if key is missing
    
    def _filter_anat(self, t1:list, t1_contrast:list, t2:list, pat, ses):
        ## process t1_contrast images
        if t1_contrast: t1_filtered = self._filter_list(t1_contrast, pat, ses)
        ## process t1 images if no contrast images
        else: t1_filtered = self._filter_list(t1, pat, ses)
        ## process t2 images
        t2_filtered = self._filter_list(t2, pat, ses)
        return t1_filtered, t2_filtered
    
    def _filter_list(self, mr: list, pat, ses) -> list:
        """
        Filters filenames in the study to identify a single T1 and T2 image in the directory.
        - Prioritizes runs over reconstructions.
        - Prioritizes sagittal reconstructions over any other reconstructions.
        - Prioritizes images that do not have "SUB" in the ImageType field in the corresponding JSON file.
        """

        best_run = None
        best_rec = None
        
        for image in mr:
            run_number = self._extract_key_value(image, 'run')
            rec_number = self._extract_key_value(image, 'rec')
            has_sub = self._has_sub_in_json(str(self.bids_set/pat/ses/'anat'/image))

            if run_number != -1:  # Prioritize images with 'run'
                if (best_run is None or run_number > self._extract_key_value(best_run, 'run')) and not has_sub:
                    best_run = image
            elif rec_number != -1:  # Consider 'rec' images only if no 'run' images exist
                if best_rec is None or rec_number > self._extract_key_value(best_rec, 'rec'):
                    best_rec = image
                elif rec_number == self._extract_key_value(best_rec, 'rec') and 'sag' in image:
                    best_rec = image  # Prioritize 'sag' if rec number is the same
                
                if has_sub:  # If the image has "SUB" in ImageType, deprioritize it
                    continue
        
        return best_run if best_run else best_rec

    def _has_sub_in_json(self, image: str) -> bool:
            """Check if the corresponding JSON file contains 'SUB' in ImageType."""
            json_path = image.replace('.nii.gz', '.json').replace('.nii', '.json')
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        metadata = json.load(f)
                        image_type = metadata.get("ImageType", [])
                        return "SUB" in image_type
                except (json.JSONDecodeError, IOError):
                    pass  # Ignore faulty JSON files
            return False
        
    def _load_image(self, path, resample=(1,1,1)):
        #print('reading image from path', path)
        img = sitk.ReadImage(path)
        if resample is not None and not img.GetSpacing == resample:
            new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(img.GetSize(), img.GetSpacing(), resample)
            ]
            img = sitk.Resample(img, new_size, outputSpacing=resample, outputDirection=img.GetDirection(), outputOrigin=img.GetOrigin(), outputPixelType=img.GetPixelID())
        return img 

    def _segment_image(self, image: sitk.Image, mode: str, fast:bool=True) -> sitk.Image:
        """
        Runs the total segmentator on an image, differentiating between mr and ct by the mode string
        returns the brain mask for the given image
        """
        if mode == 'ct':
            mask = totalsegmentator(sitk2nib(image), fast=fast, task='total', quiet=True)
            mask = nib2sitk(mask, image)
            mask = sitk.BinaryThreshold(mask, lowerThreshold=90, upperThreshold=90, insideValue=1, outsideValue=0) # get segmentation brain=90, skull=91
            return mask
        elif mode == 'mr':
            mask = totalsegmentator(sitk2nib(image), fast=fast, task='total_mr', quiet=True) # get segmentation brain = 50
            mask = nib2sitk(mask, image)
            mask = sitk.BinaryThreshold(mask, lowerThreshold=50, upperThreshold=50, insideValue=1, outsideValue=0) #  binarize segmentation
            return mask
        else:
            print(f'Invalid mode, expected ct or mr but got {mode}')
            return None

    def _prepro_ct(self, source: sitk.Image, binary_mask: sitk.Image, structs: dict, target: sitk.Image):
        """
        Preprocess the CT image and its RTs to fit into the target image space.
        Reorients, Crops, Resamples and adjusts the image origin
        """
        # extract bounding box
        label_filter = sitk.LabelShapeStatisticsImageFilter()
        label_filter.Execute(binary_mask)
        bb = label_filter.GetBoundingBox(1)  # (x, y, z, width, height, depth)
        x, y, z, w, h, d = bb
        # crop image
        source = source[x:x + w, y:y + h, z:z + d]
        # Reorient image
        source = sitk.DICOMOrient(source, sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(target.GetDirection()))
        source.SetOrigin(target.GetOrigin())
        source.SetDirection(target.GetDirection())
        # finally resample
        source = sitk.Resample(source, referenceImage=target, defaultPixelValue=-1000)
        # crop and resample mask as well
        binary_mask = binary_mask[x:x + w, y:y + h, z:z + d]
        # Reorient image
        binary_mask = sitk.DICOMOrient(binary_mask, sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(target.GetDirection()))
        binary_mask.SetOrigin(target.GetOrigin())
        binary_mask.SetDirection(target.GetDirection())
        # finally resample
        binary_mask = sitk.Resample(binary_mask, referenceImage=target, defaultPixelValue=0, interpolator=sitk.sitkNearestNeighbor)
        
        # apply the same to the rtstructs and doses
        if structs is not None:
            for key in structs.keys():
                struct = structs[key]
                struct = struct[x:x + w, y:y + h, z:z + d]
                struct = sitk.DICOMOrient(struct, sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(target.GetDirection()))
                struct.SetOrigin(target.GetOrigin())
                struct.SetDirection(target.GetDirection())
                # finally resample
                struct = sitk.Resample(struct, referenceImage=target, defaultPixelValue=0, interpolator=sitk.sitkNearestNeighbor)
                structs[key] = struct
        
        return source, binary_mask, structs

    def _register_ct2mr(self, fixed_image: sitk.Image, fixed_mask: sitk.Image, moving_image: sitk.Image, moving_mask: sitk.Image, structs: dict=None):
        """
        This horrible mess takes sitk images, normalizes and resamples them, then converts them to tensors, registers them, converts them to sitk images and finally returns the results
        """
        # crop ct to area of interest
        moving_image, moving_mask, structs = self._prepro_ct(moving_image, moving_mask, structs, fixed_image)

        fixed_mask = sitk2ants(fixed_mask)
        moving_mask = sitk2ants(moving_mask)

        reg = ants.registration(fixed_mask, moving_mask, 'Rigid', initial_transform=None)

        # warp and reconvert structure sets
        transformed_structs = {}
        for key in list(structs.keys()):
            struct = structs[key]
            # using the struct itself as fixed, because we moved it into the target image space manuslly before and this way we avoid interpolation issues with RTDose
            struct = ants2sitk(ants.apply_transforms(fixed = sitk2ants(struct), moving = sitk2ants(struct), transformlist = reg['fwdtransforms'], interpolator  = 'nearestNeighbor'))
            transformed_structs[key] = struct
        return transformed_structs
    
    def _prepro_structs(self, target, structs):
        """
        Resample structs to fit t0 target
        has to be done because the rtdose cant be transformed to t0 as it has a type mismatch
        """
        for key in structs.keys():
            struct = structs[key]
            # finally resample
            struct = sitk.Resample(struct, referenceImage=target, defaultPixelValue=0, interpolator=sitk.sitkNearestNeighbor)
            structs[key] = struct
        return structs
    
    def _register_mr2mr(self, fixed: sitk.Image, fixed_mask: sitk.Image, moving_t1: sitk.Image, moving_t1_mask:sitk.Image, moving_t2: sitk.Image, structs: dict=None):
        """
        Register the MR images to the t0 MR image
        Transforms corresponding structs if they are present
        """
        if structs is not None:
            structs = self._prepro_structs(fixed, structs)
        fixed = sitk2ants(fixed)
        fixed_mask = sitk2ants(fixed_mask)
        moving_t1 = sitk2ants(moving_t1)
        
        reg = ants.registration(fixed, moving_t1, mask=fixed_mask, type_of_transform='TRSAA', moving_mask=sitk2ants(moving_t1_mask), initial_transform=None)
        transformed_t1 = ants2sitk(ants.apply_transforms(fixed, moving_t1, transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor'))
        if moving_t2 is not None:
            moving_t2 = sitk2ants(moving_t2)
            transformed_t2 = ants2sitk(ants.apply_transforms(fixed, moving_t2, transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor'))
        else:
            transformed_t2 = None
        if structs is not None:
            transformed_structs = {}
            for key in list(structs.keys()):
                struct = structs[key]
                # using the struct itself as fixed, because we moved it into the target image space manuslly before and this way we avoid interpolation issues with RTDose
                struct = ants2sitk(ants.apply_transforms(fixed = sitk2ants(struct), moving = sitk2ants(struct), transformlist = reg['fwdtransforms'], interpolator  = 'nearestNeighbor'))
                transformed_structs[key] = struct
            return transformed_t1, transformed_t2, transformed_structs
        else:
            return transformed_t1, transformed_t2
    
class FilterRegisterMain():
    """
    Filter register main processor. Generates and runs individual registration jobs
    """
    def __init__(
            self,
            bids_set: pl.Path,
            clean_set: pl.Path,
            inclusion_criterion = {'studies >=': 0, 'obersvation period >=': 0, 'avg study interval <=': np.inf, 'rtstructs present >=': 1},
            n_jobs=5
            ):
        """
        bids_set = source directory, the bids style output from bids coiner
        clean_set = target directory
        inclusion_criterion = dict that specifies what patients are processed. default config processes any patient with at least one RTStruct in the files
        n_jobs = int, the number of parallel registration workers
          """
        self.bids_set = bids_set
        self.clean_set = clean_set
        self.inclusion_criterion = inclusion_criterion
        self.n_jobs = n_jobs
        #if self.n_jobs != 1: raise RuntimeError("Multiprocessing is not allowed because the internal segmentation engine will lead to errors. This is still a work in progress.")
        os.makedirs(clean_set, exist_ok=True)
        self.log = Printer(log_type='txt', log_prefix='Filter-Register', location=clean_set)
        self.info_format = PPFormat([ColourText('blue'), Effect('bold'), Effect('underlined')]) 

    def execute(self, test_mode=False):
        """
        Run the Preprocessor on the set

        :param test_mode -> if True will run filters and pre registration logs, but skip doing the actual registration work.

        first builds all registration jobs and saves them in a list, then runs them in parallel.
        
        Tasks:
        filter anatomical and RT studies
        sort chronologically
        remove redundancies from anatomical studies
        identify target image in each anatomical study if multiple T1 or T2 are present
        assign an anat study to each RT study by lowest timedifference
        build registration job 
        run in parallel
        """
        
        patients = [elem for elem in os.listdir(self.bids_set) if (self.bids_set/elem).is_dir() and elem.startswith('sub-')]
 
        if test_mode: self.log.warning("PatientProcessor for filtration and Registration is running in test mode, no actual output will be produced.")
        # create log file that tracks completed patients in case of error
        with open(self.clean_set/'filter_register_progress.txt', mode='a+') as progfile:
            progfile.seek(0)  # Move cursor to the beginning before reading
            processed = progfile.readlines()
            processed = [pat.replace('\n', '') for pat in processed]
            patients = [pat for pat in patients if pat not in processed]
            jobs = []
            errors = []
            # iterate patients
            self.log.tagged_print('INFO', 'Setting up registration jobs...', self.info_format)
            start = time.time()
            for pat in patients:
                try:
                    ordered_studies = self._sort_directories(pat)
                    ordered_anat, ordered_rt = self._clean_redundancies(ordered_studies, pat)
                    study_dict = self._find_corresponding_anat(ordered_anat, ordered_rt)
                    study_dict, dates = self._crop_patient_history(study_dict)
                
                    keys = list(study_dict.keys())

                    if self._discard_patient(study_dict, dates):
                        self.log.warning(f"Discarding patient {pat}")
                        progfile.write(pat+'\n')
                        continue

                
                    self.log.tagged_print('INFO', f'Processing {pat}, fulfills criterion: {self.inclusion_criterion}', self.info_format)                
                    self.log.tagged_print('MISC', f"Patient study matching: {study_dict}", self.info_format)

                    if self.n_jobs != 1:
                        jobs.append(PatientRegistrationJob(self.bids_set, self.clean_set, pat, study_dict, keys))
                        self.log.success(f"Created registration job for {pat}")
                    else:
                        reg = PatientRegistrationJob(self.bids_set, self.clean_set, pat, study_dict, keys)
                        reg.execute()

                except Exception as e:
                    self.log(f'Caught Exception in patient {pat}')
                    self.log(str(e))
                    errors.append(pat)
                    traceback.print_exc()
            stop = time.time()
            period = stop-start
            mins, secs = divmod(period, 60)
            hrs, mins = divmod(mins, 60)
            self.log.tagged_print('INFO', f'Setting up jobs took {hrs}h {mins}min {secs}s', self.info_format)
            if any(jobs):
                self.log.tagged_print('INFO', 'Running registrations', self.info_format)
                start = time.time()
                results = []
                with ProcessPoolExecutor(max_workers=self.n_jobs, mp_context=multiprocessing.get_context("spawn")) as executor:
                    futures = [executor.submit(job.execute) for job in jobs]
                    for future in as_completed(futures):
                        results.append(future.result())
                stop = time.time()
                period = stop-start
                mins, secs = divmod(period, 60)
                hrs, mins = divmod(mins, 60)
                self.log.tagged_print('INFO', f'Registration completed in {hrs}h {mins}min {secs}s', self.info_format)

                self.log.tagged_print('INFO', 'Checking for errors', self.info_format)
                for res in results:
                    if "SUCCESS" == res.split('--')[-1]:
                        self.log.success(f"""No errors to report for {res.split('--')[0]}""")
                        progfile.write(res.split('--')[0]+'\n')
                    else:
                        self.log.error(f"""Error occured in {res.split('--')[0]}, message:\n{res}""")
                        errors.append(res.split('--')[0])

            
            res_files = [f for f in os.listdir(self.clean_set) if f.startswith('sub')]
            self.log(f'{len(res_files)} Patients are in the output directory')
            self.log(f'This run handeled {len(jobs)} patients in multiprocessing')
            self.log(f'Caught exceptions in patients {errors}, fix and rerun they have not been logged in the progress file.')
                
                
    ################################################ Filtration Utilities
    def _discard_patient(self, study_dict, dates):
        keys = list(study_dict.keys())
        structs = list(study_dict.values())
        dates = list(dates.values())

        if not any(dates): # some studies have no anat, just one rts, these are to be discarded. 
            return True

        structs = len([s for s in structs if s is not None])
        observed = (dates[-1]-dates[0]).days
        studies = len(keys)
        avg_interval = self._get_avg_int(dates)

        #print(f'Structs = {structs}, Sudies = {studies}, Observed days = {observed}, Interval = {avg_interval}')

        return False if (studies >= self.inclusion_criterion['studies >='] and observed >= self.inclusion_criterion['obersvation period >='] and avg_interval <= self.inclusion_criterion['avg study interval <='] and structs >= self.inclusion_criterion['rtstructs present >=']) else True

    def _get_avg_int(self, studies:list):
        deltas = []
        for i in range(len(studies)-1):
            deltas.append((studies[i+1] - studies[i]).days)

        return sum(deltas)/len(deltas) if len(deltas)!= 0 else 0
    
    def _crop_patient_history(self, study_dict):
        """
        Removes studies from the dict that are prior to treatment
        """
        keys = list(study_dict.keys())
        # parse strings to dates again
        pattern = r"^ses-(\d{14})$"
        dates = {}
        for d in keys:
            match = re.match(pattern, d)
            if match:
                timestamp_str = match.group(1)
                # Convert the timestamp string to a datetime object
                dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                dates[d] = dt

        if any([elem for elem in list(study_dict.values()) if elem is not None]): # probably the worst subroutine ever written but well it works
            cropped_dict = {}
            cropped_dates = {}
            t0 = None
            for key in list(study_dict.keys()):
                if t0 is None:
                    if study_dict[key] is not None:
                        t0 = key
                        cropped_dict[key] = study_dict[key]
                        cropped_dates[key] = dates[key]
                else:
                    cropped_dict[key] = study_dict[key]
                    cropped_dates[key] = dates[key]
            keys = list(cropped_dict.keys())
            # remove large time gaps after the treatment
            for k in range(len(keys)-1, 1, -1):
                key = keys[k]
                if cropped_dates[keys[k-1]] + timedelta(days=6*30) < cropped_dates[key] and cropped_dict[key] is None:
                    del cropped_dates[key]
                    del cropped_dict[key]
            self.log.warning(f'Cropped patient history dict from {study_dict}\nto {cropped_dict}')
            return cropped_dict, cropped_dates

        else: # treatment day list fallback, could also be the 6 month filter fallback
            # TODO: actually implement the filter and treatment days observation.
            return study_dict, dates
      
    def _find_corresponding_anat(self, ordered_anat: list, ordered_rt: list):
        """
        Assigns an Anatomical study to each RT study according to the smallest time difference
        If no rts are found in the patient just returns the dict with all values being None
        """
        #print(f'Got anat list: {ordered_anat}\nGot rt list:{ordered_rt}')
        if any(ordered_rt) and any(ordered_anat):
            study_dict = {}

            # identify correspondences
            anat_dt = [dt for d, dt in ordered_anat]
            rt_dt = [dt for d, dt in ordered_rt]
            rt_refs = []
            for rt in rt_dt:
                rt_refs.append(min(enumerate(anat_dt), key=lambda x: abs(x[1] - rt))[0])
            # parse result dictionary
            assigned = {idx: rt[0] for rt, idx in zip(ordered_rt, rt_refs)}
            study_dict = {}
            for i, (anat_id, _) in enumerate(ordered_anat):
                study_dict[anat_id] = assigned.get(i, None)
            # rt_indexer = 0
            # for i in range(rt_refs[0], len(ordered_anat)):
            #     if i in rt_refs:
            #         study_dict[ordered_anat[i][0]] = ordered_rt[rt_indexer][0]
            #         rt_indexer += 1
            #     else:
            #         study_dict[ordered_anat[i][0]] = None
                
            return study_dict
            
        else:
            study_dict = {}
            for elem in ordered_anat:
                study_dict[elem[0]] = None
            return study_dict

    def _clean_redundancies(self, studies: list, pat: str, delta_days: int = 14):
        """
        Filters the anatomical studies, removing any redundancies i.e. when the time between images is too low 
        """
        anat = [elem for elem in studies if not (self.bids_set/pat/elem[0]/'rt').is_dir()] # include empty studies
        rt = [elem for elem in studies if (self.bids_set/pat/elem[0]/'rt').is_dir()]

        # The previous RT conversion step will create empty rt folders in the Bids structure if the conversion fails inside the platipy method
        # This little routine is meant to filter out such empty rts to correctly filter for valid data
        refiltered_rt = []
        for item in rt:
            dirl = os.listdir(self.bids_set/pat/item[0]/'rt')
            for d in dirl:
                if (self.bids_set/pat/item[0]/'rt'/d).is_dir():
                    if os.listdir(self.bids_set/pat/item[0]/'rt'/d):
                        refiltered_rt.append(item)

        # And another nested for loop filter routine to remove unneccesary structures from the RT set
        # sometimes it happens that there are structure sets with only lesions outside the brain or the entire brain as a lesion.
        # these are to be excluded
        # why are all the solutions i come up with slow af
        cleaned_rt = []
        for rtses in refiltered_rt:
            dirl = os.listdir(self.bids_set/pat/rtses[0]/'rt')
            for rtset in dirl:
                if (self.bids_set/pat/rtses[0]/'rt'/rtset).is_dir():
                    removed = []

                    ct = [ct for ct in os.listdir(self.bids_set/pat/rtses[0]/'rt') if (self.bids_set/pat/rtses[0]/'rt'/ct).is_file() and ct.endswith('- CT_reference.nii.gz')]
                    structs = [struct for struct in os.listdir(self.bids_set/pat/rtses[0]/'rt'/rtset) if 'PTV' in struct or 'GTV' in struct]
                    
                    if len(ct)>1:
                        ct = [cti for cti in ct if cti.startswith(rtset)]

                    if (self.bids_set/pat/rtses[0]/'rt'/rtset/'Struct_Brain.nii.gz').is_file(): # try loading from RTS
                        ref_brain = self._load_image(self.bids_set/pat/rtses[0]/'rt'/rtset/'Struct_Brain.nii.gz')
                    elif (self.bids_set/pat/rtses[0]/'rt'/('MASK_'+ct[0])).is_file(): # try loading from previous segmentation
                            ref_brain = self._load_image(self.bids_set/pat/rtses[0]/'rt'/('MASK_'+ct[0]))
                    else: # use total segmentator to do it
                        ref_brain = self._segment_image(self._load_image(self.bids_set/pat/rtses[0]/'rt'/ct[0]), 'ct')
                        sitk.WriteImage(ref_brain, self.bids_set/pat/rtses[0]/'rt'/('MASK_'+ct[0]))

                    ref_brain = sitk.GetArrayFromImage(ref_brain).astype(bool)
                    brain_size = ref_brain.sum()
                    # find stuff to remove
                    for struct in structs:
                        cur_struct = self._load_image(self.bids_set/pat/rtses[0]/'rt'/rtset/struct)
                        cur_struct = sitk.GetArrayFromImage(cur_struct).astype(bool)
                        lesion_size = cur_struct
                        lesion_overlap = np.bitwise_and(ref_brain, cur_struct).sum()
                        lesion_size = cur_struct.sum()
                        if not ((lesion_size == lesion_overlap) and (lesion_size <= (brain_size*0.1)) and (lesion_size != 0)): # remove condition == inverted keep condition
                            os.makedirs(self.bids_set/pat/rtses[0]/'invalid_rt'/rtset, exist_ok=True)
                            removed.append(struct) # store in to remove list
                            # do the actual removing by just rerouting the path to a dir tagged as invalid

                            os.rename(self.bids_set/pat/rtses[0]/'rt'/rtset/struct, self.bids_set/pat/rtses[0]/'invalid_rt'/rtset/struct)

                    # now if not all target file names have been removed keep the rt session
                    if not len(removed) == len(structs):
                        cleaned_rt.append(rtses)

        #rt = list(set(cleaned_rt)) # cast to set and back in order to remove duplicates this bullshit randomizes the order fuck off

        seen = set()
        rt = []
        for item in cleaned_rt:
            if item not in seen:
                seen.add(item)
                rt.append(item)
        rt.sort(key=lambda x: x[1])  # Ensure chronological order
        #if rt != cleaned_rt: self.log.warning("found duplicates in patient!")
        # Filter out directories that are within 14 days of a more recent one
        filtered_dirs = []
        while anat:
            latest = anat.pop()  # Take the latest directory
            filtered_dirs.append(latest)
            # Remove any directories within 14 days before the latest one
            anat = [(d, dt) for d, dt in anat if dt <= latest[1] - timedelta(days=delta_days)]
        
        return [d for d in sorted(filtered_dirs, key=lambda x: x[1])], rt

    def _sort_directories(self, pat: str): # courtesy of chatgpt
        """
        Finds directories in the specified base_dir that match the pattern
        ses-yyyymmddhhmmss, parses the timestamp, and returns a list of directory
        names sorted in chronological order.
        """
        base_dir = self.bids_set/pat
        # List all directories in the base directory
        dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))] # include empty studies
        pattern = r"^ses-(\d{14})$"
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

    ################################################ Registration Utilities
    def _load_image(self, path, resample=(1,1,1)):
        #print('reading image from path', path)
        img = sitk.ReadImage(path)
        if resample is not None and not img.GetSpacing == resample:
            new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(img.GetSize(), img.GetSpacing(), resample)
            ]
            img = sitk.Resample(img, new_size, outputSpacing=resample, outputDirection=img.GetDirection(), outputOrigin=img.GetOrigin(), outputPixelType=img.GetPixelID())
        return img 
               
    
    
   