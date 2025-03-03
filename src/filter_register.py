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



class PatientPreprocessor():
    def __init__(
            self,
            bids_set: pl.Path,
            clean_set: pl.Path
            ):
        self.bids_set = bids_set
        self.clean_set = clean_set

        self.log = Printer(log_type=None)
        self.info_format = PPFormat([ColourText('blue'), Effect('bold'), Effect('underlined')]) 

    def execute(self):
        """
        Run the Preprocessor on the set
        
        Tasks:
        filter anatomical and RT studies
        sort chronologically
        remove redundancies from anatomical studies
        identify target image in each anatomical study if multiple T1 or T2 are present
        assign an anat study to each RT study by lowest timedifference
        identify t0 as first anatomical with an rt
        segment MRs to obtain brainmask
        segment CTs to obtain brainmask if no Brainmask in RT
        Register RTs to corresponding MRs
        Register MRs to t0
        Save all results to new and lighter set
        """
        patients = [elem for elem in os.listdir(self.bids_set) if (self.bids_set/elem).is_dir() and elem.startswith('sub-')]
        # iterate patients
        for pat in patients:
            ordered_anat, ordered_rt = self._sort_directories(pat)

            if  not any(ordered_rt): # fallback if no RTstructs are found
                self.log.warning(f'Could not find RTSTRUCTS for patient {pat}')
                study_dict = {}
                for elem in ordered_anat:
                    study_dict[elem[0]] = None
            else:
                study_dict = self._find_corresponding_anat(ordered_anat, ordered_rt)

            t0_image = None
            t0_mask = None
            
            keys = list(study_dict.keys())


            # iterate studies
            for key in keys:
                anat_study = key
                rt_study = study_dict[key]
                if not (self.bids_set/pat/anat_study/'anat').is_dir():
                    self.log.warning('unexpectedly received an empty directory as anatomical')
                    continue
                
                # filter study files
                images = os.listdir(self.bids_set/pat/anat_study/'anat')
                t1 = [img for img in images if img.endswith('T1w.nii.gz') or img.endswith('T1wa.nii.gz')]
                t2 = [img for img in images if img.endswith('T2w.nii.gz') or img.endswith('T2wa.nii.gz')]

                # filter results if multiple t1 and t2 sequences have been coined
                t1, t2 = self._filter_anat(t1, t2)

                if t1 is None:
                    if rt_study is None:
                        self.log.fail(f'Could not find T1 in study {anat_study}')
                    else:
                        tp = 'T1' if t2 is None else 'T2'
                        tp = 'both' if t2 is None and t1 is None else tp
                        self.log.fail(f'Found RTSTRUCT but matched mr study is incomplete, missing {tp}')
                    continue

                # make output directory
                os.makedirs(self.clean_set/pat/anat_study/'anat', exist_ok=True)

                ############## set and copy t0 T1
                if (self.bids_set/pat/anat_study/'anat').is_dir() and t0_image is None:
                    self.log.tagged_print('INFO', f'Identified t0 as study day {anat_study} for patient {pat}', self.info_format)

                    # read images
                    t0_image = sitk.ReadImage(self.bids_set/pat/anat_study/'anat'/t1)
                    # save unprocessed fixed
                    sitk.WriteImage(t0_image, self.clean_set/pat/anat_study/'anat'/t1)
                    if t2 is not None:
                        sitk.WriteImage(sitk.ReadImage(self.bids_set/pat/anat_study/'anat'/t2), self.clean_set/pat/anat_study/'anat'/t2)
                    # obtain mask
                    if not (self.clean_set/pat/anat_study/'anat'/('MASK_'+t1)).is_file():
                        t0_mask = self._segment_image(t0_image, 'mr')
                        sitk.WriteImage(t0_mask, (self.clean_set/pat/anat_study/'anat'/('MASK_'+t1)))
                    else:
                        t0_mask = sitk.ReadImage((self.clean_set/pat/anat_study/'anat'/('MASK_'+t1)))
                        
                ############## if the anatomical image has a corresponding rt image do ct2mr2mr
                if rt_study is not None:
                    os.makedirs(self.clean_set/pat/anat_study/'rt', exist_ok =True)
                    # separate ct, rts and dose into lists with filenames
                    rts = os.listdir(self.bids_set/pat/rt_study/'rt')
                    ct = [elem for elem in rts if elem.endswith('CT_reference.nii.gz')]
                    rtd = [elem for elem in rts if elem.endswith('RTDOSE.nii.gz')]
                    rts = [elem for elem in rts if (self.bids_set/pat/rt_study/'rt'/elem).is_dir()]

                    if not any(ct): # fallback in case the CT is missing
                        self.log.warning(f'Could not find any ct in study {rt_study}, processing anatomical images anyway')
                        t1_image = sitk.ReadImage(self.bids_set/pat/anat_study/'anat'/t1)
                        if t2 is not None:
                            t2_image = sitk.ReadImage(self.bids_set/pat/anat_study/'anat'/t2)
                        else:
                            t2_image = None
                        
                        t1_trans, t2_trans = self._register_mr2mr(t0_image, t0_mask, t1_image, t2_image)
                        sitk.WriteImage(t1_trans, self.clean_set/pat/anat_study/'anat'/t1)
                        if t2 is not None:
                            sitk.WriteImage(t2_trans, self.clean_set/pat/anat_study/'anat'/t2)
                        self.log.success(f'Anatomical images from study {anat_study} to t0, failed to convert RTs because the reference CT is missing')
                        continue

                        
                
                    RTdict = {} # dict that stores RT files, with filepath as key and sitk.Image as value
                    # read dose images
                    if any(rtd):
                        for d in rtd:
                            RTdict[d] = sitk.ReadImage(self.bids_set/pat/rt_study/'rt'/d)
                    # read structure images
                    if any(rts):
                        for s in rts:
                            os.makedirs(self.clean_set/pat/anat_study/'rt'/s, exist_ok =True)
                            for file in os.listdir(self.bids_set/pat/rt_study/'rt'/s):
                                RTdict[s+'/'+file] = sitk.ReadImage(self.bids_set/pat/rt_study/'rt'/s/file)

                    # set ct as moving image
                    ct_image = sitk.ReadImage(self.bids_set/pat/rt_study/'rt'/ct[0])

                    # get ct segmentation
                    if (self.bids_set/pat/rt_study/'rt'/'Struct_Brain.nii.gz').is_file(): # try loading from RTS
                        ct_mask = sitk.ReadImage(self.bids_set/pat/rt_study/'rt'/'Struct_Brain.nii.gz')
                    elif (self.clean_set/pat/anat_study/'rt'/('MASK_'+ct[0])).is_file(): # try loading from previous segmentation
                        ct_mask = sitk.ReadImage((self.clean_set/pat/anat_study/'rt'/('MASK_'+ct[0])))
                    else: # use total segmentator to do it
                        ct_mask = self._segment_image(ct_image, 'ct')
                        sitk.WriteImage(ct_mask, (self.clean_set/pat/anat_study/'rt'/('MASK_'+ct[0])))
                        
                    t1_image = sitk.ReadImage(self.bids_set/pat/anat_study/'anat'/t1)

                    # segment t1
                    if not (self.clean_set/pat/anat_study/'anat'/('MASK_'+t1)).is_file():
                        t1_mask = self._segment_image(t1_image, 'mr')
                        sitk.WriteImage(t1_mask, (self.clean_set/pat/anat_study/'anat'/('MASK_'+t1)))
                    else:
                        t1_mask = sitk.ReadImage((self.clean_set/pat/anat_study/'anat'/('MASK_'+t1)))

                    if t2 is not None:
                        t2_image = sitk.ReadImage(self.bids_set/pat/anat_study/'anat'/t2)
                    else:
                        t2_image = None

                    # run registration and transform structures, move ct to corresponding mrt1
                    transformed_structs = self._register_ct2mr(t1_image, t1_mask, ct_image, ct_mask, RTdict)
                    # run registration move all to t0
                    t1_trans, t2_trans, transformed_structs = self._register_mr2mr(t0_image, t0_mask, t1_image, t2_image, transformed_structs)
                    # make output directory and save processed images
                    for key in list(transformed_structs.keys()):
                        struct = transformed_structs[key]
                        sitk.WriteImage(struct, (self.clean_set/pat/anat_study/'rt'/key))
                    sitk.WriteImage(t1_trans, self.clean_set/pat/anat_study/'anat'/t1)
                    if t2 is not None:
                        sitk.WriteImage(t2_trans, self.clean_set/pat/anat_study/'anat'/t2)
                    self.log.success(f'Moved structs, dose and anatomical images from study {anat_study} and struct study {rt_study} to t0')

                ############### if there is no corresponding rt just do mr2mr
                else:
                    t1_image = sitk.ReadImage(self.bids_set/pat/anat_study/'anat'/t1)
                    if t2 is not None:
                        t2_image = sitk.ReadImage(self.bids_set/pat/anat_study/'anat'/t2)
                    else:
                        t2_image = None
                    t1_trans, t2_trans = self._register_mr2mr(t0_image, t0_mask, t1_image, t2_image)
                    sitk.WriteImage(t1_trans, self.clean_set/pat/anat_study/'anat'/t1)
                    if t2 is not None:
                        sitk.WriteImage(t2_trans, self.clean_set/pat/anat_study/'anat'/t2)
                    self.log.success(f'Moved anatomical images from study {anat_study} to t0')
                
                

    def _filter_anat(self, t1:list, t2:list):
        ## process t1 images
        t1_filtered = self._filter_list(t1)
        ## process t2 images
        t2_filtered = self._filter_list(t2)
        return t1_filtered, t2_filtered
    
    def _filter_list(self, mr: list) -> list: # courtesy of mister gpt
        """
        filters filenames in the study to identify a single T1 and T2 image in the direcotry
        prioritizes runs over reconstructions
        prirotizes sag recs over any other recs
        """
        best_run = None
        best_rec = None
        
        for image in mr:
            run_number = self._extract_key_value(image, 'run')
            rec_number = self._extract_key_value(image, 'rec')
            
            if run_number != -1:  # Prioritize images with 'run'
                if best_run is None or run_number > self._extract_key_value(best_run, 'run'):
                    best_run = image
            elif rec_number != -1:  # Consider 'rec' images only if no 'run' images exist
                if best_rec is None or rec_number > self._extract_key_value(best_rec, 'rec'):
                    best_rec = image
                elif rec_number == self._extract_key_value(best_rec, 'rec') and 'sag' in image:
                    best_rec = image  # Prioritize 'sag' if rec number is the same
        
        return best_run if best_run else best_rec
        
    def _extract_key_value(self, filename:str, key:str):
        match = re.search(rf"_{key}-(\d+)_", filename)
        return int(match.group(1)) if match else -1  # Return -1 if key is missing


    def _segment_image(self, image: sitk.Image, mode: str) -> sitk.Image:
        """
        Runs the total segmentator on an image, differentiating between mr and ct by the mode string
        returns the brain mask for the given image
        """
        if mode == 'ct':
            mask = totalsegmentator(sitk2nib(image), fast=False, task='total', quiet=True)
            mask = nib2sitk(mask, image)
            mask = sitk.BinaryThreshold(mask, lowerThreshold=90, upperThreshold=90, insideValue=1, outsideValue=0) # get segmentation brain=90, skull=91
            return mask
        elif mode == 'mr':
            mask = totalsegmentator(sitk2nib(image), fast=False, task='total_mr', quiet=True) # get segmentation brain = 50
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

        reg = ants.registration(fixed_mask, moving_mask, 'Affine')

        # warp and reconvert structure sets
        transformed_structs = {}
        for key in list(structs.keys()):
            struct = structs[key]
            # using the struct itself as fixed, because we moved it into the target image space manuslly before and this way we avoid interpolation issues with RTDose
            struct = ants2sitk(ants.apply_transforms(fixed = sitk2ants(struct), moving = sitk2ants(struct), transformlist = reg['fwdtransforms'], interpolator  = 'nearestNeighbor'))
            transformed_structs[key] = struct
        return transformed_structs

    def _sort_directories(self, pat: list): # courtesy of chatgpt
        """
        Finds directories in the specified base_dir that match the pattern
        ses-yyyymmddhhmmss, parses the timestamp, and returns a list of directory
        names sorted in chronological order.
        """
        base_dir = self.bids_set/pat
        # Regular expression pattern to match directory names like "ses-20250224235959"
        pattern = r"^ses-(\d{14})$"
        # List all directories in the base directory
        dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and ((base_dir/d/'anat').is_dir() or (base_dir/d/'rt').is_dir())]
        
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
        return self._clean_redundancies(matching_dirs, pat)
    
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

    
    def _register_mr2mr(self, fixed: sitk.Image, fixed_mask: sitk.Image, moving_t1: sitk.Image, moving_t2: sitk.Image, structs: dict=None):
        """
        Register the MR images to the t0 MR image
        Transforms corresponding structs if they are present
        """
        if structs is not None:
            structs = self._prepro_structs(fixed, structs)
        fixed = sitk2ants(fixed)
        fixed_mask = sitk2ants(fixed_mask)
        moving_t1 = sitk2ants(moving_t1)
        reg = ants.registration(fixed, moving_t1, mask=fixed_mask)
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
    
    def _find_corresponding_anat(self, ordered_anat: list, ordered_rt: list):
        """
        Assigns an Anatomical study to each RT study according to the smallest time difference
        """
        study_dict = {}

        # identify correspondences
        anat_dt = [dt for d, dt in ordered_anat]
        rt_dt = [dt for d, dt in ordered_rt]
        rt_refs = []
        for rt in rt_dt:
            rt_refs.append(min(enumerate(anat_dt), key=lambda x: abs(x[1] - rt))[0])
        # parse result dictionary
        rt_indexer = 0
        for i in range(rt_refs[0], len(ordered_anat)):
            if i in rt_refs:
                study_dict[ordered_anat[i][0]] = ordered_rt[rt_indexer][0]
                rt_indexer += 1
            else:
                study_dict[ordered_anat[i][0]] = None
            
        return study_dict


    def _clean_redundancies(self, studies: list, pat: str, delta_days: int = 14):
        """
        Filters the anatomical studies, removing any redundancies i.e. when the time between images is too low 
        """
        anat = [elem for elem in studies if (self.bids_set/pat/elem[0]/'anat').is_dir()] # seperate liste by if they have anat or rt
        rt = [elem for elem in studies if (self.bids_set/pat/elem[0]/'rt').is_dir()]

        # Filter out directories that are within 14 days of a more recent one
        filtered_dirs = []
        while anat:
            latest = anat.pop()  # Take the latest directory
            filtered_dirs.append(latest)
            # Remove any directories within 14 days before the latest one
            anat = [(d, dt) for d, dt in anat if dt <= latest[1] - timedelta(days=delta_days)]
        
        return [d for d in sorted(filtered_dirs, key=lambda x: x[1])], rt
