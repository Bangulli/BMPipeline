### Registration Imports ###
import pathlib as pl
from PrettyPrint import *
import os
from datetime import datetime
import re
import SimpleITK as sitk
import numpy as np
from totalsegmentator.python_api import totalsegmentator
import nibabel as nib
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

        self.log = Printer()
        self.info_format = PPFormat([ColourText('blue'), Effect('bold'), Effect('underlined')]) 

    def execute(self):
        patients = [elem for elem in os.listdir(self.bids_set) if (self.bids_set/elem).is_dir()]
        # iterate patients
        for pat in patients:
            ordered_studies = self._sort_directories(self.bids_set/pat)
            fixed_image = None
            fixed_mask = None
            t0 = False
            # iterate studies
            for i, study in enumerate(ordered_studies):
                # skip studies that were taken before the first rtstruct is generated

                # assumes that the first mri is immediately before the first rtstruct, might be unstable
                if not t0:
                    if (self.bids_set/pat/ordered_studies[i+1]/'rt').is_dir():
                        t0=True
                    else:
                        continue
                # check if anat files exist in bids set
                if (self.bids_set/pat/study/'anat').is_dir():
                    # make output directory
                    os.makedirs(self.clean_set/pat/study/'anat', exist_ok=True)
                    # filter study files
                    images = os.listdir(self.bids_set/pat/study/'anat')
                    t1 = [img for img in images if img.endswith('T1w.nii.gz') or img.endswith('T1wa.nii.gz')]
                    t2 = [img for img in images if img.endswith('T2w.nii.gz') or img.endswith('T2wa.nii.gz')]

                    # filter results if multiple t1 and t2 sequences have been coined
                    t1, t2 = self._filter_anat(t1, t2)

                    # load fixed image and mask, happens in the first timepoint only, once per patient
                    if fixed_image == None:
                        # read images
                        fixed_image = sitk.ReadImage(self.bids_set/pat/study/'anat'/t1)
                        # save unprocessed fixed
                        sitk.WriteImage(fixed_image, self.clean_set/pat/study/'anat'/t1)
                        sitk.WriteImage(sitk.ReadImage(self.bids_set/pat/study/'anat'/t2), self.clean_set/pat/study/'anat'/t2)
                        # obtain mask
                        if not (self.clean_set/pat/study/'anat'/('MASK_'+t1)).is_file():
                            fixed_mask = self._segment_image(fixed_image, 'mr')
                            sitk.WriteImage(fixed_mask, (self.clean_set/pat/study/'anat'/('MASK_'+t1)))
                        else:
                            fixed_mask = sitk.ReadImage((self.clean_set/pat/study/'anat'/('MASK_'+t1)))
                    # register mr images in study to first timepoint
                    else:
                        t1_img = sitk.ReadImage(self.bids_set/pat/study/'anat'/t1)
                        t2_img = sitk.ReadImage(self.bids_set/pat/study/'anat'/t2)
                        t1_img, t2_img = self._register_mr2mr(fixed_image, fixed_mask, t1_img, t2_img)
                        sitk.WriteImage(t1_img, self.clean_set/pat/study/'anat'/t1)
                        sitk.WriteImage(t2_img, self.clean_set/pat/study/'anat'/t2)
                        
                # register structure and dose
                if (self.bids_set/pat/study/'rt').is_dir() and fixed_image is not None:
                    # separate ct, rts and dose into lists with filenames
                    rts = os.listdir(self.bids_set/pat/study/'rt')
                    ct = [elem for elem in rts if elem.endswith('CT_reference.nii.gz')]
                    rtd = [elem for elem in rts if elem.endswith('RTDOSE.nii.gz')]
                    rts = [elem for elem in rts if (self.bids_set/pat/study/'rt'/elem).is_dir()]
                
                    RTdict = {} # dict that stores RT files, with filepath as key and sitk.Image as value
                    # read dose images
                    for d in rtd:
                        RTdict[d] = sitk.ReadImage(self.bids_set/pat/study/'rt'/d)
                    # read structure images
                    for s in rts:
                        for file in os.listdir(self.bids_set/pat/study/'rt'/s):
                            RTdict[s+'/'+file] = sitk.ReadImage(self.bids_set/pat/study/'rt'/s/file)

                    # set ct as moving image
                    moving_image = sitk.ReadImage(self.bids_set/pat/study/'rt'/ct[0])

                    # get segmentation
                    if not (self.clean_set/pat/study/'rt'/('MASK_'+ct[0])).is_file():
                        moving_mask = self._segment_image(moving_image, 'ct')
                        sitk.WriteImage(moving_mask, (self.clean_set/pat/study/'rt'/('MASK_'+ct[0])))
                    else:
                        moving_mask = sitk.ReadImage((self.clean_set/pat/study/'rt'/('MASK_'+ct[0])))

                    # run registration and transform structures
                    transformed_structs = self._register_ct2mr(fixed_image, fixed_mask, moving_image, moving_mask, RTdict)

                    # make output directory and save processed images
                    os.makedirs(self.clean_set/pat/study/'rt'/s, exist_ok =True)
                    for key in list(transformed_structs.keys()):
                        struct = transformed_structs[key]
                        sitk.WriteImage(struct, (self.clean_set/pat/study/'rt'/key))
                
                elif (self.bids_set/pat/study/'rt').is_dir() and fixed_image is None:
                    self.log.warning('Found structure sets before a fixed T1 image was detected')
                

    def _filter_anat(self, t1:list, t2:list):
        # TODO write the actual function lol
        self.log.warning('Standin function for file sorting if multiple t1/t2 are found, please write the actual function this is only for test purposes')
        return t1[0], t2[0]

    def _segment_image(self, image, mode):
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

    def _prepro_ct(self, source, binary_mask, structs, target):
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

    def _register_ct2mr(self, fixed_image, fixed_mask, moving_image, moving_mask, structs=None):
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
            struct = ants2sitk(ants.apply_transforms(fixed = fixed_mask, moving = sitk2ants(struct), transformlist = reg['fwdtransforms'], interpolator  = 'nearestNeighbor'))
            transformed_structs[key] = struct
        return transformed_structs

        

    def _sort_directories(self, base_dir): # courtesy of chatgpt
        """
        Finds directories in the specified base_dir that match the pattern
        ses-yyyymmddhhmmss, parses the timestamp, and returns a list of directory
        names sorted in chronological order.
        """
        # Regular expression pattern to match directory names like "ses-20250224235959"
        pattern = r"^ses-(\d{14})$"
        
        # List all directories in the base directory
        dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
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
        return [d for d, _ in matching_dirs]
    
    def _register_mr2mr(self, fixed, fixed_mask, moving_t1, moving_t2):
        fixed = sitk2ants(fixed)
        fixed_mask = sitk2ants(fixed_mask)
        moving_t1 = sitk2ants(moving_t1)
        moving_t2 = sitk2ants(moving_t2)
        reg = ants.registration(fixed, moving_t1, mask=fixed_mask)
        transformed_t1 = ants.apply_transforms(fixed, moving_t1, transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor')
        transformed_t2 = ants.apply_transforms(fixed, moving_t2, transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor')
        return ants2sitk(transformed_t1), ants2sitk(transformed_t2)