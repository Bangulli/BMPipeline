### Registration Imports ###
from registration import input_output as io
from registration import registration as reg
from registration import warping as w
from registration import configs
from registration import utils
import torch
import pathlib as pl
from PrettyPrint import *
import os
from datetime import datetime
import re
import SimpleITK as sitk
import numpy as np
import subprocess
import torch.nn.functional as F
from totalsegmentator.python_api import totalsegmentator
import nibabel as nib

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
        for pat in patients:
            ordered_studies = self._sort_directories(self.bids_set/pat)
            fixed_image = None
            for study in ordered_studies:
                os.makedirs(self.clean_set/pat/study, exist_ok=True)
                if (self.bids_set/pat/study/'anat').is_dir():
                    os.makedirs(self.clean_set/pat/study/'anat', exist_ok=True)
                    images = os.listdir(self.bids_set/pat/study/'anat') # get all images in the study
                    t1 = [img for img in images if img.endswith('T1w.nii.gz') or img.endswith('T1wa.nii.gz')]
                    t2 = [img for img in images if img.endswith('T2w.nii.gz') or img.endswith('T2wa.nii.gz')]

                    # filter results if multiple t1 and t2 sequences have been coined
                    t1, t2 = self._filter_anat(t1, t2)

                    # register when no fixed image is present, to avoid registering it to itself
                    if fixed_image == None:
                        # read images
                        fixed_image = sitk.ReadImage(self.bids_set/pat/study/'anat'/t1)
                        # save unprocessed fixed
                        sitk.WriteImage(fixed_image, self.clean_set/pat/study/'anat'/t1)
                        # normalize fixed
                        if not (self.clean_set/pat/study/'anat'/('MASK_'+t1)).is_file():
                            fixed_mask = totalsegmentator(self._sitk2nib(fixed_image), fast=False, task='total_mr', quiet=True) # get segmentation brain = 50
                            fixed_mask = self._nib2sitk(fixed_mask, fixed_image)
                            fixed_mask = sitk.BinaryThreshold(fixed_mask, lowerThreshold=50, upperThreshold=50, insideValue=1, outsideValue=0) #  binarize segmentation
                            sitk.WriteImage(fixed_mask, (self.clean_set/pat/study/'anat'/('MASK_'+t1)))
                        else:
                            fixed_mask = sitk.ReadImage((self.clean_set/pat/study/'anat'/('MASK_'+t1)))
                        fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
                        fixed_image = sitk.RescaleIntensity(fixed_image, outputMinimum=0, outputMaximum=1)
                        

                # register structure and dose
                if (self.bids_set/pat/study/'rt').is_dir() and fixed_image is not None:
                    rts = os.listdir(self.bids_set/pat/study/'rt')
                    ct = [elem for elem in rts if elem.endswith('CT_reference.nii.gz')]
                    rtd = [elem for elem in rts if elem.endswith('RTDOSE.nii.gz')]
                    rts = [elem for elem in rts if (self.bids_set/pat/study/'rt'/elem).is_dir()]

                    if len(ct)>1:
                        print('found more than one ct in the study')
                    
                    RTdict = {}
                    for d in rtd:
                        RTdict[d] = sitk.ReadImage(self.bids_set/pat/study/'rt'/d)

                    for s in rts:
                        os.makedirs(self.clean_set/pat/study/'rt'/s, exist_ok =True)
                        for file in os.listdir(self.bids_set/pat/study/'rt'/s):
                            RTdict[s+'/'+file] = sitk.ReadImage(self.bids_set/pat/study/'rt'/s/file)

                    moving_image = sitk.ReadImage(self.bids_set/pat/study/'rt'/ct[0])
                    if not (self.clean_set/pat/study/'rt'/('MASK_'+ct[0])).is_file():
                        moving_mask = totalsegmentator(self._sitk2nib(moving_image), fast=False, task='total', quiet=True)
                        moving_mask = self._nib2sitk(moving_mask, moving_image)
                        moving_mask = sitk.BinaryThreshold(moving_mask, lowerThreshold=90, upperThreshold=90, insideValue=1, outsideValue=0) # get segmentation brain=90, skull=91
                        sitk.WriteImage(moving_mask, (self.clean_set/pat/study/'rt'/('MASK_'+ct[0])))
                    else:
                        moving_mask = sitk.ReadImage((self.clean_set/pat/study/'rt'/('MASK_'+ct[0])))
                    transformed_image, transformed_structs = self._register_ct2mr(fixed_image, fixed_mask, moving_image, moving_mask, RTdict)
                    sitk.WriteImage(transformed_image, self.clean_set/pat/study/'rt'/ct[0])
                    for key in list(transformed_structs.keys()):
                        struct = transformed_structs[key]
                        sitk.WriteImage(struct, (self.clean_set/pat/study/'rt'/key))
                

    def _filter_anat(self, t1:list, t2:list):
        # TODO write the actual function lol
        return t1[0], t2[0]
    
    def print_image_info(self, image, name):
        print(f"{name}:")
        print(f"  Origin: {image.GetOrigin()}")
        print(f"  Spacing: {image.GetSpacing()}")
        print(f"  Direction: {image.GetDirection()}")
        print(f"  Size: {image.GetSize()}")

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
        sitk.WriteImage(binary_mask, 'before_ct_mask.nii.gz')
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
        #sitk.WriteImage(moving_image, 'rawCT.nii.gz')

        # crop ct to area of interest
        moving_image, moving_mask, structs = self._prepro_ct(moving_image, moving_mask, structs, fixed_image)

        sitk.WriteImage(moving_image, 'processedCT.nii.gz')

        # normalize images
        moving_image_norm = sitk.Cast(moving_image, sitk.sitkFloat32)
        moving_image_norm = sitk.RescaleIntensity(moving_image_norm, outputMinimum=0, outputMaximum=1)

        # prepare arrays
        fixed_arr = sitk.GetArrayFromImage(fixed_image).astype(np.float32)
        moving_arr = sitk.GetArrayFromImage(moving_image_norm).astype(np.float32)
        fixed_arr = torch.from_numpy(fixed_arr).unsqueeze(0).unsqueeze(0)
        moving_arr = torch.from_numpy(moving_arr).unsqueeze(0).unsqueeze(0)

        # apply masking
        fixed_msk = sitk.GetArrayFromImage(fixed_mask).astype(np.float32)
        moving_msk = sitk.GetArrayFromImage(moving_mask).astype(np.float32)
        fixed_msk = torch.from_numpy(fixed_msk).unsqueeze(0).unsqueeze(0)
        moving_msk = torch.from_numpy(moving_msk).unsqueeze(0).unsqueeze(0)

        # prepare config
        params = configs.affine_config # Registration accuracy can be adjusted by changing the config
        device = params['device']
        # run registration on normalized images
        # we want to move CT to MR but registration from MR to CT is easier, so we do the inverse and then invert it
        displacement_field = reg.run_affine_registration(fixed_msk, moving_msk, **params)
        res = w.warp_tensor(fixed_arr.to(device), displacement_field).cpu().detach().squeeze().numpy()
        res = sitk.GetImageFromArray(res)
        res.CopyInformation(fixed_image)
        sitk.WriteImage(res, 'reverse_reg_res.nii')
        displacement_field_inverse = displacement_field*-1 # affine field inversion by flipping the vectors in the field. should be valid as in an affine transformation every vector is computed the same way

        # warp image and reconvert to sitk.Image
        # uses the non normalized image, to keep data intact
        res_moving = sitk.GetArrayFromImage(moving_image).astype(np.float32)
        res_moving = torch.from_numpy(res_moving).unsqueeze(0).unsqueeze(0)
        transformed_arr = w.warp_tensor(res_moving.to(device), displacement_field_inverse).cpu().detach().squeeze().numpy()
        transformed_image = sitk.GetImageFromArray(transformed_arr)
        transformed_image.CopyInformation(fixed_image)

        # warp and reconvert structure sets
        if structs is not None:
            transformed_structs = {}
            for key in list(structs.keys()):
                struct = sitk.GetArrayFromImage(structs[key]).astype(np.float32)
                struct = torch.from_numpy(struct).unsqueeze(0).unsqueeze(0)
                struct = w.warp_tensor(struct.to(device), displacement_field_inverse).cpu().detach().squeeze().numpy()
                struct = sitk.GetImageFromArray(struct)
                struct.CopyInformation(fixed_image)
                transformed_structs[key] = struct
            return transformed_image, transformed_structs
        else:
            return transformed_image
        

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
    
    def _sitk2nib(self, image_sitk):
        """
        Convert a SimpleITK Image to a nibabel NIfTI1Image.
        """
        # Convert SimpleITK image to numpy array
        image_np = sitk.GetArrayFromImage(image_sitk)  # Shape: (D, H, W)

        # Get spacing, origin, and direction
        spacing = image_sitk.GetSpacing()  # (sx, sy, sz)
        origin = image_sitk.GetOrigin()  # (ox, oy, oz)
        direction = image_sitk.GetDirection()  # Flattened 3x3 matrix

        # Convert SimpleITK direction (1D tuple) to a 3x3 matrix
        direction_matrix = np.array(direction).reshape(3, 3)

        # Create affine matrix (vox2ras)
        affine = np.eye(4)
        affine[:3, :3] = np.diag(spacing) @ direction_matrix  # Scale and rotate
        affine[:3, 3] = origin  # Set translation

        # Nibabel expects (H, W, D) instead of (D, H, W), so transpose axes
        image_np = np.transpose(image_np, (2, 1, 0))

        # Create Nibabel NIfTI image
        nifti_image = nib.Nifti1Image(image_np, affine)

        return nifti_image
    
    def _nib2sitk(self, image_nib, sitk_ref):
        image_np = image_nib.get_fdata()
        image_np = np.transpose(image_np, (2, 1, 0))
        image_sitk = sitk.GetImageFromArray(image_np)
        image_sitk.CopyInformation(sitk_ref)
        return image_sitk