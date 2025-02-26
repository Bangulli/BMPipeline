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
                        if not (self.clean_set/pat/study/'anat'/('MASK_'+t1)).is_file():
                            subprocess.run(
                                ['hd-bet',
                                '-i', self.bids_set/pat/study/'anat'/t1,
                                '-o', self.clean_set/pat/study/'anat'/('MASK_'+t1)]
                            )
                        # normalize fixed
                        fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
                        fixed_image = sitk.RescaleIntensity(fixed_image, outputMinimum=0, outputMaximum=1)
                        fixed_mask = sitk.ReadImage(self.clean_set/pat/study/'anat'/('MASK_'+t1))

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

                    transformed_image, transformed_structs = self._register_ct2mr(fixed_image, fixed_mask, moving_image, RTdict)
                    sitk.WriteImage(transformed_image, self.clean_set/pat/study/'rt'/ct[0])
                    for struct in list(transformed_structs.keys()):
                        sitk.WriteImage(transformed_structs[struct], self.clean_set/pat/study/'rt'/struct)
                

    def _filter_anat(self, t1:list, t2:list):
        # TODO write the actual function lol
        return t1[0], t2[0]
    
    def print_image_info(self, image, name):
        print(f"{name}:")
        print(f"  Origin: {image.GetOrigin()}")
        print(f"  Spacing: {image.GetSpacing()}")
        print(f"  Direction: {image.GetDirection()}")
        print(f"  Size: {image.GetSize()}")

    def _prepro_ct(self, source, structs, target):
        binary_mask = sitk.BinaryThreshold(source, lowerThreshold=700, upperThreshold=3000, insideValue=1, outsideValue=0)
        cc_filter = sitk.ConnectedComponentImageFilter()
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()

        # Find connected components
        cc = cc_filter.Execute(binary_mask)
        label_shape_filter.Execute(cc)

        # Get label of the largest component
        largest_label = max(label_shape_filter.GetLabels(), key=lambda l: label_shape_filter.GetPhysicalSize(l))

        # Keep only the largest component
        binary_mask = sitk.BinaryThreshold(cc, lowerThreshold=largest_label, upperThreshold=largest_label, insideValue=1, outsideValue=0)
        # Dilate the mask
        binary_mask = sitk.BinaryDilate(binary_mask, (5,5,5), sitk.sitkBall)
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
        # apply the same to the rtstructs and doses
        if structs is not None:
            for key in structs.keys():
                struct = structs[key]
                struct = struct[x:x + w, y:y + h, z:z + d]
                struct = sitk.DICOMOrient(struct, sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(target.GetDirection()))
                struct.SetOrigin(target.GetOrigin())
                struct.SetDirection(target.GetDirection())
                # finally resample
                struct = sitk.Resample(struct, referenceImage=target, defaultPixelValue=-1000)
                structs[key] = struct
        
        return source, structs
    
    def _invert_mr2ct(self, df, img, device):
        df = df.squeeze().cpu()
        H, W, D, _ = df.shape  # Get dimensions

        # Generate a grid of voxel coordinates (homogeneous)
        target_grid = torch.stack(torch.meshgrid(
            torch.linspace(0, H-1, H, device=df.device),
            torch.linspace(0, W-1, W, device=df.device),
            torch.linspace(0, D-1, D, device=df.device),
            indexing='ij'  # Ensure correct layout
        ), dim=-1).reshape(-1, 3)  # Shape [N, 3], Flattened

        # Convert relative displacement field to absolute source coordinates
        df_absolute = df * torch.tensor([H, W, D], dtype=torch.float32, device=df.device)  # Scale back to voxel space
        source_coords = target_grid + df_absolute.reshape(-1, 3)  # Add displacement to target grid

        # Add ones for homogeneous coordinates
        A_matrix = torch.cat([target_grid, torch.ones(target_grid.shape[0], 1, device=df.device)], dim=-1)  # [N, 4]
        B_matrix = source_coords  # [N, 3]

        # Solve for affine transformation using least squares
        affine_params = torch.linalg.lstsq(A_matrix, B_matrix).solution  # Shape [4,3]


        affine_matrix = affine_params.T  # [3, 4]

        # Convert to 4x4 homogeneous transformation matrix
        full_affine_matrix = torch.eye(4)
        full_affine_matrix[:3, :] = affine_matrix

        affine_inverse = torch.inverse(full_affine_matrix)[:3, :].unsqueeze(0)

        print(affine_inverse)
        
        return utils.tc_transform_to_tc_df(affine_matrix.unsqueeze(0).to(device), img.size(), device=device)

    def _register_ct2mr(self, fixed_image, fixed_mask, moving_image, structs=None):
        """
        This horrible mess takes sitk images, normalizes and resamples them, then converts them to tensors, registers them, converts them to sitk images and finally returns the results
        """
        #sitk.WriteImage(moving_image, 'rawCT.nii.gz')

        # crop ct to area of interest
        moving_image, structs = self._prepro_ct(moving_image, structs, fixed_image)
        sitk.WriteImage(moving_image, 'processedCT.nii.gz')

        # normalize images
        moving_image_norm = sitk.Cast(moving_image, sitk.sitkFloat32)
        moving_image_norm = sitk.RescaleIntensity(moving_image_norm, outputMinimum=0, outputMaximum=1)

        # prepare tensors
        fixed_arr = sitk.GetArrayFromImage(fixed_image).astype(np.float32)
        moving_arr = sitk.GetArrayFromImage(moving_image_norm).astype(np.float32)
        fixed_arr = torch.from_numpy(fixed_arr).unsqueeze(0).unsqueeze(0)
        moving_arr = torch.from_numpy(moving_arr).unsqueeze(0).unsqueeze(0)

        # prepare config
        params = configs.affine_config # Registration accuracy can be adjusted by changing the config
        device = params['device']
        # run registration on normalized images
        # we want to move CT to MR but registration from MR to CT is easier, so we do the inverse and then invert it
        displacement_field = reg.run_affine_registration(fixed_arr, moving_arr, **params)
        displacement_field_inverse = displacement_field*-1
        # displacement_field_inverse = self._invert_mr2ct(displacement_field, moving_arr, device)
        # displacement_field_inverse = displacement_field_inverse.to(device)
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
            # for key in list(structs.keys()):
            #     struct = sitk.Resample(structs[key], referenceImage=fixed_image)
            #     struct = torch.from_numpy(sitk.GetArrayFromImage(struct)).unsqueeze(0).unsqueeze(0)
            #     transformed_structs[key] = sitk.GetImageFromArray(w.warp_tensor(struct.to(device), displacement_field).cpu().detach().squeeze().numpy()).CopyInformation(fixed_image)
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