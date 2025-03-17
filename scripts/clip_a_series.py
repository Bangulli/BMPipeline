import pydicom
import os
import numpy as np
import pathlib as pl
import SimpleITK as sitk
import matplotlib.pyplot as plt
from totalsegmentator.python_api import totalsegmentator
import ants
import nibabel as nib

def sitk2ants(sitk_image: "SimpleITK.Image") -> ants.ANTsImage:
    """
    Converts a given SimpleITK image into an ANTsPy image

    Parameters
    ----------
        img: SimpleITK.Image

    Returns
    -------
        ants_image: ANTsImage
    """
    import SimpleITK as sitk

    ndim = sitk_image.GetDimension()

    if ndim < 3:
        print("Dimensionality is less than 3.")
        return None

    direction = np.asarray(sitk_image.GetDirection()).reshape((3, 3))
    spacing = list(sitk_image.GetSpacing())
    origin = list(sitk_image.GetOrigin())

    data = sitk.GetArrayViewFromImage(sitk_image)

    ants_img: ants.ANTsImage = ants.from_numpy(
        data=data.ravel(order="F").reshape(data.shape[::-1]),
        origin=origin,
        spacing=spacing,
        direction=direction,
    )

    return ants_img


def ants2sitk(ants_image: ants.ANTsImage) -> "SimpleITK.Image":
    """
    Converts a given ANTsPy image into an SimpleITK image

    Parameters
    ----------
        ants_image: ANTsImage

    Returns
    -------
        img: SimpleITK.Image
    """

    import SimpleITK as sitk

    data = ants_image.view()
    shape = ants_image.shape

    sitk_img = sitk.GetImageFromArray(data.ravel(order="F").reshape(shape[::-1]))
    sitk_img.SetOrigin(ants_image.origin)
    sitk_img.SetSpacing(ants_image.spacing)
    sitk_img.SetDirection(ants_image.direction.flatten())
    return sitk_img

def sitk2nib(image_sitk):
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
    
def nib2sitk(image_nib, sitk_ref):
    image_np = image_nib.get_fdata()
    image_np = np.transpose(image_np, (2, 1, 0))
    image_sitk = sitk.GetImageFromArray(image_np)
    image_sitk.CopyInformation(sitk_ref)
    return image_sitk

def process_dicom(file_path):
    try:
        ds = pydicom.dcmread(file_path, force=True)  # Attempt to read DICOM without relying on extension
        
        # Check if the DICOM file is an MRI image
        if ds.Modality == "MR":
            if hasattr(ds, "PixelData"):
                # Convert pixel data to numpy array
                pixel_array = ds.pixel_array.astype(np.float32)
                
                # Clip negative values to zero
                pixel_array[pixel_array < 0] = 0
                
                # Convert back to original dtype
                ds.PixelData = pixel_array.astype(ds.pixel_array.dtype).tobytes()
                
                # Save modified DICOM file (overwrite original)
                ds.save_as(file_path)
                print(f"Processed: {file_path}")
    except Exception as e:
        print(f"Skipping {file_path}: Not a valid DICOM file or error occurred ({e})")

if __name__ == '__main__':

    target_series = pl.Path('/home/lorenz/data/series_to_fix/pat-214_ses-20220912/1200_Sagittal 3D FSPGR Gado reformats')

    for slice in os.listdir(target_series):
        process_dicom(target_series/slice)

    reader = sitk.ImageSeriesReader()
    dcm_files = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(target_series)
    reader.SetFileNames(dcm_files)
    img = reader.Execute()

    sitk.WriteImage(img, str(target_series)+'.nii.gz')

    mask = totalsegmentator(sitk2nib(img), fast=False, task='total_mr', quiet=True) # get segmentation brain = 50
    mask = nib2sitk(mask, img)
    mask = sitk.BinaryThreshold(mask, lowerThreshold=50, upperThreshold=50, insideValue=1, outsideValue=0) #  binarize segmentation

    sitk.WriteImage(mask, str(target_series)+'_total_seg_mask.nii.gz')

    # img = sitk.ReadImage('/mnt/nas6/data/Target/PROCESSED_mrct1000_nobatch/sub-PAT0214/ses-20220912124733/anat/sub-PAT0214_ses-20220912124733_run-1_T1w.nii.gz')#'/mnt/nas6/data/Target/batch_copy/struct filter exp/clean/sub-PAT0085/ses-20221025132459/anat/sub-PAT0085_ses-20221025132459_run-1_T1w.nii.gz')

    # # filter = sitk.MedianImageFilter()
    # # img = filter.Execute(img)

    # img_arr = sitk.GetArrayFromImage(img)

    # hist, bin_edges = np.histogram(img_arr)

    # # Plot histogram
    # plt.figure(figsize=(8, 6))
    # plt.plot(bin_edges[:-1], hist, lw=2)
    # plt.xlabel("Intensity Value")
    # plt.ylabel("Frequency")
    # plt.title("Image Histogram")
    # plt.grid()
    # plt.savefig('histogrm_new.png')
    # plt.show()


    
