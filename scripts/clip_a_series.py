import pydicom
import os
import numpy as np
import pathlib as pl
import SimpleITK as sitk
import matplotlib.pyplot as plt


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

    target_series = pl.Path('/home/lorenz/data/series_to_fix/pat-85_ses-20221025132459/1100_Sagittal 3D FSPGR Gado reformats')

    # for slice in os.listdir(target_series):
    #     process_dicom(target_series/slice)

    reader = sitk.ImageSeriesReader()
    dcm_files = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(target_series)
    reader.SetFileNames(dcm_files)
    img = reader.Execute()

    img = sitk.ReadImage('/mnt/nas6/data/Target/batch_copy/struct filter exp/clean/sub-PAT0015/ses-20190324124058/anat/MASK_sub-PAT0015_ses-20190324124058_run-2_T1w.nii.gz')#'/mnt/nas6/data/Target/batch_copy/struct filter exp/clean/sub-PAT0085/ses-20221025132459/anat/sub-PAT0085_ses-20221025132459_run-1_T1w.nii.gz')

    # filter = sitk.MedianImageFilter()
    # img = filter.Execute(img)

    img_arr = sitk.GetArrayFromImage(img)

    hist, bin_edges = np.histogram(img_arr)

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.plot(bin_edges[:-1], hist, lw=2)
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.title("Image Histogram")
    plt.grid()
    plt.savefig('histogram_damaged_clip.png')
    plt.show()


    sitk.WriteImage(img, str(target_series)+'.nii.gz')
