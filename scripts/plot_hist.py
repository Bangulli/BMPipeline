import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np

img = sitk.ReadImage('/mnt/nas6/data/Target/batch_copy/reseg_test/set/sub-PAT0085/ses-20231122180927/anat/sub-PAT0085_ses-20231122180927_acq-sag_run-2_T1w.nii.gz')


img_arr = sitk.GetArrayFromImage(img)

hist, bin_edges = np.histogram(img_arr)

# Plot histogram
plt.figure(figsize=(8, 6))
plt.plot(bin_edges[:-1], hist, lw=2)
plt.xlabel("Intensity Value")
plt.ylabel("Frequency")
plt.title("Image Histogram")
plt.grid()
plt.savefig('X_histogrm.png')
plt.show()