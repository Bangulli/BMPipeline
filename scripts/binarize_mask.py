import SimpleITK as sitk

input = '/mnt/nas6/data/Target/PROCESSED_mrct1000_nobatch/sub-PAT1083/ses-20200707023358/mets/metastasis_labels_3_class.nii.gz'
output = '/mnt/nas6/data/Target/PROCESSED_mrct1000_nobatch/sub-PAT1083/ses-20200707023358/mets/metastasis_labels_1_class.nii.gz'

mask = sitk.ReadImage(input)
binary = sitk.GetArrayFromImage(mask)
binary[binary == 2] = 0 # ignore edema
binary[binary != 0] = 1 # binarize tumor & necrosis
binary = sitk.GetImageFromArray(binary)
binary.CopyInformation(mask)
sitk.WriteImage(binary, output)