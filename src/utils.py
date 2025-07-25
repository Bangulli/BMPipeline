import ants
import numpy as np
import SimpleITK as sitk
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