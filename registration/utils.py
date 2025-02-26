### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union

### External Imports ###
import numpy as np
import torch as tc
import torch.nn.functional as F

### Internal Imports ###

########################

def normalize(tensor : tc.Tensor) -> tc.Tensor:
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

def generate_grid(tensor_size: tc.Tensor, device=None):
    """
    Generates the identity grid for a given tensor size.

    Parameters
    ----------
    tensor_size : tc.Tensor or tc.Size
        The tensor size used to generate the regular grid
    device : str
        The device used for resampling (e.g. "cpu" or "cuda:0")
    
    Returns
    ----------
    grid : tc.Tensor
        The regular grid (relative for warp_tensor with align_corners=False)
    """
    identity_transform = tc.eye(len(tensor_size)-1, device=device)[:-1, :].unsqueeze(0)
    identity_transform = tc.repeat_interleave(identity_transform, tensor_size[0], dim=0)
    grid = F.affine_grid(identity_transform, tensor_size, align_corners=False)
    return grid

def tc_transform_to_tc_df(transformation: tc.Tensor, size: tc.Size, device: str=None):
    """
    Transforms the transformation tensor into the displacement field tensor.

    Parameters
    ----------
    transformation : tc.Tensor
        The transformation tensor (B x transformation size (2x3 or 3x4))
    size : tc.Tensor (or list, or tuple)
        The desired displacement field size
    device : str
        The device used for resampling (e.g. "cpu" or "cuda:0")
    Returns
    ----------
    resampled_displacement_field: tc.Tensor
        The resampled displacement field (BxYxXxZxD)
    """
    device = device if device is not None else transformation.device
    deformation_field = F.affine_grid(transformation, size=size, align_corners=False) #.to(device)
    size = (deformation_field.size(0), 1) + deformation_field.size()[1:-1]
    grid = generate_grid(size, device=device)
    displacement_field = deformation_field - grid
    return displacement_field

def create_pyramid(tensor: tc.Tensor, num_levels: int, device: str=None, mode: str='bilinear'):
    """
    Creates the resolution pyramid of the input tensor (assuming uniform resampling step = 2).

    Parameters
    ----------
    tensor : tc.Tensor
        The input tensor
    num_levels: int
        The number of output levels
    device : str
        The device used for the calculation (e.g. "cpu" or "cuda:0")
    mode : str
        The interpolation mode ("bilinear" or "nearest")
    
    Returns
    ----------
    pyramid: list of tc.Tensor
        The created resolution pyramid

    TO DO: Add optional Gaussian filtering to avoid aliasing for deep pyramids
    """
    device = device if device is not None else tensor.device
    pyramid = []
    for i in range(num_levels):
        if i == num_levels - 1:
            pyramid.append(tensor)
        else:
            current_size = tensor.size()
            new_size = (int(current_size[j]/(2**(num_levels-i-1))) if j > 1 else current_size[j] for j in range(len(current_size)))
            new_size = tc.Size(new_size)
            new_tensor = resample_tensor(tensor, new_size, device=device, mode=mode)
            pyramid.append(new_tensor)
    return pyramid


def resample_tensor(tensor: tc.Tensor, new_size: tc.Tensor, device: str=None, mode: str='bilinear'):
    """
    Resamples the input tensor to a given, new size (may be used both for down and upsampling).
    Uses F.grid_sample for the structured data interpolation (only linear and nearest supported).
    Be careful - the autogradient calculation is possible only with mode set to "bilinear".

    Parameters
    ----------
    tensor : tc.Tensor
        The tensor to be resampled (BxYxXxZxD)
    new_size : tc.Tensor (or list, or tuple)
        The resampled tensor size
    device : str
        The device used for resampling (e.g. "cpu" or "cuda:0")
    mode : str
        The interpolation mode ("bilinear" or "nearest")

    Returns
    ----------
    resampled_tensor : tc.Tensor
        The resampled tensor (Bxnew_sizexD)
    """
    device = device if device is not None else tensor.device
    sampling_grid = generate_grid(new_size, device=device)
    resampled_tensor = F.grid_sample(tensor, sampling_grid, mode=mode, padding_mode='zeros', align_corners=False)
    return resampled_tensor



def resample_displacement_field(displacement_field: tc.Tensor, new_size: tc.Tensor, device: str=None, mode: str='bilinear'):
    """
    Resamples the given displacement field.
    Uses F.grid_sample for the structured data interpolation (only linear and nearest supported).
    Be careful - the function will not maintain the gradient history (independently of the mode used).

    Parameters
    ----------
    displacement_field : tc.Tensor
        The PyTorch displacement field to be resampled (BxYxXxZxD)
    new_size : tc.Tensor (or list, or tuple)
        The resampled tensor size
    device : str
        The device used for resampling (e.g. "cpu" or "cuda:0")
    mode : str
        The interpolation mode ("bilinear" or "nearest")

    Returns
    ----------
    resampled_displacement_field: tc.Tensor
        The resampled displacement field (BxYxXxZxD)
    """
    device = device if device is not None else displacement_field.device
    sampling_grid = generate_grid((1,) + new_size[:-1], device=device) #.to(device)
    resampled_displacement_field = tc.zeros(new_size, device=device) #.to(device)
    size = displacement_field.size()
    ndim = len(size) - 2
    for i in range(size[-1]):
        if ndim == 2:
            resampled_displacement_field[:, :, :, i] = F.grid_sample(displacement_field[:, :, :, i].unsqueeze(0), sampling_grid, mode=mode, padding_mode='zeros', align_corners=False)[0]
        elif ndim == 3:
            resampled_displacement_field[:, :, :, :, i] = F.grid_sample(displacement_field[:, :, :, :, i].unsqueeze(0), sampling_grid, mode=mode, padding_mode='zeros', align_corners=False)[0]
        else:
            raise ValueError("Unsupported number of dimensions.")
    return resampled_displacement_field


def tc_size_to_df_size(tensor : tc.Tensor):
    """
    Calculates the desired displacement field size based on the input tensor.
    
    Parameters
    ----------
    tensor : tc.Tensor
        The input tensor

    Returns
    ----------
    size: tuple
        The displacement field size
    """
    tsize = tensor.size()
    ndim = len(tsize) - 2
    size = (tsize[0], ) + (tuple(list(tsize[2:],))) + (ndim,)
    return size


def np_df_to_tc_df(displacement_field_np: np.ndarray, device: str="cpu"):
    """
    Convert the displacement field in NumPy to the displacement field in PyTorch (assuming uniform spacing and align_corners set to false).

    Parameters
    ----------
    displacement_field_np : np.ndarray
        The NumPy displacment field (DxYxXxZ)

    Returns
    ----------
     displacement_field_tc : tc.Tensor
        The PyTorch displacement field (1xZxXxYxD)
    """
    shape = displacement_field_np.shape
    ndim = len(shape) - 1
    if ndim == 2:
        displacement_field_tc = tc.from_numpy(displacement_field_np.copy())
        displacement_field_tc = displacement_field_tc.permute(1, 2, 0).unsqueeze(0)
        temp_df_copy = displacement_field_tc.clone()
        displacement_field_tc[:, :, :, 0] = temp_df_copy[:, :, :, 0] / (shape[2]) * 2.0
        displacement_field_tc[:, :, :, 1] = temp_df_copy[:, :, :, 1] / (shape[1]) * 2.0
    if ndim == 3:
        displacement_field_tc = tc.from_numpy(displacement_field_np.copy())
        displacement_field_tc = displacement_field_tc.permute(1, 2, 3, 0).unsqueeze(0)
        temp_df_copy = displacement_field_tc.clone()
        displacement_field_tc[:, :, :, :, 0] = temp_df_copy[:, :, :, :, 2] / (shape[3]) * 2.0
        displacement_field_tc[:, :, :, :, 1] = temp_df_copy[:, :, :, :, 0] / (shape[2]) * 2.0
        displacement_field_tc[:, :, :, :, 2] = temp_df_copy[:, :, :, :, 1] / (shape[1]) * 2.0
    return displacement_field_tc.to(device)