### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union

### External Imports ###
import torch as tc
import torch.nn.functional as F

### Internal Imports ###
import utils as u

########################

def warp_tensor(tensor: tc.Tensor, displacement_field: tc.Tensor, grid: tc.Tensor=None, device: str=None, mode: str='bilinear'):
    """
    Transforms a tensor with a given displacement field.
    Uses F.grid_sample for the structured data interpolation (only linear and nearest supported).
    Be careful - the autogradient calculation is possible only with mode set to "bilinear".

    Parameters
    ----------
    tensor : tc.Tensor
        The tensor to be transformed (BxYxXxZxD)
    displacement_field : tc.Tensor
        The PyTorch displacement field (BxYxXxZxD)
    grid : tc.Tensor (optional)
        The input identity grid (optional - may be provided to speed-up the calculation for iterative algorithms)
    device : str
        The device used for warping (e.g. "cpu" or "cuda:0")
    mode : str
        The interpolation mode ("bilinear" or "nearest")

    Returns
    ----------
    transformed_tensor : tc.Tensor
        The transformed tensor (BxYxXxZxD)
    """
    device = device if device is not None else tensor.device
    if grid is None:
        grid = u.generate_grid(tensor.size(), device=device)
    sampling_grid = grid + displacement_field
    transformed_tensor = F.grid_sample(tensor, sampling_grid, mode=mode, padding_mode='zeros', align_corners=False)
    return transformed_tensor

def transform_tensor(tensor: tc.Tensor, sampling_grid: tc.Tensor, grid: tc.Tensor=None, device: str=None, mode: str='bilinear'):
    """
    Transforms a tensor with a given sampling grid.
    Uses F.grid_sample for the structured data interpolation (only linear and nearest supported).
    Be careful - the autogradient calculation is possible only with mode set to "bilinear".

    Parameters
    ----------
    tensor : tc.Tensor
        The tensor to be transformed (BxYxXxZxD)
    sampling_grid : tc.Tensor
        The PyTorch sampling grid
    grid : tc.Tensor (optional)
        The input identity grid (optional - may be provided to speed-up the calculation for iterative algorithms)
    device : str
        The device used for warping (e.g. "cpu" or "cuda:0")
    mode : str
        The interpolation mode ("bilinear" or "nearest")

    Returns
    ----------
    transformed_tensor : tc.Tensor
        The transformed tensor (BxYxXxZxD)
    """
    transformed_tensor = F.grid_sample(tensor, sampling_grid, mode=mode, padding_mode='zeros', align_corners=False)
    return transformed_tensor


def compose_displacement_fields(source, displacement_field_1, displacement_field_2, device=None):
    device = device if device is not None else source.device
    size = displacement_field_1.size()
    sampling_grid = u.generate_grid(source.size(), device=device) #NOTE: in other version was: `sampling_grid = generate_grid(source, (1,) + size[0:-1])`
    composed_displacement_field = tc.zeros(size).type_as(source)
    ndim = len(size) - 2
    for i in range(size[-1]):
        composed_displacement_field[:, :, :, :, i] = F.grid_sample((sampling_grid[:, :, :, :, i] + displacement_field_1[:, :, :, :, i]).unsqueeze(0), sampling_grid + displacement_field_2, padding_mode='zeros', align_corners=False)[0]
    composed_displacement_field = composed_displacement_field - sampling_grid
    return composed_displacement_field