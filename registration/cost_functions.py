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

class MutualInformation(tc.nn.Module):
    """
    Mutual Information
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = tc.autograd.Variable(tc.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        y_pred = tc.clamp(y_pred, 0., self.max_clip)
        y_true = tc.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = tc.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = tc.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = tc.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = tc.exp(- self.preterm * tc.square(y_true - vbc))
        I_a = I_a / tc.sum(I_a, dim=-1, keepdim=True)

        I_b = tc.exp(- self.preterm * tc.square(y_pred - vbc))
        I_b = I_b / tc.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = tc.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = tc.mean(I_a, dim=1, keepdim=True)
        pb = tc.mean(I_b, dim=1, keepdim=True)

        papb = tc.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = tc.sum(tc.sum(pab * tc.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)

class MaskedMutualInformation(tc.nn.Module):
    """
    Mutual Information
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MaskedMutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = tc.autograd.Variable(tc.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred, y_true_mask, y_pred_mask):
        y_pred = tc.clamp(y_pred, 0., self.max_clip)
        y_true = tc.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = tc.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = tc.unsqueeze(y_pred, 2)

        # prep mask
        if y_true_mask is None:
            mask = y_pred_mask
        elif y_pred_mask is None:
            mask = y_true_mask
        else:
            mask = y_true_mask*y_pred_mask

        mask = mask.view(mask.shape[0], -1)
        mask = tc.unsqueeze(mask, 2)

        # crop imgs
        y_true = self._crop_to_bbox(y_true, mask)
        y_pred = self._crop_to_bbox(y_pred, mask)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = tc.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = tc.exp(- self.preterm * tc.square(y_true - vbc))
        I_a = I_a / tc.sum(I_a, dim=-1, keepdim=True)

        I_b = tc.exp(- self.preterm * tc.square(y_pred - vbc))
        I_b = I_b / tc.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = tc.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = tc.mean(I_a, dim=1, keepdim=True)
        pb = tc.mean(I_b, dim=1, keepdim=True)

        papb = tc.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = tc.sum(tc.sum(pab * tc.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred, y_true_mask, y_pred_mask):
        return -self.mi(y_true, y_pred, y_true_mask, y_pred_mask)

    def _crop_to_bbox(self, image: tc.Tensor, mask: tc.Tensor):
        """
        Crops the input 3D image tensor to the bounding box of the mask.
        
        Args:
            image (torch.Tensor): The input 3D image tensor of shape (C, D, H, W) or (D, H, W).
            mask (torch.Tensor): The binary mask tensor of the same shape as the image (excluding channels).

        Returns:
            cropped_image (torch.Tensor): The cropped image tensor.
            bbox (tuple): Bounding box coordinates (z_min, z_max, y_min, y_max, x_min, x_max).
        """

        # Find the coordinates where the mask is 1
        mask_coords = tc.nonzero(mask, as_tuple=False)  # Get all foreground coordinates (N, 3)

        if mask_coords.shape[0] == 0:
            raise ValueError("Mask contains no foreground elements (all zeros).")

        # Get min and max along each dimension (Z, Y, X)
        z_min, y_min, x_min = mask_coords.min(dim=0)[0]
        z_max, y_max, x_max = mask_coords.max(dim=0)[0]

        # Crop the image based on the computed bounding box
        if image.dim() == 4:  # If image has channels (C, D, H, W)
            cropped_image = image[:, z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        else:  # If image is (D, H, W)
            cropped_image = image[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]

        return cropped_image

class localMutualInformation(tc.nn.Module):
    """
    Local Mutual Information for non-overlapping patches
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32, patch_size=5):
        super(localMutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = tc.autograd.Variable(tc.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers
        self.patch_size = patch_size

    def local_mi(self, y_true, y_pred):
        y_pred = tc.clamp(y_pred, 0., self.max_clip)
        y_true = tc.clamp(y_true, 0, self.max_clip)

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = tc.reshape(self.vol_bin_centers, o).cuda()

        """Making image paddings"""
        if len(list(y_pred.size())[2:]) == 3:
            ndim = 3
            x, y, z = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            z_r = -z % self.patch_size
            padding = (z_r // 2, z_r - z_r // 2, y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        elif len(list(y_pred.size())[2:]) == 2:
            ndim = 2
            x, y = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            padding = (y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        else:
            raise Exception('Supports 2D and 3D but not {}'.format(list(y_pred.size())))
        y_true = F.pad(y_true, padding, "constant", 0)
        y_pred = F.pad(y_pred, padding, "constant", 0)

        """Reshaping images into non-overlapping patches"""
        if ndim == 3:
            y_true_patch = tc.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_true_patch = tc.reshape(y_true_patch, (-1, self.patch_size ** 3, 1))

            y_pred_patch = tc.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_pred_patch = tc.reshape(y_pred_patch, (-1, self.patch_size ** 3, 1))
        else:
            y_true_patch = tc.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 3, 5)
            y_true_patch = tc.reshape(y_true_patch, (-1, self.patch_size ** 2, 1))

            y_pred_patch = tc.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 3, 5)
            y_pred_patch = tc.reshape(y_pred_patch, (-1, self.patch_size ** 2, 1))

        """Compute MI"""
        I_a_patch = tc.exp(- self.preterm * tc.square(y_true_patch - vbc))
        I_a_patch = I_a_patch / tc.sum(I_a_patch, dim=-1, keepdim=True)

        I_b_patch = tc.exp(- self.preterm * tc.square(y_pred_patch - vbc))
        I_b_patch = I_b_patch / tc.sum(I_b_patch, dim=-1, keepdim=True)

        pab = tc.bmm(I_a_patch.permute(0, 2, 1), I_b_patch)
        pab = pab / self.patch_size ** ndim
        pa = tc.mean(I_a_patch, dim=1, keepdim=True)
        pb = tc.mean(I_b_patch, dim=1, keepdim=True)

        papb = tc.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = tc.sum(tc.sum(pab * tc.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()

    def forward(self, y_true, y_pred):
        return -self.local_mi(y_true, y_pred)
    
class maskedLocalMutualInformation(tc.nn.Module):
    """
    Local Mutual Information for non-overlapping patches
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32, patch_size=5):
        super(maskedLocalMutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = tc.autograd.Variable(tc.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers
        self.patch_size = patch_size

    def local_mi(self, y_true, y_true_mask, y_pred, y_pred_mask):
        y_pred = tc.clamp(y_pred, 0., self.max_clip)
        y_true = tc.clamp(y_true, 0, self.max_clip)

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = tc.reshape(self.vol_bin_centers, o).cuda()

        """Making image paddings"""
        if len(list(y_pred.size())[2:]) == 3:
            ndim = 3
            x, y, z = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            z_r = -z % self.patch_size
            padding = (z_r // 2, z_r - z_r // 2, y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        else:
            raise Exception('Supports 3D but not {}'.format(list(y_pred.size())))
        y_true = F.pad(y_true, padding, "constant", 0)
        y_pred = F.pad(y_pred, padding, "constant", 0)


        if y_true_mask is None:
            y_pred_mask = F.pad(y_pred_mask, padding, "constant", 0)
            sampling_mask = y_pred_mask
        elif y_pred_mask is None:
            y_true_mask = F.pad(y_true_mask, padding, "constant", 0)
            sampling_mask = y_true_mask
        else:
            y_true_mask = F.pad(y_true_mask, padding, "constant", 0)
            y_pred_mask = F.pad(y_pred_mask, padding, "constant", 0)
            sampling_mask = y_true_mask * y_pred_mask

        """Reshaping images into non-overlapping patches"""
        if ndim == 3:
            # prepare target patches
            y_true_patch = tc.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_true_patch = tc.reshape(y_true_patch, (-1, self.patch_size ** 3, 1))

            # prepare source patches
            y_pred_patch = tc.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_pred_patch = tc.reshape(y_pred_patch, (-1, self.patch_size ** 3, 1))

            # prepare mask patches
            mask_patch = tc.reshape(sampling_mask, (sampling_mask.shape[0], sampling_mask.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            mask_patch = mask_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            mask_patch = tc.reshape(mask_patch, (-1, self.patch_size ** 3, 1))

            valid_patches = mask_patch.sum(dim=1) > 80

            y_pred_patch = y_pred_patch[valid_patches.squeeze()]

            y_true_patch = y_true_patch[valid_patches.squeeze()]

        """Compute MI"""
        I_a_patch = tc.exp(- self.preterm * tc.square(y_true_patch - vbc))
        I_a_patch = I_a_patch / tc.sum(I_a_patch, dim=-1, keepdim=True)

        I_b_patch = tc.exp(- self.preterm * tc.square(y_pred_patch - vbc))
        I_b_patch = I_b_patch / tc.sum(I_b_patch, dim=-1, keepdim=True)

        pab = tc.bmm(I_a_patch.permute(0, 2, 1), I_b_patch)
        pab = pab / self.patch_size ** ndim
        pa = tc.mean(I_a_patch, dim=1, keepdim=True)
        pb = tc.mean(I_b_patch, dim=1, keepdim=True)

        papb = tc.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = tc.sum(tc.sum(pab * tc.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()

    def forward(self, y_true, y_true_mask, y_pred, y_pred_mask):
        return -self.local_mi(y_true, y_true_mask, y_pred, y_pred_mask)


def mi_global(
    sources: tc.Tensor,
    targets: tc.Tensor,
    device: Union[str, tc.device, None]="cpu",
    **params : dict) -> tc.Tensor:
    """
    Parameters
    ----------
    sources : tc.Tensor(Bx1xMxN)
        The source tensor
    targest : tc.Tensor (Bx1xMxN)
        The target target
    device : str
        The device where source/target are placed
    params : dict
        Additional cost function parameters

    Returns
    ----------
    ncc : float
        The negative of normalized cross-correlation (average across batches)
    """
    sources = (sources - tc.min(sources)) / (tc.max(sources) - tc.min(sources))
    targets = (targets - tc.min(targets)) / (tc.max(targets) - tc.min(targets))
    if sources.size() != targets.size():
        raise ValueError("Shape of both the tensors must be the same.")
    MI = MutualInformation(**params)
    mi = MI(sources, targets)
    return mi
    
    
def mi_local(
    sources: tc.Tensor,
    targets: tc.Tensor,
    device: Union[str, tc.device, None]="cpu",
    **params : dict) -> tc.Tensor:
    """
    Parameters
    ----------
    sources : tc.Tensor(Bx1xMxN)
        The source tensor
    targest : tc.Tensor (Bx1xMxN)
        The target target
    device : str
        The device where source/target are placed
    params : dict
        Additional cost function parameters

    Returns
    ----------
    ncc : float
        The negative of normalized cross-correlation (average across batches)
    """
    sources = (sources - tc.min(sources)) / (tc.max(sources) - tc.min(sources))
    targets = (targets - tc.min(targets)) / (tc.max(targets) - tc.min(targets))
    if sources.size() != targets.size():
        raise ValueError("Shape of both the tensors must be the same.")
    MI = localMutualInformation(**params)
    mi = MI(sources, targets)
    return mi

def mi_masked_local(
    sources: tc.Tensor,
    targets: tc.Tensor,
    source_masks: tc.Tensor,
    target_masks: tc.Tensor,
    device: Union[str, tc.device, None]="cpu",
    **params : dict) -> tc.Tensor:
    """
    Parameters
    ----------
    sources : tc.Tensor(Bx1xMxN)
        The source tensor
    targest : tc.Tensor (Bx1xMxN)
        The target target
    device : str
        The device where source/target are placed
    params : dict
        Additional cost function parameters

    Returns
    ----------
    ncc : float
        The negative of normalized cross-correlation (average across batches)
    """
    sources = (sources - tc.min(sources)) / (tc.max(sources) - tc.min(sources))
    targets = (targets - tc.min(targets)) / (tc.max(targets) - tc.min(targets))
    if sources.size() != targets.size():
        raise ValueError("Shape of both the volume tensors must be the same.")
    MI = maskedLocalMutualInformation(**params)
    mi = MI(sources, source_masks, targets, target_masks)
    return mi

def mi_masked_global(
    sources: tc.Tensor,
    targets: tc.Tensor,
    source_masks: tc.Tensor,
    target_masks: tc.Tensor,
    device: Union[str, tc.device, None]="cpu",
    **params : dict) -> tc.Tensor:
    """
    Parameters
    ----------
    sources : tc.Tensor(Bx1xMxN)
        The source tensor
    targest : tc.Tensor (Bx1xMxN)
        The target target
    device : str
        The device where source/target are placed
    params : dict
        Additional cost function parameters

    Returns
    ----------
    ncc : float
        The negative of normalized cross-correlation (average across batches)
    """
    sources = (sources - tc.min(sources)) / (tc.max(sources) - tc.min(sources))
    targets = (targets - tc.min(targets)) / (tc.max(targets) - tc.min(targets))
    if sources.size() != targets.size():
        raise ValueError("Shape of both the tensors must be the same.")
    MI = MaskedMutualInformation(**params)
    mi = MI(sources, targets, source_masks, target_masks)
    return mi