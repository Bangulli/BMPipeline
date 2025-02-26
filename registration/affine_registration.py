### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Callable, Union

### External Imports ###
import torch as tc
import torch.nn.functional as F
import torch.optim as optim

### Internal Imports ###
import utils as u
import warping as w

########################


def affine_registration(
    source: tc.Tensor,
    target: tc.Tensor,
    num_levels: int,
    used_levels: int,
    num_iters: list,
    learning_rate: float,
    cost_function: Callable[[tc.Tensor, tc.Tensor, dict], float],
    cost_function_params: dict={}, 
    device: Union[str, tc.device, None] ="cpu",
    initial_transform : Union[tc.Tensor, None] = None,
    echo: bool=False,
    return_best: bool=False) -> tc.Tensor:
    """
    Performs the affine registration using the instance optimization technique (a prototype).

    Parameters
    ----------
    source : tc.Tensor
        The source tensor (1x1 x size)
    target : tc.Tensor
        The target tensor (1x1 x size)
    num_levels : int
        The number of resolution levels
    used_levels : int
        The number of actually used resolution levels (must be lower (or equal) than the num_levels)
    num_iters : int
        The nubmer of iterations per resolution
    learning_rate : float
        The learning rate for the optimizer
    cost_function : Callable[tc.Tensor, tc.Tensor, dict] -> float
        The cost function being optimized
    cost_function_params : dict (default: {})
        The optional cost function parameters
    device : str
        The device used for warping (e.g. "cpu" or "cuda:0")

    Returns
    ----------
    transformation : tc.Tensor
        The affine transformation matrix (1 x transformation_size (2x3 or 3x4))
    """
    ndim = len(source.size()) - 2
    if initial_transform is None:
        if ndim == 2:
            transformation = tc.zeros((1, 2, 3), dtype=source.dtype, device=device)
            transformation[0, 0, 0] = 1.0
            transformation[0, 1, 1] = 1.0
            transformation = transformation.detach().clone()
            transformation.requires_grad = True
        elif ndim == 3:
            transformation = tc.zeros((1, 3, 4), dtype=source.dtype, device=device)
            transformation[0, 0, 0] = 1.0
            transformation[0, 1, 1] = 1.0
            transformation[0, 2, 2] = 1.0
            transformation = transformation.detach().clone()
            transformation.requires_grad = True
        else:
            raise ValueError("Unsupported number of dimensions.")
    else:
        transformation = initial_transform.detach().clone()
        transformation.requires_grad = True

    optimizer = optim.AdamW([transformation], learning_rate, weight_decay=0.05)
    source_pyramid = u.create_pyramid(source, num_levels=num_levels)
    target_pyramid = u.create_pyramid(target, num_levels=num_levels)
    if return_best:
        best_transformation = transformation.clone()
        best_cost = 1000.0
    for j in range(used_levels):
        current_source = source_pyramid[j]
        current_target = target_pyramid[j]
        for i in range(num_iters[j]):
            with tc.set_grad_enabled(True):
                sampling_grid = F.affine_grid(transformation, size=current_source.size(), align_corners=False)
                warped_source = w.transform_tensor(current_source, sampling_grid, device=device)
                cost = cost_function(warped_source, current_target, device=device, **cost_function_params)    
                cost.backward()
                optimizer.step()
                current_cost = cost.item()
            optimizer.zero_grad()
            if echo:
                print(f"Level: {j+1}/{used_levels}, Iter: {i+1}/{num_iters[j]}, Current cost: {current_cost}")
            if return_best:
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_transformation = transformation.clone()
    if return_best:
        return best_transformation
    else:
        return transformation