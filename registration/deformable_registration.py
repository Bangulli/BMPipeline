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

def nonrigid_registration(source: tc.Tensor, target: tc.Tensor, num_levels: int, used_levels: int, num_iters: int,
    learning_rate: float, alpha: float,
    cost_function: Callable[[tc.Tensor, tc.Tensor, dict], float], regularization_function: Callable[[tc.Tensor, dict], float],
    cost_function_params: dict={}, regularization_function_params: dict={},
    penalty_function: Callable=None, penalty_function_params: dict={},
    initial_displacement_field: tc.Tensor = None,
    device: str="cpu", echo: bool=False, vecint: int=0, scaler=None):
    """
    Performs the nonrigid registration using the instance optimization technique (a prototype).
    For the real-time DirectX implementation the autograd must be replaced by an analytical gradient calculation
    and the optimization should be implemented using matrix-free operations and a quasi-Newton algorithm.

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
    alpha : float
        The regularization weight
    cost_function : Callable[tc.Tensor, tc.Tensor, dict] -> float
        The cost function being optimized
    regularization_function : Callable[tc.Tensor,  dict] -> float
        The regularization function
    cost_function_params : dict (default: {})
        The optional cost function parameters
    regularization_function_params : dict (default: {})
        The optional regularization function parameters
    penalty_function : Callable
        The optional penalty function (must be differntiable)
    penalty_function_params : dict(default: {})
        The optional penalty function parameters
    initial_displacement_field : tc.Tensor (default None)
        The initial displacement field (e.g. resulting from the initial, affine registration)
    device : str
        The device used for warping (e.g. "cpu" or "cuda:0")

    TO DO: early stoping technique
    TO DO: add scheduler

    Returns
    ----------
    displacement_field : tc.Tensor
        The calculated displacement_field (to be applied using warp_tensor from utils_tc)
    """
    ndim = len(source.size()) - 2
    source_pyramid = u.create_pyramid(source, num_levels=num_levels, device=device)
    target_pyramid = u.create_pyramid(target, num_levels=num_levels, device=device)
    for j in range(used_levels):
        current_source = source_pyramid[j]
        current_target = target_pyramid[j]
        if j == 0:
            if initial_displacement_field is None:
                if ndim == 2:
                    displacement_field = tc.zeros((1, current_source.size(2), current_source.size(3), 2), dtype=source.dtype, device=device, requires_grad=True)
                elif ndim == 3:
                    displacement_field = tc.zeros((1, current_source.size(2), current_source.size(3), current_source.size(4), 3), dtype=source.dtype, device=device, requires_grad=True)
                else:
                    raise ValueError("Unsupported number of dimensions.")
            else:
                displacement_field = u.resample_displacement_field(initial_displacement_field, u.tc_size_to_df_size(current_source), device=device).detach().clone()
                displacement_field.requires_grad = True
            optimizer = optim.Adam([displacement_field], learning_rate)
            if scaler is not None:
                current_scaler = u.resample_tensor(scaler.unsqueeze(0), current_source.size(), device=device)
                print(f"Scaler size: {current_scaler.size()}") if echo else None
                cost_function_params['mask'] = current_scaler
        else:
            displacement_field = u.resample_displacement_field(displacement_field, u.tc_size_to_df_size(current_source), device=device).detach().clone()
            displacement_field.requires_grad = True
            optimizer = optim.Adam([displacement_field], learning_rate)
            if scaler is not None:
                current_scaler = u.resample_tensor(scaler.unsqueeze(0), current_source.size(), device=device)
                print(f"Scaler size: {current_scaler.size()}") if echo else None
                cost_function_params['mask'] = current_scaler
        for i in range(num_iters[j]):
            with tc.set_grad_enabled(True):
                warped_source = w.warp_tensor(current_source, displacement_field, device=device)
                cost = cost_function(warped_source, current_target, device=device, **cost_function_params)   
                reg = regularization_function(displacement_field, device=device, **regularization_function_params)
                loss = cost + alpha[j]*reg
                if penalty_function is not None:
                    loss = loss + penalty_function(penalty_function_params) 
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            print(f"Level: {j+1}/{used_levels} Iter: {i+1}/{num_iters[j]} Current cost: {cost.item()} Current reg: {reg.item()} Current loss: {loss.item()}") if echo else None
    if used_levels != num_levels:
        displacement_field = u.resample_displacement_field(displacement_field, u.tc_size_to_df_size(source), device=device)
    return displacement_field