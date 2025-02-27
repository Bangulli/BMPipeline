### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union

### External Imports ###
import numpy as np
import scipy.ndimage as nd
import torch as tc
import SimpleITK as sitk

### Internal Imports ###
import affine_registration as ar
import masked_affine_registration as mar
import deformable_registration as dr
import utils as u
import warping as w

########################

def center_of_mass_initialization(source : tc.Tensor, target : tc.Tensor):
    source_np = source.cpu().detach().numpy()[0, 0]
    target_np = target.cpu().detach().numpy()[0, 0]
    source_com = np.array(nd.center_of_mass(source_np), dtype=np.float32)
    target_com = np.array(nd.center_of_mass(target_np), dtype=np.float32)
    translation_vector = target_com - source_com
    y_shape, x_shape, z_shape = source_np.shape
    x_grid, y_grid, z_grid = np.meshgrid(np.arange(x_shape), np.arange(y_shape), np.arange(z_shape))
    displacement_field_np = np.stack((y_grid, x_grid, z_grid)).astype(np.float32)
    displacement_field_np[0, :, :, :] = -translation_vector[1]
    displacement_field_np[1, :, :, :] = -translation_vector[0]
    displacement_field_np[2, :, :, :] = -translation_vector[2]
    displacement_field_tc = u.np_df_to_tc_df(displacement_field_np)
    displacement_field_tc = displacement_field_tc.to(source.device)
    warped_source = w.warp_tensor(source, displacement_field_tc, mode='nearest')
    return warped_source, displacement_field_tc

def run_masked_affine_registration(source : tc.Tensor, source_mask: tc.Tensor, target : tc.Tensor, target_mask: tc.Tensor, **params):
    ### Parse Params ###
    device = params['device']
    echo = params['echo']
    registration_shape = params['registration_shape']

    num_levels = params['num_levels_aff']
    used_levels = params['used_levels_aff']
    num_iters = params['num_iters_aff']
    learning_rate = params['learning_rate_aff']
    cost_function = params['cost_function_aff']
    cost_function_params = params['cost_function_params_aff']

    ### Prepare Volumes ###
    source = u.normalize(source.to(device))
    target = u.normalize(target.to(device))
    if source_mask is not None:
        source_mask = source_mask.to(device)
    target_mask = target_mask.to(device)

    com_source, com_df = center_of_mass_initialization(source, target)
    if source_mask is not None:
        com_source_mask = w.warp_tensor(source_mask, com_df, mode='nearest')
    res_source = u.resample_tensor(com_source, registration_shape)
    res_target = u.resample_tensor(target, registration_shape)
    if source_mask is not None:
        res_source_mask = u.resample_tensor(com_source_mask, registration_shape, mode='nearest')
    else:
        res_source_mask = None
    res_target_mask = u.resample_tensor(target_mask, registration_shape, mode='nearest')

    ### Run Registration ###

    affine_transform = mar.masked_affine_registration(
        res_source,
        res_target,
        res_source_mask,
        res_target_mask,
        num_levels,
        used_levels,
        num_iters,
        learning_rate,
        cost_function,
        cost_function_params,
        device=device,
        echo=echo)
    
    displacement_field = u.tc_transform_to_tc_df(affine_transform, source.size(), device=device)
    displacement_field = w.compose_displacement_fields(source, com_df, displacement_field)
    return displacement_field



def run_affine_deformable_registration(source : tc.Tensor, target : tc.Tensor, **params):
    ### Parse Params ###
    device = params['device']
    echo = params['echo']
    registration_shape = params['registration_shape']
    
    num_levels_aff = params['num_levels_aff']
    used_levels_aff = params['used_levels_aff']
    num_iters_aff = params['num_iters_aff']
    learning_rate_aff = params['learning_rate_aff']
    cost_function_aff = params['cost_function_aff']
    cost_function_params = params['cost_function_params_aff']

    num_levels = params['num_levels']
    used_levels = params['used_levels']
    num_iters = params['num_iters']
    learning_rate = params['learning_rate']
    cost_function = params['cost_function']
    cost_function_params = params['cost_function_params']
    alpha = params['alpha']
    regularization_function = params['regularization_function']
    regularization_function_params = params['regularization_function_params']
    penalty_function = params['penalty_function']
    penalty_function_params = params['penalty_function_params']

    ### Prepare Volumes ###
    source = u.normalize(source.to(device))
    target = u.normalize(target.to(device))

    ### Run Registration ###
    com_source, com_df = center_of_mass_initialization(source, target)
    res_source = u.resample_tensor(com_source, registration_shape)
    res_target = u.resample_tensor(target, registration_shape)

    com_df = u.resample_displacement_field(com_df, u.tc_size_to_df_size(res_source))

    affine_transform = ar.affine_registration(
        res_source,
        res_target,
        num_levels_aff,
        used_levels_aff,
        num_iters_aff,
        learning_rate_aff,
        cost_function_aff,
        cost_function_params,
        device=device,
        echo=echo)
    displacement_field = u.tc_transform_to_tc_df(affine_transform, res_source.size(), device=device)


    affine_source = w.warp_tensor(res_source, displacement_field)
    nonrigid_displacement_field = dr.nonrigid_registration(
        affine_source.detach(),
        res_target,
        num_levels,
        used_levels,
        num_iters,
        learning_rate,
        alpha,
        cost_function,
        regularization_function,
        cost_function_params,
        regularization_function_params,
        penalty_function,
        penalty_function_params,
        device=device, 
        echo=echo)
    
    temp_df = w.compose_displacement_fields(affine_source, com_df, displacement_field)
    displacement_field = w.compose_displacement_fields(affine_source, temp_df, nonrigid_displacement_field)
    displacement_field = u.resample_displacement_field(displacement_field, u.tc_size_to_df_size(source))
    return displacement_field

def run_affine_registration(source : tc.Tensor, target : tc.Tensor, **params):
    ### Parse Params ###
    device = params['device']
    echo = params['echo']
    registration_shape = params['registration_shape']

    num_levels = params['num_levels_aff']
    used_levels = params['used_levels_aff']
    num_iters = params['num_iters_aff']
    learning_rate = params['learning_rate_aff']
    cost_function = params['cost_function_aff']
    cost_function_params = params['cost_function_params_aff']

    ### Prepare Volumes ###
    source = u.normalize(source.to(device))
    target = u.normalize(target.to(device))

    com_source, com_df = center_of_mass_initialization(source, target)
    res_source = u.resample_tensor(com_source, registration_shape, mode='nearest')
    res_target = u.resample_tensor(target, registration_shape, mode='nearest')

    ### Run Registration ###

    affine_transform = ar.affine_registration(
        res_source,
        res_target,
        num_levels,
        used_levels,
        num_iters,
        learning_rate,
        cost_function,
        cost_function_params,
        device=device,
        echo=echo)
    displacement_field = u.tc_transform_to_tc_df(affine_transform, source.size(), device=device)
    displacement_field = w.compose_displacement_fields(source, com_df, displacement_field)
    return displacement_field
