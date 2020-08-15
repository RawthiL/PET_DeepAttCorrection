#!/usr/bin/python
# -*- coding: iso-8859-15 -*-


import os, sys
import numpy as np
import pydicom as dicom

from pydicom.dataset import Dataset, FileDataset

# Used to accelerate couch removal task
from joblib import Parallel, delayed
import multiprocessing

from scipy import ndimage
from skimage import morphology
from skimage import measure

import tensorflow as tf

import datetime
import time

from DeepAttCorr_lib import file_manage_utils as File_mng


X = 0
Y = 1
Z = 2

# Linear attenuation coefficient of water at 120keV and 511keV
mu_agua_120kev = 1.607e-01
mu_agua_511kev = 9.687e-02 # cm^-1
rel_conv = mu_agua_511kev/mu_agua_120kev

# Defines
CT_CAHNNELS = 1
LIM_READ_SIZE = 2
LABELS_CAHNNELS = 4
PET_CHANNELS = 1


#-----------------------------------------------------------------------------#
#--------------- Data specific functions -------------------------------------#
#-----------------------------------------------------------------------------#

# Unpack study
def get_sample_axial_FOV_limits(path_to_samples):
    """Get the FOV limits from a study
    """
    
    # Get all slice paths
    list_paths = sorted(os.listdir(path_to_samples))
    # Get number of slices to load
    num_slices = len(list_paths)
        
    # Iterate and get axial FOV limits
    locations = np.zeros((num_slices))
    for idx, filename in enumerate(list_paths):
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            RefDs = dicom.read_file(os.path.join(path_to_samples,filename))
            # Get location
            locations[idx] = RefDs.SliceLocation
    
    # Order loations
    sorted_locs = np.sort(locations)
    
    # Return limits
    return (sorted_locs[0], sorted_locs[-1])




def preprocess_sample(ct_volume, 
                      pet_volume, 
                      LIM_SUP_PET, 
                      LIM_INFO_CT =  0.005, 
                      HU_Lung_upper_limit = -500.0, 
                      HU_Parenchyma_lower_limit = 10.0, 
                      HU_Parenchyma_upper_limit = 90.0, 
                      HU_Bone_low_limit = 90.0, 
                      HU_Bone_upper_limit = 1300.0,
                      HU_Fluids_Fat_lower_limit = -120.0,
                      HU_Fluids_Fat_upper_limit = 10.0,
                      USE_FAT_THRESHOLD = False,
                      USE_HIST_THRESHOLD = True):
    """Process a sample to get the labels of the tissues
    
        Arguments:
        ct_volume --- CT volumetric image
        pet_volume --- PET volumetric image (registred to CT)
        LIM_SUP_PET --- Limit of the PET normalized activity
        
        Outputs:
        input_out --- Thresholded PET image
        labels_out --- Tissue label mask
    """
    

    
    # Get volume shape
    vol_shape = ct_volume.shape
    
    # Create output volumes
    input_out = np.copy(pet_volume)
    labels_out = np.zeros((vol_shape[0],vol_shape[1],vol_shape[2],4)) # X,Y,Z,Mat. // (Bone, Parenchyma, Fluids-Fat, Air)
    
    # Saturate using histogram limit
    if USE_HIST_THRESHOLD:
        input_out[input_out > LIM_SUP_PET] = LIM_SUP_PET
            
    # Get parenchyma labels
    # https://en.wikipedia.org/wiki/Hounsfield_scale
    Limit_mu_lung = ((HU_Lung_upper_limit/1000.0)+1)*mu_agua_120kev/10.0
    
    Limit_mu_fluid_fat_low = ((HU_Fluids_Fat_lower_limit/1000.0)+1)*mu_agua_120kev/10.0
    Limit_mu_fluid_fat_up = ((HU_Fluids_Fat_upper_limit/1000.0)+1)*mu_agua_120kev/10.0
    
    Limit_mu_parenchyma_low = ((HU_Parenchyma_lower_limit/1000.0)+1)*mu_agua_120kev/10.0
    Limit_mu_parenchyma_up = ((HU_Parenchyma_upper_limit/1000.0)+1)*mu_agua_120kev/10.0
    
    Limit_mu_bone_low = ((HU_Bone_low_limit/1000.0)+1)*mu_agua_120kev/10.0
    Limit_mu_bone_up = ((HU_Bone_upper_limit/1000.0)+1)*mu_agua_120kev/10.0
    

    # Get all parenchyma exept lung
    if USE_FAT_THRESHOLD:
        ct_volume[ct_volume < Limit_mu_fluid_fat_low] = 0
    else:
        ct_volume[ct_volume < Limit_mu_lung] = 0
        
        
    # Create auxiliary masks
    parenchyma_mask = np.ones(ct_volume.shape)
    bone_mask = np.ones(ct_volume.shape)
    fluid_fat_mask = np.ones(ct_volume.shape)

    # Get all pixels with fat, parenchyma or bone
    parenchyma_mask[ct_volume>0] = 1 
    # Erase all pixels harder or softer than fluid_fat
    fluid_fat_mask[ct_volume>Limit_mu_fluid_fat_up] = 0 
    fluid_fat_mask[ct_volume<Limit_mu_fluid_fat_low] = 0 
    # Erase all pixels harder or softer than flesh or muscle
    parenchyma_mask[ct_volume>Limit_mu_parenchyma_up] = 0 
    parenchyma_mask[ct_volume<Limit_mu_parenchyma_low] = 0 
    # Erase all pixels below the bone range
    bone_mask[ct_volume<Limit_mu_bone_low] = 0
    

    # Close fluid_fat mask
    fluid_fat_mask = ndimage.morphology.binary_dilation(fluid_fat_mask, structure=morphology.ball(2)).astype(np.float32) # ball 2
    fluid_fat_mask = ndimage.morphology.binary_erosion(fluid_fat_mask, structure=morphology.ball(2)) # ball 2
    
    # Close parenchyma mask
    parenchyma_mask = ndimage.morphology.binary_dilation(parenchyma_mask, structure=morphology.ball(2)).astype(np.float32) # ball 2
    parenchyma_mask = ndimage.morphology.binary_erosion(parenchyma_mask, structure=morphology.ball(2)) # ball 2

    # Open bone mask 
    bone_mask = ndimage.morphology.binary_erosion(bone_mask, structure=morphology.ball(1)) 
    bone_mask = ndimage.morphology.binary_dilation(bone_mask, structure=morphology.ball(2)).astype(np.float32)# ball 2

    # Erase possible double-labeled pixels, with prevalence of harder tisues
    fluid_fat_mask[(parenchyma_mask+bone_mask)>0] = 0
    parenchyma_mask[bone_mask>0] = 0
    
    # Include all pixels to label structure and create air label
    labels_out[:,:,:,0] = bone_mask
    labels_out[:,:,:,1] = parenchyma_mask
    labels_out[:,:,:,2] = fluid_fat_mask
    labels_out[:,:,:,3] = np.ones(vol_shape) - (bone_mask + parenchyma_mask + fluid_fat_mask)
    
    # Return procesed sample
    return (input_out, labels_out)

def CT_couch_removal_mask(ct_volume, low_cut = -400, high_cut = 400, limits_CT = [0,0], samples_use = 10, n_corr_disk = 4):
    """Removes couch structure from CT volume
    
     based on bandi2011automated
     Peter Bandi et al. - Automated patient couch removal algorithm on CT images
    """
    
    
    n_corr_disk = int(n_corr_disk)
    samples_use = int(samples_use)
    
    X_size = ct_volume.shape[0]
    Y_size = ct_volume.shape[1]
    Z_size = ct_volume.shape[2]
    
    # Get the limits of the image with CT couch
    if limits_CT[0] == limits_CT[1]:
        limits_CT[0] = 0
        limits_CT[1] = Z_size-1
        
    # Get N equally separated axial samples of the CT with couch  
    # [S_i] e [M_p]
    axial_samples = np.linspace(limits_CT[0], limits_CT[1], samples_use, endpoint=False, dtype=np.int32)
    sampled_vol = ct_volume[:,:,axial_samples]
    
    # Cut the data
    # [S_i^c] e [M_p^c]
    sampled_vol[sampled_vol > high_cut] = high_cut
    sampled_vol[sampled_vol < low_cut] = low_cut
    
    # Normalize the data
    sampled_vol = (sampled_vol - sampled_vol.min()) / (sampled_vol.max() - sampled_vol.min())
    
    # Create [m_i] 
    m_i = np.zeros((X_size,Y_size,samples_use))
    aux_vol = np.zeros((X_size+2*n_corr_disk,Y_size+2*n_corr_disk,samples_use))
    aux_vol[n_corr_disk:-n_corr_disk,n_corr_disk:-n_corr_disk,:] = sampled_vol
    kernel_shape = np.ones((2*n_corr_disk,2*n_corr_disk))
    kernel_gain = 1.0 / ((2.0*n_corr_disk + 1.0)**2.0)
    for z_idx in range(0,samples_use):
        for x_idx in range(0,X_size):
            for y_idx in range(0,Y_size):
                patch_aux = aux_vol[x_idx:x_idx+(n_corr_disk*2), 
                                    y_idx:y_idx+(n_corr_disk*2), 
                                    z_idx]
                sum_val = np.sum(np.multiply(patch_aux, 
                                             kernel_shape))
                m_i[x_idx,y_idx,z_idx] = sum_val * kernel_gain

    # Calculate [C]
    C = np.zeros((X_size,Y_size))
    for x_idx in range(0,X_size):
        for y_idx in range(0,Y_size):
            mult_field = np.ones((n_corr_disk*2,n_corr_disk*2))
            for z_idx in range(0,samples_use):
                s_i_c = aux_vol[x_idx:x_idx+(n_corr_disk*2), 
                                y_idx:y_idx+(n_corr_disk*2), 
                                z_idx]
                mult_field = np.multiply(mult_field, s_i_c - m_i[x_idx,y_idx,z_idx] )
            C[x_idx,y_idx] = np.sum(mult_field)
            
    # Generate mask [B]
    B = np.copy(C)
    B[C !=0 ] = 1
    
    # Open mask 
    B = ndimage.morphology.binary_erosion(B, structure=morphology.disk(2)) 
    B = ndimage.morphology.binary_dilation(B, structure=morphology.disk(4)).astype(np.float32)
    
    # Segmentate objects [B^n]
    B_n = measure.label(B, background=0)
    
    # Get distance to objects
    blobs_clases = np.unique(B_n)
    dist2center = np.zeros(len(blobs_clases))
    for x_idx in range(0,X_size):
        for y_idx in range(0,Y_size):
            distancia = ((x_idx-(X_size/2.0))**2+(y_idx-(Y_size/2.0))**2)**4
            clase = B_n[x_idx,y_idx]
            if clase != 0:
                dist2center[clase] = dist2center[clase] + float(distancia)

    # Select most distant object
    B_d = np.copy(B_n)
    B_d[B_n != np.argmax(dist2center)] = 0
    
    # Fill mask 
    B_d = ndimage.morphology.binary_dilation(B_d, structure=morphology.disk(15)).astype(np.float32)
    B_d = ndimage.morphology.binary_fill_holes(B_d)
    couch_mask = ndimage.morphology.binary_erosion(B_d, structure=morphology.disk(13)) 
    
            
    return couch_mask
               
def calc_mi_val(index_act, X_size,Y_size, Z_size, n_corr_disk, aux_vol, kernel_shape, kernel_gain):
    
    (x_idx, y_idx,) = np.unravel_index(index_act, (X_size,Y_size))
    m_i = np.zeros((Z_size))
    for z_idx in range(0,Z_size):
       
        patch_aux = aux_vol[x_idx:x_idx+(n_corr_disk*2), 
                            y_idx:y_idx+(n_corr_disk*2), 
                            z_idx]
        sum_val = np.sum(np.multiply(patch_aux, 
                                     kernel_shape))
        m_i[z_idx] = sum_val * kernel_gain
        
    return m_i


def calc_C_val(index_act, X_size,Y_size,samples_use, n_corr_disk, aux_vol, m_i):
    
    (x_idx, y_idx) = np.unravel_index(index_act, (X_size,Y_size))

    mult_field = np.ones((n_corr_disk*2,n_corr_disk*2))
    for z_idx in range(0,samples_use):
        s_i_c = aux_vol[x_idx:x_idx+(n_corr_disk*2), 
                        y_idx:y_idx+(n_corr_disk*2), 
                        z_idx]
        mult_field = np.multiply(mult_field, s_i_c - m_i[x_idx,y_idx,z_idx] )
    return np.sum(mult_field)


   
    
def CT_couch_removal_mask_multicore(ct_volume, low_cut = -400, 
                                    high_cut = 400, 
                                    limits_CT = [0,0], 
                                    samples_use = 10, 
                                    n_corr_disk = 4, 
                                    B_n_dilatation_disk_rad = 4,
                                    B_n_erosion_disk_rad = 2,
                                    B_d_dilatation_disk_rad = 15,
                                    B_d_erosion_disk_rad = 13,
                                    return_all = False):
    """Removes couch structure from CT volume (multicore version)
    
     based on bandi2011automated
     Peter Bandi et al. - Automated patient couch removal algorithm on CT images
    """

    n_cores = multiprocessing.cpu_count()
    
    n_corr_disk = int(n_corr_disk)
    samples_use = int(samples_use)
    
    X_size = ct_volume.shape[0]
    Y_size = ct_volume.shape[1]
    Z_size = ct_volume.shape[2]
    
    # Get the limits of the image with CT couch
    if limits_CT[0] == limits_CT[1]:
        limits_CT[0] = 0
        limits_CT[1] = Z_size-1
        
    # Get N equally separated axial samples of the CT with couch  
    # [S_i] e [M_p]
    axial_samples = np.linspace(limits_CT[0], limits_CT[1], samples_use, endpoint=False, dtype=np.int32)
    sampled_vol = ct_volume[:,:,axial_samples]
    
    # Cut the data
    # [S_i^c] e [M_p^c]
    sampled_vol[sampled_vol > high_cut] = high_cut
    sampled_vol[sampled_vol < low_cut] = low_cut
    
    # Normalize the data
    sampled_vol = (sampled_vol - sampled_vol.min()) / (sampled_vol.max() - sampled_vol.min())
    
    # Pre-Calculate [m_i] 
    aux_vol = np.zeros((X_size+2*n_corr_disk,Y_size+2*n_corr_disk,samples_use))
    aux_vol[n_corr_disk:-n_corr_disk,n_corr_disk:-n_corr_disk,:] = sampled_vol
    kernel_shape = np.ones((2*n_corr_disk,2*n_corr_disk))
    kernel_gain = 1.0 / ((2.0*n_corr_disk + 1.0)**2.0)
    
    m_i = Parallel(n_jobs=n_cores)(delayed(calc_mi_val)(index_act, 
                                                        X_size,
                                                        Y_size, 
                                                        samples_use, 
                                                        n_corr_disk, 
                                                        aux_vol, 
                                                        kernel_shape, 
                                                        kernel_gain) for index_act in range(0,(X_size*Y_size)))
    m_i = np.array(m_i)
    m_i = np.reshape(m_i, (X_size,Y_size,samples_use))

    # Calculate [C]
    C = Parallel(n_jobs=n_cores)(delayed(calc_C_val)(index_act, 
                                                     X_size,
                                                     Y_size, 
                                                     samples_use, 
                                                     n_corr_disk, 
                                                     aux_vol, 
                                                     m_i) for index_act in range(0,(X_size*Y_size)))
    C = np.array(C)
    C = np.reshape(C, (X_size,Y_size))
            
    # Generate mask [B]
    B = np.copy(C)
    B[C !=0 ] = 1
    
    # Open mask 
    B = ndimage.morphology.binary_erosion(B, structure=morphology.disk(B_n_erosion_disk_rad)) 
    B = ndimage.morphology.binary_dilation(B, structure=morphology.disk(B_n_dilatation_disk_rad)).astype(np.float32)
    
    # Segmentate objects [B^n]
    B_n = measure.label(B, background=0)
    
    # Get distance to objects
    blobs_clases = np.unique(B_n)
    dist2center = np.zeros(len(blobs_clases))
    pixel_num = np.zeros(len(blobs_clases))
    for x_idx in range(0,X_size):
        for y_idx in range(0,Y_size):
            #distancia = ((x_idx-(X_size/2.0))**2+(y_idx-(Y_size/2.0))**2)**4
            distancia = ((x_idx-(X_size/2.0))**2)**4
            clase = B_n[x_idx,y_idx]
            if clase != 0:
                dist2center[clase] = dist2center[clase] + float(distancia)
                pixel_num[clase] = pixel_num[clase] + 1
    pixel_num[0] = 1
    for idx_clase in range(0,len(blobs_clases)):
        dist2center[idx_clase] = dist2center[idx_clase]/pixel_num[idx_clase]
    dist2center[0] = 0

    # Select most distant object
    B_d = np.copy(B_n)
    B_d[B_n != np.argmax(dist2center)] = 0
    
    # Fill mask 
    B_d = ndimage.morphology.binary_dilation(B_d, structure=morphology.disk(B_d_dilatation_disk_rad)).astype(np.float32)
    B_d = ndimage.morphology.binary_fill_holes(B_d)
    couch_mask = ndimage.morphology.binary_erosion(B_d, structure=morphology.disk(B_d_erosion_disk_rad)) 
    
    if return_all:
        return couch_mask, B_d, B_n, dist2center, C, m_i
    else:
        return couch_mask
  
    
    
#-----------------------------------------------------------------------------#
#--------------- Volumetric transformation functions -------------------------#
#-----------------------------------------------------------------------------#
def normalize_volume_single(volume, Voxel_size_INPUT, Objective_size, Voxel_size_NORM, order_interpol = 2):
    """Normalize the voxels size and span of a volumetric image
    """

    # Calculate transformation parameters
    X_zoom = Voxel_size_INPUT[0]/Voxel_size_NORM[0]
    Y_zoom = Voxel_size_INPUT[1]/Voxel_size_NORM[1]
    Z_zoom = Voxel_size_INPUT[2]/Voxel_size_NORM[2]

    # Transform Volumes
    out_aux = ndimage.interpolation.zoom(input=volume, zoom=[X_zoom,Y_zoom,Z_zoom], order=order_interpol)
    shape_aux = out_aux.shape
    
    for i in range(3):
        if (shape_aux[i] > Objective_size[i]):
            ini = int(np.ceil((shape_aux[i]-Objective_size[i])/2))
            fin = int(np.floor((shape_aux[i]-Objective_size[i])/2))
            if i == 0:
                out_aux = out_aux[ini:-fin,:,:]
            elif i == 1:
                out_aux = out_aux[:,ini:-fin,:]
            elif i == 2:
                out_aux = out_aux[:,:,ini:-fin]
    shape_aux = out_aux.shape

    # Get limits
    limits = get_image_limits_Z(Objective_size, shape_aux)
    
    # Include them centered in the output array
    for i in range(3):
        if (Objective_size[i]-shape_aux[i])<0:
            raise ValueError("Input image outside normalization limits")
            
    out_bye = np.pad(out_aux,((np.floor((Objective_size[0]-shape_aux[0])/2.0).astype(np.int32),
                    np.ceil((Objective_size[0]-shape_aux[0])/2.0).astype(np.int32)),
                           (np.floor((Objective_size[1]-shape_aux[1])/2.0).astype(np.int32),
                            np.ceil((Objective_size[1]-shape_aux[1])/2.0).astype(np.int32)),
                               (np.floor((Objective_size[2]-shape_aux[2])/2.0).astype(np.int32),
                                np.ceil((Objective_size[2]-shape_aux[2])/2.0).astype(np.int32))), mode='constant')
    
    return out_bye, limits

def get_image_limits_Z(objective_size, input_size):
    
    limits = np.zeros((2))
    limits[0] = np.floor((objective_size[2]-input_size[2])/2.0).astype(np.int32)
    limits[1] = objective_size[2] - np.ceil((objective_size[2]-input_size[2])/2.0).astype(np.int32)
    
    return limits


def normalize_volume_size(ct_input, pet_input, locations_CT, locations_PET, Voxel_size_INPUT, Voxel_size_NORM, Objective_size):
    """Normalize the size of a pair CT and PET volumetric images
    """
    
     # Get limits
    locations_PET = np.sort(locations_PET)
    locations_CT = np.sort(locations_CT)
    loc_min = np.max([locations_CT[0],locations_PET[0]])
    loc_max = np.min([locations_CT[-1],locations_PET[-1]])
    # Get axial indexes included
    CT_ini_aux = np.argwhere(locations_CT>loc_min)[0][0]
    CT_fin_aux = np.argwhere(locations_CT<loc_max)[-1][0]
    PET_ini_aux = np.argwhere(locations_PET>loc_min)[0][0]
    PET_fin_aux = np.argwhere(locations_PET<loc_max)[-1][0]
    # crop
    pet_input = pet_input[:,:,PET_ini_aux:PET_fin_aux]
    ct_input = ct_input[:,:,CT_ini_aux:CT_fin_aux]
    
    # Transform Volumes and include them centered in the output array    
    ct_output, ct_image_limits = normalize_volume_single(ct_input, Voxel_size_INPUT[0,:], Objective_size, Voxel_size_NORM)
    del(ct_input)
    pet_output, pet_image_limits = normalize_volume_single(pet_input, Voxel_size_INPUT[1,:], Objective_size, Voxel_size_NORM)
    del(pet_input)
    
    limites_abs = np.array([max(pet_image_limits[0], ct_image_limits[0]), min(pet_image_limits[1], ct_image_limits[1])])
    
    return (ct_output, pet_output, limites_abs)




#-----------------------------------------------------------------------------#
#--------------- Batch generation functions ----------------------------------#
#-----------------------------------------------------------------------------#


def tf_read_raw_sample(data_record, sizes_data):
    """Read a non-modified sample from the TFrecord file
        
        data_record -- TFrecord object
        sizes_data -- size of the data to be read
    """

    X_size = sizes_data[0]
    Y_size = sizes_data[1]
    Z_size = sizes_data[2]
    IN_READ_SIZE = int(X_size*Y_size*Z_size)*PET_CHANNELS
    LABEL_READ_SIZE = int(X_size*Y_size*Z_size)*LABELS_CAHNNELS
    CT_READ_SIZE = int(X_size*Y_size*Z_size)*CT_CAHNNELS
    
    # Define features
    read_features = {
        'input': tf.io.FixedLenFeature([IN_READ_SIZE], dtype=tf.float32),
        'label': tf.io.FixedLenFeature([LABEL_READ_SIZE], dtype=tf.float32),
        'ct': tf.io.FixedLenFeature([CT_READ_SIZE], dtype=tf.float32),
        'limits_Z': tf.io.FixedLenFeature([LIM_READ_SIZE], dtype=tf.float32)
    }

    # Read sample
    sample = tf.io.parse_single_example(data_record, read_features)
    
    # load to tensors
    t_input = tf.reshape(sample.get('input'), (X_size,Y_size,Z_size))
    t_label = tf.reshape(sample.get('label'), (X_size,Y_size,Z_size, LABELS_CAHNNELS))
    t_ct = tf.reshape(sample.get('ct'), (X_size,Y_size,Z_size))
    t_limits = sample.get('limits_Z')
    
    return t_input, t_label, t_ct, t_limits



def tf_read_sample_file(data_record, sizes_data, sizes_input, 
                        noise_std = 0.025, gain_rnd_low = 1.0, 
                        gain_rnd_high = 1.0001, hist_shift_rnd_low = -0.1, 
                        hist_shift_rnd_high = 0.1, not_modified = False, 
                        not_cropped = False, not_transformed = False, 
                        pixel_value_mod = True, cdf_sampler_coef=[np.float64(1.0)]):
    """Read sample from TFrecord and apply transformations
    """

    if not_modified:
        not_cropped = True
        not_transformed = True
        pixel_value_mod = False
        
    # Get info size
    voxels_X = sizes_input[0]
    voxels_Y = sizes_input[1]
    voxels_Z = sizes_input[2]
    
    # Read sample
    t_input, t_label, t_ct, t_limits = tf_read_raw_sample(data_record, sizes_data)
    
    
    # Crop volume to given limmits
    initial_pos_0 = tf.cast((sizes_data[0]-sizes_input[0])/2, dtype=tf.int32)
    initial_pos_1 = tf.cast((sizes_data[1]-sizes_input[1])/2, dtype=tf.int32)
    initial_pos_2 = tf.cast((sizes_data[2])/2, dtype=tf.int32)
    
    # cropping on axial plane is always centered, on Z axis a random slice is chosen
    min_margin = tf.cast(t_limits[0]+(sizes_input[2]/2), dtype=tf.int32)
    max_margin = tf.cast(t_limits[1]-(sizes_input[2]/2), dtype=tf.int32)
    
       
    
    # if the requested size is larger than the un-padded volume, we center the slice
    rnd_Z = tf.cond(min_margin >= max_margin,
                    true_fn  = lambda : initial_pos_2,
                    false_fn = lambda : get_sample_center(min_margin, max_margin, cdf_sampler_coef))

    # If it is the non-modified version, use the central slice
    if not_cropped:
        rnd_Z = initial_pos_2
    
    # Get slice
    t_input, t_label, t_ct = tf_get_sample_slice(t_input, 
                                                 t_label, 
                                                 t_ct, 
                                                 sizes_input, 
                                                 initial_pos_0, 
                                                 initial_pos_1, 
                                                 rnd_Z)
    
    # Apply transformations
    if not not_transformed:

        # Random rotation angle in the axial plane
        rnd_angle = tf.random.uniform([],minval=0,maxval=np.pi,dtype=tf.float32)

        # Rotate complete volumes to avoid as much crop as possible
        t_input = tf.transpose(t_input, perm=[3, 0, 1, 2]) # Shape for rotation compatibility
        t_input = tf.contrib.image.rotate(t_input, rnd_angle, interpolation='BILINEAR')
        t_input = tf.transpose(t_input, perm=[1, 2, 3, 0]) # back to the original form 

        t_label = tf.transpose(t_label, perm=[3, 0, 1, 2]) # Shape for rotation compatibility
        t_label = tf.contrib.image.rotate(t_label, rnd_angle, interpolation='BILINEAR')
        t_label = tf.transpose(t_label, perm=[1, 2, 3, 0]) # back to the original form 

        t_ct = tf.transpose(t_ct, perm=[3, 0, 1, 2]) # Shape for rotation compatibility
        t_ct = tf.contrib.image.rotate(t_ct, rnd_angle, interpolation='BILINEAR')
        t_ct = tf.transpose(t_ct, perm=[1, 2, 3, 0]) # back to the original form 


    if pixel_value_mod:
        # Apply random gain normalization
        rnd_hist_shift = tf.random.uniform([], minval=hist_shift_rnd_low, maxval=hist_shift_rnd_high, dtype=tf.float32)
        t_input = t_input+rnd_hist_shift # Shift histogram
        t_input = tf.clip_by_value(t_input, 0.0, 1.0) # Clip between 0 and 1
        t_input = (t_input - tf.reduce_min(t_input)) / (tf.reduce_max(t_input) - tf.reduce_min(t_input)) # Expand histogram again
        # Add random noise to input
        noise = tf.random.normal(shape=tf.shape(t_input), mean=0.0, stddev=noise_std, dtype=tf.float32)
        t_input = t_input + noise

        # Clip image value between 0.0 and 1.0
        t_input = tf.clip_by_value( t_input, 0.0, 1.0)
    else:
        t_input = (t_input - tf.reduce_min(t_input)) / (tf.reduce_max(t_input) - tf.reduce_min(t_input)) # Expand histogram again
    

    return t_input, t_label, t_ct


def tf_get_keras_sample(data_record, sizes_data, sizes_input, noise_std = 0.0, gain_rnd_low = 1.0, gain_rnd_high = 1.0001, hist_shift_rnd_low = 0.0, hist_shift_rnd_high = 0.0001, not_modified = False, not_cropped = False, not_transformed = False, pixel_value_mod = True, cdf_sampler_coef=[]):
    """Read sample from TFrecord and apply transformations (Keras wrapper)
    """
    
    t_input, t_label, t_ct = tf_read_sample_file(data_record, sizes_data, sizes_input, noise_std, gain_rnd_low, gain_rnd_high, hist_shift_rnd_low, hist_shift_rnd_high, not_modified, not_cropped, not_transformed, pixel_value_mod, cdf_sampler_coef)
    
    inputs = t_input


    outputs = tf.concat([t_ct, t_label], axis = -1)

    return inputs, outputs


    
def get_sample_center(min_val, max_val, sampler_coef):
    """Returns a randomly selected sample center.
    """
    
    max_val = tf.cast(max_val,tf.float64)
    min_val = tf.cast(min_val,tf.float64)
    rand_center = tf.random.uniform([],minval=0.0,maxval=1.0)    
    rand_center = (tf.math.polyval(sampler_coef, tf.cast(rand_center,tf.float64))*(max_val-min_val))+min_val
    rand_center = tf.cast(rand_center,tf.int32)
                   
        
    return rand_center

def tf_get_sample_slice(t_input, t_label, t_ct, sizes_input, initial_pos_0, initial_pos_1, rnd_Z):
    """Returns a slice from a full sized sample
    
        Arguments:
        t_input --- NAC PET input volume
        t_label --- Tissue labels input volume
        t_ct --- CT input volume
        sizes_input --- Networ input size
        initial_pos_0 --- x initial position
        initial_pos_1 --- y initial position
        rnd_Z --- Random slice centyer (z axis)
        
        Returns:
        t_input --- NAC PET slice of Network input size
        t_label --- Tissue slice of Network input size
        t_ct --- CT slice of Network input size
    """
    
    # Get info size
    voxels_X = sizes_input[0]
    voxels_Y = sizes_input[1]
    voxels_Z = sizes_input[2]
    margin_Z = tf.cast(sizes_input[2]/2,tf.int32)
    
    t_input = tf.reshape(t_input[initial_pos_0:initial_pos_0+sizes_input[0],
                                 initial_pos_1:initial_pos_1+sizes_input[1],
                                 rnd_Z-margin_Z:rnd_Z+margin_Z], 
                         (voxels_X,voxels_Y,voxels_Z, PET_CHANNELS))
    
    t_label = tf.reshape(t_label[initial_pos_0:initial_pos_0+sizes_input[0],
                                 initial_pos_1:initial_pos_1+sizes_input[1],
                                 rnd_Z-margin_Z:rnd_Z+margin_Z,
                                 :], 
                         (voxels_X,voxels_Y,voxels_Z, LABELS_CAHNNELS))
    
    t_ct    = tf.reshape(t_ct   [initial_pos_0:initial_pos_0+sizes_input[0],
                                 initial_pos_1:initial_pos_1+sizes_input[1],
                                 rnd_Z-margin_Z:rnd_Z+margin_Z], 
                         (voxels_X,voxels_Y,voxels_Z, CT_CAHNNELS))
    
    return t_input, t_label, t_ct




#-----------------------------------------------------------------------------#
#--------------- DICOM functions ---------------------------------------------#
#-----------------------------------------------------------------------------#

def saveDICOM_sliced_CT(image_np, voxel_size, image_position_patient, output_path, output_name, 
                        orig_Z_pos = 0, patient_ID_in = 'SynthAttCorr_Test', StudyInstanceUID='', SeriesInstanceUID='',
                        TYPE_INT = np.int16, vervose = True):
    '''Saves a 3D numpy array as a CT DICOM file.
    '''
    
    # Get info type
    info_bits = np.iinfo(TYPE_INT)
    Vol_CT_max_val = np.max(image_np)

    # Create output folder
    OUTPUT_DCM_FOLDER = os.path.join(output_path,output_name)
    File_mng.check_create_path('OUTPUT_DCM_FOLDER', OUTPUT_DCM_FOLDER, clear_folder=True)

    # get current time
    now = datetime.datetime.now()
    
    # Number of slices to save
    num_CT_slices = image_np.shape[Z]

    # Save each slice
    for idx_slice_CT in range(num_CT_slices):

        print('Slice: %d/%d       '%(idx_slice_CT+1,num_CT_slices), end='')
        print('', end='\r')

        OUTPUT_DCM = os.path.join(OUTPUT_DCM_FOLDER,'%04d.dcm'%idx_slice_CT)

        slice_aux = image_np[:,:,idx_slice_CT]

        # This code was taken from the output of a valid CT file
        # I do not know what the long dotted UIDs mean, but this code works...
        file_meta = Dataset()
        # (0008, 0016) SOP Class UID                       UI: CT Image Storage
        file_meta.MediaStorageSOPClassUID = 'CT Image Storage'
        # (0008, 0018) SOP Instance UID                    UI: 1.3.6.1.4.1.14519.5.2.1.3320.3273.139372384049551777032016175435
        file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.14519.5.2.1.3320.3273.139372384049551777032016175435'

    #     file_meta.ImplementationClassUID = '1.2.840.10008.5.1.4.1.1.20'

        file_meta.TransferSyntaxUID      = dicom.uid.ImplicitVRLittleEndian

        ds = FileDataset(OUTPUT_DCM, {},file_meta = file_meta,preamble=b'\x00'*128)
        ds.Modality = 'CT'
        ds.ContentDate = str(datetime.date.today()).replace('-','')
        ds.ContentTime = str(time.time()) #milliseconds since the epoch


        # (0020, 000d) Study Instance UID                  UI: 1.3.6.1.4.1.14519.5.2.1.3320.3273.231445815234682119538863169983
        if StudyInstanceUID == '':
            StudyInstanceUID = '1.3.6.1.4.1.14519.5.2.1.3320.3273.231445815234682119538863169983'
        ds.StudyInstanceUID =  StudyInstanceUID
        # (0020, 000e) Series Instance UID                 UI: 1.3.6.1.4.1.14519.5.2.1.3320.3273.330516103388011054891344582212
        if SeriesInstanceUID == '':
            SeriesInstanceUID = '1.3.6.1.4.1.14519.5.2.1.3320.3273.330516103388011054891344582212'
        ds.SeriesInstanceUID = SeriesInstanceUID

        # (0008, 1155) Referenced SOP Instance UID         UI: 1.3.6.1.4.1.14519.5.2.1.3320.3273.275604822259752169794323488440
        ds.SOPInstanceUID =    '1.3.6.1.4.1.14519.5.2.1.3320.3273.2756048222597%d'%np.random.randint(low=52169794323488440, high=92169794323488440)
        # ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.20'

        # These are the necessary imaging components of the FileDataset object.

        # (0028, 0002) Samples per Pixel                   US: 1
        ds.SamplesPerPixel = 1
        # (0028, 0004) Photometric Interpretation          CS: 'MONOCHROME2'
        ds.PhotometricInterpretation = "MONOCHROME2"

        # (0028, 0103) Pixel Representation                US: 1
        ds.PixelRepresentation = 1

        # (0028, 0100) Bits Allocated                      US: 16
        ds.BitsAllocated = 16
        # (0028, 0101) Bits Stored                         US: 16
        ds.BitsStored = 16
        # (0028, 0102) High Bit                            US: 15
        ds.HighBit = 15

        # (0028, 0010) Rows                                US: 512
        ds.Rows = slice_aux.shape[X]
        # (0028, 0011) Columns                             US: 512
        ds.Columns = slice_aux.shape[Y]

        # (0028, 0030) Pixel Spacing                       DS: ['0.976562', '0.976562']
        ds.PixelSpacing = [str(voxel_size[X]), str(voxel_size[Y])]

        # (0018, 0088) Spacing Between Slices              DS: "3.260000"
        ds.SpacingBetweenSlices = b'00000'

        # (0018, 0050) Slice Thickness                     DS: "1.250000"
        ds.SliceThickness = str(voxel_size[Z])

        ds.ImageOrientationPatient = ['1.000000', '0.000000', '0.000000', '0.000000', '1.000000', '0.000000']

        ds.ImagePositionPatient = [str(image_position_patient[X]).encode(), 
                                   str(image_position_patient[Y]).encode(), 
                                   str(orig_Z_pos+(idx_slice_CT*voxel_size[Z])).encode()]





        # Otros agregados

        # (0008, 0050) Accession Number                    SH: ''
        ds.AccessionNumber = ''
        # (0008, 0022) Acquisition Date                    DA: '20001007'
        ds.AcquisitionDate = now.strftime("%Y%m%d")
        # (0008, 0032) Acquisition Time                    TM: '110409.543667'
        ds.AcquisitionTime = now.strftime("%H%M%S")

        # (0008, 0008) Image Type                          CS: ['ORIGINAL', 'PRIMARY', 'AXIAL']
        ds.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
        # (0008, 0012) Instance Creation Date              DA: '20001007'
        ds.InstanceCreationDate = now.strftime("%Y%m%d")
        # (0008, 0013) Instance Creation Time              TM: '110451'
        ds.InstanceCreationTime = now.strftime("%H%M%S")
        # (0020, 0013) Instance Number                     IS: "207"
        ds.InstanceNumber = str(idx_slice_CT)

        ds.Manufacturer = 'UTN-UTT-CNEA-CONICET'

        # (0010, 0020) Patient ID                          LO: 'C3N-02285'
        ds.PatientID = patient_ID_in
        # (0010, 0010) Patient's Name                      PN: 'C3N-02285'
        ds.PatientName = output_name

        # (0018, 5100) Patient Position                    CS: 'HFS'
        ds.PatientPosition = 'HFS'
        # (0018, 1030) Protocol Name                       LO: '4.1 PET_WB LOW BMI'
        ds.ProtocolName = 'Synth. CT'

        # (0018, 1100) Reconstruction Diameter             DS: "500.000000"
        ds.ReconstructionDiameter = '500.000000'
        # (0008, 0090) Referring Physician's Name          PN: ''
        ds.ReferringPhysicianName = ''

        # (0008, 0021) Series Date                         DA: '20001007'
        ds.SeriesDate = now.strftime("%Y%m%d")
        ds.SeriesNumber = ''
        # (0008, 0031) Series Time                         TM: '110352'
        ds.SeriesTime = now.strftime("%H%M%S")
        # (0020, 1041) Slice Location                      DS: "-216.500"
        ds.SliceLocation = str(orig_Z_pos+(idx_slice_CT*voxel_size[Z]))
        # (0008, 0020) Study Date                          DA: '20001007'
        ds.StudyDate = now.strftime("%Y%m%d")
        # (0008, 1030) Study Description                   LO: 'PET WB LOW BMI'
        ds.StudyDescription = 'synthetic CT created from NAC PET'
        ds.StudyID = ''
        # (0008, 0030) Study Time                          TM: '110246'
        ds.StudyTime = now.strftime("%H%M%S")


        # (0028, 1052) Rescale Intercept                   DS: "-1024"
        ds.RescaleIntercept = "-1024"
        # (0028, 1053) Rescale Slope                       DS: "1"
        ds.RescaleSlope = "1"#str(Vol_CT_max_val/info_bits.max)
        # (0028, 1054) Rescale Type                        LO: 'HU'
        ds.RescaleType = 'HU'

    #     slice_aux = ((slice_aux/Vol_CT_max_val)*info_bits.max)+1024
    #     slice_aux = (((slice_aux+1024)/Vol_CT_max_val)*info_bits.max)
        slice_aux = (slice_aux+1024).astype(np.int16)
    #     print(slice_aux.max(),slice_aux.min())


        # volume_aux = np.swapaxes(volume_aux,0,1)
        # volume_aux = np.swapaxes(volume_aux,0,2)

        ds.PixelData = slice_aux.tostring()

        ds.save_as(OUTPUT_DCM)



    
    print('done.')
    
    return