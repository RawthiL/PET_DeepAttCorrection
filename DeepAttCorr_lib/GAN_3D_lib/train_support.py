#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

# External
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import numpy as np
import matplotlib.pyplot as plt

import os, shutil, sys

from dataclasses import dataclass

from scipy import ndimage
from scipy import signal
from skimage import measure


# Internal
from DeepAttCorr_lib.GAN_3D_lib import layers
from DeepAttCorr_lib.GAN_3D_lib import losses
from DeepAttCorr_lib import file_manage_utils as File_mng


#-----------------------------------------------------------------------------
#---------------------- Training functions -----------------------------------
#-----------------------------------------------------------------------------


def train_step_generator_conditional_3D_GAN_tf(images_ref_dist,
                                               images_obj_dist, 
                                               g_model_dist, 
                                               d_model_dist, 
                                               train_g_loss,
                                               train_g_comp_loss,
                                               optimizer_gen,
                                               mirrored_strategy,
                                               norm_by_size=False,
                                              K_comp_loss=0.0):
    '''Performs a training step of the generator network using a multi GPU strategy
    '''
    
    def step_fn(images_ref, images_obj, g_model, d_model, train_g_loss, 
                train_g_comp_loss, optimizer_gen, norm_by_size, K_comp_loss):
        
        n_batch = images_ref.shape[0]  
        img_res = images_ref.shape[1]*images_ref.shape[2]*images_ref.shape[3]
        if norm_by_size:
            SIZE_REG = img_res
        else:
            SIZE_REG = 1.0
        

        # Real samples
        (V_real_ref, 
         V_real_obj, 
         y_real) = generate_real_3D_conditional_samples_tf(images_ref, images_obj, 0)
        

        
        with tf.GradientTape() as gen_tape: 
       
            # Fake samples               
            V_fake = g_model(V_real_ref)
            y_fake = tf.constant(-1.0, shape=(n_batch,1), dtype=tf.float32)
            
            # Evaluate fake samples
            d_fake_y_pred = d_model([V_real_ref, V_fake])
            
            # Gen. loss
            g_loss = losses.wasserstein_loss_tf(d_fake_y_pred, y_real)
            g_loss = tf.reduce_sum(g_loss) * (1. / (n_batch))

            # MSE
            g_loss_comp = tf.reduce_sum(tf.pow((V_fake - V_real_obj),2),[1,2,3,4])
            g_loss_comp = (g_loss_comp*SIZE_REG)
            g_loss_comp = tf.reduce_sum(g_loss_comp) * (1. / (n_batch))
            
            
            g_loss_total = g_loss+K_comp_loss*g_loss_comp
            g_loss_total = tf.reduce_sum(g_loss_total) * (1. / (n_batch))
            
        

        grad_gen = gen_tape.gradient(g_loss_total, g_model.trainable_variables)
        optimizer_gen.apply_gradients(zip(grad_gen, g_model.trainable_variables))

            
        train_g_loss(g_loss)
        train_g_comp_loss(g_loss_comp)
        


        return
    
    mirrored_strategy.experimental_run_v2(step_fn, args=(images_ref_dist,
                                                         images_obj_dist, 
                                                         g_model_dist, 
                                                         d_model_dist, 
                                                         train_g_loss,
                                                         train_g_comp_loss,
                                                         optimizer_gen,
                                                         norm_by_size,
                                                        K_comp_loss))
    
    
    return


def train_step_discriminator_conditional_3D_GAN_tf(images_ref_dist, 
                                                   images_obj_dist, 
                                                   g_model_dist, 
                                                   d_model_dist, 
                                                   train_d1_loss,
                                                   train_d2_loss,
                                                   train_dgrad_loss,
                                                   optimizer_disc,
                                                   mirrored_strategy,
                                                   K_grad=10.0,
                                                   cross_sample_loss = False):
    '''Performs a training step of the discrimniator network using a multi GPU strategy
    '''
    
    def step_fn(images_ref, images_obj, g_model, d_model, train_d1_loss, train_d2_loss, train_dgrad_loss, optimizer_disc, K_grad, cross_sample_loss):
        
        n_batch = images_ref.shape[0]         

        # Real samples
        (V_real_ref, 
         V_real_obj, 
         y_real) = generate_real_3D_conditional_samples_tf(images_ref,images_obj, 0)
        
        
        with tf.GradientTape() as disc_tape:  
        
            # Fake samples
            V_fake = g_model(V_real_ref)
            y_fake = tf.constant(-1.0, shape=(n_batch,1), dtype=tf.float32)
        
            # Gradient penalty
            if True:
                epsilon = tf.random.uniform([V_real_obj.shape[0], 1, 1, 1, 1], 0.0, 1.0)
                V_hat = V_real_obj*epsilon + (1-epsilon)*V_fake

                with tf.GradientTape() as grad_pen_tape:    
                    grad_pen_tape.watch(V_hat)
                    d_hat_y_pred = d_model([V_real_ref, V_hat])

                grad_d_hat = grad_pen_tape.gradient(d_hat_y_pred, V_hat)

                grad_d_hat_slopes = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(grad_d_hat), axis=[1, 2, 3]))

                gradient_penalty = tf.reduce_mean(tf.square((grad_d_hat_slopes - 1.)))
            else:
                gradient_penalty = 0.0
            
            # Disc. loss
            d_real_y_pred = d_model([V_real_ref, V_real_obj])
            d_loss1 = losses.wasserstein_loss_tf(d_real_y_pred, y_real)
            
            d_fake_y_pred = d_model([V_real_ref, V_fake])
            d_loss2 = losses.wasserstein_loss_tf(d_fake_y_pred, y_fake)
            
            if cross_sample_loss:
                # Reorder reference batch in inverse order
                V_real_ref_reorder = tf.reverse(V_real_ref, axis=[0])
                
                d_cross_sample_y_pred = d_model([V_real_ref_reorder, V_real_obj])
                d_loss3 = losses.wasserstein_loss_tf(d_cross_sample_y_pred, y_fake)
                
            else:
                d_loss3 = 0.0
                
            d_loss_grad = K_grad*gradient_penalty 
            d_loss_disc = d_loss1 + d_loss2 + d_loss3 + d_loss_grad
            
            d_loss1 = tf.reduce_sum(d_loss1) * (1. / (n_batch))
            d_loss2 = tf.reduce_sum(d_loss2) * (1. / (n_batch))
            d_loss_grad = tf.reduce_sum(d_loss_grad) * (1. / (n_batch))
            
            
        grad_disc = disc_tape.gradient(d_loss_disc, d_model.trainable_variables)
        optimizer_disc.apply_gradients(zip(grad_disc, d_model.trainable_variables))
            
        train_d1_loss(d_loss1)
        train_d2_loss(d_loss2)
        train_dgrad_loss(d_loss_grad)

        return
    
    mirrored_strategy.experimental_run_v2(step_fn, args=(images_ref_dist,
                                                         images_obj_dist, 
                                                         g_model_dist, 
                                                         d_model_dist, 
                                                         train_d1_loss,
                                                         train_d2_loss,
                                                         train_dgrad_loss,
                                                         optimizer_disc,
                                                         K_grad,
                                                         cross_sample_loss,))
    
    return




def train_step_generator_and_segmentator_conditional_3D_GAN_tf(images_ref_dist,
                                                               images_obj_dist, 
                                                               labels_obj_dist,
                                                               g_model_dist, 
                                                               d_model_dist, 
                                                               s_model_dist,
                                                               train_g_loss,
                                                               train_g_comp_loss,
                                                               train_s1_loss,
                                                               train_s2_loss,
                                                               optimizer_gen,
                                                               mirrored_strategy,
                                                               K_comp_loss = 1.0,
                                                               K_comp_segm_loss = 10.0,
                                                               norm_by_size = True,
                                                               split_train = False,
                                                               train_segm = True):
    '''Performs a training step of the generator and segmentator networks using a multi GPU strategy
    '''
    
    def step_fn(images_ref, images_obj, labels_obj, g_model, d_model, s_model, train_g_loss, train_g_comp_loss, train_s1_loss, train_s2_loss, optimizer_gen, K_comp_loss, norm_by_size, split_train, train_segm):
        
        n_batch = images_ref.shape[0]  
        img_res = images_ref.shape[1]*images_ref.shape[2]*images_ref.shape[3]
        if norm_by_size:
            SIZE_REG = img_res
        else:
            SIZE_REG = 1.0
        

        # Real samples
        (V_real_ref, 
         V_real_obj, 
         y_real) = generate_real_3D_conditional_samples_tf(images_ref, images_obj, 0)
        
        V_real_ref_segm = images_ref
        V_real_obj_segm = images_obj
        V_labels_obj_segm = labels_obj
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as segm_tape: 
                      
            # Segmentate samples
            V_compound = s_model(V_real_ref_segm)
            V_texture, V_segm = tf.split(V_compound, [1,4], axis=-1)

            # DICE Loss
            s_loss_1 = losses.DICE_loss(V_labels_obj_segm, V_segm)
            s_loss_1 = (1.0-s_loss_1)
            # L2 Loss
            s_loss_2 = tf.reduce_sum(tf.pow((V_real_obj_segm - V_texture),2),[1,2,3,4])
            
            s_loss_1 = tf.reduce_sum(s_loss_1) * (1. / (n_batch))
            s_loss_2 = tf.reduce_sum(s_loss_2) * (1. / (n_batch))
            
            s_loss = s_loss_1 + (K_comp_segm_loss*s_loss_2 )

            # Fake samples               
            V_fake = g_model(V_real_ref)
            y_fake = tf.constant(-1.0, shape=(n_batch,1), dtype=tf.float32)
            
            # Evaluate fake samples
            d_fake_y_pred = d_model([V_real_ref, V_fake])
            
            # Gen. loss
            g_loss = losses.wasserstein_loss_tf(d_fake_y_pred, y_real)

            g_loss_comp = tf.reduce_sum(tf.pow((V_fake - V_real_obj),2),[1,2,3,4])
            g_loss_comp = (g_loss_comp*SIZE_REG)
            
            g_loss_total = g_loss + K_comp_loss*s_loss
            
            g_loss_comp = tf.reduce_sum(g_loss_comp) * (1. / (n_batch))
            g_loss = tf.reduce_sum(g_loss) * (1. / (n_batch))
            g_loss_total = tf.reduce_sum(g_loss_total) * (1. / (n_batch))
            
        
        if train_segm and not split_train:
            grad_gen = gen_tape.gradient(g_loss_total, g_model.trainable_variables)
            optimizer_gen.apply_gradients(zip(grad_gen, g_model.trainable_variables))
        elif not train_segm and not split_train:
            grad_gen = gen_tape.gradient(g_loss, g_model.trainable_variables)
            optimizer_gen.apply_gradients(zip(grad_gen, g_model.trainable_variables))
        elif split_train:
            s_model.trainable=False
            grad_gen = gen_tape.gradient(g_loss, g_model.trainable_variables)
            optimizer_gen.apply_gradients(zip(grad_gen, g_model.trainable_variables))
            
            if train_segm:
                s_model.trainable=True
                grad_segm = segm_tape.gradient(s_loss, s_model.trainable_variables)
                optimizer_gen.apply_gradients(zip(grad_segm, s_model.trainable_variables))

        
            
        train_g_loss(g_loss)
        train_g_comp_loss(g_loss_comp)
        
        train_s1_loss(s_loss_1)
        train_s2_loss(s_loss_2)

        return
    
    mirrored_strategy.experimental_run_v2(step_fn, args=(images_ref_dist,
                                                         images_obj_dist, 
                                                         labels_obj_dist,
                                                         g_model_dist, 
                                                         d_model_dist, 
                                                         s_model_dist,
                                                         train_g_loss,
                                                         train_g_comp_loss,
                                                         train_s1_loss,
                                                         train_s2_loss,
                                                         optimizer_gen,
                                                         K_comp_loss,
                                                         norm_by_size,
                                                         split_train, 
                                                         train_segm,))
    
    
    return



#-----------------------------------------------------------------------------
#---------------------- Real Sample Generation -------------------------------
#-----------------------------------------------------------------------------
@tf.function
def low_sample_and_normalination(images_in, low_sample_pow):
    '''Performs a downsampling using average pooling in tensorflow
    '''
    for i in range(low_sample_pow):
        images_in = tf.nn.avg_pool3d(images_in,
                         [1,2,2,2,1], 
                         [1,2,2,2,1], 
                         'SAME')

    images_in = (images_in-0.5)*2.0
    
    return images_in

@tf.function
def generate_real_3D_conditional_samples_tf(images_in_condition, images_in_objective, low_sample_pow):
    '''Creates a pair of NAC PET - CT co-registered samples for the NN
    '''
    
    batch_size = images_in_condition.shape[0]
    
    # Lowsample inputs
    volumes_in_condition = low_sample_and_normalination(images_in_condition, low_sample_pow)
    volumes_in_objective = low_sample_and_normalination(images_in_objective, low_sample_pow)
        
    return volumes_in_condition, volumes_in_objective, tf.constant(1.0, shape=(batch_size,1), dtype=tf.float32)




#-----------------------------------------------------------------------------
#---------------------- Validation and Plotting functions --------------------
#-----------------------------------------------------------------------------

def compute_whole_volume(in_NAC_PET, input_size, g_model, limits_act, margin, sample_num=0, segm_net=False, s_model='', criticize=False, d_model='', slice_val_cap_use=1.0):
    """Compute a whole image using multiple calls of the NN model.
    """
    
    # Get info size
    voxels_X = input_size[0]
    voxels_Y = input_size[1]
    voxels_Z = input_size[2]
    
    
    data_size = in_NAC_PET.shape
        
    voxels_X_Full = data_size[0]
    voxels_Y_Full = data_size[1]
    voxels_Z_Full = data_size[2]
    
    if limits_act[0] == 0:
        limits_act[0] = margin
    if limits_act[1] == data_size[2]:
        limits_act[1] = data_size[2]-margin

    synth_CT_whole = np.zeros([data_size[0],data_size[1],data_size[2]])
    sementation_whole = np.zeros((data_size[0],data_size[1],data_size[2],4))
    promedio = np.zeros([data_size[0],data_size[1],data_size[2]])

    len_tot = len(range(int(limits_act[0])-margin, int(limits_act[1])+margin, margin))


    weight_mod = signal.gaussian(voxels_Z, 4, sym=True)
    weight_vol = np.tile(weight_mod, (data_size[0],data_size[1],1))

    slice_score = 0
    count = 0
    slice_count = 0
    for Z_ini in range(int(limits_act[0])-margin, int(limits_act[1])+margin, margin):
        
        slice_count += 1

        print('(Sample %d) Slice: %d/%d       '%(sample_num+1, slice_count,len_tot), end='')
        print('', end='\r')

        if Z_ini+voxels_Z > voxels_Z_Full:
            break
        else:
            Z_fin = Z_ini+voxels_Z

        min_act = 9e9

        # Crop Single slice   
        initial_pos_0 = 0
        initial_pos_1 = 0
        in_NAC_PET_slice = tf.reshape(in_NAC_PET[initial_pos_0:initial_pos_0+voxels_X,
                                                 initial_pos_1:initial_pos_1+voxels_Y,
                                                 Z_ini:Z_ini+voxels_Z], 
                                      (voxels_X,voxels_Y,voxels_Z, 1))


        in_NAC_PET_slice = tf.expand_dims(in_NAC_PET_slice, axis=0)

        in_NAC_PET_slice_np = in_NAC_PET_slice.numpy()

        # Prepare slice
        in_NAC_PET_slice = shift_image_values(in_NAC_PET_slice_np, slice_val_cap_use)
        in_NAC_PET_network_slice = (in_NAC_PET_slice-0.5)*2.0
        

        # Get synthetic CT
        synth_CT_slice = g_model(in_NAC_PET_network_slice, training=False)
        
        # Test if is an 3D Unet model
        standard_3d_unet = False
        if (synth_CT_slice.shape[-1] == 5) and segm_net:
            standard_3d_unet = True
            synth_CT_slice, _ = tf.split(synth_CT_slice, [1,4], axis=-1)

        # Criticize slice
        if criticize:
            input_net = list()
            input_net.append(in_NAC_PET_network_slice)
            input_net.append(synth_CT_slice)
            aux_score = d_model(input_net, training=False).numpy()
            if not np.isnan(aux_score):
                slice_score += aux_score
                count += 1

        if not standard_3d_unet:
            synth_CT_slice_np = np.copy(np.squeeze(synth_CT_slice.numpy()))
            synth_CT_slice_np = (synth_CT_slice_np/2.0)+0.5

            synth_CT_whole[:,:,Z_ini:Z_fin] += np.multiply(synth_CT_slice_np,weight_vol)
            promedio[:,:,Z_ini:Z_fin] += (1.0*weight_mod)

        # Segmentate
        if segm_net:
            out_v_net_segm_slice = s_model(in_NAC_PET_slice)
            X_texture_slice, X_segm_slice = tf.split(out_v_net_segm_slice, [1,4], axis=-1)
            segm_slice_np = np.copy(np.squeeze(X_segm_slice.numpy()))
            # Get class map
            class_map = np.argmax(segm_slice_np, axis = 3)

            weight_mod_slice= np.reshape(weight_mod, [1,1,32])
            weight_mod_slice = np.repeat(weight_mod_slice, 128, axis = 0)
            weight_mod_slice = np.repeat(weight_mod_slice, 128, axis = 1)
            sementation_whole[:,:,Z_ini+margin:Z_fin-margin,0] += (class_map[:,:,margin:-margin] == 0).astype(np.float32)*weight_mod_slice[:,:,margin:-margin]
            sementation_whole[:,:,Z_ini+margin:Z_fin-margin,1] += (class_map[:,:,margin:-margin] == 1).astype(np.float32)*weight_mod_slice[:,:,margin:-margin]
            sementation_whole[:,:,Z_ini+margin:Z_fin-margin,2] += (class_map[:,:,margin:-margin] == 2).astype(np.float32)*weight_mod_slice[:,:,margin:-margin]
            sementation_whole[:,:,Z_ini+margin:Z_fin-margin,3] += (class_map[:,:,margin:-margin] == 3).astype(np.float32)*weight_mod_slice[:,:,margin:-margin]

            if standard_3d_unet:
                synth_CT_slice_np = np.copy(np.squeeze(X_texture_slice.numpy()))
#                 synth_CT_slice_np = (synth_CT_slice_np/2.0)+0.5

                synth_CT_whole[:,:,Z_ini:Z_fin] += np.multiply(synth_CT_slice_np,weight_vol)
                promedio[:,:,Z_ini:Z_fin] += (1.0*weight_mod)

    # Normalize
    prom_divide = np.copy(promedio)
    prom_divide[prom_divide==0] = 1.0
    synth_CT_whole = np.divide(synth_CT_whole,prom_divide)
    synth_CT_whole[promedio == 0] = 0

    synth_CT_whole[:,:,:int(limits_act[0])] = 0
    synth_CT_whole[:,:,int(limits_act[1]):] = 0

    synth_CT_whole[synth_CT_whole < 0 ] = 0
    synth_CT_whole[synth_CT_whole > 1] = 1


    if segm_net and criticize:
        return synth_CT_whole, sementation_whole, (slice_score/count)
    if segm_net:
        return synth_CT_whole, sementation_whole
    if criticize:
        return synth_CT_whole, (slice_score/count)
        
    return synth_CT_whole

def shift_image_values(test_input, current_limit):
    """Shift volume voxel values
    """
    
    out_img = np.copy(test_input)
    
    # Get maximum 
    in_max = test_input.max()
    # Set new maximum value
    new_max = current_limit*in_max
    # Cap image to new value
    out_img[test_input>new_max] = new_max
    # Re-Normalize
    if (new_max-out_img.min()) != 0:
        out_img = (out_img-out_img.min())/(new_max-out_img.min())
    else:
        out_img = (out_img-out_img.min())*0.0
        
    return out_img


def add_tensorboard_2Dimage(image_in_2D, name_img, step_in, min_val=0.0, max_val=1.0):
    """Adds a 2D image to the tensorboard output.
    """
    voxels_X = image_in_2D.shape[0]
    voxels_Y = image_in_2D.shape[1]

    image_in_2D = (image_in_2D-min_val)/(max_val-min_val)

    image_in = tf.reshape(image_in_2D, (1,voxels_X, voxels_Y, 1))

    tf.summary.image(name_img, image_in, step=step_in)

    return




def plot_images3D_conditional(X_real_ref, X_real_obj, X_fake, 
                              image_plot = 0, segm_net = False, X_segm='', X_real_segm='', dpi_use=75,
                              add_tensorboard=False, epoch=0):
    """Plots the output of the model and the objective. Optionally adds them to the tensorboard out.
    """
    
    slice_X = int(X_real_ref.shape[1]/2)
    slice_Y = int(X_real_ref.shape[2]/2)
    slice_Z = int(X_real_ref.shape[3]/2)

    fig, ax = plt.subplots(nrows=1, ncols=3, dpi=dpi_use, sharex=True, sharey=True)
    ax[0].imshow(X_real_ref[image_plot,slice_X,:,:,0], vmin=-1.0, vmax=1.0, aspect='equal')
    ax[0].axis('off')       
    ax[1].imshow(X_real_ref[image_plot,:,slice_Y,:,0], vmin=-1.0, vmax=1.0, aspect='equal')
    ax[1].axis('off')
    ax[2].imshow(X_real_ref[image_plot,:,:,slice_Z,0], vmin=-1.0, vmax=1.0, aspect='equal')
    ax[2].axis('off')
    plt.tight_layout()
    plt.plot()
    
    if add_tensorboard:
        add_tensorboard_2Dimage(np.squeeze(X_real_ref[image_plot,slice_X,:,:,0]), 'NAC_PET_axial', epoch, min_val=-1.0, max_val=1.0)
        add_tensorboard_2Dimage(np.squeeze(X_real_ref[image_plot,:,slice_Y,:,0]), 'NAC_PET_sagital', epoch, min_val=-1.0, max_val=1.0)
        add_tensorboard_2Dimage(np.squeeze(X_real_ref[image_plot,:,:,slice_Z,0]), 'NAC_PET_coronal', epoch, min_val=-1.0, max_val=1.0)
            
            
    
    fig, ax = plt.subplots(nrows=2, ncols=3, dpi=dpi_use, sharex=True, sharey=True)
    ax[0][0].imshow(X_real_obj[image_plot,slice_X,:,:,0], vmin=-1.0, vmax=1.0, aspect='equal')
    ax[0][0].axis('off')
    ax[0][1].imshow(X_real_obj[image_plot,:,slice_Y,:,0], vmin=-1.0, vmax=1.0, aspect='equal')
    ax[0][1].axis('off')
    ax[0][2].imshow(X_real_obj[image_plot,:,:,slice_Z,0], vmin=-1.0, vmax=1.0, aspect='equal')
    ax[0][2].axis('off')
    
    if add_tensorboard:
        add_tensorboard_2Dimage(np.squeeze(X_real_obj[image_plot,slice_X,:,:,0]), 'CT_axial', epoch, min_val=-1.0, max_val=1.0)
        add_tensorboard_2Dimage(np.squeeze(X_real_obj[image_plot,:,slice_Y,:,0]), 'CT_sagital', epoch, min_val=-1.0, max_val=1.0)
        add_tensorboard_2Dimage(np.squeeze(X_real_obj[image_plot,:,:,slice_Z,0]), 'CT_coronal', epoch, min_val=-1.0, max_val=1.0)
    
    
    ax[1][0].imshow(X_fake[image_plot,slice_X,:,:,0], vmin=-1.0, vmax=1.0, aspect='equal')
    ax[1][0].axis('off')
    ax[1][1].imshow(X_fake[image_plot,:,slice_Y,:,0], vmin=-1.0, vmax=1.0, aspect='equal')
    ax[1][1].axis('off')
    ax[1][2].imshow(X_fake[image_plot,:,:,slice_Z,0], vmin=-1.0, vmax=1.0, aspect='equal')
    ax[1][2].axis('off')
    
    if add_tensorboard:
        add_tensorboard_2Dimage(np.squeeze(X_fake[image_plot,slice_X,:,:,0]), 'sCT_axial', epoch, min_val=-1.0, max_val=1.0)
        add_tensorboard_2Dimage(np.squeeze(X_fake[image_plot,:,slice_Y,:,0]), 'sCT_sagital', epoch, min_val=-1.0, max_val=1.0)
        add_tensorboard_2Dimage(np.squeeze(X_fake[image_plot,:,:,slice_Z,0]), 'sCT_coronal', epoch, min_val=-1.0, max_val=1.0)
    
    plt.tight_layout()
    plt.plot()
    
    if segm_net:
        num_class = X_segm.shape[-1]
        X_segm = np.argmax(X_segm, axis=-1)
        X_real_segm = np.argmax(X_real_segm, axis=-1)
        
        fig, ax = plt.subplots(nrows=2, ncols=3, dpi=dpi_use, sharex=True, sharey=True)
        ax[0][0].imshow(X_segm[image_plot,slice_X,:,:], vmin=0.0, vmax=num_class-1, aspect='equal')
        ax[0][0].axis('off')
        ax[0][1].imshow(X_segm[image_plot,:,slice_Y,:], vmin=0.0, vmax=num_class-1, aspect='equal')
        ax[0][1].axis('off')
        ax[0][2].imshow(X_segm[image_plot,:,:,slice_Z], vmin=0.0, vmax=num_class-1, aspect='equal')
        ax[0][2].axis('off')
        
        if add_tensorboard:
            add_tensorboard_2Dimage(np.squeeze(X_segm[image_plot,slice_X,:,:,0]), 'Labels_Obj_axial', epoch, 0, min_val=0.0, max_val=1.0)
            add_tensorboard_2Dimage(np.squeeze(X_segm[image_plot,:,slice_Y,:,0]), 'Labels_Obj_sagital', epoch, 0, min_val=0.0, max_val=1.0)
            add_tensorboard_2Dimage(np.squeeze(X_segm[image_plot,:,:,slice_Z,0]), 'Labels_Obj_coronal', epoch, 0, min_val=0.0, max_val=1.0)



        ax[1][0].imshow(X_real_segm[image_plot,slice_X,:,:], vmin=0.0, vmax=num_class-1, aspect='equal')
        ax[1][0].axis('off')
        ax[1][1].imshow(X_real_segm[image_plot,:,slice_Y,:], vmin=0.0, vmax=num_class-1, aspect='equal')
        ax[1][1].axis('off')
        ax[1][2].imshow(X_real_segm[image_plot,:,:,slice_Z], vmin=0.0, vmax=num_class-1, aspect='equal')
        ax[1][2].axis('off')
        
        if add_tensorboard:
            add_tensorboard_2Dimage(np.squeeze(X_real_segm[image_plot,slice_X,:,:,0]), 'Labels_Gen_axial', epoch, 0, min_val=0.0, max_val=1.0)
            add_tensorboard_2Dimage(np.squeeze(X_real_segm[image_plot,:,slice_Y,:,0]), 'Labels_Gen_sagital', epoch, 0, min_val=0.0, max_val=1.0)
            add_tensorboard_2Dimage(np.squeeze(X_real_segm[image_plot,:,:,slice_Z,0]), 'Labels_Gen_coronal', epoch, 0, min_val=0.0, max_val=1.0)

        plt.tight_layout()
        plt.plot()

    return

def validate_whole_volume(g_model, d_model, dataset_validation, input_size, segm_net = False, s_model='', single_image=False, all_images_out=False, MARGIN = 3 ):
    """Perform validation of a model on a full scale sample
    
        This function takes a validation dataset as imput and applies the NN
        on a full scale sample (by slicing). Then it calculates some metrics
        and plots the results.
        Used for training progress loggin.
    """
    
    
    error_val = 0.0
    d_value = 0.0
    
    CT_INPUT_images = list()
    PET_INPUT_images = list()
    LABELS_INPUT_images = list()
    CT_SYNTH_images = list()
    SEGMENTED_images = list()
    SCORE_images = list()
    PSNR_images = list()
    ME_images = list()
    NMSE_images = list()
    NCC_images = list()
    
    batch_carry = 0
    for in_NAC_PET, in_LABELS, in_CT, in_limits in dataset_validation:
        
        batch_size = in_NAC_PET.shape[0]
        
        
        for idx_batch in range(batch_size):
            limits_act = in_limits[idx_batch,:]

            if segm_net:
                (synth_CT_whole,
                 sementation_whole,
                 score_whole) = compute_whole_volume(in_NAC_PET[idx_batch,:,:,:], 
                                                     input_size,
                                                     g_model, 
                                                     limits_act, 
                                                     MARGIN, 
                                                     sample_num=idx_batch+batch_carry, 
                                                     segm_net=True, 
                                                     s_model=s_model, 
                                                     criticize=True, 
                                                     d_model=d_model)
            else:
                (synth_CT_whole,
                 score_whole) = compute_whole_volume(in_NAC_PET[idx_batch,:,:,:], 
                                                     input_size,
                                                     g_model, 
                                                     limits_act, 
                                                     MARGIN, 
                                                     sample_num=idx_batch+batch_carry, 
                                                     segm_net=False, 
                                                     s_model=s_model, 
                                                     criticize=True, 
                                                     d_model=d_model)

            # Metrics
            X_in = in_CT[idx_batch,:,:,:].numpy()
            Y_in = synth_CT_whole
            # PSNR
            psnr_aux = measure.compare_psnr(X_in, Y_in)
            # ME
            me_aux = np.sum((X_in-Y_in)[X_in>0]/(X_in[X_in>0]))/(X_in[X_in>0].size)
            # NMSE
            nmse_aux = np.sqrt(np.sum((X_in-Y_in)**2))/(np.sqrt(np.sum(X_in**2)))
            # NCC
            X_in_zm = X_in-np.mean(X_in)
            Y_in_zm = Y_in-np.mean(Y_in)
            ncc_aux = (np.sum(X_in_zm*Y_in_zm)/X_in.size) / (np.std(X_in)*np.std(Y_in))


            # Save
            SCORE_images.append(score_whole)
            PSNR_images.append(psnr_aux)
            ME_images.append(me_aux)
            NMSE_images.append(nmse_aux)
            NCC_images.append(ncc_aux)
            
            normal_out_in_LABELS = np.expand_dims(in_LABELS[idx_batch,:,:,:,:].numpy(), axis=0)
            normal_out_in_CT = (np.expand_dims(np.expand_dims(in_CT[idx_batch,:,:,:].numpy(), axis=0),axis=-1)-0.5)*2.0
            normal_out_in_NAC_PET = (np.expand_dims(np.expand_dims(in_NAC_PET[idx_batch,:,:,:].numpy(), axis=0),axis=-1)-0.5)*2.0
            if segm_net:
                normal_out_synth_CT_whole = (np.expand_dims(np.expand_dims(synth_CT_whole, axis=0),axis=-1)-0.5)*2.0
                normal_out_sementation_whole = np.expand_dims(sementation_whole, axis=0)
            else:
                normal_out_synth_CT_whole = (np.expand_dims(np.expand_dims(synth_CT_whole, axis=0),axis=-1)-0.5)*2.0
                normal_out_sementation_whole = np.zeros(normal_out_in_LABELS.shape)
            
            if all_images_out:
                LABELS_INPUT_images.append(normal_out_in_LABELS)
                CT_INPUT_images.append(normal_out_in_CT)
                PET_INPUT_images.append(normal_out_in_NAC_PET)
                CT_SYNTH_images.append(normal_out_synth_CT_whole)
                SEGMENTED_images.append(normal_out_sementation_whole)
            else:
                PET_INPUT_images = normal_out_in_NAC_PET
                CT_INPUT_images = normal_out_in_CT
                CT_SYNTH_images = normal_out_synth_CT_whole
                SEGMENTED_images = normal_out_sementation_whole
                LABELS_INPUT_images = normal_out_in_LABELS

            if single_image:
                break
        
        batch_carry += batch_size
        
        if single_image:
            break

    
    return PET_INPUT_images, CT_INPUT_images, LABELS_INPUT_images, CT_SYNTH_images, SEGMENTED_images, SCORE_images, PSNR_images, ME_images, NMSE_images, NCC_images



def show_images(dataset_use, gen_model, show_labels_prob = True, calc_losses = True):
    """Show progress of the NN, plotting objective and generated CTs and labels
    """
    
    
    for a, b in dataset_use:

        c = gen_model(a, training=False)
        
        slice_plot_z = int(a.shape[-1]/2)
        
        y_true_texture, y_true_labels = tf.split(b, [1,4], axis=-1)
        y_pred_texture, y_pred_labels = tf.split(c, [1,4], axis=-1)

        plt.figure(dpi=100)
        plt.subplot(2,3,1)
        plt.imshow(a[0,:,:,slice_plot_z,0])
        plt.axis('off')
        plt.subplot(2,3,2)
        plt.imshow(y_true_texture[0,:,:,slice_plot_z,0])
        plt.axis('off')
        plt.subplot(2,3,3)
        plt.imshow(np.argmax(y_true_labels[0,:,:,slice_plot_z,:], axis=-1))
        plt.axis('off')
        plt.subplot(2,3,5)
        plt.imshow(y_pred_texture[0,:,:,slice_plot_z,0])
        plt.axis('off')
        plt.subplot(2,3,6)
        plt.imshow(np.argmax(y_pred_labels[0,:,:,slice_plot_z,:], axis=-1))
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        
        if show_labels_prob:
            plt.figure(dpi=100)
            plt.subplot(2,4,1)
            plt.imshow(y_pred_labels[0,:,:,slice_plot_z,0])
            plt.axis('off')
            plt.subplot(2,4,2)
            plt.imshow(y_pred_labels[0,:,:,slice_plot_z,1])
            plt.axis('off')
            plt.subplot(2,4,3)
            plt.imshow(y_pred_labels[0,:,:,slice_plot_z,2])
            plt.axis('off')
            plt.subplot(2,4,4)
            plt.imshow(y_pred_labels[0,:,:,slice_plot_z,3])
            plt.axis('off')

            plt.subplot(2,4,5)
            plt.imshow(y_true_labels[0,:,:,slice_plot_z,0])
            plt.axis('off')
            plt.subplot(2,4,6)
            plt.imshow(y_true_labels[0,:,:,slice_plot_z,1])
            plt.axis('off')
            plt.subplot(2,4,7)
            plt.imshow(y_true_labels[0,:,:,slice_plot_z,2])
            plt.axis('off')
            plt.subplot(2,4,8)
            plt.imshow(y_true_labels[0,:,:,slice_plot_z,3])
            plt.axis('off')

            plt.tight_layout()
            plt.show()
            
        if calc_losses:
            dice_loss = losses.DICE_loss(y_true_labels, y_pred_labels)
            dice_loss = 1.0-dice_loss
            L2_loss = tf.reduce_sum(tf.pow((y_true_texture - y_pred_texture),2),[1,2,3,4])
            total = dice_loss+(losses.k_coupling*L2_loss)
            print('Loss:\tDice: %0.4f\tL2: %0.4f\t\tTotal: %0.4f'%(dice_loss[0].numpy(), L2_loss.numpy()[0], total.numpy()[0]))

        break
        
  

  
def plot_loss(loss_plot, epoch, forma='rows', dpi_use = 75, titulos=list(), skip_n_initial=0):
    
    num_plots = loss_plot.shape[1]
    
    plt.figure(dpi = dpi_use)
    
    if forma == 'rows':
        plt_x = num_plots
        plt_y = 1
    if forma == 'cols':
        plt_x = 1
        plt_y = num_plots
    if forma == 'square':
        plt_x = np.ceil(num_plots/2)
        plt_y = np.floor(num_plots/2)
    if forma == '2_cols':
        plt_y = 2
        plt_x = np.ceil(num_plots/plt_y)
    if forma == '3_cols':
        plt_y = 3
        plt_x = np.ceil(num_plots/plt_y)
        
    
    for idx_plot in range(num_plots):
        plt.subplot(plt_x,plt_y,idx_plot+1)
        plt.plot(np.arange(0,epoch)[skip_n_initial:],loss_plot[skip_n_initial:epoch,idx_plot])    
        plt.grid('on')
        if len(titulos) == num_plots:
            plt.title(titulos[idx_plot])
    
    
    plt.tight_layout()
    plt.show()
    
    return
#-----------------------------------------------------------------------------
#---------------------- Checkpoint and Logging Functions ---------------------
#-----------------------------------------------------------------------------

def add_tensorboard_3Dimage(image_in_3D, name_img, step_in, channel, min_val=0.0, max_val=1.0):
    '''Creates 3 slices from a volume and adds them as images to tensorboard
    '''
    
    voxels_X = image_in_3D.shape[1]
    voxels_Y = image_in_3D.shape[2]
    voxels_Z = image_in_3D.shape[3]
    
    image_in_3D = (image_in_3D-min_val)/(max_val-min_val)
    
    image_in_axial = image_in_3D[0,:,:,int(voxels_Z/2),channel]
    image_in_axial = tf.reshape(image_in_axial, (1,voxels_X, voxels_Y, 1))

    image_in_sagital = image_in_3D[0,:,int(voxels_Y/2),:,channel]
    image_in_sagital = tf.reshape(image_in_sagital, (1,voxels_X, voxels_Z, 1))
    
    image_in_coronal = image_in_3D[0,int(voxels_X/2),:,:,channel]
    image_in_coronal = tf.reshape(image_in_coronal, (1,voxels_Y, voxels_Z, 1))
        
    with tf.name_scope(name_img):
        tf.summary.image(name_img+"_axial", image_in_axial, step=step_in)
        tf.summary.image(name_img+"_sagital", image_in_sagital, step=step_in)
        tf.summary.image(name_img+"_coronal", image_in_coronal, step=step_in)

    return



def save_progressive_model(generator_model_list, critic_model_list, full_res_shape, 
                           SAVE_NAME, CHECKPOINT_PATH_ROOT, epoch, save_limit=-1, use_keras=True):
    '''Saves a progressive model with resolution information
    '''
    
    CHECKPOINT_PATH_EPOCH = os.path.join(CHECKPOINT_PATH_ROOT,'epoch_%d'%epoch)
    File_mng.check_create_path('CHECKPOINT_PATH_EPOCH', CHECKPOINT_PATH_EPOCH, clear_folder=True) 
    
    
    if save_limit < 0:
        num_models = len(generator_model_list)
    else:
        num_models = save_limit+1
    
    for idx_model in range(num_models):
        
        generator_model_save = generator_model_list[idx_model][0]
        critic_model_save = critic_model_list[idx_model][0]


        save_ProGAN_models_3D(generator_model_save, 
                              critic_model_save, 
                              CHECKPOINT_PATH_EPOCH, 
                              full_res_shape, 
                              low_sample_pow=len(generator_model_list)-idx_model, 
                              fade_in=False,
                              use_keras=use_keras)



        generator_model_save = generator_model_list[idx_model][1]
        critic_model_save = critic_model_list[idx_model][1]


        save_ProGAN_models_3D(generator_model_save, 
                              critic_model_save, 
                              CHECKPOINT_PATH_EPOCH, 
                              full_res_shape, 
                              low_sample_pow=len(generator_model_list)-idx_model, 
                              fade_in=True,
                              use_keras=use_keras)

        
        
        print(">Saved resolution model %d to disk"%idx_model)
        
        

def save_ProGAN_models_3D(g_model, d_model, save_path, full_res_shape, low_sample_pow=0, 
                       fade_in=False, use_keras=True):
    # Fade in append
    append_fade = ''
    if fade_in:
        append_fade = '_fadeIn'
        
    # Calculate shape
    shape_X = int((full_res_shape[0]/(2**(low_sample_pow-1))))
    shape_Y = int((full_res_shape[1]/(2**(low_sample_pow-1))))
    shape_Z = int((full_res_shape[2]/(2**(low_sample_pow-1))))
    
    # Build folder name
    save_folder = os.path.join(save_path, 'shape_%dx%dx%d'%(shape_X, shape_Y, shape_Z))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    # Save Generator
    filename_generator = 'model_generator'+append_fade
    save_model(g_model, save_folder, filename_generator, use_keras=use_keras)
    # Save Critic
    filename_discriminator = 'model_discriminator'+append_fade
    save_model(d_model, save_folder, filename_discriminator, use_keras=use_keras)
        
    
    
    print('>Saved:\n\t%s\n\t%s\n\t\tin %s' % (filename_generator, 
                                                filename_discriminator, 
                                                save_folder))
        
    
    return

def save_multiple_models(model_list, model_names, save_path, base_name, name_prefix = '', use_keras=True):
    """Save a list of models to a folder
    """
    
    # Build folder name
    save_folder = os.path.join(save_path, base_name+name_prefix)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # save the models
    for idx_model in range(len(model_list)):
        
        save_model(model_list[idx_model], 
                   os.path.join(save_folder,model_names[idx_model]), 
                   model_names[idx_model], 
                   use_keras=use_keras)

    
        print('>Saved:\n\t%s\tin %s' % (model_names[idx_model],save_folder))
        
        
    return
    
def load_multiple_models(model_list, model_names, load_path, use_keras=True, custom_obj_dict=[]):
    """Load a list of models from a folder
    """
    
    # load the models
    for idx_model in range(len(model_list)):
        
        model_list[idx_model] = load_model(load_path, 
                                           model_names[idx_model], 
                                           use_keras=use_keras, 
                                           custom_obj_dict=custom_obj_dict)
    
        print('>Loaded:\n\t%s\tfrom %s' % (filename_load,load_path))
        
        
    return



def save_model(model_save, save_path, save_name, use_keras=True):
    """Save model using Keras API (single .h5 file) OR a HDF5 for weights and JSON for model.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if use_keras:
        model_save.save(os.path.join(save_path,save_name+".h5"))

    else:
        # serialize model to JSON
        model_json = model_save.to_json()
        with open(os.path.join(save_path,save_name+".json"), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model_save.save_weights(os.path.join(save_path,save_name+".h5"))
    
    return

def load_model(load_path, load_name, use_keras=True, custom_obj_dict=[]):
    """Load model using Keras API (single .h5 file) OR from a HDF5 file for weights and a JSON file for model  
    """

    if use_keras:
        loaded_model = keras.models.load_model(os.path.join(load_path,load_name+".h5"), custom_objects=custom_obj_dict)

    else:
        MODEL_NAME = load_name+'.json'
        WEIGHTS_NAME = load_name+'.h5'
        # load json and create model
        json_file = open(os.path.join(load_path,load_name), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(os.path.join(load_path,load_name))
    
    return loaded_model
 
    