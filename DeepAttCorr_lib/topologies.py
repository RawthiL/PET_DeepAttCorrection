#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

# External
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import numpy as np

from dataclasses import dataclass

# Internal
from DeepAttCorr_lib.GAN_3D_lib import layers

# Defaults
CONV_CHANNELS_GEN = 32
CONV_CHANNELS_DISC = 32
CONV_LAYERS_PER_BLOCK_GEN = 2
CONV_LAYERS_PER_BLOCK_DISC = 2
CONV_LAYERS_KERNEL_SIZE = 3


#-----------------------------------------------------------------------------
#-------------------- Parameter Structures -----------------------------------
#-----------------------------------------------------------------------------

        
@dataclass
class Gen_param_structure:

    # Each block
    block_conv_channels: int = CONV_CHANNELS_GEN
    block_conv_layers: int = CONV_LAYERS_PER_BLOCK_GEN
    block_kernel_size: int = CONV_LAYERS_KERNEL_SIZE
    
    conv_out_channels: int = 1
    out_kernel_size: int = 1
      
    # Input size of the condition or latent vector
    latent_dim: int = 128

    # Pixel normalization
    use_PixelNorm: bool = True
    
    # Kernel initializer parameters
    initializer_std: float = 1.0
    use_He_scale: bool = True
    lrmul_block: float = 1.0
    gain_block: float = np.sqrt(2)
        
    # Weight constraints
    use_norm_contrain_scale: bool = False
    block_max_norm_contrain: float = 1.0
        
    rand_in_dim: int = (0,0,0)
    latent_conv_channels: int = CONV_CHANNELS_GEN
    latent_conv_layers: int = CONV_LAYERS_PER_BLOCK_GEN
    latent_kernel_size: int = 3   
        
    # Batch normalization
    use_BatchNorm: bool = False
        
    input_dense_neurons_mult: int = 16
    input_conv_channels: int = CONV_CHANNELS_GEN
    input_conv_layers: int = CONV_LAYERS_PER_BLOCK_GEN
    input_kernel_size: int = 3
    lrmul_ini: float = 1.0
    gain_ini: float = np.sqrt(2)
    
    n_blocks: int = 0
        
    use_skip_connections: bool = True
        
    # Segmentation
    use_tanh_out: bool = False
    use_sigmoid_out: bool = False
    segmentation_output: bool = False
    segmentation_classes: int = 0
    segmentation_layers: int = 0
    conv_segmentation_channels: int = 0
    segmentation_kernel_size: int = 0
        




#-----------------------------------------------------------------------------
#-------------------- 3D V-Net Generator Network -----------------------------
#-----------------------------------------------------------------------------   

# define generator models
def define_3D_Vnet_generator(param_struct):
    # weight initialization
    init = keras.initializers.RandomNormal(stddev=param_struct.initializer_std)
    # weight constraint
    if param_struct.use_norm_contrain_scale:
        const = keras.constraints.max_norm(param_struct.block_max_norm_contrain)
    else:
        const = None 
        
    model_list = list()
    
    if len(param_struct.latent_dim) != 4:
        print('Input must be a 3D image of shape [X,Y,Z,Chn].')
        return
    
    # Input
    Input_latent = keras.Input(shape=param_struct.latent_dim, name="Input_blk")
    t_latent = Input_latent
    
    # ------------------------- Initial Block -------------------------------------------
    input_dim = t_latent.shape[-1]
    g = keras.layers.Conv3D(param_struct.block_conv_channels[0], 
                               param_struct.input_kernel_size, 
                               padding='same', 
                               use_bias=False,
                               kernel_initializer=init, 
                               kernel_constraint=const,
                              name='Conv3D_ini_gen')(t_latent)
    if param_struct.use_He_scale:
        g = layers.HeScale([param_struct.input_kernel_size,
                         param_struct.input_kernel_size,
                         param_struct.input_kernel_size], 
                        input_dim,
                    lrmul=param_struct.lrmul_ini,
                    gain=param_struct.gain_ini,
                   name='HeScale_ini_gen')(g) # Add He scaling
    g = layers.BiasLayer(name='Bias_ini_gen')(g) # Add bias 
    g = keras.layers.LeakyReLU(alpha=0.2,
                                 name='LeakyReLu_ini_gen')(g)
    if param_struct.use_PixelNorm:
        g = layers.PixelNormalization(name='Pixel_norm_ini_ini_gen')(g)
    elif param_struct.use_BatchNorm:
        g = layers.BatchNormalization(name='Batch_norm_ini_ini_gen')(g)
    
    naming_layer = keras.layers.Lambda(lambda x: x, name='Initial_end_gen')
    g = naming_layer (g)
    
    # ------------------------- Down Blocks ------------------------------------------
     # Apply old v layers and out
    skip_connections = list()
    for idx_block in range(param_struct.n_blocks-1,-1,-1):

        # down block
        for idx_blk_layer in range(param_struct.n_blocks):
            input_dim = g.shape[-1]
            g = keras.layers.Conv3D(param_struct.block_conv_channels[param_struct.n_blocks-idx_block-1], 
                                       param_struct.block_kernel_size, 
                                       padding='same', 
                                       use_bias=False,
                                       kernel_initializer=init, 
                                       kernel_constraint=const,
                                      name='Conv3D_dwn_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))(g)
            if param_struct.use_He_scale:
                g = layers.HeScale([param_struct.block_kernel_size,
                             param_struct.block_kernel_size,
                             param_struct.block_kernel_size], 
                            input_dim,
                        lrmul=param_struct.lrmul_block,
                        gain=param_struct.gain_block,
                           name='HeScale_dwn_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))(g) # Add He scaling
            g = layers.BiasLayer(name='Bias_dwn_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))(g) # Add bias 
            g = keras.layers.LeakyReLU(alpha=0.2,
                                         name='LeakyReLu_dwn_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))(g)
            if param_struct.use_PixelNorm:
                g = layers.PixelNormalization(name='Pixel_norm_dwn_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))(g)
            elif param_struct.use_BatchNorm:
                g = layers.BatchNormalization(name='Batch_norm_dwn_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))(g)

        # Detail connection
        naming_layer = keras.layers.Lambda(lambda x: x, name='detail_tensor_%d_gen'%(idx_block))
        d_g = naming_layer (g)
        skip_connections.append(d_g)

        # Downsampling
        g = keras.layers.AveragePooling3D(name='DownSample_blk-%d_gen'%(idx_block))(g)    
    
    
    # ------------------------- Base Block ----------------------------------------------
    
    for idx_blk_layer in range(param_struct.block_conv_layers):
        input_dim = g.shape[-1]
        g = keras.layers.Conv3D(param_struct.block_conv_channels[param_struct.n_blocks-1], 
                                   param_struct.block_kernel_size, 
                                   padding='same', 
                                   use_bias=False,
                                   kernel_initializer=init, 
                                   kernel_constraint=const,
                                  name='Conv3D_base_l-%d_gen'%(idx_blk_layer))(g)
        if param_struct.use_He_scale:
            g = layers.HeScale([param_struct.block_kernel_size,
                         param_struct.block_kernel_size,
                         param_struct.block_kernel_size], 
                        input_dim,
                    lrmul=param_struct.lrmul_block,
                    gain=param_struct.gain_block,
                       name='HeScale_base_l-%d_gen'%(idx_blk_layer))(g) # Add He scaling
        g = layers.BiasLayer(name='Bias_base_l-%d_gen'%(idx_blk_layer))(g) # Add bias 
        g = keras.layers.LeakyReLU(alpha=0.2,
                                     name='LeakyReLu_base_l-%d_gen'%(idx_blk_layer))(g)
        if param_struct.use_PixelNorm:
            g = layers.PixelNormalization(name='Pixel_norm_base_l-%d_gen'%(idx_blk_layer))(g)
        elif param_struct.use_BatchNorm:
            g = layers.BatchNormalization(name='Batch_norm_base_l-%d_gen'%(idx_blk_layer))(g)
        
    # ------------------------- Add Up Block --------------------------------------------
    for idx_block in range(param_struct.n_blocks):
    
        g = keras.layers.UpSampling3D(name='UpSample_blk-%d_gen'%(idx_block))(g)

        if param_struct.use_skip_connections:
            # Cancatenate input with last level
            g = keras.layers.concatenate([g,skip_connections[param_struct.n_blocks-idx_block-1]], 
                                         axis=-1, 
                                         name='DetailAdd_blk-%d_gen'%(idx_block))

        for idx_blk_layer in range(param_struct.block_conv_layers):
            input_dim = g.shape[-1]
            g = keras.layers.Conv3D(param_struct.block_conv_channels[param_struct.n_blocks-idx_block-1], 
                                       param_struct.block_kernel_size, 
                                       padding='same', 
                                       use_bias=False,
                                       kernel_initializer=init, 
                                       kernel_constraint=const,
                                      name='Conv3D_up_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))(g)
            if param_struct.use_He_scale:
                g = layers.HeScale([param_struct.block_kernel_size,
                             param_struct.block_kernel_size,
                             param_struct.block_kernel_size], 
                            input_dim,
                        lrmul=param_struct.lrmul_block,
                        gain=param_struct.gain_block,
                           name='HeScale_up_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))(g) # Add He scaling
            g = layers.BiasLayer(name='Bias_up_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))(g) # Add bias 
            g = keras.layers.LeakyReLU(alpha=0.2,
                                         name='LeakyReLu_up_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))(g)
            if param_struct.use_PixelNorm:
                g = layers.PixelNormalization(name='Pixel_norm_up_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))(g)
            elif param_struct.use_BatchNorm:
                g = layers.BatchNormalization(name='Batch_norm_up_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))(g)
            

    
       
    # ------------------------- Add Output Block ----------------------------------------
     
    # conv 1x1, output block
    input_dim = g.shape[-1]
    out_image = keras.layers.Conv3D(param_struct.conv_out_channels, 
                                       param_struct.out_kernel_size, 
                                       padding='same', 
                                       kernel_initializer=init, 
                                       kernel_constraint=const,
                                       use_bias=False,
                                      name='Conv3D_out_gen')(g)
    if param_struct.use_He_scale:
        out_image = layers.HeScale([param_struct.out_kernel_size,
                         param_struct.out_kernel_size,
                         1], 
                       input_dim, 
                            name='He_Scale_out_gen')(out_image) # Add He scaling
    out_image = layers.BiasLayer(name='Bias_out_gen')(out_image) # Add bias    
    if param_struct.use_tanh_out:
        out_image = K.tanh(out_image)
    elif param_struct.use_sigmoid_out:
        out_image = K.sigmoid(out_image)
        
        
        
    if param_struct.segmentation_output:
        
        segm_out = g
        
        for idx_segm_layer in range(param_struct.segmentation_layers):
            segm_out = keras.layers.Conv3D(param_struct.conv_segmentation_channels, 
                                       param_struct.segmentation_kernel_size, 
                                       padding='same', 
                                       kernel_initializer=init, 
                                       kernel_constraint=const,
                                       use_bias=True,
                                       activation=K.relu,
                                       name='Conv3D_segm_l-%d_gen'%idx_segm_layer)(segm_out)
            if param_struct.use_BatchNorm:
                segm_out = layers.BatchNormalization(name='Batch_segm_l-%d_gen'%idx_segm_layer)(segm_out)
            
        segm_out = keras.layers.Conv3D(param_struct.segmentation_classes, 
                                       param_struct.segmentation_kernel_size, 
                                       padding='same', 
                                       kernel_initializer=init, 
                                       kernel_constraint=const,
                                       use_bias=True,
                                       activation=K.relu,
                                       name='Conv3D_segm_output_gen')(segm_out)
        
        segm_out = K.softmax(segm_out, axis = -1)
    
    
        # define model
        model = keras.Model(Input_latent, K.concatenate([out_image, segm_out],axis = -1), name="Gen_V-Net")
    else:
        # define model
        model = keras.Model(Input_latent, out_image, name="Gen_V-Net")
    
    return model