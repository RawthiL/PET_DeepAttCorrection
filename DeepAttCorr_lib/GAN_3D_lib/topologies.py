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
        


                
@dataclass
class Disc_param_structure:
    conditional: bool = False
    
    input_shape: int = (0,0,1)

    use_minibatch_stdev: bool = True
        
    use_norm_contrain_scale: bool = False
    block_max_norm_contrain: float = 1.0
        
    block_conv_channels: int = CONV_CHANNELS_DISC
    block_conv_layers: int = CONV_LAYERS_PER_BLOCK_DISC
    block_kernel_size: int = 3
        
    downSampling_layers: int = 0

    initializer_std: float = 0.02
    use_He_scale: bool = False
    lrmul_block: float = 1.0
    gain_block: float = np.sqrt(2)
    lrmul_dense: float = 1.0
    gain_dense: float = np.sqrt(2)
    lrmul_ini: float = 1.0
    gain_ini: float = np.sqrt(2)
        
    input_kernel_size: int = 1
    
    n_blocks: int = 0



@dataclass
class ProGAN_param_structure:

    block_conv_channels: int = CONV_CHANNELS_GEN
    block_conv_layers: int = CONV_LAYERS_PER_BLOCK_GEN
    block_kernel_size: int = 3
    
    conv_out_channels: int = 1
    out_kernel_size: int = 1
               
    growing_style: bool = False
    latent_dim: int = 128
    fixed_latent_dim: int = 128
    
    use_PixelNorm: bool = True
    
    initializer_std: float = 1.0
    use_He_scale: bool = True
    lrmul_block: float = 1.0
    gain_block: float = np.sqrt(2)
        
    use_norm_contrain_scale: bool = False
    block_max_norm_contrain: float = 1.0
        
    rand_in_dim: int = (0,0,0)
    latent_conv_channels: int = CONV_CHANNELS_GEN
    latent_conv_layers: int = CONV_LAYERS_PER_BLOCK_GEN
    latent_kernel_size: int = 3   
        
        
    use_BatchNorm: bool = False
        
    input_dense_neurons_mult: int = 16
    input_conv_channels: int = CONV_CHANNELS_GEN
    input_conv_layers: int = CONV_LAYERS_PER_BLOCK_GEN
    input_kernel_size: int = 3
    lrmul_ini: float = 1.0
    gain_ini: float = np.sqrt(2)
    
    n_blocks: int = 0
        
    use_skip_connections: bool = True
        
#-----------------------------------------------------------------------------
#-------------------- 3D V-Net Generator Network -----------------------------
#-----------------------------------------------------------------------------   

# define generator models
def define_3D_Vnet_generator(param_struct):
    """Creates a 3D V-Net (or U-Net) neural network
    
    This function receives as input a structure containing all required data 
    (reffer to structure above)
    
    It returns a 3D U-Net Keras model of arbitrary size.
    """
    
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



# define generator models
def define_3D_Convolutional_generator(param_struct, name_appendix=''):
    """Creates a 3D convolutional neural network
    
    This function receives as input a structure containing all required data 
    (reffer to structure above)
    
    It returns a convolutional model (no skip connections) Keras model of arbitrary size.
    """
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
    
    # ------------------------- Convolutional Blocks ------------------------------------------
    for idx_blk_layer in range(param_struct.n_blocks):
        input_dim = g.shape[-1]
        g = keras.layers.Conv3D(param_struct.block_conv_channels[idx_blk_layer], 
                                   param_struct.block_kernel_size, 
                                   padding='same', 
                                   use_bias=False,
                                   kernel_initializer=init, 
                                   kernel_constraint=const,
                                  name='Conv3D_l-%d_gen'%(idx_blk_layer))(g)
        if param_struct.use_He_scale:
            g = layers.HeScale([param_struct.block_kernel_size,
                         param_struct.block_kernel_size,
                         param_struct.block_kernel_size], 
                        input_dim,
                    lrmul=param_struct.lrmul_block,
                    gain=param_struct.gain_block,
                       name='HeScale_l-%d_gen'%(idx_blk_layer))(g) # Add He scaling
        g = layers.BiasLayer(name='Bias_l-%d_gen'%(idx_blk_layer))(g) # Add bias 
        g = keras.layers.LeakyReLU(alpha=0.2,
                                     name='LeakyReLu_l-%d_gen'%(idx_blk_layer))(g)
        if param_struct.use_PixelNorm:
            g = layers.PixelNormalization(name='Pixel_norm_l-%d_gen'%(idx_blk_layer))(g)
        elif param_struct.use_BatchNorm:
            g = layers.BatchNormalization(name='Batch_norm_l-%d_gen'%(idx_blk_layer))(g)

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
        model = keras.Model(Input_latent, K.concatenate([out_image, segm_out],axis = -1), name="Gen_V-Net"+name_appendix)
    else:
        # define model
        model = keras.Model(Input_latent, out_image, name="Gen_V-Net"+name_appendix)
    
    return model





#-----------------------------------------------------------------------------
#-------------------- 3D Progressive V-Net Generator Network -----------------
#-----------------------------------------------------------------------------   

def add_3D_prog_Vnet_generator_block(old_model, param_struct, idx_block):
    '''Adds a convolutional block for a progressive growing 3D U-Net
    
        The function receives a model and adds a greater resolution 
        convolutional block on top of it. 
        It returns the new NN, with greater resolution, and the fade-in model 
        for training.
    '''
    
    # weight initialization
    init = keras.initializers.RandomNormal(stddev=param_struct.initializer_std)
    # weight constraint
    if param_struct.use_norm_contrain_scale:
        const = keras.constraints.max_norm(param_struct.block_max_norm_contrain)
    else:
        const = None

    # get shape of existing model
    in_shape = list(old_model.input.shape)
    # define new input shape as double the size
    input_shape = (in_shape[-4]*2, in_shape[-3]*2, in_shape[-2]*2, in_shape[-1])
    Input_latent = keras.Input(shape=input_shape, name="Input_condition_blk-%d_gen"%idx_block)
    
    t_latent = Input_latent
    
    # -------------------------  New Initial Block -------------------------------------------
    input_dim = t_latent.shape[-1]
    intial_channels_idx = param_struct.n_blocks-idx_block-2
    intial_channels_idx = intial_channels_idx if intial_channels_idx > 0 else 0
    g = keras.layers.Conv3D(param_struct.block_conv_channels[intial_channels_idx], 
                               param_struct.input_kernel_size, 
                               padding='same', 
                               use_bias=False,
                               kernel_initializer=init, 
                               kernel_constraint=const,
                              name='Conv3D_blk-%d_ini_gen'%(idx_block))(t_latent)
    if param_struct.use_He_scale:
        g = layers.HeScale([param_struct.input_kernel_size,
                         param_struct.input_kernel_size,
                         param_struct.input_kernel_size], 
                        input_dim,
                    lrmul=param_struct.lrmul_ini,
                    gain=param_struct.gain_ini,
                   name='HeScale_blk-%d_ini_gen'%(idx_block))(g) # Add He scaling
    g = layers.BiasLayer(name='Bias_blk-%d_ini_gen'%(idx_block))(g) # Add bias 
    g_init = keras.layers.LeakyReLU(alpha=0.2,
                                 name='LeakyReLu_blk-%d_ini_gen'%(idx_block))(g)
    if param_struct.use_PixelNorm:
        g = layers.PixelNormalization(name='Pixel_norm_blk-%d_gen'%(idx_block))(g)
    
    naming_layer = keras.layers.Lambda(lambda x: x, name='Initial_end_%d_gen'%(idx_block))
    g = naming_layer (g_init)
    
    # ------------------------ New Down Block -------------------------------------------
    
    
    
    
    for idx_blk_layer in range(param_struct.block_conv_layers):
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
        
    naming_layer = keras.layers.Lambda(lambda x: x, name='detail_tensor_%d_gen'%(idx_block))
    d_g = naming_layer (g)
    
    g = keras.layers.AveragePooling3D(name='DownSample_blk-%d_gen'%(idx_block))(g)
    
    fade_g = g
        
    # ------------------------ Place old bocks ------------------------------------------
    
    # All layers excluding input layer and output
    skip_connections = list()
    for old_block_indx in range(idx_block-1,0-1,-1):
               
        # down block
        for idx_blk_layer in range(param_struct.input_conv_layers):
            old_dwn_conv = old_model.get_layer('Conv3D_dwn_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            if param_struct.use_He_scale:
                old_dwn_HeScale = old_model.get_layer('HeScale_dwn_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            old_dwn_bias = old_model.get_layer('Bias_dwn_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            old_dwn_leakyReLu = old_model.get_layer('LeakyReLu_dwn_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            if param_struct.use_PixelNorm:
                old_dwn_pixelNorm = old_model.get_layer('Pixel_norm_dwn_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            
            
            g = old_dwn_conv(g)
            if param_struct.use_He_scale:
                g = old_dwn_HeScale(g)
            g = old_dwn_bias(g)
            g = old_dwn_leakyReLu(g)
            if param_struct.use_PixelNorm:
                g = old_dwn_pixelNorm(g)
        
        skip_connections.append(g)
        
        # downsample
        old_downsample = old_model.get_layer('DownSample_blk-%d_gen'%(old_block_indx))
        g = old_downsample(g)
        
        

    # Base layer
    if old_block_indx == 0:
        for idx_blk_layer in range(param_struct.input_conv_layers):
            old_base_conv = old_model.get_layer('Conv3D_base_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            if param_struct.use_He_scale:
                old_base_HeScale = old_model.get_layer('HeScale_base_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            old_base_bias = old_model.get_layer('Bias_base_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            old_base_leakyReLu = old_model.get_layer('LeakyReLu_base_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            if param_struct.use_PixelNorm:
                old_base_pixelNorm = old_model.get_layer('Pixel_norm_base_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))

            g = old_base_conv(g)
            if param_struct.use_He_scale:
                g = old_base_HeScale(g)
            g = old_base_bias(g)
            g = old_base_leakyReLu(g)
            if param_struct.use_PixelNorm:
                g = old_base_pixelNorm(g)

    # up blocks
    for old_block_indx in range(idx_block):
        
        # Upsample
        old_upsample = old_model.get_layer('UpSample_blk-%d_gen'%(old_block_indx))
        g = old_upsample(g)
        
        if param_struct.use_skip_connections:
            # Cancatenate input with last level
            old_detail_Add = old_model.get_layer('DetailAdd_blk-%d_gen'%(old_block_indx))
            g = old_detail_Add([g,skip_connections[idx_block-1-old_block_indx]])
            
        for idx_blk_layer in range(param_struct.input_conv_layers):
            old_up_conv = old_model.get_layer('Conv3D_up_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            if param_struct.use_He_scale:
                old_up_HeScale = old_model.get_layer('HeScale_up_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            old_up_bias = old_model.get_layer('Bias_up_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            old_up_leakyReLu = old_model.get_layer('LeakyReLu_up_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            if param_struct.use_PixelNorm:
                old_up_pixelNorm = old_model.get_layer('Pixel_norm_up_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            
            g = old_up_conv(g)
            if param_struct.use_He_scale:
                g = old_up_HeScale(g)
            g = old_up_bias(g)
            g = old_up_leakyReLu(g)
            if param_struct.use_PixelNorm:
                g = old_up_pixelNorm(g)
            
        

        
    # ------------------------ New Up Block ------------------------------------------------
    
    g = keras.layers.UpSampling3D(name='UpSample_blk-%d_gen'%(idx_block))(g)
    
    if param_struct.use_skip_connections:
        # Cancatenate input with last level
        g = keras.layers.concatenate([g,d_g], axis=-1, name='DetailAdd_blk-%d_gen'%(idx_block))
    
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
    
    naming_layer = keras.layers.Lambda(lambda x: x, name='Blocks_end_%d_gen'%(idx_block))
    g = naming_layer (g)
    
    block_end_up = g
            
    # ------------------------ New Output Layer -----------------------------------------
    input_dim = g.shape[-1]
    out_image = keras.layers.Conv3D(param_struct.conv_out_channels, 
                                       param_struct.out_kernel_size, 
                                       padding='same', 
                                       kernel_initializer=init, 
                                       kernel_constraint=const,
                                       use_bias=False,
                                      name='Conv3D_blk-%d_out_gen'%(idx_block))(g)
    if param_struct.use_He_scale:
        out_image = layers.HeScale([param_struct.out_kernel_size,
                         param_struct.out_kernel_size,
                         1], 
                       input_dim, 
                            name='He_Scale_blk-%d_out_gen'%(idx_block))(out_image) # Add He scaling
    out_image = layers.BiasLayer(name='Bias_blk-%d_out_gen'%(idx_block))(out_image) # Add bias    
            
    # ------------------------ Define New Model -----------------------------------------
    model1 = keras.Model(Input_latent, out_image, name="Gen_blk-%d"%(idx_block))
    
    
    
    # ------------------------ Define Fade-In Model -------------------------------------
    
    # Fade input
    g = tf.keras.layers.AveragePooling3D(name='DownSample_fadeIn_blk-%d_gen'%(idx_block))(Input_latent)
    old_layer_names = [layer.name for layer in old_model.layers]
    old_input_layers_num = old_layer_names.index('Initial_end_%d_gen'%(idx_block-1))
    for i in range(1, old_input_layers_num):
        g = old_model.layers[i](g)
    g = layers.WeightedSum(name='WSum_fadeInput_blk-%d_out_gen'%(idx_block))([g, fade_g])
    
    # Apply old v layers and out
    skip_connections = list()
    for old_block_indx in range(idx_block-1,0-1,-1):
               
        # down block
        for idx_blk_layer in range(param_struct.input_conv_layers):
            old_dwn_conv = old_model.get_layer('Conv3D_dwn_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            if param_struct.use_He_scale:
                old_dwn_HeScale = old_model.get_layer('HeScale_dwn_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            old_dwn_bias = old_model.get_layer('Bias_dwn_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            old_dwn_leakyReLu = old_model.get_layer('LeakyReLu_dwn_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            if param_struct.use_PixelNorm:
                old_dwn_pixelNorm = old_model.get_layer('Pixel_norm_dwn_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            
            
            g = old_dwn_conv(g)
            if param_struct.use_He_scale:
                g = old_dwn_HeScale(g)
            g = old_dwn_bias(g)
            g = old_dwn_leakyReLu(g)
            if param_struct.use_PixelNorm:
                g = old_dwn_pixelNorm(g)
        
        skip_connections.append(g)
        
        # downsample
        old_downsample = old_model.get_layer('DownSample_blk-%d_gen'%(old_block_indx))
        g = old_downsample(g)
        
        

    # Base layer
    if old_block_indx == 0:
        for idx_blk_layer in range(param_struct.input_conv_layers):
            old_base_conv = old_model.get_layer('Conv3D_base_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            if param_struct.use_He_scale:
                old_base_HeScale = old_model.get_layer('HeScale_base_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            old_base_bias = old_model.get_layer('Bias_base_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            old_base_leakyReLu = old_model.get_layer('LeakyReLu_base_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            if param_struct.use_PixelNorm:
                old_base_pixelNorm = old_model.get_layer('Pixel_norm_base_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))

            g = old_base_conv(g)
            if param_struct.use_He_scale:
                g = old_base_HeScale(g)
            g = old_base_bias(g)
            g = old_base_leakyReLu(g)
            if param_struct.use_PixelNorm:
                g = old_base_pixelNorm(g)

    # up blocks
    for old_block_indx in range(idx_block):
        
        old_upsample = old_model.get_layer('UpSample_blk-%d_gen'%(old_block_indx))
        g = old_upsample(g)
        
        if param_struct.use_skip_connections:
            # Cancatenate input with last level
            old_detail_Add = old_model.get_layer('DetailAdd_blk-%d_gen'%(old_block_indx))
            g = old_detail_Add([g,skip_connections[idx_block-1-old_block_indx]])
            
        for idx_blk_layer in range(param_struct.input_conv_layers):
            old_up_conv = old_model.get_layer('Conv3D_up_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            if param_struct.use_He_scale:
                old_up_HeScale = old_model.get_layer('HeScale_up_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            old_up_bias = old_model.get_layer('Bias_up_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            old_up_leakyReLu = old_model.get_layer('LeakyReLu_up_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            if param_struct.use_PixelNorm:
                old_up_pixelNorm = old_model.get_layer('Pixel_norm_up_blk-%d_l-%d_gen'%(old_block_indx,idx_blk_layer))
            
            g = old_up_conv(g)
            if param_struct.use_He_scale:
                g = old_up_HeScale(g)
            g = old_up_bias(g)
            g = old_up_leakyReLu(g)
            if param_struct.use_PixelNorm:
                g = old_up_pixelNorm(g)
            
    old_last_blk_out = g

    # Old out level
    old_out_conv = old_model.get_layer('Conv3D_blk-%d_out_gen'%(idx_block-1))
    if param_struct.use_He_scale:
        old_out_HeScale = old_model.get_layer('He_Scale_blk-%d_out_gen'%(idx_block-1))
    old_out_bias = old_model.get_layer('Bias_blk-%d_out_gen'%(idx_block-1))
    
    g = old_out_conv(g)
    if param_struct.use_He_scale:
        g = old_out_HeScale(g)
    g = old_out_bias(g)
    upsampled_out = keras.layers.UpSampling3D(name='UpSample_fadeIn_blk-%d_gen'%(idx_block))(g)
    
    # New Up Block
    g = old_last_blk_out
    new_upsample = model1.get_layer('UpSample_blk-%d_gen'%(idx_block))
    g = new_upsample(g)

    if param_struct.use_skip_connections:
        # Cancatenate input with last level
        new_detail_Add = model1.get_layer('DetailAdd_blk-%d_gen'%(idx_block))
        g = new_detail_Add([g,d_g])

    for idx_blk_layer in range(param_struct.input_conv_layers):
        new_up_conv = model1.get_layer('Conv3D_up_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))
        if param_struct.use_He_scale:
            new_up_HeScale = model1.get_layer('HeScale_up_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))
        new_up_bias = model1.get_layer('Bias_up_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))
        new_up_leakyReLu = model1.get_layer('LeakyReLu_up_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))
        if param_struct.use_PixelNorm:
            new_up_pixelNorm = model1.get_layer('Pixel_norm_up_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))
            

        g = new_up_conv(g)
        if param_struct.use_He_scale:
            g = new_up_HeScale(g)
        g = new_up_bias(g)
        g = new_up_leakyReLu(g)
        if param_struct.use_PixelNorm:
            g = new_up_pixelNorm(g)
    
    
    # New out level
    # look for the end of last block of current model
    current_layer_names = [layer.name for layer in model1.layers]
    current_out_blk_idx = current_layer_names.index('Blocks_end_%d_gen'%(idx_block))
    for i in range(current_out_blk_idx+1, len(model1.layers)):
        g = model1.layers[i](g)
    out_image_new_fade = g
    
    # Fade output
    merged_out = layers.WeightedSum(name='WSum_fadeOutput_blk-%d_out_gen'%(idx_block))([upsampled_out, out_image_new_fade])
    
    model2 = keras.Model(Input_latent, merged_out, name="Gen_blk-%d_fadeIn"%(idx_block))
    
        
    return [model1, model2]



# define generator models
def define_3D_prog_Vnet_generator(param_struct):
    """Creates a 3D U-Net progressive growing convolutional neural network
    
    This function receives as input a structure containing all required data 
    (reffer to structure above)
    
    It returns a list of Keras models, composing a 3D U-Net of arbitrary size 
    at different resolution scales.
    """
    
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
    
    
    idx_block = 0
    
    # Input
    Input_latent = keras.Input(shape=param_struct.latent_dim, name="Input_condition_blk-%d_gen"%idx_block)
    t_latent = Input_latent
    
    # ------------------------- Initial Block -------------------------------------------
    input_dim = t_latent.shape[-1]
    g = keras.layers.Conv3D(param_struct.block_conv_channels[param_struct.n_blocks-idx_block-2], 
                               param_struct.input_kernel_size, 
                               padding='same', 
                               use_bias=False,
                               kernel_initializer=init, 
                               kernel_constraint=const,
                              name='Conv3D_blk-%d_ini_gen'%(idx_block))(t_latent)
    if param_struct.use_He_scale:
        g = layers.HeScale([param_struct.input_kernel_size,
                         param_struct.input_kernel_size,
                         param_struct.input_kernel_size], 
                        input_dim,
                    lrmul=param_struct.lrmul_ini,
                    gain=param_struct.gain_ini,
                   name='HeScale_blk-%d_ini_gen'%(idx_block))(g) # Add He scaling
    g = layers.BiasLayer(name='Bias_blk-%d_ini_gen'%(idx_block))(g) # Add bias 
    g = keras.layers.LeakyReLU(alpha=0.2,
                                 name='LeakyReLu_blk-%d_ini_gen'%(idx_block))(g)
    if param_struct.use_PixelNorm:
        g = layers.PixelNormalization(name='Pixel_norm_ini_blk-%d_ini_gen'%(idx_block))(g)
    
    naming_layer = keras.layers.Lambda(lambda x: x, name='Initial_end_%d_gen'%(idx_block))
    g = naming_layer (g)
    
    # ------------------------- Add Down Block ------------------------------------------

    for idx_blk_layer in range(param_struct.block_conv_layers):
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
        
    naming_layer = keras.layers.Lambda(lambda x: x, name='detail_tensor_%d_gen'%(idx_block))
    d_g = naming_layer (g)
    
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
                                  name='Conv3D_base_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))(g)
        if param_struct.use_He_scale:
            g = layers.HeScale([param_struct.block_kernel_size,
                         param_struct.block_kernel_size,
                         param_struct.block_kernel_size], 
                        input_dim,
                    lrmul=param_struct.lrmul_block,
                    gain=param_struct.gain_block,
                       name='HeScale_base_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))(g) # Add He scaling
        g = layers.BiasLayer(name='Bias_base_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))(g) # Add bias 
        g = keras.layers.LeakyReLU(alpha=0.2,
                                     name='LeakyReLu_base_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))(g)
        if param_struct.use_PixelNorm:
            g = layers.PixelNormalization(name='Pixel_norm_base_blk-%d_l-%d_gen'%(idx_block,idx_blk_layer))(g)
        
    
    # ------------------------- Add Up Block --------------------------------------------
    
    g = keras.layers.UpSampling3D(name='UpSample_blk-%d_gen'%(idx_block))(g)
    
    if param_struct.use_skip_connections:
        # Cancatenate input with last level
        g = keras.layers.concatenate([g,d_g], axis=-1, name='DetailAdd_blk-%d_gen'%(idx_block))
    
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
    
    
    
    naming_layer = keras.layers.Lambda(lambda x: x, name='Blocks_end_%d_gen'%(idx_block))
    g = naming_layer (g)
    
    # ------------------------- Add Output Block ----------------------------------------
     
    # conv 1x1, output block
    input_dim = g.shape[-1]
    out_image = keras.layers.Conv3D(param_struct.conv_out_channels, 
                                       param_struct.out_kernel_size, 
                                       padding='same', 
                                       kernel_initializer=init, 
                                       kernel_constraint=const,
                                       use_bias=False,
                                      name='Conv3D_blk-%d_out_gen'%(0))(g)
    if param_struct.use_He_scale:
        out_image = layers.HeScale([param_struct.out_kernel_size,
                         param_struct.out_kernel_size,
                         1], 
                       input_dim, 
                            name='He_Scale_blk-%d_out_gen'%(0))(out_image) # Add He scaling
    out_image = layers.BiasLayer(name='Bias_blk-%d_out_gen'%(0))(out_image) # Add bias       
        
    # define model
    model = keras.Model(Input_latent, out_image, name="Gen_blk_%d"%(0))
    # store model
    model_list.append([model, model])
    # create submodels
    for i in range(1, param_struct.n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_3D_prog_Vnet_generator_block(old_model, param_struct, i)
        # store model
        model_list.append(models)
    return model_list



#-----------------------------------------------------------------------------
#-------------------- 3D Discriminator Network -------------------------------
#-----------------------------------------------------------------------------    


def add_3D_discriminator_block(old_model, param_struct, idx_block):
    """Adds a convolutional block to a discriminator network
    
        Parameters:
        old_model --- model to be enlarged by one convolutional block
        param_struct --- model parameters
        idx_block --- position of the block (resution index)
        
        Returns:
        model1 --- updated model
        model2 --- updated model with fade-In structure (for progressive growing GANs)
    """
    # Number of end convolutional blocks
    end_conv_blocks = param_struct.n_blocks - param_struct.downSampling_layers
    
    # weight initialization
    init = keras.initializers.RandomNormal(stddev=param_struct.initializer_std)
    # weight constraint
    if param_struct.use_norm_contrain_scale:
        const = keras.constraints.max_norm(param_struct.block_max_norm_contrain)
    else:
        const = None
    
    intial_channels_idx = param_struct.n_blocks-end_conv_blocks-idx_block-2
    intial_channels_idx = intial_channels_idx if intial_channels_idx > 0 else 0
    
    if param_struct.conditional:
        in_unk_shape = old_model.input[0].shape
        in_ref_shape = old_model.input[1].shape
        # define new input shape as double the size
        in_unk_shape_new = (in_unk_shape[-4]*2, in_unk_shape[-3]*2, in_unk_shape[-2]*2, in_unk_shape[-1])
        in_ref_shape_new = (in_unk_shape[-4]*2, in_ref_shape[-3]*2, in_ref_shape[-2]*2, in_ref_shape[-1])
        # Create new inputs
        in_unk_image = keras.Input(shape=in_unk_shape_new, name='Iniput_unk_blk-%d_disc'%idx_block)
        in_ref_image = keras.Input(shape=in_ref_shape_new, name='Iniput_ref_blk-%d_disc'%idx_block)
        # Concatenate both images at channel level
        in_image = keras.layers.concatenate([in_unk_image,in_ref_image], axis=-1, 
                                            name='Iniput_concat_blk-%d_disc'%idx_block)
    else:    
        # get shape of existing model
        in_shape = list(old_model.input.shape)
        # define new input shape as double the size
        input_shape = (in_shape[-4]*2, in_shape[-3]*2, in_shape[-2]*2, in_shape[-1])
        in_image = keras.Input(shape=input_shape, name='Iniput_blk-%d_disc'%idx_block)
        
        
    # define new input processing layer
    input_dim = in_image.shape[-1]
    d = keras.layers.Conv3D(param_struct.block_conv_channels[intial_channels_idx], 
                               param_struct.input_kernel_size, 
                               padding='same', 
                               use_bias=False,
                               kernel_initializer=init, 
                               kernel_constraint=const,
                              name='Conv3D_blk-%d_ini_disc'%(idx_block))(in_image)
    if param_struct.use_He_scale:
        d = layers.HeScale([param_struct.input_kernel_size,
                         param_struct.input_kernel_size,
                         1], 
                        input_dim,
                    lrmul=param_struct.lrmul_ini,
                    gain=param_struct.gain_ini,
                   name='HeScale_blk-%d_ini_disc'%(idx_block))(d) # Add He scaling
    d = layers.BiasLayer(name='Bias_blk-%d_ini_disc'%(idx_block))(d) # Add bias 
    d = keras.layers.LeakyReLU(alpha=0.2,
                                 name='LeakyReLu_blk-%d_ini_disc'%(idx_block))(d)
    # define new block
    for idx_blk_layer in range(param_struct.block_conv_layers):
        input_dim = d.shape[-1]
        d = keras.layers.Conv3D(param_struct.block_conv_channels[param_struct.n_blocks-end_conv_blocks-idx_block-1], 
                                   param_struct.block_kernel_size, 
                                   padding='same', 
                                   use_bias=False,
                                   kernel_initializer=init, 
                                   kernel_constraint=const,
                                  name='Conv3D_blk-%d_l-%d_disc'%(idx_block,idx_blk_layer))(d)
        if param_struct.use_He_scale:
            d = layers.HeScale([param_struct.block_kernel_size,
                         param_struct.block_kernel_size,
                         1], 
                        input_dim,
                    lrmul=param_struct.lrmul_block,
                    gain=param_struct.gain_block,
                       name='HeScale_blk-%d_l-%d_disc'%(idx_block,idx_blk_layer))(d) # Add He scaling
        d = layers.BiasLayer(name='Bias_blk-%d_l-%d_disc'%(idx_block,idx_blk_layer))(d) # Add bias 
        d = keras.layers.LeakyReLU(alpha=0.2,
                                     name='LeakyReLu_blk-%d_l-%d_disc'%(idx_block,idx_blk_layer))(d)
    d = keras.layers.AveragePooling3D(name='DownSample_blk-%d_disc'%(idx_block))(d)
    block_new = d
    # look for the end of initial block of last model
    old_layer_names = [layer.name for layer in old_model.layers]
    old_input_layers_num = old_layer_names.index('LeakyReLu_blk-%d_ini_disc'%(idx_block-1))
    # skip the input, 1x1 and activation for the old model
    for i in range(old_input_layers_num+1, len(old_model.layers)):
        d = old_model.layers[i](d)
    # define straight-through model
    if param_struct.conditional:
        model1 = keras.Model([in_unk_image,in_ref_image], d, name="Disc_blk-%d"%(idx_block))
    else:
        model1 = keras.Model(in_image, d, name="Disc_blk-%d"%(idx_block))
    
    
    # downsample the new larger image
    downsample = keras.layers.AveragePooling3D(name='DownSample_input_blk-%d_disc'%(idx_block))(in_image)
    # connect old input processing to downsampled new input
    block_old = old_model.get_layer('Conv3D_blk-%d_ini_disc'%(idx_block-1))(downsample)
    if param_struct.use_He_scale:
        block_old = old_model.get_layer('HeScale_blk-%d_ini_disc'%(idx_block-1))(block_old)
    block_old = old_model.get_layer('Bias_blk-%d_ini_disc'%(idx_block-1))(block_old)
    block_old = old_model.get_layer('LeakyReLu_blk-%d_ini_disc'%(idx_block-1))(block_old)
    
    # fade in output of old model input layer with new input
    d = layers.WeightedSum(name='WSum_fadeIn_blk-%d_disc'%(idx_block))([block_old, block_new])
    # skip the input, 1x1 and activation for the old model
    for i in range(old_input_layers_num+1, len(old_model.layers)):
        d = old_model.layers[i](d)
    # define straight-through model
    if param_struct.conditional:
        model2 = keras.Model([in_unk_image,in_ref_image], d, name="Disc_blk-%d_fadeIn"%(idx_block))
    else:
        model2 = keras.Model(in_image, d, name="Disc_blk-%d_fadeIn"%(idx_block))
    return [model1, model2]

# define the discriminator models for each image resolution
def define_discriminator_3D(param_struct):
    """Creates a 3D convolutional discriminator network
    """
    
    # Number of end convolutional blocks
    end_conv_blocks = param_struct.n_blocks - param_struct.downSampling_layers
    
    
    # weight initialization
    init = keras.initializers.RandomNormal(stddev=param_struct.initializer_std)
    # weight constraint
    if param_struct.use_norm_contrain_scale:
        const = keras.constraints.max_norm(param_struct.block_max_norm_contrain)
    else:
        const = None
    model_list = list()
    idx_block = 0
    # base model input
    if param_struct.conditional:
        in_unk_image = keras.Input(shape=param_struct.input_shape, name='Iniput_unk_blk-%d_disc'%idx_block)
        in_ref_image = keras.Input(shape=param_struct.input_shape, name='Iniput_ref_blk-%d_disc'%idx_block)
        # Concatenate both images at channel level
        in_image = keras.layers.concatenate([in_unk_image,in_ref_image], axis=-1,
                                            name='Iniput_concat_blk-%d_disc'%idx_block)
    else:
        in_image = keras.Input(shape=param_struct.input_shape, name='Iniput_blk-%d_disc'%idx_block)
    
    
    # conv 1x1
    input_dim = in_image.shape[-1]
    d = keras.layers.Conv3D(param_struct.block_conv_channels[param_struct.n_blocks-end_conv_blocks-idx_block-2], 
                               param_struct.input_kernel_size, 
                               padding='same', 
                               use_bias=False,
                               kernel_initializer=init, 
                               kernel_constraint=const,
                               name='Conv3D_blk-%d_ini_disc'%(0))(in_image)
    if param_struct.use_He_scale:
        d = layers.HeScale([param_struct.input_kernel_size,
                         param_struct.input_kernel_size,
                         1], 
                        input_dim,
                    lrmul=param_struct.lrmul_ini,
                    gain=param_struct.gain_ini,
                   name='HeScale_blk-%d_ini_disc'%(0))(d) # Add He scaling
    d = layers.BiasLayer(name='Bias_blk-%d_ini_disc'%(0))(d) # Add bias 
    d = keras.layers.LeakyReLU(alpha=0.2, name='LeakyReLu_blk-%d_ini_disc'%(0))(d)
    
    # conv 3x3 (output block)
    if param_struct.use_minibatch_stdev:
        d = layers.MinibatchStdev(name='MinibatchStdev_blk-%d_disc'%(0))(d)
        
    for idx_blk_layer in range(param_struct.block_conv_layers):
        input_dim = d.shape[-1]
        d = keras.layers.Conv3D(param_struct.block_conv_channels[param_struct.n_blocks-end_conv_blocks-idx_block-1], 
                                   param_struct.block_kernel_size, 
                                   padding='same', 
                                   use_bias=False,
                                   kernel_initializer=init, 
                                   kernel_constraint=const,
                                   name='Conv3D_blk-%d_l-%d_disc'%(0,idx_blk_layer))(d)
        if param_struct.use_He_scale:
            d = layers.HeScale([param_struct.block_conv_channels,
                         param_struct.block_conv_channels,
                         1], 
                        input_dim,
                    lrmul=param_struct.lrmul_block,
                    gain=param_struct.gain_block,
                       name='HeScale_blk-%d_l-%d_disc'%(0,idx_blk_layer))(d) # Add He scaling
        d = layers.BiasLayer(name='Bias_blk-%d_l-%d_disc'%(0,idx_blk_layer))(d) # Add bias 
        d = keras.layers.LeakyReLU(alpha=0.2,
                                     name='LeakyReLu_blk-%d_l-%d_disc'%(0,idx_blk_layer))(d)
        # conv (4x4?)
        
        
    # ---------------------- Output Convolutional layers--------------------------------
    for idx_downSampl in range(0, end_conv_blocks):
        for idx_conv_layer in range(param_struct.block_conv_layers):
            input_dim = d.shape[-1]
            d = keras.layers.Conv3D(param_struct.block_conv_channels[param_struct.n_blocks-end_conv_blocks-idx_downSampl-1], 
                                       param_struct.block_kernel_size, 
                                       padding='same', 
                                       use_bias=False,
                                       kernel_initializer=init, 
                                       kernel_constraint=const,
                                       name='Conv3D_end_blk-%d_l-%d_disc'%(idx_downSampl,idx_conv_layer))(d)
            if param_struct.use_He_scale:
                d = layers.HeScale([param_struct.block_conv_channels,
                             param_struct.block_conv_channels,
                             1], 
                            input_dim,
                        lrmul=param_struct.lrmul_block,
                        gain=param_struct.gain_block,
                           name='HeScale_end_blk-%d_l-%d_disc'%(idx_downSampl,idx_conv_layer))(d) # Add He scaling
            d = layers.BiasLayer(name='Bias_end_blk-%d_l-%d_disc'%(idx_downSampl,idx_conv_layer))(d) # Add bias 
            d = keras.layers.LeakyReLU(alpha=0.2,
                                         name='LeakyReLu_end_blk-%d_l-%d_disc'%(idx_downSampl,idx_conv_layer))(d)
    
    # ---------------------- Output Dense layer --------------------------------
    d = keras.layers.Flatten(name='Flatten_output_disc')(d)
    input_dim = d.shape[-1]
    d = keras.layers.Dense(1,
                                     kernel_constraint=const,
                            bias_initializer='ones',
                           #activation=K.tanh,
                           #activation=K.relu,
                                     name='Dense_1_output_disc')(d)
    
    out_class = d
    # define model
    if param_struct.conditional:
        model = keras.Model([in_unk_image,in_ref_image], out_class, name="Disc_blk-%d"%(0))
    else:
        model = keras.Model(in_image, out_class, name="Disc_blk-%d"%(0))
    # store model
    model_list.append([model, model])
    # create submodels
    for i in range(1, param_struct.n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_3D_discriminator_block(old_model, param_struct, i)
        # store model
        model_list.append(models)
    return model_list


