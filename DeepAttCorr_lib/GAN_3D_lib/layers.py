#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

# External
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import numpy as np

from dataclasses import dataclass

# Internal


#-----------------------------------------------------------------------------
#-------------------- Get Upsampled Slice layer ------------------------------
#-----------------------------------------------------------------------------
        
class GetUpsampledSlice(keras.layers.Layer):

    # initialize the layer
    def __init__(self, **kwargs):
        super(GetUpsampledSlice, self).__init__(**kwargs)
        
    # perform the operation
    def call(self, inputs):

        upsmp = inputs[3]   
        slice_ax = inputs[1][0,0]
        slice_num = inputs[2][0,0]
        batched_slices = self.get_upsample_slice_tf(inputs[0], slice_ax, slice_num, upsmp)
        
        return batched_slices

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        dim_px = input_shape[2]
        num_chan = input_shape[-1]
        return [None, dim_px, dim_px, num_chan]
    
    def get_config(self):
        base_config = super(GetUpsampledSlice, self).get_config()
        return base_config
    
    
    def get_upsample_slice_tf(self, volume_in, slice_ax, slice_num, upsmp):
    

        orig_size = volume_in.shape[2]
        end_size = orig_size*(2**upsmp)
        # Get original fractional slice
        orig_fractional_slice = K.cast(slice_num*orig_size, volume_in.dtype) / K.cast(end_size, volume_in.dtype)
        # Get original size slice (rounded down)
        slice_orig = tf.math.floordiv(slice_num*orig_size,end_size)
        # Get fraction
        fractional_value = orig_fractional_slice-K.cast(slice_orig, volume_in.dtype)


        ini_mat = tf.gather(volume_in, slice_orig, axis=1)
        fin_mat = tf.gather(volume_in, slice_orig+1, axis=1)
        a_mat = fin_mat-ini_mat 
        b_mat = fin_mat-ini_mat 
        sliced_in_A = fractional_value*a_mat + b_mat
        sliced_in_A = tf.reshape(sliced_in_A, [-1, orig_size,orig_size,volume_in.shape[-1]])

        ini_mat = tf.gather(volume_in, slice_orig, axis=2)
        in_mat = tf.gather(volume_in, slice_orig+1, axis=2)
        a_mat = fin_mat-ini_mat 
        b_mat = fin_mat-ini_mat 
        sliced_in_B = fractional_value*a_mat + b_mat
        sliced_in_B = tf.reshape(sliced_in_B, [-1, orig_size,orig_size,volume_in.shape[-1]])

        ini_mat = tf.gather(volume_in, slice_orig, axis=3)
        fin_mat = tf.gather(volume_in, slice_orig+1, axis=3)
        a_mat = fin_mat-ini_mat 
        b_mat = ini_mat 
        sliced_in_C = fractional_value*a_mat + b_mat
        sliced_in_C = K.reshape(sliced_in_C, [-1, orig_size,orig_size,volume_in.shape[-1]])

        sliced_in = tf.concat([sliced_in_A, sliced_in_B, sliced_in_C], axis=0)

        return sliced_in



#-----------------------------------------------------------------------------
#-------------------- Fixed Latent Input layer -------------------------------
#-----------------------------------------------------------------------------

class FixedLatentSpace(keras.layers.Layer):
    
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(FixedLatentSpace, self).__init__(**kwargs)

    def build(self, inputs):
        # Create a trainable weight variable for this layer.
        self.FixedLatentSpace = self.add_weight(name='FixedLatentSpace',
                                      shape=(self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(FixedLatentSpace, self).build(self.output_dim)  # Be sure to call this at the end

    def call(self, inputs):

        return self.FixedLatentSpace*inputs[0,0]

    def compute_output_shape(self, inputs):
        return self.output_dim
    
    def get_config(self):
        base_config = super(FixedLatentSpace, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config
    
#-----------------------------------------------------------------------------
#-------------------- 3D Noise Addition layer --------------------------------
#----------------------------------------------------------------------------- 
class NoiseAddition3D(keras.layers.Layer):

    # initialize the layer
    def __init__(self, **kwargs):
        super(NoiseAddition3D, self).__init__(**kwargs)
        
    def build(self, inputs):
        
        # inputs: [feature_map]
        n_layers = inputs[-1]
        
        # Create a trainable weight variable for this layer.
        self.W_noise = self.add_weight(name='W_noise',
                                      shape=([1,1,1, n_layers, n_layers]),
                                      initializer='zeros',
                                      trainable=True,
                                      dtype=np.float32)

        
        super(NoiseAddition3D, self).build(n_layers)  # Be sure to call this at the end

    # perform the operation
    def call(self, inputs):
               
        # Get noise
        layer_noise = K.random_normal(shape=[inputs.shape[1],
                                                           inputs.shape[2],
                                                           inputs.shape[3],
                                                           inputs.shape[4]],
                                            mean=0.0,
                                            stddev=1.0,
                                                    dtype=np.float32)

        # Noise affine transform modulation
        transf_noise = tf.matmul(layer_noise, self.W_noise)

        # Add noise
        noised = inputs+transf_noise
        
        return noised

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def get_config(self):
        base_config = super(NoiseAddition3D, self).get_config()
        return base_config
    
    
#-----------------------------------------------------------------------------
#-------------------- 2D Noise Addition layer --------------------------------
#-----------------------------------------------------------------------------     
class NoiseAddition2D(keras.layers.Layer):

    # initialize the layer
    def __init__(self, **kwargs):
        super(NoiseAddition2D, self).__init__(**kwargs)
        
    def build(self, inputs):
        
        # inputs: [feature_map]
        n_layers = inputs[-1]
        
        # Create a trainable weight variable for this layer.
        self.W_noise = self.add_weight(name='W_noise',
                                      shape=([1,1, n_layers, n_layers]),
                                      initializer='zeros',
                                      trainable=True,
                                      dtype=np.float32)

        
        super(NoiseAddition2D, self).build(n_layers)  # Be sure to call this at the end

    # perform the operation
    def call(self, inputs):
               
        # Get noise
        layer_noise = K.random_normal(shape=[inputs.shape[1],
                                                           inputs.shape[2],
                                                           inputs.shape[3]],
                                            mean=0.0,
                                            stddev=1.0,
                                                    dtype=np.float32)

        # Noise affine transform modulation
        transf_noise = tf.matmul(layer_noise, self.W_noise)

        # Add noise
        noised = inputs+transf_noise
        
        return noised

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def get_config(self):
        base_config = super(NoiseAddition2D, self).get_config()
        return base_config
    
#-----------------------------------------------------------------------------
#-------------------- 3D Style modulation layer ------------------------------
#----------------------------------------------------------------------------- 
class StyleModulation3D(keras.layers.Layer):

    # initialize the layer
    def __init__(self, **kwargs):
        super(StyleModulation3D, self).__init__(**kwargs)
        
    def build(self, inputs):
        
        # inputs: [feature_map, latent_style]
        n_layers = inputs[0][-1]
        n_latent_in = inputs[1][-1]
        
        # Create a trainable weight variable for this layer.
        self.W_mean = self.add_weight(name='W_mean',
                                      shape=([n_latent_in, n_layers]),
                                      initializer='he_normal',
                                      trainable=True,
                                     dtype=np.float32)
        self.b_mean = self.add_weight(name='b_mean',
                                      shape=(n_layers),
                                      initializer='zeros',
                                      trainable=True,
                                     dtype=np.float32)
        
        # Create a trainable weight variable for this layer.
        self.W_std = self.add_weight(name='W_std',
                                      shape=([n_latent_in, n_layers]),
                                      initializer='he_normal',
                                      trainable=True,
                                    dtype=np.float32)
        self.b_std = self.add_weight(name='b_std',
                                      shape=(n_layers),
                                      initializer='zeros',
                                      trainable=True,
                                    dtype=np.float32)
        
        
        super(StyleModulation3D, self).build(n_layers)  # Be sure to call this at the end

    # perform the operation
    def call(self, inputs):
        
        # inputs: [feature_map, latent_style]
        n_layers = inputs[0].shape[-1]

        # Get style modulation
        bias_style = tf.matmul(inputs[1], self.W_mean)+self.b_mean
        #bias_style = (inputs[1]*self.W_mean)+self.b_mean
        bias_style = keras.layers.Reshape([1, 1, 1, n_layers],
                                          name = 'style_bias_reshape')(bias_style)
        std_style = tf.matmul(inputs[1], self.W_std)+self.b_std
        #std_style = (inputs[1]*self.W_std)+self.b_std
        std_style = keras.layers.Reshape([1, 1, 1, n_layers],
                                          name = 'style_std_reshape')(std_style)
        
        # Stylize
        stylized = (inputs[0]*(std_style+1.0))+bias_style
        
        return stylized

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def get_config(self):
        base_config = super(StyleModulation3D, self).get_config()
        return base_config
    
#-----------------------------------------------------------------------------
#-------------------- 2D Style modulation layer ------------------------------
#-----------------------------------------------------------------------------  
class StyleModulation2D(keras.layers.Layer):

    # initialize the layer
    def __init__(self, **kwargs):
        super(StyleModulation2D, self).__init__(**kwargs)
        
    def build(self, inputs):
        
        # inputs: [feature_map, latent_style]
        n_layers = inputs[0][-1]
        n_latent_in = inputs[1][-1]
        
        # Create a trainable weight variable for this layer.
        self.W_mean = self.add_weight(name='W_mean',
                                      shape=([n_latent_in, n_layers]),
                                      initializer='he_normal',
                                      trainable=True,
                                     dtype=np.float32)
        self.b_mean = self.add_weight(name='b_mean',
                                      shape=(n_layers),
                                      initializer='zeros',
                                      trainable=True,
                                     dtype=np.float32)
        
        # Create a trainable weight variable for this layer.
        self.W_std = self.add_weight(name='W_std',
                                      shape=([n_latent_in, n_layers]),
                                      initializer='he_normal',
                                      trainable=True,
                                    dtype=np.float32)
        self.b_std = self.add_weight(name='b_std',
                                      shape=(n_layers),
                                      initializer='zeros',
                                      trainable=True,
                                    dtype=np.float32)
        
        
        super(StyleModulation2D, self).build(n_layers)  # Be sure to call this at the end

    # perform the operation
    def call(self, inputs):
        
        # inputs: [feature_map, latent_style]
        n_layers = inputs[0].shape[-1]

        # Get style modulation
        bias_style = tf.matmul(inputs[1], self.W_mean)+self.b_mean
        #bias_style = (inputs[1]*self.W_mean)+self.b_mean
        bias_style = keras.layers.Reshape([1, 1, n_layers],
                                          name = 'style_bias_reshape')(bias_style)
        std_style = tf.matmul(inputs[1], self.W_std)+self.b_std
        #std_style = (inputs[1]*self.W_std)+self.b_std
        std_style = keras.layers.Reshape([1, 1, n_layers],
                                          name = 'style_std_reshape')(std_style)
        
        # Stylize
        stylized = (inputs[0]*(std_style+1.0))+bias_style
        
        return stylized

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def get_config(self):
        base_config = super(StyleModulation2D, self).get_config()
        return base_config


#-----------------------------------------------------------------------------
#-------------------- 3Dto2D Style modulation layer --------------------------
#-----------------------------------------------------------------------------  
class StyleModulation3Dto2D(keras.layers.Layer):

    # initialize the layer
    def __init__(self, **kwargs):
        super(StyleModulation3Dto2D, self).__init__(**kwargs)
        self.n_latent_in = 0
        self.n_layers = 0
        
    def build(self, inputs):
        
        # inputs: [feature_map, latent_style]
        self.n_layers = inputs[0][-1]
        self.n_latent_in = inputs[1][-1]
        
        # Create a trainable weight variable for this layer.
        self.W_mean = self.add_weight(name='W_mean',
                                      shape=([self.n_latent_in, self.n_layers]),
                                      initializer='he_normal',
                                      trainable=True,
                                     dtype=np.float32)
        self.b_mean = self.add_weight(name='b_mean',
                                      shape=(self.n_layers),
                                      initializer='zeros',
                                      trainable=True,
                                     dtype=np.float32)
        
        # Create a trainable weight variable for this layer.
        self.W_std = self.add_weight(name='W_std',
                                      shape=([self.n_latent_in, self.n_layers]),
                                      initializer='he_normal',
                                      trainable=True,
                                    dtype=np.float32)
        self.b_std = self.add_weight(name='b_std',
                                      shape=(self.n_layers),
                                      initializer='zeros',
                                      trainable=True,
                                    dtype=np.float32)
        
        
        super(StyleModulation3Dto2D, self).build(self.n_layers)  # Be sure to call this at the end

    # perform the operation
    def call(self, inputs):
        
        # inputs: [feature_map, latent_style]

        # Each style batch to each one of the three 2D slices
        # batch = 2
        # batch 2D = batch*3
        # Style = [A,B]
        # Style 2D = [A,A,A,B,B,B]
        expanded_style = tf.stack([inputs[1], inputs[1], inputs[1]], axis=1)
        expanded_style = tf.reshape(expanded_style, shape=[-1,self.n_latent_in])
        
        # Get style modulation
        bias_style = tf.matmul(expanded_style, self.W_mean)+self.b_mean
        bias_style = keras.layers.Reshape([1, 1, self.n_layers],
                                          name = 'style_bias_reshape')(bias_style)
        std_style = tf.matmul(expanded_style, self.W_std)+self.b_std
        std_style = keras.layers.Reshape([1, 1, self.n_layers],
                                          name = 'style_std_reshape')(std_style)
        
        # Stylize
        stylized = (inputs[0]*(std_style+1.0))+bias_style
        
        return stylized

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def get_config(self):
        base_config = super(StyleModulation3Dto2D, self).get_config()
        return base_config
#-----------------------------------------------------------------------------
#-------------------- 3D Instance Normalization layer ------------------------
#-----------------------------------------------------------------------------    
class InstanceNormalization3D(keras.layers.Layer):

    # initialize the layer
    def __init__(self, **kwargs):
        super(InstanceNormalization3D, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        # Get the mean and std from the feature map
        inputs_mean = keras.backend.mean(inputs, axis=[1,2,3], keepdims=True)
        inputs_std = keras.backend.std(inputs, axis=[1,2,3], keepdims=True)
        
        # Normalize
        normalized = (inputs-inputs_mean+1.0e-8)/inputs_std
        
        return normalized

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        base_config = super(InstanceNormalization3D, self).get_config()
        return base_config
    
#-----------------------------------------------------------------------------
#-------------------- 2D Instance Normalization layer ------------------------
#-----------------------------------------------------------------------------    
class InstanceNormalization2D(keras.layers.Layer):

    # initialize the layer
    def __init__(self, **kwargs):
        super(InstanceNormalization2D, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        # Get the mean and std from the feature map
        inputs_mean = keras.backend.mean(inputs, axis=[1,2], keepdims=True)
        inputs_std = keras.backend.std(inputs, axis=[1,2], keepdims=True)
        
        # Normalize
        normalized = (inputs-inputs_mean+1.0e-8)/inputs_std
        
        return normalized

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        base_config = super(InstanceNormalization2D, self).get_config()
        return base_config
    
    
#-----------------------------------------------------------------------------
#-------------------- He-Scalling layer --------------------------------------
#-----------------------------------------------------------------------------
class HeScale(keras.layers.Layer):

    # initialize the layer
    def __init__(self, 
                 kernel_size, 
                 input_dim, 
                 gain=float(np.sqrt(2)),  
                 lrmul=float(1.0), 
                 **kwargs):
        
        super(HeScale, self).__init__(**kwargs)
        #self.gain = K.cast(gain, "float32")
        self.gain = float(gain)
        #self.lrmul = (K.cast(lrmul, "float32")
        self.lrmul = float(lrmul)
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        

    # perform the operation
    def call(self, inputs):
        fan_in = keras.backend.prod(self.kernel_size)*self.input_dim # [kernel, kernel, kernel, in, out] 

        fan_in = keras.backend.cast(fan_in, "float32")
        
        he_std = self.gain / keras.backend.sqrt(fan_in) # He init

        runtime_coef = he_std * self.lrmul
        return  inputs * runtime_coef
        

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        base_config = super(HeScale, self).get_config()
        base_config['gain'] = self.gain
        base_config['lrmul'] = self.lrmul
        base_config['kernel_size'] = self.kernel_size
        base_config['input_dim'] = self.input_dim
        return base_config
    

#-----------------------------------------------------------------------------
#-------------------- Bias Addition layer ------------------------------------
#-----------------------------------------------------------------------------
class BiasLayer(keras.layers.Layer):

    # initialize the layer
    def __init__(self, **kwargs):
        super(BiasLayer, self).__init__(**kwargs)
        
    def build(self, inputs):
        bias_dim = [inputs[-1]]
        
        
        # Create a trainable bias variable for this layer.
        self.bias = self.add_weight(name='bias',
                                      shape=bias_dim,
                                      initializer=keras.initializers.RandomNormal(stddev=0.02),
                                      trainable=True)
        super(BiasLayer, self).build(self.bias)  # Be sure to call this at the end

    # perform the operation
    def call(self, inputs):
        return inputs + self.bias

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        base_config = super(BiasLayer, self).get_config()
        return base_config
    

#-----------------------------------------------------------------------------
#----------- pixel-wise feature vector normalization layer -------------------
#-----------------------------------------------------------------------------
class PixelNormalization(keras.layers.Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        # calculate square pixel values
        values = inputs**2.0
        # calculate the mean pixel values
        mean_values = K.mean(values, axis=-1, keepdims=True)
        # ensure the mean is not zero
        mean_values += 1.0e-8
        # calculate the sqrt of the mean squared value (L2 norm)
        l2 = K.sqrt(mean_values)
        # normalize values by the l2 norm
        normalized = inputs / l2
        return normalized

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        base_config = super(PixelNormalization, self).get_config()
        return base_config
    
#-----------------------------------------------------------------------------
#---------------------- weighted sum output ----------------------------------
#-----------------------------------------------------------------------------
class WeightedSum(keras.layers.Add):
    # init with default value
    def __init__(self, alpha=float(0.0), **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = K.variable(float(alpha), name='ws_alpha', dtype="float32")
        

    # output a weighted sum of inputs
    def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        assert (len(inputs) == 2)
        # ((1-a) * input1) + (a * input2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output
    
    def get_config(self):
        base_config = super(WeightedSum, self).get_config()
        base_config['alpha'] = self.alpha
        return base_config
    
# update the alpha value on each instance of WeightedSum

def update_fadein(models, step, n_steps):
    # calculate current alpha (linear from 0 to 1)
    alpha = float(step) / float(n_steps - 1)
    # update the alpha for each model
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                K.set_value(layer.alpha, alpha)
    
#-----------------------------------------------------------------------------
#---------------------- mini-batch standard deviation layer ------------------
#-----------------------------------------------------------------------------
class MinibatchStdev(keras.layers.Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        # calculate the mean value for each pixel across channels
        mean = K.mean(inputs, axis=0, keepdims=True)
        # calculate the squared differences between pixel values and mean
        squ_diffs = K.square(inputs - mean)
        # calculate the average of the squared differences (variance)
        mean_sq_diff = K.mean(squ_diffs, axis=0, keepdims=True)
        # add a small value to avoid a blow-up when we calculate stdev
        mean_sq_diff += 1e-8
        # square root of the variance (stdev)
        stdev = K.sqrt(mean_sq_diff)
        # calculate the mean standard deviation across each pixel coord
        mean_pix = K.mean(stdev, keepdims=True)
        # scale this up to be the size of one input feature map for each sample
        shape = K.shape(inputs)
        output = K.tile(mean_pix, (shape[0], shape[1], shape[2], shape[3], 1))
        # concatenate with the output
        combined = K.concatenate([inputs, output], axis=-1)
        return combined

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        # create a copy of the input shape as a list
        input_shape = list(input_shape)
        # add one to the channel dimension (assume channels-last)
        input_shape[-1] += 1
        # convert list to a tuple
        return tuple(input_shape)
    
    def get_config(self):
        base_config = super(MinibatchStdev, self).get_config()
        return base_config