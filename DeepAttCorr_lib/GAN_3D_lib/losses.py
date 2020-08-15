#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

# External
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import numpy as np

from dataclasses import dataclass

# Defines
k_coupling = 0.0001


#-----------------------------------------------------------------------------
#---------------------- wasserstein loss -------------------------------------
#-----------------------------------------------------------------------------
def wasserstein_loss(y_true, y_pred):
    """WGAN Loss Keras
    """
    return tf.keras.backend.mean(y_true * y_pred)

@tf.function
def wasserstein_loss_tf(y_true, y_pred):
    """WGAN Loss Tensorflow
    """
    return tf.math.reduce_mean(tf.multiply(y_true, y_pred))

#-----------------------------------------------------------------------------
#---------------------- DICE loss --------------------------------------------
#-----------------------------------------------------------------------------

def DICE_loss(y_true, y_pred):
    """Calculate DICE loss for a giben sample batch
    """
    
    # Get number of clases
    n_class = y_pred.shape[-1]
    
    # Element-wise multiplication of output and truth
    inter_multi = tf.multiply(y_pred, y_true)
    # Sumation
    mult_sum = tf.reduce_sum(inter_multi,[1,2,3])
    
    # Inter-class
    sum_class_1 = tf.reduce_sum(tf.cast(tf.pow(y_pred,2), tf.float32),[1,2,3])
    sum_class_2 = tf.reduce_sum(tf.cast(tf.pow(y_true,2), tf.float32),[1,2,3])
    # Dice-loss
    dice_per_class = tf.math.divide(tf.multiply(tf.cast(2, tf.float32),mult_sum) , tf.add(sum_class_1,sum_class_2))

    return tf.reduce_sum(dice_per_class, -1)/n_class

#-----------------------------------------------------------------------------
#---------------------- V-Net compound loss ----------------------------------
#-----------------------------------------------------------------------------
@tf.function
def Vnet_compound_loss(y_true, y_pred):
    """Calculates the loss composed of the DICE and L2 loss
    """

    y_true_texture, y_true_labels = tf.split(y_true, [1,4], axis=-1)
    y_pred_texture, y_pred_labels = tf.split(y_pred, [1,4], axis=-1)
    
    dice_loss = DICE_loss(y_true_labels, y_pred_labels)

    dice_loss = 1.0-dice_loss # MINIMIZAR!

    L2_loss = tf.reduce_sum(tf.pow((y_true_texture - y_pred_texture),2),[1,2,3,4])
    
    
    return (dice_loss+(k_coupling*L2_loss))

