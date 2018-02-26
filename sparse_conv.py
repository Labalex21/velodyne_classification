# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:44:06 2017

@author: schlichting
"""

import tensorflow as tf

def sparse_conv(tensor,binary_mask = None,filters=32,kernel_size=3,strides=2):

    if binary_mask == None: #first layer has no binary mask
        b,h,w,c = tensor.get_shape()
        channels=tf.split(tensor,c,axis=3)
        #assume that if one channel has no information, ALL CHANNELS HAVE NO INFORMATION
        binary_mask = tf.where(tf.equal(channels[0], tf.cast(0,dtype=tf.float32)), tf.zeros_like(channels[0]), tf.ones_like(channels[0])) #mask should only have the size of (B,H,W,1)

    features = tf.multiply(tensor,binary_mask)
    features = tf.layers.conv2d(features, filters=filters, kernel_size=kernel_size, strides=(strides, strides), trainable=True, use_bias=False, padding="same")

    norm = tf.layers.conv2d(binary_mask, filters=filters,kernel_size=kernel_size,strides=(strides, strides),kernel_initializer=tf.ones_initializer(),trainable=False,use_bias=False,padding="same")
    norm = tf.where(tf.equal(norm,0),tf.zeros_like(norm),tf.reciprocal(norm)) # Berechnet Kehrwert, falls Wert != 0
    _,_,_,bias_size = norm.get_shape() # Bestimmt notwendige Länge von b

    b = tf.Variable(tf.constant(0.01, shape=[bias_size]))
    #feature = tf.multiply(features,norm)+b
    feature = features+b
    mask = tf.layers.max_pooling2d(binary_mask,strides = strides,pool_size=3,padding="same")

    return feature,mask