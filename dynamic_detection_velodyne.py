# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from tflearn import conv_2d as conv, conv_2d_transpose as deconv,max_pool_2d as maxpool,dropout
from tflearn.activations import leaky_relu as lrelu
import tensorflow as tf
import time
import file_handler as fh
#import cv2
import numpy as np
from sparse_conv import sparse_conv


# Reset graph
tf.reset_default_graph()

#dir_data = "X:/Proc/Velodyne_Puck/20180201_icsens_innenstadt/imgs/"
dir_data = "../data/imgs/"
dir_imgs_training = dir_data + "training/"
dir_labels_training = dir_data + "labels_training/"
dir_imgs_testing = dir_data + "testing/"
dir_records = dir_data + "records/"
#path_model = "X:/Proc/Velodyne_Puck/20180201_icsens_innenstadt/models/conv_dyn_velodyne.ckpt"
path_model = "../data/models/conv_dyn_velodyne.ckpt"

# input data parameters
epochs = 200
batch_size = 30

# images parameters
max_dist = 40
height = 900
width = 16
image_shape = [height,width]
label_shape = image_shape

# network parameters
keep_prob = 0.5
learning_rate = 0.001

def create_network(keep_prob,x,labels):
    print(x.get_shape())
    x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1], name='reshape_image1')
    mask = tf.reshape(labels, [tf.shape(labels)[0], tf.shape(labels)[1], tf.shape(labels)[2], 1], name='reshape_mask1')
    network = tf.to_float(x)
    print(network.get_shape())
    
#     sparse convolutions
    network, mask = sparse_conv(x, mask, filters = 32, kernel_size = 3, strides = 1)
    network = lrelu(network)
    network, mask = sparse_conv(network, mask, filters = 32, kernel_size = 3, strides = 1)
    network = lrelu(network)
    network = maxpool(network, 2, strides=[2,2], name='maxpool1')
    network = lrelu(network)
    mask = maxpool(mask, 2, strides=[2,2], name='maxpool_mask2')
    print(network.get_shape())

    network, mask = sparse_conv(network, binary_mask=mask, filters = 64, kernel_size = 3, strides = 1)
    network = lrelu(network)
    network, mask = sparse_conv(network, mask, filters = 64, kernel_size = 3, strides = 1)
    network = lrelu(network)
    network = maxpool(network, 2, strides=[2,2], name='maxpool2')
    network = lrelu(network)
    mask = maxpool(mask, 2, strides=[2,2], name='maxpool_mask2')
    
    print(network.get_shape())
    network, mask = sparse_conv(network, mask, filters = 128, kernel_size = 3, strides = 1)
    network = lrelu(network)
    network, mask = sparse_conv(network, mask, filters = 128, kernel_size = 3, strides = 1)
    network = lrelu(network)
    network, mask = sparse_conv(network, mask, filters = 128, kernel_size = 3, strides = 1)
    network = lrelu(network)
    network = maxpool(network, 2, strides=[2,2], name='maxpool3')
    network = lrelu(network)
    mask = maxpool(mask, 2, strides=[2,2], name='maxpool_mask3') 
    print(network.get_shape())

    network, mask = sparse_conv(network, mask, filters = 256, kernel_size = 3, strides = 1)
    network = lrelu(network)
    network, mask = sparse_conv(network, mask, filters = 256, kernel_size = 3, strides = 1)
    network = lrelu(network)
    network, mask = sparse_conv(network, mask, filters = 256, kernel_size = 3, strides = 1)
    network = lrelu(network)
    network = maxpool(network, 2, strides=[2,2], name='maxpool4')
    network = lrelu(network)
    mask = maxpool(mask, 2, strides=[2,2], name='maxpool_mask4')
    print(network.get_shape())    

    #deconvolution
    network = deconv(network, 112, 1,[int(np.ceil(height/8)),int(width/8)], activation='relu', strides=[2,2], name='deconv1')
    print(network.get_shape())
    network = dropout(network,keep_prob)
    network = deconv(network,64, 1,[int(height/4),int(width/4)], activation='relu', strides=[2,2], name='deconv2')
    print(network.get_shape())
    network = dropout(network, keep_prob)
    network = deconv(network, 32, 1,[int(height/2),int(width/2)], activation='relu', strides=[2,2], name='deconv3')
    print(network.get_shape())
    network = dropout(network, keep_prob)
    network = deconv(network, 1, 1,[height,width], activation='relu', strides=[2,2], name='deconv4')
    output = tf.reshape(network, [tf.shape(network)[0], tf.shape(network)[1], tf.shape(network)[2]], name='reshape2')
    print(output.get_shape())

    return output, mask

def train():
    print("start training...")
    with tf.Session()  as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
    
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        total_batch = int(number_batches/batch_size)
        print("total_batch:",total_batch)
        start = time.time()
        for e in range(epochs):
            print("epoch",e)
            for i in range(total_batch):
                start2 = time.time()
                current_loss,_,img,pred, mask_values = sess.run([loss, optimizer,x, output, mask])
                            
                elapsed = time.time() - start
                elapsed2 = time.time() - start2
                if i % 20 == 0:
                    print("epoch {}/{}".format(e+1,epochs),
                          "| batch: {}/{}".format(i+1,total_batch),
                          "| current los:",current_loss,
                          "| El. time: ", "{:.2f}".format(elapsed), "s",
                          "| Batch time: ", "{:.2f}".format(elapsed2), "s")
                    
         
        coord.request_stop()
        coord.join(threads)
        
        # Save model
        save_path = saver.save(sess, path_model)
        print("Model saved in file: %s" % save_path)

x, labels, number_batches = fh.read_tfrecord(dir_records, image_shape, batch_size = batch_size,num_epochs=epochs)

print("number_batches: ",number_batches)

#output,mask = create_network(keep_prob, x, tf.ones([batch_size, 31, 1160]))
output,mask = create_network(keep_prob, x, labels)

# loss
loss = tf.reduce_mean(tf.pow(x - output, 2))

# optimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

train()
#test_prediction()