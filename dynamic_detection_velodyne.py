# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow as tf
import time
import file_handler as fh
#import cv2
import numpy as np


# Reset graph
tf.reset_default_graph()

dir_data = "X:/Proc/Velodyne_Puck/20180201_icsens_innenstadt/imgs/"
#dir_data = "../data/imgs/"
dir_imgs_training = dir_data + "training/"
dir_labels_training = dir_data + "labels_training/"
dir_imgs_testing = dir_data + "testing/"
dir_records = dir_data + "records/"
path_model = "X:/Proc/Velodyne_Puck/20180201_icsens_innenstadt/models/conv_dyn_velodyne.ckpt"
#path_model = "../data/models/conv_dyn_velodyne.ckpt"

# input data parameters
epochs = 200
batch_size = 100

# images parameters
max_dist = 40
height = 900
width = 16
image_shape = [height,width]
label_shape = image_shape

# network parameters
keep_prob = 0.5
learning_rate = 0.0001

n_features = 32
patch_size = 3
strides = [1, 1, 1, 1]

def create_network(x):
    print('input: ',x.get_shape())
    x = tf.reshape(x, [tf.shape(x)[0], height, width, 1], name='reshape_image1')
    x = tf.to_float(x)/max_dist
    print('x:     ',x.get_shape())
    
    conv1 = tflearn.conv_2d(x,n_features,patch_size,strides, padding = 'same', activation = 'leaky_relu')
    print('conv1: ', conv1.get_shape())
    
    conv2 = tflearn.conv_2d(conv1,n_features,patch_size,strides, padding = 'same', activation = 'leaky_relu')
    print('conv2: ', conv2.get_shape())
    
    conv3 = tflearn.conv_2d(conv2,n_features,patch_size,strides, padding = 'same', activation = 'leaky_relu')
    print('conv3: ', conv3.get_shape())
    
    conv4 = tflearn.conv_2d(conv3,n_features,patch_size,strides, padding = 'same', activation = 'leaky_relu')
    print('conv4: ', conv4.get_shape())
    
    
    tconv1 = tflearn.conv_2d_transpose(conv4,n_features,patch_size,conv3.get_shape().as_list()[1:4], padding = 'same', activation = 'leaky_relu')
    print('tconv1:', tconv1.get_shape())
    
    tconv2 = tflearn.conv_2d_transpose(tconv1,n_features,patch_size,conv2.get_shape().as_list()[1:4], padding = 'same', activation = 'leaky_relu')
    print('tconv2:', tconv2.get_shape())
    
    tconv3 = tflearn.conv_2d_transpose(tconv2,n_features,patch_size,conv1.get_shape().as_list()[1:4], padding = 'same', activation = 'leaky_relu')
    print('tconv3:', tconv3.get_shape())
    
    tconv4 = tflearn.conv_2d_transpose(tconv3,1,patch_size,x.get_shape().as_list()[1:4], padding = 'same', activation = 'leaky_relu')
    print('tconv4:', tconv4.get_shape())

    output = tf.reshape(tconv4,[-1,height,width])
    print('output:', tconv4.get_shape())

    return output

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
                current_loss,img,pred = sess.run([loss,x, output])
                            
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

output = create_network(x)

# loss
loss = tf.reduce_mean(tf.pow(labels - output, 2))

# optimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

train()
#test_prediction()