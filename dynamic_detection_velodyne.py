# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow as tf
import time
import file_handler as fh
import cv2
import numpy as np
import datetime as dt

# Reset graph
tf.reset_default_graph()

# log file
log_filename = "../data/logs/log_classi_" + dt.datetime.now().strftime("%Y%m%d_%H_%M_%S") + ".txt"
log_file = open(log_filename,"w")
log_file.write("start\n")
log_file.flush()

#dir_data = "D:/Velodyne/20180201_icsens_innenstadt/imgs/"
#dir_test = "D:/Velodyne/20180201_icsens_innenstadt/imgs/result_detection/"
dir_test = "../data/20180201/result_detection/"
dir_data = "../data/20180201/"
dir_imgs_training = dir_data + "training/"
dir_labels_training = dir_data + "labels_training/"
dir_imgs_testing = dir_data + "testing/"
dir_records = dir_data + "records_dyn/"
#path_model = "D:/Velodyne/20180201_icsens_innenstadt/models/conv_dyn_velodyne.ckpt"
path_model = "../data/models/conv_dyn_velodyne_50.ckpt"

# input data parameters
epochs = 200
batch_size = 50

# images parameters
max_dist = 40
height = 900
width = 16
image_shape = [height,width]
label_shape = image_shape

# network parameters
keep_prob = 0.5
learning_rate = 0.0002

n_features = 32
patch_size = 3
strides = [1, 1, 1, 1]

def create_network(x,y):
    print('input: ',x.get_shape())
    x = tf.reshape(x, [tf.shape(x)[0], height, width, 1], name='reshape_image1')
    print(x)
    x = tf.to_float(x)/max_dist
    print(x)
    print('x:     ',x.get_shape())
    
    conv1 = tflearn.conv_2d(x,n_features,patch_size,strides, padding = 'same', activation = 'leaky_relu', name='conv1')
    print('conv1: ', conv1.get_shape())
    maxPool1 = tflearn.layers.conv.max_pool_2d (conv1, 2, padding='same')
    print('mPool1:', maxPool1.get_shape())
    
    conv2 = tflearn.conv_2d(maxPool1,n_features,patch_size,strides, padding = 'same', activation = 'leaky_relu', name='conv2')
    print('conv2: ', conv2.get_shape())
    maxPool2 = tflearn.layers.conv.max_pool_2d (conv2, 2, padding='same')
    print('mPool2:', maxPool2.get_shape())
    
    conv3 = tflearn.conv_2d(maxPool2,n_features,patch_size,strides, padding = 'same', activation = 'leaky_relu', name='conv3')
    print('conv3: ', conv3.get_shape())
    maxPool3 = tflearn.layers.conv.max_pool_2d (conv3, 2, padding='same')
    print('mPool3:', maxPool3.get_shape())
    
    conv4 = tflearn.conv_2d(maxPool3,n_features,patch_size,strides, padding = 'same', activation = 'leaky_relu')
#    print('conv4: ', conv4.get_shape())
#    maxPool4 = tflearn.layers.conv.max_pool_2d (conv4, 2, padding='same')
#    print('mpool4:', conv4.get_shape())
#    
#    
#    fc1 = tflearn.fully_connected(conv3, 225*4*n_features, activation = 'leaky_relu')
    fc1 = tflearn.fully_connected(conv4, 5000, activation = 'leaky_relu')
    tfc1 = tflearn.fully_connected(fc1, 225*4*n_features, activation = 'leaky_relu')
    print('fc1: ', fc1.get_shape())
    tfc1 = tf.reshape(tfc1, [-1, 225, 4, n_features])
    print('tfc1: ', tfc1.get_shape())
    
#    last = fully_connected(tfc2, tf.transpose(weights['wfc1']), biases['b3_dec'])
#    # tfc2 = tf.reshape(tfc2, [-1, 1160*2, 1, n_features])
#    last = tf.reshape(last, [-1, 1160, 1, n_features])
#    print('tfc3: ', last.get_shape())
    
#    upsample1 = tflearn.upsample_2d(maxPool4,2)
#    print('usamp1:', upsample1.get_shape())
#    tconv1 = tflearn.conv_2d_transpose(fc1,n_features*4,patch_size,maxPool3.get_shape().as_list()[1:4], padding = 'same', activation = 'leaky_relu')
#    print('tconv1:', tconv1.get_shape())
    
#    upsample2 = tflearn.upsample_2d(tconv1,2)
#    print('usamp2:', upsample2.get_shape())
    tconv2 = tflearn.conv_2d_transpose(tfc1,n_features,patch_size,maxPool2.get_shape().as_list()[1:4], padding = 'same', activation = 'leaky_relu', name='deconv2')
    print('tconv2:', tconv2.get_shape())
    
    upsample3 = tflearn.upsample_2d(tconv2,2)
    print('usamp3:', upsample3.get_shape())
    tconv3 = tflearn.conv_2d_transpose(upsample3,n_features,patch_size,maxPool1.get_shape().as_list()[1:4], padding = 'same', activation = 'leaky_relu', name='deconv3')
    print('tconv3:', tconv3.get_shape())
    
    upsample4 = tflearn.upsample_2d(tconv3,2)
    print('usamp4:', upsample4.get_shape())
    tconv4 = tflearn.conv_2d_transpose(upsample4,2,patch_size,y.get_shape().as_list()[1:4], padding = 'same', activation = 'leaky_relu', name='deconv4')
    print('tconv4:', tconv4.get_shape())

    output = tf.nn.softmax(tf.reshape(tconv4,[-1,height,width,2]), name="softmax_tensor")
    print('output:', output.get_shape())

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
                _,current_loss,imgs,preds,l, accuracy = sess.run([optimizer,loss,x, output, y, accr])
                
                elapsed = time.time() - start
                elapsed2 = time.time() - start2
                if i % 20 == 0:
                    current_string = "epoch: " + str(e+1) + " iteration: " + str(i+1) + "current los: " + str(current_loss) + " accuracy: " + str(accuracy) + "\n"
                    log_file.write(current_string)
                    log_file.flush()
                    
            for i in range(imgs.shape[0]):
                filename_input = dir_test +  str(i) + "_input.png"
                filename_labels = dir_test +  str(i) + "_labels.png"
                filename_output = dir_test +  str(i)  + "_output.png"
                cv2.imwrite(filename_input, imgs[i]*max_dist)
                cv2.imwrite(filename_labels, l[i,:,:,0]*255)
                cv2.imwrite(filename_output, preds[i,:,:,0]*255)

            save_path = saver.save(sess, path_model)
                    
         
        coord.request_stop()
        coord.join(threads)
        
        # Save model
        save_path = saver.save(sess, path_model)
        print("Model saved in file: %s" % save_path)

log_file.write("create network\n")
log_file.flush()
x, labels, number_batches = fh.read_tfrecord(dir_records, image_shape, batch_size = batch_size,num_epochs=epochs)
y = tf.reshape(labels, [tf.shape(labels)[0], height, width, 1], name='reshape_labels')
y = tf.concat([y, 1-y],axis = 3)
print("number_batches: ",number_batches)
current_string = "number batches: " + str(number_batches) + "\n"
log_file.write(current_string)
log_file.flush()

output = create_network(x,y)

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = y))
#loss = tf.reduce_mean(tf.pow(y - output, 2))

# optimizer
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Accuracy
corr = tf.equal(tf.argmax(y,3), tf.argmax(output, 3)) 
accr = tf.reduce_mean(tf.cast(corr, "float"))

log_file.write("train\n")
log_file.flush()
train()
log_file.close()
#test_prediction()
