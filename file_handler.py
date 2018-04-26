# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:45:48 2017

@author: schlichting
"""

import numpy as np
import os, glob, sys, re
import tensorflow as tf

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
  
  
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  
def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def split_image(img, labels):
    new_img = np.reshape(img,[5, 31, img.shape[1]])
    new_labels = np.reshape(labels,[5, 31, img.shape[1]])
    
    return new_img, new_labels

def files_in_folder(directory):
    filenames = glob.glob(os.path.join(directory, '*.*'))
    filenames.sort(key = natural_keys)
    return np.array(filenames)
  
def write_tfrecord(writer, input, output):
    for i in range(input.shape[0]):
        
        # print how many images are saved every 100 images
#        if not i % 20:
#            print('Train data: {}/{}'.format(i, input.shape[0]))
#            sys.stdout.flush()
        
        img1 = input[i]
        img2 = output[i]
        
        feature = {'train/input': _bytes_feature(tf.compat.as_bytes(img1.tostring())),
                   'train/output': _bytes_feature(tf.compat.as_bytes(img2.tostring()))}

        example= tf.train.Example(features=tf.train.Features(feature = feature))
        writer.write(example.SerializeToString())

def read_tfrecord(folder, image_shape, batch_size = 100, num_epochs = 100):
    feature = {'train/input': tf.FixedLenFeature([], tf.string),
               'train/output': tf.FixedLenFeature([], tf.string)}
        
    info_filenames = glob.glob(os.path.join(folder, '*.info'))
    number_batches = 0
    for filename in info_filenames:
        with open(filename,'r') as f:
            l = f.readline()
            number_batches += int(l)
                                             
    path_records = folder + "*.tfrecord"
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(path_records), num_epochs=num_epochs)
    
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/input'], tf.float64)
    image = tf.to_float(image)
    
    # Reshape image data into the original shape
    image = tf.reshape(image, image_shape, name='reshape_image')
    image = tf.to_float(image)
    
    # Cast label data into bool
    label = tf.decode_raw(features['train/output'], tf.float64)
    #label = tf.multiply(label-1,-1)
    label = tf.reshape(label, image_shape, name='reshape_label')
    label = tf.to_float(label)

    # Creates batches by randomly shuffling tensors    
    images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=2147483647, allow_smaller_final_batch = True, num_threads=1, min_after_dequeue=2147483646)
    
    return images, labels, number_batches

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]
    
def get_velodyne_img(filename):
    res_az = 0.4*100 # 0.4 deg times 100
    max_dist = 40
    img_dist = np.zeros([900,16])
    img_int = np.zeros([900,16])
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            values = [float(x) for x in line.strip().split(',')]
            if len(values) < 1:
                continue
            row = np.mod(900-int(values[5]/res_az)+225,900)
            col = 15-int((values[6]+15)/2)
            img_dist[row,col] = values[4] # distance
            img_int[row,col] = values[3] # intensity
    return img_dist, img_int
    
def read_images(directory, max_dist = 80, max_images = -1):
    
    filenames = glob.glob(os.path.join(directory, '*.txt'))
    filenames.sort(key = natural_keys)
    trajectory = []
    images = []
    counter = 0
    for filename in filenames:
        (ranges, _, _, traj) = read_data_traj(filename, skip_values = 999999, max_dist = max_dist, asNumpyArray=True)
        if len(ranges) < 1:
            continue
        images.append(ranges/max_dist)
        print(filename, ranges.shape, traj.shape)
        trajectory.append(traj)
        if max_images >= 0:
            counter += 1
            if counter > max_images:
                break
    images = np.array(images)
    trajectory = np.array(trajectory)
    return images, trajectory
  
def read_labels(directory, max_images = -1):
    
    filenames = glob.glob(os.path.join(directory, '*.txt'))
    filenames.sort(key = natural_keys)
    images_labels = []
    counter = 0
    for filename in filenames:
        print(filename)
        with open(filename, 'r') as f:    
            labels = []
            for line in f:
                values = [int(x) for x in line.strip().split(',')]
                if len(values) < 1:
                    continue
                values = values[0:1160]
                labels.append(values)
            if len(labels) < 1:
                continue
            images_labels.append(np.array(labels))
            if max_images >= 0:
                counter += 1
                if counter > max_images:
                    break
    images_labels = np.multiply(np.array(images_labels)-1,-1)
    return images_labels
