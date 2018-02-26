# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:17:35 2018

@author: schlichting
"""

import numpy as np
import os, glob, re
import file_handler as fh
import tensorflow as tf

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def get_files_folder(folder):
    filenames = glob.glob(os.path.join(folder, '*.txt'))
    filenames.sort(key = natural_keys)
    
    return np.array(filenames)

def get_velodyne_training(filename):
    res_az = 0.4*100 # 0.4 deg times 100
#    max_dist = 40
    img_dist = np.zeros([900,16])
    img_int = np.zeros([900,16])
    labels = np.zeros([900,16])
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            values = [float(x) for x in line.strip().split(',')]
            if len(values) < 1:
                continue
            row = np.mod(900-int(values[5]/res_az)+300,900)
            col = 15-int((values[6]+15)/2)
            img_dist[row,col] = values[4] # distance
            img_int[row,col] = values[3] # intensity
            labels[row,col] = values[7] # static (0) or dynamic (1)
    return np.array(img_dist), np.array(img_int), np.array(labels)

def get_velodyne_imgs(filename):
    res_az = 0.4*100 # 0.4 deg times 100
    max_dist = 40
    img_dist = np.zeros([900,16,3])
    img_int = np.zeros([900,16,3])
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            values = [float(x) for x in line.strip().split(',')]
            if len(values) < 1:
                continue
            row = np.mod(900-int(values[5]/res_az)+300,900)
            col = 15-int((values[6]+15)/2)
            dyn = bool(values[7])
            if dyn:
                img_dist[row,col,0] = 0 # distance
                img_dist[row,col,1] = 0 # distance
                img_dist[row,col,2] = 255 # distance
                img_int[row,col,0] = 0 # distance
                img_int[row,col,1] = 0 # distance
                img_int[row,col,2] = 255 # distance
            else:
                img_dist[row,col,0:3] = values[4]/max_dist # distance
                img_int[row,col,0:3] = values[3]/255 # intensity
    return img_dist, img_int

#velodyne_scans_folder = "../20180201_icsens_innenstadt/points/scansDynamic/"
velodyne_scans_folder = "X:/Proc/Velodyne_Puck/20180201_icsens_innenstadt/scans_dynamic/"
velodyne_img_folder = "X:/Proc/Velodyne_Puck/20180201_icsens_innenstadt/imgs/"
filenames_velodyne = get_files_folder(velodyne_scans_folder)

record_path = "X:/Proc/Velodyne_Puck/20180201_icsens_innenstadt/imgs/records/train.tfrecord"

images_dist = []
images_int = []
labels = []

# open the TFRecords file
writer = tf.python_io.TFRecordWriter(record_path)

#for i in range(filenames_velodyne.shape[0]):
for i in range(50):
    img_dist, img_int, img_labels = get_velodyne_training(filenames_velodyne[i])
    images_dist.append(img_dist)
    images_int.append(img_int)
    labels.append(img_labels)
    if len(images_dist) > 10-1:
        images_dist = np.array(images_dist)
        images_int = np.array(images_int)
        labels = np.array(labels)
        fh.write_tfrecord(writer,images_dist,labels)
        images_dist = []
        images_int = []
        labels = []
        
images_dist = np.array(images_dist)
images_int = np.array(images_int)
labels = np.array(labels)
fh.write_tfrecord(writer,images_dist,labels)
writer.close()

#    string_dist = '../20180201_icsens_innenstadt/imgs/scan_dist_' + str(i) + '.jpg'
#    string_int = '../20180201_icsens_innenstadt/imgs/scan_int_' + str(i) + '.jpg'
#    cv2.imwrite(string_dist,img_dist*255)
#    cv2.imwrite(string_int,img_int*255)