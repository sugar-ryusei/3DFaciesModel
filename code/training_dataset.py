import numpy as np
import os
import cv2
import glob
from natsort import natsorted

#import time
import math
import tensorflow as tf
import random

class TrainingDataset(object):

    def __init__(self,
                 direc,
                 batch_size,
                 augmentation
                 ):
    
        self.direc = direc
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.data_paths = natsorted(glob.glob(f'{self.direc}/data/*.npy'))
        self.label_paths = natsorted(glob.glob(f'{self.direc}/label/*.png'))
        self.init_len_data = len(self.data_paths)
        self.data_length = len(self.data_paths)

    def load_label(self, label_file):
        image = tf.io.read_file(label_file)
        image = tf.io.decode_image(image)
        image = image[:,:,0:3]
        image = tf.cast(image, tf.int64)
        image = self.to_one_hot(image)
        return image

    def load_data_label(self, data, label):
        return data, label

    def normalize_color(self, image):
        image[:,:,-3:] = (image[:,:,-3:]/127.5) - 1
        return image

    def normalize_z(self, data, n):
        ran_max = max([np.max(i[:,:,n]) for i in data])
        ran_min = min([np.min(i[:,:,n]) for i in data])
        ran = ran_max - ran_min
        mid_ran = (ran_max + ran_min)/2
        data = [self.normalize_temp(i, n, ran, mid_ran) for i in data]
        return data

    def normalize_temp(self, data, n, ran, mid_ran):
        data[:,:,n] = (data[:,:,n]/ran*2) - mid_ran/ran*2
        return data

    def im2vec(self, filename):
        im = cv2.imread(filename)
        im = np.dot(im[:,:,0], 1000000) + np.dot(im[:,:,1], 1000) + im[:,:,2]
        im = im.ravel(order='F')
        return im

    def set_classes(self):
        tmp = [self.im2vec(i) for i in self.label_paths]
        im = np.concatenate(tmp)
        
        im_u, con = np.unique(im, return_counts=True)
        num_classes = len(im_u)

        self.palette = im_u.astype(np.uint32)
        self.classes = num_classes

    @tf.function()
    def to_one_hot(self, image):
        a = image[:,:,2] * (tf.ones(tf.shape(image[:,:,0]), tf.int64)*1000000)
        b = image[:,:,1] * (tf.ones(tf.shape(image[:,:,1]), tf.int64)*1000)
        c = image[:,:,0] * tf.ones(tf.shape(image[:,:,2]), tf.int64)
        image = a + b + c
        image = tf.cast(image, tf.int64)

        x = []
        for i,p in enumerate(self.palette):
            wh = tf.where(image==p, 1, 0)
            wh = tf.cast(wh, dtype=tf.float32)
            x.append(wh)
        output = tf.stack(x, axis=-1)
        return output

    def load_dataset(self, shuffle=True):
        data = [np.load(data_file).astype(np.float32) for data_file in self.data_paths]
        data = self.normalize_z(data, 0)
        data = [self.normalize_color(i) for i in data]
        ##
        labels = [self.load_label(label_file) for label_file in self.label_paths]
        ds = tf.data.Dataset.from_tensor_slices((data, labels))
        ##
        ds = ds.map(
            self.load_data_label,
            num_parallel_calls=tf.data.AUTOTUNE
            )
        if shuffle:
            ds = ds.shuffle(buffer_size=len(self.data_paths))
        if self.augmentation:
            ds = ds.map(
                lambda x, y: self.augment((x, y)),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    @tf.function()
    def augment(self, data_label):
        data, label = data_label

        lot = tf.random.uniform(shape=[1])
        if lot>0.5:
            data = tf.image.flip_left_right(data)
            label = tf.image.flip_left_right(label)

        lot = tf.random.uniform(shape=[1])
        if lot>0.5:
            data = tf.image.flip_up_down(data)
            label = tf.image.flip_up_down(label)

        lot = tf.random.uniform(shape=[1])
        if lot>0.5:
            data = self.noise(data)
            label = label

        lot = tf.random.uniform(shape=[1])
        if lot>0.5:
            rotate_time = random.randint(1,3)
            data = tf.image.rot90(data, k=rotate_time)
            label = tf.image.rot90(label, k=rotate_time)

        return data, label

    def noise(self, data):
        no = tf.random.normal(shape=tf.shape(data), mean=0.0, stddev=0.2, dtype=tf.float32)
        noise_data = tf.add(data, no)
        return noise_data

def main():
    td = TrainingDataset()
    td.set_classes()
    x = td.load_dataset()
    x = iter(x)
    print(next(x))

if __name__ == '__main__':
    main()