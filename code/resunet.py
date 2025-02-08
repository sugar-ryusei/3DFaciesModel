import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

import tensorflow as tf

import os
import glob


class ResUNet(object):
    """docstring for ResUNet"""
    def __init__(self, input_shape, classes):
        #super(ResUNet, self).__init__()
        self.input_shape = input_shape
        self.classes = classes
        self.initializer = 'he_normal'
        self.regularizer = tf.keras.regularizers.l2(1e-4)

# Deep ResU-Net preactivation ----------------------------------------------------
    def stem(self, inputs, filters):
        sk = tf.keras.layers.Conv2D(filters, (3, 3), strides=1, padding='same',
            use_bias=True,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer)(inputs)
        sk = tf.keras.layers.BatchNormalization()(sk)

        x = tf.keras.layers.Conv2D(filters, (3, 3), strides=1, padding='same',
            use_bias=True,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters, (3, 3), strides=1, padding='same',
            use_bias=True,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Add()([x, sk])
        x = tf.keras.layers.Activation('relu')(x)

        return x

    def resunit(self, inputs, filters, skip=True, name=""):
        x = tf.keras.layers.Conv2D(filters//4, (1, 1), strides=1, padding='same',
            use_bias=True,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            name=name+"_1_conv")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters//4, (3, 3), strides=1, padding='same',
            use_bias=True,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            name=name+"_2_conv")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters, (1, 1), strides=1, padding='same',
            use_bias=True,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            name=name+"_3_conv")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if skip:
            sk = tf.keras.layers.Conv2D(filters, (1, 1), strides=1, 
                padding='same',
                use_bias=True,
                kernel_initializer=self.initializer,
                kernel_regularizer=self.regularizer,
                name=name+"_0_conv")(inputs)
            sk = tf.keras.layers.BatchNormalization()(sk)
            x = tf.keras.layers.Add()([x, sk])
            x = tf.keras.layers.Activation('relu')(x)

        return x

    def bridge(self, inputs, filters):
        x = tf.keras.layers.Conv2D(filters//2, (1, 1), strides=1, 
            padding='same',
            use_bias=True,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters//2, (3, 3), strides=1, 
            padding='same',
            use_bias=True,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)        

        x = tf.keras.layers.Conv2D(filters, (1, 1), strides=1,
            padding='same',
            use_bias=True,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        sk = tf.keras.layers.Conv2D(filters, (1, 1), strides=1,
            padding='same',
            use_bias=True,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer)(inputs)
        sk = tf.keras.layers.BatchNormalization()(sk)

        x = tf.keras.layers.Add()([x, sk])

        x = tf.keras.layers.Activation('relu')(x)

        return x

    def architecture(self, n1=64):
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]

        inputs = tf.keras.layers.Input(shape=self.input_shape)

        # encoder
        e1 = self.stem(inputs, filters=filters[0])

        e2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(e1)
        e2 = self.resunit(e2, filters=filters[2], skip=True, name="e_2")

        e3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(e2)
        e3 = self.resunit(e3, filters=filters[3], skip=True, name="e_3")

        e4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(e3)
        e4 = self.resunit(e4, filters=filters[4], skip=True, name="e_4")

        e5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(e4)
        e5 = self.resunit(e5, filters=filters[5], skip=True, name="e_5")

        # bridge
        br = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(e5)
        br = self.bridge(br, filters=filters[5])

        # decoder
        d5 = tf.keras.layers.UpSampling2D(size=(2, 2))(br)
        d5 = tf.keras.layers.concatenate([d5, e5], axis=-1)
        d5 = self.resunit(d5, filters=filters[5], skip=True, name="d_5")

        d4 = tf.keras.layers.UpSampling2D(size=(2, 2))(d5)
        d4 = tf.keras.layers.concatenate([d4, e4], axis=-1)
        d4 = self.resunit(d4, filters=filters[4], skip=True, name="d_4")

        d3 = tf.keras.layers.UpSampling2D(size=(2, 2))(d4)
        d3 = tf.keras.layers.concatenate([d3, e3], axis=-1)
        d3 = self.resunit(d3, filters=filters[3], skip=True, name="d_3")

        d2 = tf.keras.layers.UpSampling2D(size=(2, 2))(d3)
        d2 = tf.keras.layers.concatenate([d2, e2], axis=-1)
        d2 = self.resunit(d2, filters=filters[2], skip=True, name="d_2")

        d1 = tf.keras.layers.UpSampling2D(size=(2, 2))(d2)
        d1 = tf.keras.layers.concatenate([d1, e1], axis=-1)
        d1 = self.resunit(d1, filters=filters[1], skip=True, name="d_1")

        x = tf.keras.layers.Conv2D(self.classes, (1, 1), strides=1, 
            padding='same', 
            kernel_initializer=self.initializer, 
            kernel_regularizer=self.regularizer)(d1)
        outputs = tf.keras.layers.Activation('softmax')(x)

        #transfer learning based on imagenet
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        resnet = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
        
        weights = resnet.get_layer('conv2_block1_1_conv').get_weights()
        model.get_layer('e_2_1_conv').set_weights(weights)
        weights = resnet.get_layer('conv2_block1_2_conv').get_weights()
        model.get_layer('e_2_2_conv').set_weights(weights)
        weights = resnet.get_layer('conv2_block1_3_conv').get_weights()
        model.get_layer('e_2_3_conv').set_weights(weights)
        weights = resnet.get_layer('conv2_block1_0_conv').get_weights()
        model.get_layer('e_2_0_conv').set_weights(weights)
        weights = resnet.get_layer('conv3_block1_1_conv').get_weights()
        model.get_layer('e_3_1_conv').set_weights(weights)
        weights = resnet.get_layer('conv3_block1_2_conv').get_weights()
        model.get_layer('e_3_2_conv').set_weights(weights)
        weights = resnet.get_layer('conv3_block1_3_conv').get_weights()
        model.get_layer('e_3_3_conv').set_weights(weights)
        weights = resnet.get_layer('conv3_block1_0_conv').get_weights()
        model.get_layer('e_3_0_conv').set_weights(weights)
        weights = resnet.get_layer('conv4_block1_1_conv').get_weights()
        model.get_layer('e_4_1_conv').set_weights(weights)
        weights = resnet.get_layer('conv4_block1_2_conv').get_weights()
        model.get_layer('e_4_2_conv').set_weights(weights)
        weights = resnet.get_layer('conv4_block1_3_conv').get_weights()
        model.get_layer('e_4_3_conv').set_weights(weights)
        weights = resnet.get_layer('conv4_block1_0_conv').get_weights()
        model.get_layer('e_4_0_conv').set_weights(weights)
        weights = resnet.get_layer('conv5_block1_1_conv').get_weights()
        model.get_layer('e_5_1_conv').set_weights(weights)
        weights = resnet.get_layer('conv5_block1_2_conv').get_weights()
        model.get_layer('e_5_2_conv').set_weights(weights)
        weights = resnet.get_layer('conv5_block1_3_conv').get_weights()
        model.get_layer('e_5_3_conv').set_weights(weights)
        weights = resnet.get_layer('conv5_block1_0_conv').get_weights()
        model.get_layer('e_5_0_conv').set_weights(weights)
        
        return model