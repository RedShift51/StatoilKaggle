#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 09:56:08 2017

@author: alex
"""

import tensorflow as tf, numpy as np

class ConvLayer2D():
    def __init__(self, scope, kernel_shape = [2,2,3,3], num_filters = 1, \
                 padding = 'SAME', strides = [1,1,1,1], nonlin = 'relu'):
        self.weights = []
        self.padding = padding
        self.nonlin = nonlin
        self.strides = strides
        with tf.variable_scope(scope):
            for i in range(num_filters):
                with tf.variable_scope(str(i)):
                    #print(i)
                    self.weights.append(\
                            [tf.get_variable('weights', shape = kernel_shape, \
                            initializer = tf.glorot_normal_initializer()), \
                            tf.get_variable('bias', shape = kernel_shape[-1], \
                            initializer = tf.zeros_initializer())])
                    #print(self.weights)
    
    def transform(self, x):
        ans = tf.concat([tf.nn.conv2d(x, self.weights[i][0], padding = self.padding, \
                        strides = self.strides) + self.weights[i][1] \
                        for i in range(len(self.weights))], axis = 3)

        if self.nonlin == 'relu':
            return tf.nn.relu(ans)
        elif self.nonlin == 'sigmoid':
            return tf.nn.sigmoid(ans)
        else:
            return tf.nn.relu(ans)

    def __call__(self, x):
        return self.transform(x)

def block(scope, Xtf):
    Xtf = ConvLayer2D('conv1', kernel_shape=[2,2,5,1], num_filters=16)(Xtf)
    Xtf = tf.nn.max_pool(Xtf, ksize = [1,2,2,1], strides = [1,2,2,1], \
                    padding = 'SAME')
    Xtf = tf.contrib.layers.batch_norm(Xtf)

    Xtf = ConvLayer2D('conv2', kernel_shape=[2,2,16,1], num_filters=16)(Xtf)
    Xtf = tf.nn.max_pool(Xtf, ksize = [1,2,2,1], strides = [1,2,2,1], \
                    padding = 'SAME')
    Xtf = tf.contrib.layers.batch_norm(Xtf)

    Xtf = ConvLayer2D('conv3', kernel_shape=[2,2,16,1], num_filters=16)(Xtf)
    Xtf = tf.nn.max_pool(Xtf, ksize = [1,2,2,1], strides = [1,2,2,1], \
                    padding = 'SAME')
    Xtf = tf.contrib.layers.batch_norm(Xtf)
    Xtf = tf.contrib.layers.flatten(Xtf)
    Xtf = tf.contrib.layers.fully_connected(Xtf, num_outputs = 100)
    Xtf = tf.nn.relu(Xtf)
    ans = tf.contrib.layers.fully_connected(Xtf, num_outputs = 2, \
                                            activation_fn = None)
    return ans

def iterate_minibatches(X, Y, vol = 5):
    vol = np.random.randint(low=0, high=len(X), size=vol)
    return X[vol,:,:,:], np.array(Y[vol])

def data_processing(data):
    X, Y = [],[]
    for i in range(len(data)):
        X1 = np.concatenate([np.reshape(np.array(data[i]['band_1']),(75,75,1)), \
                    np.reshape(np.array(data[i]['band_2']),(75,75,1)), \
                    np.reshape(np.array(data[i]['band_1']),(75,75,1))*\
                    np.reshape(np.array(data[i]['band_2']),(75,75,1)), \
                    np.reshape(np.array(data[i]['band_1']),(75,75,1))*\
                    np.reshape(np.array(data[i]['band_1']),(75,75,1)), \
                    np.reshape(np.array(data[i]['band_2']),(75,75,1))*\
                    np.reshape(np.array(data[i]['band_2']),(75,75,1))], axis = 2)
        X.append(np.expand_dims(X1, axis=0))
        Y.append(data[i]['is_iceberg'])
    X = np.concatenate(X)
    Y = np.array(Y)
    return X, Y

