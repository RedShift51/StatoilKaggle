#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:09:58 2017

@author: alex
"""

import tensorflow as tf
import numpy as np, matplotlib.pyplot as plt, json
from sklearn.model_selection import cross_val_score, train_test_split

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

all_data = json.load(open('/home/alex/kaggle/icebergs/train.json','r'))

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

data, Y_data = data_processing(all_data)
del all_data

train_nums = np.unique(np.random.randint(low=0,high=len(data),size=int(len(data))))
test_nums = np.array(list(set(list(range(len(data)))) - set(list(train_nums))))

tf.reset_default_graph()
Xtf = tf.placeholder(dtype=tf.float32, shape=[None,75,75,5])
Ytf = tf.placeholder(dtype=tf.float32, shape = [None,2])

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

ans = block('tree', Xtf)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Ytf,logits=ans))

#Xtf = tf.nn.sigmoid(ConvLayer2D('conv1', kernel_shape = [3,3,3,6], num_filters = 6)(Xtf))
print(ans.get_shape().as_list())


init = tf.global_variables_initializer()
optimizer = tf.train.AdamOptimizer().minimize(loss)

loss_list = []
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())#init)
    for i in range(60):
        X,Yvs = iterate_minibatches(data,Y_data, 100)
        Y = np.zeros([100,2])
        Y[np.arange(100), 1-Yvs] = 1
        #X = np.expand_dims(X, axis = 0)
        #print(X.shape, Y.shape)
        for j in range(3):
            _,a=sess.run([optimizer, loss], feed_dict = {Xtf:X,Ytf:Y})
            loss_list.append(a)
        print(i, a)
    saver.save(sess, '/home/alex/kaggle/icebergs/modelv1.ckpt')


plt.plot(loss_list)
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Crossentropy score')
