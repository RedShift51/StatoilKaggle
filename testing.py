#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 00:05:26 2017

@author: alex
"""

import tensorflow as tf
import numpy as np, pandas as pd, os
from storage import block, ConvLayer2D

select_list = os.listdir('/home/alex/kaggle/icebergs/data')


tf.reset_default_graph()
Xtf = tf.placeholder(dtype=tf.float32, shape=[None,75,75,5])
Ytf = tf.placeholder(dtype=tf.float32, shape = [None,2])

ans = block('tree', Xtf)
#print(ans.get_shape().as_list())
#ans = ans

X = np.load(open('/home/alex/kaggle/icebergs/data/4499Xtest.json.npy','rb'))
ID = np.load(open('/home/alex/kaggle/icebergs/data/4499IDtest.json.npy','rb'))

predicted = {}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph\
        ('/home/alex/kaggle/icebergs/modelv1.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint\
              ('/home/alex/kaggle/icebergs/.'))
    #print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    for i in range(len(X)):
        mark = sess.run(tf.argmax(ans, 1), feed_dict = {Xtf:X[i:i+1,:,:,:]})
        predicted[ID[i]] = mark
        print(len(predicted))





