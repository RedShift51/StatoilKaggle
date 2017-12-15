#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 21:25:15 2017

@author: alex
"""

""" Here we will process big json to big numpy array """

import numpy as np, json

def data_processing(data):
    X, Z = [], []
    for i0,i in enumerate(range(len(data))):
        X1 = np.concatenate([np.reshape(np.array(data[i]['band_1']),(75,75,1)), \
                    np.reshape(np.array(data[i]['band_2']),(75,75,1)), \
                    np.reshape(np.array(data[i]['band_1']),(75,75,1))*\
                    np.reshape(np.array(data[i]['band_2']),(75,75,1)), \
                    np.reshape(np.array(data[i]['band_1']),(75,75,1))*\
                    np.reshape(np.array(data[i]['band_1']),(75,75,1)), \
                    np.reshape(np.array(data[i]['band_2']),(75,75,1))*\
                    np.reshape(np.array(data[i]['band_2']),(75,75,1))], axis = 2)
        X.append(np.expand_dims(X1, axis=0))
        #Y.append(data[i]['is_iceberg'])
        Z.append(data[i]['id'])
        if (i0 + 1) % 1500 == 0:
            X = np.concatenate(X)
            #Y = np.array(Y)
            Z = np.array(Z)
            np.save('/home/alex/kaggle/icebergs/data/'+str(i)+'Xtest.json',X)
            np.save('/home/alex/kaggle/icebergs/data/'+str(i)+'IDtest.json',Z)
            #json.dump({'data':X, 'ID_data':Z},\
            #   open('/home/alex/kaggle/icebergs/data/'+str(i)+'test.json','w'))    
            X, Z = [], []
        print(i0, len(data))
    #X = np.concatenate(X)
    #Y = np.array(Y)
    #Z = np.array(Z)
    #json.dump({'data':X, 'ID_data':Z},\
    #           open('/home/alex/kaggle/icebergs/data/'+str(i)+'test.json','w'))
    X = np.concatenate(X)
    #Y = np.array(Y)
    Z = np.array(Z)       
    np.save('/home/alex/kaggle/icebergs/data/'+str(i)+'Xtest.json',X)
    np.save('/home/alex/kaggle/icebergs/data/'+str(i)+'IDtest.json',Z)        
    return X, Z

all_data = json.load(open('/home/alex/kaggle/icebergs/test.json','r'))

data, ID_data = data_processing(all_data)
del all_data




