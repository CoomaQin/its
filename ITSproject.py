# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 20:26:50 2022

@author: 49762
"""
import numpy as np
import pandas as pd


kp, od = None, None
with open('kp.npy', 'rb') as f:
    kp = np.load(f, allow_pickle = True)
with open('od_info.npy', 'rb') as f:
    od = np.load(f, allow_pickle = True)

timestamp = pd.read_csv('Timestamp.csv')


a = np.arange(1630098945367,1630099005334,1/3*100)

df = pd.DataFrame({'frame':a, 'key point':kp, 'bounding box': od})

timestamp['frame'] = 0
timestamp['key point'] = None
timestamp['bounding box'] = None



for i in range(len(timestamp)):
    index = np.argmin(abs(timestamp['timestamp'][i] - a))
    timestamp['frame'][i] = index
    timestamp['key point'][i] = df['key point'][index]
    timestamp['bounding box'][i] = df['bounding box'][index]


timestamp.to_csv('Timestamp_Frame.csv')