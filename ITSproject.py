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
number_of_lane = len(max(kp, key = len))
timestamp = pd.read_csv('Timestamp.csv')
a = np.arange(1630098945367,1630099005334,1/3*100)
#df = pd.DataFrame({'frame':a, 'key point':kp, 'bounding box': od})

#print(kp[0][0])

timestamp['frame'] = 0
timestamp['bounding box'] = None
for j in range(number_of_lane):
    name = 'key point ' + str(j)
    timestamp[name] = None

for i in range(len(timestamp)):
    index = np.argmin(abs(timestamp['timestamp'][i] - a))
    timestamp['frame'][i] = index
#    timestamp['key point'][i] = df['key point'][index]
    timestamp['bounding box'][i] = od[index]
    for j in range(number_of_lane):
        name = 'key point ' + str(j)
        try:
            timestamp[name][i] = kp[index][j]
        except Exception:
            timestamp[name][i] = np.NaN

timestamp.to_csv('Timestamp_Frame.csv')