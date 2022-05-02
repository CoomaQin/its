# -*- coding: utf-8 -*-
"""
Created on Mon May  2 09:51:58 2022

@author: 49762
"""

import pandas as pd
import datetime
import numpy as np


csv_path = "./its.csv"
table = pd.read_csv(csv_path)
tf_path = "./Timestamp_Frame.csv"
tf = pd.read_csv(tf_path)



time = str(table['timestamp'][0])[:10]
table_name = datetime.datetime.fromtimestamp(int(time))
table_name = table_name.strftime('%Y-%m-%d %H-%M-%S')
table['device_name'] = table_name
device = table['device_name']
table = table.drop(['Unnamed: 0', 'device_name','Unnamed: 0.1', 'key point'], axis=1)
table.insert(0, 'device_name', device)
table.insert(26, 'key point 1', tf['key point 0'])
table.insert(27, 'key point 2', tf['key point 1'])
table.insert(28, 'key point 3', tf['key point 2'])
table.insert(29, 'key point 4', tf['key point 3'])


table.to_csv(table_name + '.csv', index=False)