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
table = table.drop(['Unnamed: 0', 'device_name','Unnamed: 0.1'], axis=1)
table.insert(0, 'device_name', device)
table.to_csv(table_name + '.csv', index=False)

kp_table = pd.DataFrame(table, columns = ['device_name', 'timestamp', 'frame'])

kp_table.insert(3, 'key point 1', tf['key point 0'])
kp_table.insert(4, 'key point 2', tf['key point 1'])
kp_table.insert(5, 'key point 3', tf['key point 2'])
kp_table.insert(6, 'key point 4', tf['key point 3'])
kp_table.to_csv(table_name + ' key point.csv', index=False)


ob_table = pd.DataFrame(table, columns = ['device_name', 'timestamp', 'frame','bounding box', 'right', 'left', 'current'])
ob_table['bounding box'] = ob_table['bounding box'].map(lambda x: str(x)[:-1])
ob_table['bounding box'] = ob_table['bounding box'].map(lambda x: str(x)[1:])

df_split_row = ob_table.drop('bounding box', axis=1).join(
    ob_table['bounding box'].str.split('],', expand=True).stack().reset_index(level=1, drop=True).rename('bounding box'))
df_split_row['bounding box'] = df_split_row['bounding box'].str.split('[').str[1]
df_split_row['bounding box'] = df_split_row['bounding box'].str.split(']').str[0]
df_split_row['vehicle index'] = df_split_row['bounding box'].str.split(',').str[0]
df_split_row['vehicle type'] = df_split_row['bounding box'].str.split(',').str[1]
df_split_row['left_top_x'] = df_split_row['bounding box'].str.split(',').str[2]
df_split_row['left_top_y'] = df_split_row['bounding box'].str.split(',').str[3]
df_split_row['right_bottom_x'] = df_split_row['bounding box'].str.split(',').str[4]
df_split_row['right_bottom_y'] = df_split_row['bounding box'].str.split(',').str[5]
df_split_row = df_split_row.reset_index()

df_split_row['veh location'] = None
for i in range(len(df_split_row)):
    if str(df_split_row['vehicle index'][i]) in str(df_split_row['left'][i]):
        df_split_row['veh location'][i] = 'left'
    if str(df_split_row['vehicle index'][i]) in str(df_split_row['right'][i]):
        df_split_row['veh location'][i] = 'right'
    if str(df_split_row['vehicle index'][i]) in str(df_split_row['current'][i]):
        df_split_row['veh location'][i] = 'current'

df_split_row = df_split_row.drop(['index','left', 'right','current'], axis=1)
df_split_row.to_csv(table_name + ' bounding box.csv', index=False)