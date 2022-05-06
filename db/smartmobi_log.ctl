LOAD DATA 
INFILE "2021-08-27 17-15-45.csv"
truncate 
INTO TABLE smartmobi_log
FIELDS TERMINATED BY ","  OPTIONALLY ENCLOSED BY '"' 
trailing nullcols(
device_name,
unix_ts, 
attitude_roll,
attitude_pitch, 
attitude_yaw, 
gravity_x, 
gravity_y, 
gravity_z, 
gyroscope_x,
gyroscope_y,
gyroscope_z,
user_acc_x, 
user_acc_y, 
user_acc_z, 
magnetic_field_x,
magnetic_field_y, 
magnetic_field_z, 
rotation_rate_x , 
rotation_rate_y ,
rotation_rate_z,
latitude,
longitude,
altitude, 
speed, 
course,
frame,
key_point_1 char(4000) NULLIF key_point_1=BLANKS,
key_point_2 char(4000) NULLIF key_point_2=BLANKS,
key_point_3 char(4000) NULLIF key_point_3=BLANKS,
key_point_4 char(4000) NULLIF key_point_4=BLANKS,
bounding_box char(4000) NULLIF bounding_box=BLANKS,
right_lane char(4000) NULLIF right_lane=BLANKS,
left_lane char(4000) NULLIF left_lane=BLANKS,
current_lane char(4000) NULLIF current_lane=BLANKS
)
