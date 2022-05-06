LOAD DATA 
INFILE "2021-08-27 17-15-45 bounding box.csv"
truncate 
INTO TABLE smartmobi_log_bounding_box
FIELDS TERMINATED BY ","  OPTIONALLY ENCLOSED BY '"' 
trailing nullcols(
device_name,
unix_ts , 
frame,
bounding_box  char(4000) NULLIF bounding_box=BLANKS,
vehicle_index NULLIF vehicle_index=BLANKS,
vehicle_type NULLIF vehicle_type=BLANKS,
left_top_x NULLIF left_top_x=BLANKS,
left_top_y NULLIF left_top_y=BLANKS,
right_bottom_x NULLIF right_bottom_x=BLANKS,
right_bottom_y NULLIF right_bottom_y=BLANKS,
veh_loaction char(4000) NULLIF veh_loaction=BLANKS)
