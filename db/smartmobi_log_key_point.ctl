LOAD DATA 
INFILE "2021-08-27 17-15-45 key point.csv"
truncate 
INTO TABLE smartmobi_log_key_point
FIELDS TERMINATED BY ","  OPTIONALLY ENCLOSED BY '"' 
trailing nullcols(
device_name,
unix_ts,
frame,
key_point_1 char(4000) NULLIF key_point_1=BLANKS,
key_point_2 char(4000) NULLIF key_point_2=BLANKS,
key_point_3 char(4000) NULLIF key_point_3=BLANKS,
key_point_4 char(4000) NULLIF key_point_4=BLANKS)
