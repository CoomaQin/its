Drop table smartmobi_log;
create table smartmobi_log(
device_name varchar2(200),
unix_ts number, --unix
attitude_roll number, --radians
attitude_pitch number, --radians
attitude_yaw number, --radians
gravity_x number, --G
gravity_y number, --G
gravity_z number, --G
gyroscope_x number,
gyroscope_y number,
gyroscope_z number,
user_acc_x number, --G
user_acc_y number, --G 
user_acc_z number, --G
magnetic_field_x number,--microteslas
magnetic_field_y number, --(microteslas)
magnetic_field_z number, --(microteslas)
rotation_rate_x number, --radians/s
rotation_rate_y number, --radians/s
rotation_rate_z number, --radians/s
latitude number, --(degree)
longitude number, --(degree)
altitude number, --(meter)
speed number, --(m/s)
course number, --(degree)
frame number,
key_point_1  varchar2(4000),
key_point_2  varchar2(4000),
key_point_3  varchar2(4000),
key_point_4  varchar2(4000),
bounding_box  varchar2(4000),
right_lane  varchar2(4000),
left_lane  varchar2(4000),
current_lane  varchar2(4000)
);
select * from smartmobi_log;

create table smartmobi_log_key_point(
device_name varchar2(200),
unix_ts number, --unix
frame number,
key_point_1  varchar2(4000),
key_point_2  varchar2(4000),
key_point_3  varchar2(4000),
key_point_4  varchar2(4000)
);
select * from smartmobi_log_key_point;

create table smartmobi_log_bounding_box(
device_name varchar2(200),
unix_ts number, 
frame number,
bounding_box  varchar2(4000),
vehicle_index number,
vehicle_type number,
left_top_x number,
left_top_y number,
right_bottom_x number,
right_bottom_y number,
veh_loaction  varchar2(4000)
);
select * from smartmobi_log_bounding_box;


--case 1
Select unix_ts, frame, vehicle_index, veh_loaction
from smartmobi_log_bounding_box
where veh_loaction = 'current' and frame <= 450
order by frame asc;

Select vehicle_index
from smartmobi_log_bounding_box
where veh_loaction = 'current' and frame <= 450
Group by vehicle_index;



--case 2
Select count(smartmobi_log.unix_ts) as total_row,  count(smartmobi_log_bounding_box.unix_ts) as veh_2_occur_row
from smartmobi_log
left join smartmobi_log_bounding_box
on smartmobi_log.unix_ts = smartmobi_log_bounding_box.unix_ts
where smartmobi_log_bounding_box.vehicle_index = 2 and smartmobi_log.unix_ts <= 1630098960367;

Select distinct(veh_loaction), vehicle_index
from smartmobi_log_bounding_box
where vehicle_index = 2 and unix_ts <= 1630098960367;




--case 3
create or replace view speed_table as
SELECT smartmobi_log.unix_ts, smartmobi_log.frame, smartmobi_log.speed, 
smartmobi_log_bounding_box.unix_ts AS BD_UNIX_TS, smartmobi_log_bounding_box.frame BD_Frame, smartmobi_log_bounding_box.vehicle_index, smartmobi_log_bounding_box.veh_loaction,
smartmobi_log.speed - (LAG(smartmobi_log.speed) OVER (ORDER BY smartmobi_log_bounding_box.frame)) as delta_speed
FROM smartmobi_log
LEFT JOIN smartmobi_log_bounding_box
ON smartmobi_log.unix_ts = smartmobi_log_bounding_box.unix_ts
where vehicle_index = 2 and smartmobi_log.frame <= 450
ORDER BY smartmobi_log_bounding_box.frame ASC;

select * from speed_table
where delta_speed <>0 or delta_speed is null;
