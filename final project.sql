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