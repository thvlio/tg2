# 1205e

# timestamps for train-test-validation splitting
start1=1557697952
t0=$(($start1 + 861))
t1=$(($start1 + 1047))
t2=$(($start1 + 1110))
t3=$(($start1 + 1320))
t4=$(($start1 + 1326))
t5=$(($start1 + 1383))
t6=$(($start1 + 1390))
t7=$(($start1 + 1439))

# split the bags
rosbag filter bags/2019-05-12-18-52-31.bag bags/1205e/train1.bag "t.secs >= $(($t0-1)) and t.secs <= $(($t1+1)) and (topic == 'pose' or topic == 'joy')"
rosbag filter bags/2019-05-12-18-52-31.bag bags/1205e/train2.bag "t.secs >= $(($t2-1)) and t.secs <= $(($t3+1)) and (topic == 'pose' or topic == 'joy')"
rosbag filter bags/2019-05-12-18-52-31.bag bags/1205e/test.bag "t.secs >= $(($t4-1)) and t.secs <= $(($t5+1)) and (topic == 'pose' or topic == 'joy')"
rosbag filter bags/2019-05-12-18-52-31.bag bags/1205e/val.bag "t.secs >= $(($t6-1)) and t.secs <= $(($t7+1)) and (topic == 'pose' or topic == 'joy')"

# 1205n

# timestamps for train-test-validation splitting
start1=1557700005
t0=$(($start1 + 182))
t1=$(($start1 + 662))
t2=$(($start1 + 1524))
t3=$(($start1 + 1908))
t4=$(($start1 + 1052))
t5=$(($start1 + 1524))
t6=$(($start1 + 684))
t7=$(($start1 + 1052))

# # split the bags
rosbag filter bags/2019-05-12-19-26-44.bag bags/1205n/train1.bag "t.secs >= $(($t0-1)) and t.secs <= $(($t1+1)) and (topic == 'pose' or topic == 'joy')"
rosbag filter bags/2019-05-12-19-26-44.bag bags/1205n/train2.bag "t.secs >= $(($t2-1)) and t.secs <= $(($t3+1)) and (topic == 'pose' or topic == 'joy')"
rosbag filter bags/2019-05-12-19-26-44.bag bags/1205n/test.bag "t.secs >= $(($t4-1)) and t.secs <= $(($t5+1)) and (topic == 'pose' or topic == 'joy')"
rosbag filter bags/2019-05-12-19-26-44.bag bags/1205n/val.bag "t.secs >= $(($t6-1)) and t.secs <= $(($t7+1)) and (topic == 'pose' or topic == 'joy')"

# 1205p

# timestamps for train-test-validation splitting
start1=1557701939
t0=$(($start1 + 4))
t1=$(($start1 + 212))
t2=$(($start1 + 383))
t3=$(($start1 + 453))
t4=$(($start1 + 313))
t5=$(($start1 + 383))
t6=$(($start1 + 233))
t7=$(($start1 + 313))

# split the bags
rosbag filter bags/2019-05-12-19-58-58.bag bags/1205p/train1.bag "t.secs >= $(($t0-1)) and t.secs <= $(($t1+1)) and (topic == 'pose' or topic == 'joy')"
rosbag filter bags/2019-05-12-19-58-58.bag bags/1205p/train2.bag "t.secs >= $(($t2-1)) and t.secs <= $(($t3+1)) and (topic == 'pose' or topic == 'joy')"
rosbag filter bags/2019-05-12-19-58-58.bag bags/1205p/test.bag "t.secs >= $(($t4-1)) and t.secs <= $(($t5+1)) and (topic == 'pose' or topic == 'joy')"
rosbag filter bags/2019-05-12-19-58-58.bag bags/1205p/val.bag "t.secs >= $(($t6-1)) and t.secs <= $(($t7+1)) and (topic == 'pose' or topic == 'joy')"
