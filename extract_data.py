# import necessary packages
import progressbar
# import quaternion
import datetime
import argparse
import pandas as pd
import numpy as np
import os

# import ros packages
import rosbag
import roslib
import rospy

# import message types
from sensor_msgs.msg import Joy #, CompressedImage
from nav_msgs.msg import Odometry
# from p2os_msgs.msg import SonarArray

def extract_data(args):
    ''' Extracts data from .bag file given. '''

    # open the bag file
    print('Loading .bag file')
    bag = rosbag.Bag(args['bag'], 'r')

    # assumption:
    # the matching message will have been received within 150 ms of the pivot message
    # this is used to narrow the search for the matching message
    t_tol = rospy.Duration(secs=0, nsecs=150000000) # 150000000 ns = 150 ms

    # start and end time for the data extraction
    start_at = rospy.Time(secs=int(bag.get_start_time()) + 1, nsecs=0) # 1 second offset for the start
    end_at = rospy.Time(secs=int(bag.get_end_time()) - 1, nsecs=0) # -1 second offset for the end

    # pose is the slowest topic
    pose_bagmsgs = bag.read_messages(topics='pose', start_time=start_at, end_time=end_at)

    # joy is faster and is out of sync, which is why a tolerance is needed
    joy_bagmsgs = bag.read_messages(topics='joy', start_time=start_at-t_tol, end_time=end_at+t_tol)

    # load the messages into memory
    print('Loading messages into memory')
    pose_bagmsgs = [p for p in pose_bagmsgs]
    joy_bagmsgs = [j for j in joy_bagmsgs]

    # last messages saved. this will be used to obtain incremental
    # measures and ground truth for joystick measures
    last_pose_msg = None
    last_joy_msg = None
    second_last_pose_msg = None
    second_last_joy_msg = None

    # setup the progress bar
    c = 0
    c_max = len(pose_bagmsgs)
    bar = progressbar.ProgressBar(max_value=c_max)

    # create pandas dataframe for the data
    data = pd.DataFrame(columns=[
        'dtns', # 'dts', 'dtns'
        # 'dx', 'dy', 'dq.w', 'dq.z'
        'x', 'y', ####
        'vx', 'dvx', 'vz', 'dvz',
        'jx', 'djx', 'jy', 'djy', # 'jb', 'djb'
        'jx_gt', 'jy_gt' # 'jb_gt'
    ])

    # counter for the joy list
    joy_idx = 0

    # read messages from the pose topic
    print('Processing and saving data')
    for pose_bagmsg in pose_bagmsgs:

        # get the pose message and it's timestamp
        pose_msg = pose_bagmsg.message
        pose_t = pose_bagmsg.timestamp # pose_msg.header.stamp # when it was generated/captured

        # read from the joy topic
        joy_msg = None
        min_diff = rospy.Duration(secs=10, nsecs=0) # starting the minimum with 10 seconds
        for joy_bagmsg in joy_bagmsgs[joy_idx:]:
            joy_t = joy_bagmsg.timestamp # joy_bagmsg.message.header.stamp
            diff_t = abs(pose_t-joy_t)
            if diff_t < min_diff or joy_msg is None:
                joy_msg = joy_bagmsg.message
                min_diff = diff_t
                joy_idx += 1
            else:
                break

        # to proceed, we need a last message for pose and joy
        if last_pose_msg is not None and last_joy_msg is not None:

            # and we also need a second last to get the ground truth
            if second_last_pose_msg is not None and second_last_joy_msg is not None:

                # extract the relevant data from the messages
                row = {}

                # datetime
                row['datetime'] = datetime.datetime.fromtimestamp(pose_t.secs) \
                    + datetime.timedelta(microseconds=pose_t.nsecs//1000)

                # time
                ti = last_pose_msg.header.stamp - second_last_pose_msg.header.stamp
                # row['dts'] = ti.secs
                row['dtns'] = ti.nsecs

                # position
                '''
                po = second_last_pose_msg.pose.pose.position
                pn = last_pose_msg.pose.pose.position
                row['dx'] = pn.x - po.x
                row['dy'] = pn.y - po.y
                '''
                row['x'] = last_pose_msg.pose.pose.position.x
                row['y'] = last_pose_msg.pose.pose.position.y

                # orientation
                '''
                qo = second_last_pose_msg.pose.pose.orientation
                qn = last_pose_msg.pose.pose.orientation
                qo = np.quaternion(qo.w, qo.x, qo.y, qo.z)
                qn = np.quaternion(qn.w, qn.x, qn.y, qn.z)
                qi = qn/qo
                row['dq.w'] = qi.w
                row['dq.z'] = qi.z
                '''

                # velocity
                row['vx'] = last_pose_msg.twist.twist.linear.x
                row['dvx'] = last_pose_msg.twist.twist.linear.x - second_last_pose_msg.twist.twist.linear.x
                row['vz'] = last_pose_msg.twist.twist.angular.z
                row['dvz'] = last_pose_msg.twist.twist.angular.z - second_last_pose_msg.twist.twist.angular.z

                # joy (data)
                row['jx'] = last_joy_msg.axes[0]
                row['djx'] = last_joy_msg.axes[0] - second_last_joy_msg.axes[0]
                row['jy'] = last_joy_msg.axes[1]
                row['djy'] = last_joy_msg.axes[1] - second_last_joy_msg.axes[1]
                # row['jb'] = last_joy_msg.buttons[0]
                # row['djb'] = last_joy_msg.buttons[0] - second_last_joy_msg.buttons[0]

                # joy (ground-truth)
                row['jx_gt'] = joy_msg.axes[0]
                row['jy_gt'] = joy_msg.axes[1]
                # row['jb_gt'] = joy_msg.buttons[0]
                
                # append the row to the data
                data = data.append(pd.DataFrame(row, index=[0]), ignore_index=True, sort=False)

            # save the messages for the next iteration
            second_last_pose_msg = last_pose_msg
            second_last_joy_msg = last_joy_msg

        # save the messages for the next iteration
        last_pose_msg = pose_msg
        last_joy_msg = joy_msg

        # update the progress
        c += 1
        bar.update(c)

    # mark the processing as finished
    bar.finish()

    # save the data to disk
    set_folder = os.path.join('data', args['dest'], args['set'])
    if not os.path.isdir(set_folder):
        os.makedirs(set_folder)
    if args['append']:
        data = pd.concat([pd.read_csv(os.path.join(set_folder, 'data.csv')), data])
    data.to_csv(path_or_buf=os.path.join(set_folder, 'data.csv'), header=True, index=False)
        
    # close the bag file
    bag.close()

def main():
    ''' Main function. '''

    # setup the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', type=str,
        help='path to bag file')
    ap.add_argument('--set', type=str,
        help='indicates set (train, val, test)')
    ap.add_argument('--dest', type=str,
        help='destination folder')
    ap.add_argument('--append', action='store_true',
        help='append data to existing .csv')
    args = vars(ap.parse_args())

    # print the arguments
    print('File: {}'.format(args['bag']))
    print('Set: {}'.format(args['set']))

    # extract the data from bag the file
    extract_data(args)

# if it is the main file
if __name__ == '__main__':
    main()
