# import necessary packages
import progressbar
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
from tg2.msg import Info

def extract_results(args):
    ''' Extracts results from .bag file given. '''

    # open the bag file
    print('Loading .bag file')
    bag = rosbag.Bag(args['bag'], 'r')

    # info and pose are the topics we want
    info_bagmsgs = bag.read_messages(topics='info')
    pose_bagmsgs = bag.read_messages(topics='pose')

    # load the messages into memory
    print('Loading messages into memory')
    info_bagmsgs = [i for i in info_bagmsgs]
    pose_bagmsgs = [p for p in pose_bagmsgs]

    # last message saved. this will be used to obtain incremental measures
    last_info_msg = None
    last_info_t = None

    # setup the progress bar
    c = 0
    c_max = len(info_bagmsgs)
    bar = progressbar.ProgressBar(max_value=c_max)

    # create pandas dataframe for the results
    results_info = pd.DataFrame(columns=[
        'datetime', 'ddatetime',
        'infos', 'enabled',
        'network', 'method', 'thresh', 'diff',
        'acting', 'moving', 'time',
        'vx', 'vz',
        'jb', 'jpx', 'jpy', 'jcx', 'jcy', 'jox', 'joy'
    ])

    # read messages from the info topic
    print('Processing and saving info results')
    for info_bagmsg in info_bagmsgs:

        # get the info message and it's timestamp
        info_msg = info_bagmsg.message
        info_t = info_bagmsg.timestamp

        # to proceed, we need a last message
        if last_info_msg is not None and last_info_t is not None:

            # extract the relevant data from the messages
            row = {}

            # datetime
            row['datetime'] = datetime.datetime.fromtimestamp(info_t.secs) \
                + datetime.timedelta(microseconds=info_t.nsecs//1000)
            row['ddatetime'] = datetime.datetime.fromtimestamp(last_info_t.secs) \
                + datetime.timedelta(microseconds=last_info_t.nsecs//1000) \
                - row['datetime']

            # information
            row['infos'] = info_msg.infos
            row['enabled'] = info_msg.enabled
            row['network'] = info_msg.network
            row['method'] = info_msg.method
            row['thresh'] = int(info_msg.thresh)
            row['diff'] = info_msg.diff
            row['acting'] = info_msg.acting
            row['moving'] = 1 if (info_msg.vx != 0.0 or info_msg.vz != 0.0) else 0
            row['time'] = info_msg.time
            row['vx'] = info_msg.vx; row['vz'] = info_msg.vz
            row['jb'] = info_msg.jb
            row['jpx'] = info_msg.jpx; row['jpy'] = info_msg.jpy
            row['jcx'] = info_msg.jcx; row['jcy'] = info_msg.jcy
            row['jox'] = info_msg.jox; row['joy'] = info_msg.joy

            # append the row to the results
            results_info = results_info.append(pd.DataFrame(row, index=[0]), ignore_index=True, sort=False)

        # save the message for the next iteration
        last_info_msg = info_msg
        last_info_t = info_t

        # update the progress
        c += 1
        bar.update(c)

    # mark the processing as finished
    bar.finish()

    # setup the progress bar
    c = 0
    c_max = len(pose_bagmsgs)
    bar = progressbar.ProgressBar(max_value=c_max)

    # create pandas dataframe for the results
    results_pose = pd.DataFrame(columns=[
        'datetime',
        'x', 'y', 'th',
        'vx', 'vz'
    ])

    # read messages from the pose topic
    print('Processing and saving pose results')
    for pose_bagmsg in pose_bagmsgs:

        # get the info message and it's timestamp
        pose_msg = pose_bagmsg.message
        pose_t = pose_bagmsg.timestamp

        # extract the relevant data from the messages
        row = {}

        # datetime
        row['datetime'] = datetime.datetime.fromtimestamp(pose_t.secs) \
            + datetime.timedelta(microseconds=pose_t.nsecs//1000)

        # information
        row['x'] = pose_msg.pose.pose.position.x
        row['y'] = pose_msg.pose.pose.position.y
        q = pose_msg.pose.pose.orientation
        row['th'] = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        row['vx'] = pose_msg.twist.twist.linear.x
        row['vz'] = pose_msg.twist.twist.angular.z

        # append the row to the results
        results_pose = results_pose.append(pd.DataFrame(row, index=[0]), ignore_index=True, sort=False)

        # update the progress
        c += 1
        bar.update(c)

    # mark the processing as finished
    bar.finish()

    # save the results to disk
    name = args['dest'].split('_', 1)[0]
    test = args['dest'].split('_', 1)[1]
    res_folder = os.path.join('results', name)
    if not os.path.isdir(res_folder):
        os.makedirs(res_folder)
    if args['append']:
        results_info = pd.concat([pd.read_csv(os.path.join(res_folder, f'{test}.info.csv')), results_info])
        results_pose = pd.concat([pd.read_csv(os.path.join(res_folder, f'{test}.pose.csv')), results_pose])
    results_info.to_csv(path_or_buf=os.path.join(res_folder, f'{test}.info.csv'), header=True, index=False)
    results_pose.to_csv(path_or_buf=os.path.join(res_folder, f'{test}.pose.csv'), header=True, index=False)
        
    # close the bag file
    bag.close()

def main():
    ''' Main function. '''

    # setup the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--bag', type=str,
        help='path to bag file')
    ap.add_argument('-d', '--dest', type=str,
        help='destination folder')
    ap.add_argument('-a', '--append', action='store_true',
        help='append results to existing .csv')
    args = vars(ap.parse_args())

    # print the arguments
    print('File: {}'.format(args['bag']))
    print('Dest: {}'.format(args['dest']))

    # extract the results from bag the file
    extract_results(args)

# if it is the main file
if __name__ == '__main__':
    main()
