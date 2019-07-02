#!/usr/bin/env python

# import necessary packages
import tensorflow as tf
import pandas as pd
import numpy as np
import rospy
import time
import os
import keras.backend.tensorflow_backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import model_from_json

# import ROS messages
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tg2.msg import Info

class Node:
    ''' ROS Node. '''

    def __init__(self):
        ''' Create the necessary node stuff. '''

        # data storage
        self.joy_msg = None
        self.prev_joy_msg = None
        self.pose_msg = None
        self.prev_pose_msg = None
        self.jp = None
        self.jc = None
        self.jo = None
        self.diff = 0.0
        self.pred_time = 0.0

        # button mapping config
        self.en_idx = 4
        self.off_idx = 8 
        self.sw_net_idx = 9
        self.sw_method_idx = 0
        self.sw_thresh_idx = 1
        self.marker_idx = 5

        # more config
        self.use_net = False
        self.net_idx = 0
        self.method_idx = 0
        self.thresh_idx = 0
        self.acting = False

        # TF stuff
        print('Configuring TF session')
        config_tf = tf.ConfigProto()
        config_tf.gpu_options.allow_growth = True
        session = tf.Session(config=config_tf)
        K.set_session(session)

        # ROS stuff
        print('Configuring ROS node')
        rospy.init_node('correct', anonymous=True)
        rospy.Subscriber("/joyous", Joy, self.callback_joy)
        rospy.Subscriber("/pose", Odometry, self.callback_pose)
        self.pub = rospy.Publisher('/joy', Joy, queue_size=10)
        self.pub_info = rospy.Publisher('/info', Info, queue_size=10)
        self.rate = rospy.Rate(20)

        # config
        self.model_strings = ['L128-96-64-32_B32_I3', 'L96-64-32_B64_I0']
        self.dataset_names = ['predio', 'oito']
        self.thresholds = [0.05, 0.20]
        self.methods = ['net', 'mean']
        proj_path = os.path.join('/home', 'thulio', 'projects', 'pioneer')
        data_folders = [os.path.join(proj_path, 'data', dn) for dn in self.dataset_names]
        model_folders = [os.path.join(proj_path, 'models', dn, ms) for (dn, ms) in zip(self.dataset_names, self.model_strings)]

        # loading scalers
        print('Loading scalers')
        self.scalers_x = [joblib.load(os.path.join(df, 'scaler_x.pkl')) for df in data_folders]
        self.scalers_y = [joblib.load(os.path.join(df, 'scaler_y.pkl')) for df in data_folders]

        # load model
        print('Loading serialized models')
        self.models = []
        for mf in model_folders:
            with open(os.path.join(mf, 'model.json'), 'r') as json_file:
                model_json = json_file.read()
            self.models.append(model_from_json(model_json))

        # load best model weights
        print('Loading best model weights')
        for mf, model in zip(model_folders, self.models):
            best_model_path = os.path.join(mf, 'best-model.hdf5')
            model.load_weights(best_model_path)

    def publish_info(self):
        ''' Publish the information. '''

        # create the info message
        info_msg = Info()
        info_msg.infos = 'Network: {}, Dataset: {}, Method: {}, Thresh: {}'.format(
            self.model_strings[self.net_idx], self.dataset_names[self.net_idx],
            self.methods[self.method_idx], self.thresholds[self.thresh_idx]
        ) if self.use_net else 'None'
        info_msg.enabled = int(self.use_net)
        info_msg.network = self.net_idx if self.use_net else -1
        info_msg.method = self.method_idx if self.use_net else -1
        info_msg.thresh = self.thresh_idx if self.use_net else -1
        info_msg.diff = self.diff if self.use_net else 0.0
        info_msg.acting = int(self.acting) if self.use_net else -1
        info_msg.moving = 1 if self.pose_msg.twist.twist.linear.x != 0.0 or self.pose_msg.twist.twist.angular.z != 0.0 else 0
        info_msg.marker = int(self.joy_msg.buttons[self.marker_idx])
        info_msg.time = self.pred_time
        info_msg.vx = self.pose_msg.twist.twist.linear.x; info_msg.vz = self.pose_msg.twist.twist.angular.z
        info_msg.jb = self.jp[2]
        info_msg.jpx = self.jp[0]; info_msg.jpy = self.jp[1]
        info_msg.jcx = self.jc[0]; info_msg.jcy = self.jc[1]
        info_msg.jox = self.jo[0]; info_msg.joy = self.jo[1]

        # publish the message
        self.pub_info.publish(info_msg)

        # log some info
        rospy.loginfo(info_msg)

    def callback_pose(self, data):
        ''' Pose callback function. '''

        # save the data
        self.prev_pose_msg = self.pose_msg
        self.pose_msg = data

    def callback_joy(self, data):
        ''' Joystick callback function. '''

        # save the data
        self.prev_joy_msg = self.joy_msg
        self.joy_msg = data

        # check if a switch button was pressed
        if self.prev_joy_msg is not None and self.joy_msg is not None:

            # switch off/on
            if self.joy_msg.buttons[self.off_idx] == 1 and self.prev_joy_msg.buttons[self.off_idx] == 0:
                self.use_net = not self.use_net
                self.publish_info()
            
            # switch between networks
            if self.joy_msg.buttons[self.sw_net_idx] == 1 and self.prev_joy_msg.buttons[self.sw_net_idx] == 0:
                self.net_idx = (self.net_idx + 1) % len(self.models)
                self.publish_info()

            # switch between methods
            if self.joy_msg.buttons[self.sw_method_idx] == 1 and self.prev_joy_msg.buttons[self.sw_method_idx] == 0:
                self.method_idx = (self.method_idx + 1) % len(self.methods)
                self.publish_info()

            # switch between thresholds
            if self.joy_msg.buttons[self.sw_thresh_idx] == 1 and self.prev_joy_msg.buttons[self.sw_thresh_idx] == 0:
                self.thresh_idx = (self.thresh_idx + 1) % len(self.thresholds)
                self.publish_info()

    def run(self):
        ''' Run function. '''

        # run while the node is not shutdown
        while not rospy.is_shutdown():

            # if the necessary data has been read
            if None not in [self.pose_msg, self.prev_pose_msg, self.joy_msg, self.prev_joy_msg]:

                # if not using a network
                if not self.use_net:

                    # publish the original data
                    self.pub.publish(self.joy_msg)

                    # joy (data)
                    jx = self.joy_msg.axes[0]
                    jy = self.joy_msg.axes[1]
                    jb = self.joy_msg.buttons[self.en_idx]

                    # vectors
                    self.jp = np.array([jx, jy, jb])
                    self.jc = self.jp
                    self.jo = self.jp

                    # set flags
                    self.acting = False

                    # publish info
                    self.publish_info()

                # if using a network
                else:

                    # extract the data from the pose and joy messages
                    # dtns, vx, dvx, vz, dvz, jx, djx, jy, djy

                    # time the prediction
                    tstart = time.time()

                    # time
                    ti = self.pose_msg.header.stamp - self.prev_pose_msg.header.stamp
                    # dts = ti.secs
                    dtns = ti.nsecs

                    # velocity
                    vx = self.pose_msg.twist.twist.linear.x
                    dvx = self.pose_msg.twist.twist.linear.x - self.prev_pose_msg.twist.twist.linear.x
                    vz = self.pose_msg.twist.twist.angular.z
                    dvz = self.pose_msg.twist.twist.angular.z - self.prev_pose_msg.twist.twist.angular.z

                    # joy (data)
                    jx = self.joy_msg.axes[0]
                    djx = self.joy_msg.axes[0] - self.prev_joy_msg.axes[0]
                    jy = self.joy_msg.axes[1]
                    djy = self.joy_msg.axes[1] - self.prev_joy_msg.axes[1]
                    jb = self.joy_msg.buttons[self.en_idx]

                    # run the prediction
                    features = [[dtns, vx, dvx, vz, dvz, jx, djx, jy, djy]]
                    features = self.scalers_x[self.net_idx].transform(features).reshape(1, 1, 9)
                    correction = self.models[self.net_idx].predict(features)
                    correction = self.scalers_y[self.net_idx].inverse_transform(correction)
                    cjx = correction[0][0]; cjy = correction[0][1]

                    # vectors
                    self.jp = np.array([jx, jy, jb])
                    if self.methods[self.method_idx] == 'net':
                        self.jc = np.array([cjx, cjy, jb])
                    elif self.methods[self.method_idx] == 'mean':
                        self.jc = np.array([(jx + cjx)/2, (jy + cjy)/2, jb])

                    # check the difference and set flags
                    self.diff = np.linalg.norm((self.jp-self.jc))/np.sqrt(2)/2
                    if self.acting is False:
                        if self.diff >= self.thresholds[self.thresh_idx]:
                            self.jo = self.jc
                            self.acting = True
                        else:
                            self.jo = self.jp
                            self.acting = False
                    else:
                        if self.diff <= self.thresholds[self.thresh_idx]:
                            self.jo = self.jp
                            self.acting = False
                        else:
                            self.jo = self.jc
                            self.acting = True

                    # construct new joy message
                    new_joy_msg = Joy(
                        axes=[self.jo[0], self.jo[1], 0.0, 0.0, 0.0, 0.0, 0.0],
                        buttons=[0, 0, 0, 0, int(self.jo[2]), 0, 0, 0, 0, 0, 0, 0, 0]
                    )

                    # measure time
                    self.pred_time = time.time() - tstart

                    # publish the filtered data
                    self.pub.publish(new_joy_msg)

                    # publish the info
                    self.publish_info()

            # sleep
            self.rate.sleep()

def main():
    ''' Main function. '''
    
    # run the ROS node
    node = Node()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass

# if it is the main file
if __name__ == '__main__':
    main()
