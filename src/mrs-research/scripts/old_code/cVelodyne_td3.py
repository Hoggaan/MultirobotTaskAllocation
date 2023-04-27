from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from squaternion import Quaternion
from sensor_msgs.msg import LaserScan, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry

import time
import subprocess
import rospy
from numpy import inf
import numpy as np
import random
import math
import os

class GazeboEnv:
    '''Supper Class for all Gazebo Environments'''
    metadata = {'render.modes':['human']}
    def __init__(self, launchfile, hight, width, nchannels):
        self.odomX = 0
        self.odomY = 0

        self.goalX = 1
        self.goalY = 0

        self.upper = 5.0
        self.lower = -5.0
        self.velodyne_data = np.ones(20) * 10
        self.last_laser = None
        self.last_odom = None

        self.set_self_state = ModelState()
        self.set_self_state.model_name = 'burger'
        self.set_self_state.pose.position.x = 0.
        self.set_self_state.pose.position.y = 0.
        self.set_self_state.pose.position.z = 0.
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0
        self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
        self.gaps = [[-1.6, -1.57 + 3.14 / 20]] # Unidentified
        for m in range(19):
            self.gaps.append([self.gaps[m][1], self.gaps[m][1] + 3.14 / 20])
        self.gaps[-1][-1] += 0.03 # Not get it
    
        port = '11311'
        subprocess.Popen(['roscore', '-p', port])

        print('Roscore Launched!')

        # Launch the simulation with the given launchfile name
        rospy.init_node('gym', anonymous = True)
        if launchfile.startswith('/'):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', launchfile)
        if not os.path.exists(fullpath):
            raise IOError('file '+fullpath+' Doese not exist!')
        
        subprocess.Popen(['roslaunch', '-p', port, fullpath])
        print('Gazebo launched!')

        self.gzclient_pid = 0

        # Setup the ros puplishers and subscribers
        self.vel_pub = rospy.publisher('/cmd_vel', Twist, queue_size=1)
        self.set_state = rospy.publisher('gazebo/set_model_state', ModelState, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        topic = 'vis_mark_array'
        self.publisher = rospy.Publisher(topic, MarkerArray, queue_size=3)
        topic2 = 'vis_mark_array2'
        self.publisher2 = rospy.Publisher(topic2, MarkerArray, queue_size=1)
        topic3 = 'vis_mark_array3'
        self.publisher3 = rospy.Publisher(topic3, MarkerArray, queue_size=1)
        topic4 = 'vis_mark_array4'
        self.publisher4 = rospy.Publisher(topic4, MarkerArray, queue_size=1)

        self.velodyne = rospy.Subscriber('/velodyne_points', PointCloud2, self.velodyne_callback, queue_size=1)
        self.laser = rospy.Subscriber('/scan', LaserScan, self.laser_callback, queue_size=1)
        self.odom = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)
        
    # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # range as state representation
    def velodyne_callback(self,v):
        data = list(pc2.read_points(v, skip_nans=False, field_names = ('x', 'y', 'z'))) # What is the input. Then check Pc2
        self.velodyne_data = np.ones(20) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0 # Not Get It. 
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1,2) + math.pow(0,2))
                beta = math.acos(dot / (mag1, mag2)) * np.sign(data[i][1]) # * -1 # Not get it at all
                dist = math.sqrt(data[i][0]**2 + data[i][1]**2 + data[i][2]**3)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][i]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    def laser_callback(self, scan):
        self.last_laser = scan 
    
    def odom_callback(self, od_data):
        self.last_odom = od_data
    
    # Detect a collision from laser data
    def calculate_observation(self, data):
        min_range = 0.3
        min_laser = 2
        done = False
        col  = False

        for i, item in enumerate(data.ranges): # What is data ranges?
            if min_laser > data.ranges[i]:
                min_laser = data.ranges[i]
            if (min_range > data.ranges[i] > 0):
                done = True
                col = True
        return done, col, min_laser
    

    # Perform an action and read a new state
    def step(self, act):
        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = act[0]
        vel_cmd.angular.z = act[1]
        self.vel_pub.Publish(vel_cmd)

        target = False
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print('/gazebo/unpause_physics service call failed!')
        
        time.sleep(1.0)
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print('/gazebo/pause_physics service call failed!')
        
        data = self.last_laser
        dataOdom = self.last_odom
        laser_state = np.array(data.ranges[:])
        v_state = []
        v_state[:] = self.velodyne_data
        laser_state = [v_state]

        done, col, min_laser = self.velodyne_callback(data)

        # Calculate robot heading from odometry data 
        self.odomX = dataOdom.pose.pose.position.x
        self.odomY = dataOdom.pose.pose.position.y
        quaternion = Quaternion( # Needs to understand what is going on
            dataOdom.pose.pose.oreintation.w,
            dataOdom.pose.pose.oreintation.x,
            dataOdom.pose.pose.oreintation.y,
            dataOdom.pose.pose.oreintation.z)
        ueler = quaternion.to_ueler(degrees = False)
        angle = round(euler[2], 4)

        # Calculate distance to the gaol from the robot
        Dist = math.sqrt(math.pow(self.odomX - self.goalX, 2) \
            + math.pow(self.odomY - self.goalY))

        # Calculate the angle distance between the robots heading and heading towords the goal.
        skewX = self.goalX - self.odomX
        skewY = self.goalY - self.odomY
        dot = skewX * 1 + skewY * 0
        mag1 = math.sqrt(math.pow(skewX, 2) + math.pow(skewY, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skewY < 0:
            if skewX < 0:
                beta = -beta
            else:
                beta = 0 - beta
        beta2 = (beta - angle)
        if beta2 > np.pi:
            beta2 = np.pi - beta2
            beta2 = -np.pi - beta2
        if beta2 < -np.pi:
            beta2 = -np.pi - beta2
            beta2 = np.pi - beta2
        
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goalX
        marker.pose.position.y = self.goalY
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(act[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(act[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

        markerArray4 = MarkerArray()
        marker4 = Marker()
        marker4.header.frame_id = "odom"
        marker4.type = marker.CUBE
        marker4.action = marker.ADD
        marker4.scale.x = 0.1  # abs(act2)
        marker4.scale.y = 0.1
        marker4.scale.z = 0.01
        marker4.color.a = 1.0
        marker4.color.r = 1.0
        marker4.color.g = 0.0
        marker4.color.b = 0.0
        marker4.pose.orientation.w = 1.0
        marker4.pose.position.x = 5
        marker4.pose.position.y = 0.4
        marker4.pose.position.z = 0

        markerArray4.markers.append(marker4)
        self.publisher4.publish(markerArray4)

        '''Generate the reward'''
        r3 = lambda x: 1 - x if x < 1 else 0.0
        reward = act[0] / 2 - abs(act[1]) / 2 - r3(min(laser_state[0])) / 2
        self.distOld = Dist

        # Detect if the goal has been reached and give large positive reward
        if Dist < 0.3:
            target = True
            done = True
            self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) \
                + math.pow(self.odomY - self.goalY, 2))
            reward = 80
        
        # Detect if collision has happened and give a large negative reward 
        if col:
            reward = -100
        
        toGoal = [Dist, beta, act[0], act[1]]
        state = np.append(laser_state, toGoal)
        return state, reward, done, target

    def reset(self):
        # Resets the state of the environment and return an initial observation
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("/gazebo/reset_simulation service call failed!")
        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0., 0., angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        chk = False
        while not chk:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            chk = check_pos(x,y)
        object_state.pose.position.x  = x
        object_state.pose.position.y = y
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odomX = object_state.pose.position.x 
        self.odomY = object_state.pose.position.y

        self.change_goal()
        self.random_box()
        self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))

        data = None
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print('/gazebo/unpause_physics service call failed!')
        
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=0.5)
            except:
                pass

        laser_state = np.array(data.ranges[:])
        laser_state[laser_state == inf] = 10
        laser_state = binning(0, laser_state, 20)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        Dist = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))

        skewX = self.goalX - self.odomX
        skewY = self.goalY - self.odomY
        dot = skewX * 1 + skewY * 0
        mag1 = math.sqrt(math.pow(skewX, 2) + math.pow(skewY, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skewY < 0:
            if skewX < 0:
                beta = -beta
            else:
                beta = 0 - beta
        beta2 = (beta - angle)

        if beta2 > np.pi:
            beta2 = np.pi - beta2
            beta2 = -np.pi - beta2
        if beta2 < -np.pi:
            beta2 = -np.pi - beta2
            beta2 = np.pi - beta2

        toGoal = [Dist, beta2, 0.0, 0.0]
        state = np.append(laser_state, toGoal)
        return state
    
    # Place a new goal and check if it's location is not on one of the obstacles
    def change_goal(self):
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004
        
        gOK = False
        
        while not gOK:
            self.goalX = self.odomX + random.uniform(self.upper, self.lower)
            self.goalY = self.odomY + random.uniform(self.upper, self.lower)
            gOK = check_pos(self.goalX, self.goalY)
        

        










    











        



