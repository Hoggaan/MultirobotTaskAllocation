# Import modules
import math
import subprocess
import os
import time
import numpy as np
from squaternion import Quaternion

import rospy
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import Odometry


class Gazebo_Env:
    #Initialize and launch
    def __init__(self, launchfile):
        self.odomX = 0.0
        self.odomY = 0.0

        self.odom1X = 0.0
        self.odom1Y = 0.0

        self.odom2X = 0.0
        self.odom2Y = 0.0

        self.odom2X = 0.0
        self.odom2Y = 0.0

        self.odom3X = 0.0
        self.odom3Y = 0.0

        self.odom4X = 0.0
        self.odom4Y = 0.0

        self.goal1X = 1
        self.goal1Y = 0.0

        self.goal2X = 2
        self.goal2Y = 0.0

        self.goal3X = 3
        self.goal3Y = 0.0

        self.goal4X = 4
        self.goal4Y = 0.0

        self.goal5X = 3
        self.goal5Y = 1.0

        self.goal6X = 3
        self.goal6Y = 5.0


        self.set_self_state = ModelState()
        self.set_self_state.model_name = 'Robot1'
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 0.0

        self.set_self_state2 = ModelState()
        self.set_self_state2.model_name = 'Robot2'
        self.set_self_state2.pose.position.x = 1.0
        self.set_self_state2.pose.position.y = 1.0
        self.set_self_state2.pose.position.z = 0.0
        self.set_self_state2.pose.orientation.x = 0.0
        self.set_self_state2.pose.orientation.y = 0.0
        self.set_self_state2.pose.orientation.z = 0.0
        self.set_self_state2.pose.orientation.w = 0.0

        self.set_self_state2 = ModelState()
        self.set_self_state2.model_name = 'Robot3'
        self.set_self_state2.pose.position.x = 2.0
        self.set_self_state2.pose.position.y = 2.0
        self.set_self_state2.pose.position.z = 0.0
        self.set_self_state2.pose.orientation.x = 0.0
        self.set_self_state2.pose.orientation.y = 0.0
        self.set_self_state2.pose.orientation.z = 0.0
        self.set_self_state2.pose.orientation.w = 0.0




        self.distOld = math.sqrt(math.pow(self.odomX - self.goalX) + \
             math.pow(self.odomY - self.goalY))
        
        # Run ross master - roscore
        port = '11311'
        subprocess.Popen(["roscore", "-p", port])
        print("Roscore launched!")

        #Launch the simulation with the given launchfile
        rospy.init_node('hoggaan', anonymous=True)
        if launchfile.startswith('/'):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not os.path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")
        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        # self.gzclient_pid = 0

        # Visualization Topics
        topic1 = 'vis_mark_array1'
        topic2 = 'vis_mark_array2'
        topic3 = 'vis_mark_array3'
        topic4 = 'vis_mark_array4'

        # Setup the ROS publishers and subscribers
        self.vel_pub1 = rospy.Publisher('robot1/cmd_vel', Twist, queue_size=1)
        self.vel_pub2 = rospy.Publisher('robot2/cmd_vel', Twist, queue_size=1)
        self.set_state1 = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=10)
        self.set_state2 = rospy.Publisher('gazebo/set_model_state2', ModelState, queue_size=10)
        self.publisher1 = rospy.Publisher(topic1, MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher(topic2, MarkerArray, queue_size=1)  
        self.publisher3 = rospy.Publisher(topic3, MarkerArray, queue_size=1)
        self.publisher4 = rospy.Publisher(topic4, MarkerArray, queue_size=1)
        # self.velodyne = rospy.Subscriber('/velodyne_points', PointCloud2, self.velodyne_callback, queue_size=1)
        self.laser1 = rospy.Subscriber('robot/scan', LaserScan, self.laser_callback1, queue_size=1)
        self.laser2 = rospy.Subscriber('robot2/scan', LaserScan, self.laser_callback2, queue_size=1)
        self.odom1 = rospy.Subscriber('robot1/odom', Odometry, self.odom_callback1, queue_size=1)
        self.odom2 = rospy.Subscriber('robot2/odom', Odometry, self.odom_callback2, queue_size=1)

        # Calculate Distance Data 

    def laser_callback1(self, scan):
        self.last_laser = scan
    
    def odom_callback1(self, od_data):
        self.last_odom = od_data

    # Detect a collision from laser data
    def calculate_observation(self, data):
        min_range = 0.3
        min_laser = 2
        done = False
        col = False

        for i, item in enumerate(data.ranges):
            if min_laser > data.ranges[i]:
                min_laser = data.ranges[i]
            if (min_range > data.ranges[i] > 0):
                done = True
                col = True
        return done, col, min_laser
    
    # Calculate the robot heading from Odometry 
    def heading(self, dataOdom):
        self.odomX = dataOdom.pose.position.x
        self.odomY = dataOdom.pose.position.y
        quaternion = Quaternion(
            dataOdom.pose.pose.orientation.w,
            dataOdom.pose.pose.orientation.x,
            dataOdom.pose.pose.orientaiton.y,
            dataOdom.pose.pose.orientation.z)
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
        return euler, angle
    
    # Calculate the angle of the robot heading and the angle to the goal.
    def angleMeasurement(self, angle):
        skewX = self.goalX - self.odomX
        skewY = self.goalY - self.odomY
        dot = skewX * 1 + skewY * 0
        mag1 = math.sqrt(math.pow(skewX, 2) + math.pow(skewY, 2))
        mag2 = math.sqrt(math.pow(1,2) + math.pow(0,2))
        beta = math.acos(dot/ (mag1 * mag2))

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
            beta2 = np.pi  - beta2
        return beta2

    # Perform an action and read new state
    def step(self, act):
        # Publish the robot action
        vel_cmd  = Twist()
        vel_cmd.linear.x = act[0]
        vel_cmd.angular.z = act[1]
        self.vel_pub1.publish(vel_cmd)

        target = False
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed! ")
        

        time.sleep(0.1)

    
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed! ")
        

        data = self.last_laser
        dataOdom = self.last_odom
        laser_state = np.array(data.ranges[:])
        v_state = []
        # v_stat[:] = self.velodyne_data[:]
        laser_state = [v_state]

        done, col, min_laser = self.calculate_observation(data)

        # Calculate robot heading from odometry data
        euler, angle = self.heading(dataOdom)

        # Calculate distance to the goal from the robot
        Dist = math.sqrt(math.pow(self.odomX - self.goalX, 2) +\
            math.pow(self.odomY - self.goalY))
        
        
        # Calculate the angle distance between the robot's heading
        # And the heading toward the goal
        beta2 = self.angleMeasurement(angle)
        
        
        # publish the visualization data in RViz 
        # Pass


        #Reward
        reward = 0

        self.distOld = Dist 

        toGoal = [Dist, beta2, act[0], act[2]]
        state = np.append(laser_state, toGoal)
        return state, reward, done, target
    
    def reset(self):
        # Resets the state of the environment and returns 
        # an Initial observation.
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed! ")
        
        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0., 0., angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        chk = False
        while not chk:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            chk = check_pos(x, y) # Function that checks if the 
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.position.z = z    Originally commented!
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odomX = object_state.pose.position.x
        self.odomY = object_state.pose.position.y

        # Two custom Methods for this software
        self.change_goal()
        self.random_box()

        self.distOld = math.sqrt(math.pow(self.odomX - self.goalX) +\
            math.pow(self.odomY - self.goalY))
        
        data = None
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed! ")
        
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=0.5)
            except:
                pass
        laser_state = np.array(data.ranges[:])
        laser_state[laser_state == np.inf] = 10
        laser_state = binning(0, laser_state, 20)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        
        Dist = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))

        beta2 = self.angleMeasurement(angle)

        toGoal = [Dist, beta2, 0.0, 0.0]
        state = np.append(laser_state, toGoal)
        return state

    # Change the goal method












