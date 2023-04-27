#Import modules
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

                # Define the starting points of 3 robots using model state
        self.robot1_state = ModelState()
        self.robot1_state.model_name = "robot1"
        self.robot1_state.pose.position.x = 0
        self.robot1_state.pose.position.y = 0
        self.robot1_state.pose.position.z = 0

        self.robot2_state = ModelState()
        self.robot2_state.model_name = "robot2"
        self.robot2_state.pose.position.x = 1
        self.robot2_state.pose.position.y = 1
        self.robot2_state.pose.position.z = 0

        self.robot3_state = ModelState()
        self.robot3_state.model_name = "robot3"
        self.robot3_state.pose.position.x = 2
        self.robot3_state.pose.position.y = 2
        self.robot3_state.pose.position.z = 0