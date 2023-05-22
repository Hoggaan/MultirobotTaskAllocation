import rospy
import numpy as np
import subprocess
import os
import time
import math
import csv
from datetime import timedelta
import datetime

from geometry_msgs.msg import PoseStamped
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, SpawnModel
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import DeleteModel

class MultiRobotEnv:
    def __init__(self,launchfile, num_robots, num_goals):
        self.num_robots = num_robots
        self.num_goals = num_goals
        self.robot_positions = self.generate_robot_positions(num_robots)
        self.robot_orientations = np.zeros((num_robots, 3)) 
        self.robot_laser_data = np.zeros((num_robots, 360))
        self.goals = self.generate_goal_positions(num_goals)

        self.start_time = None
        
        self.observations = None

        self.last_distances = self.distances()
        self.available_goals = self.goals

        # Laser scan parameters
        self.num_laser_readings = 360
        self.max_laser_range = 50.0  # meters
        self.min_laser_range = 0.2   # meters
        self.laser_threshold = 0.2   # meters

        # Define the goal model SDF string
        self.goal_sdf = """
        <sdf version="1.6">
        <model name="goal">
            <static>true</static>
            <link name="link">
            <visual name="visual">
                <geometry>
                <sphere>
                    <radius>0.1</radius>
                </sphere>
                </geometry>
                <material>
                <ambient>0 1 0 1</ambient>
                <diffuse>0 1 0 1</diffuse>
                </material>
            </visual>
            <collision name="collision">
                <geometry>
                <sphere>
                    <radius>0.1</radius>
                </sphere>
                </geometry>
            </collision>
            </link>
        </model>
        </sdf>
        """


        # Launch the Gazebo simulation
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
        time.sleep(5)
        self.main()
        # Wait for the Gazebo service to be available
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        # Set the initial positions of the robots in the Gazebo simulation
        for i in range(self.num_robots):
            model_state = ModelState()
            model_state.model_name = "robot_{}".format(i)
            model_state.pose.position.x = self.robot_positions[i][0]
            model_state.pose.position.y = self.robot_positions[i][1]
            model_state.pose.position.z = 0
            model_state.pose.orientation.x = 0.0
            model_state.pose.orientation.y = 0.0
            model_state.pose.orientation.z = 0.0
            model_state.pose.orientation.w = 1.0
            self.set_model_state(model_state)
        
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        # Subscribe to the laser and odometry topics for each robot
        self.laser_subs = []
        self.odom_subs = []
        self.cmd_vel_pubs = []
        self.robot_poses = {}
        for i in range(self.num_robots):
            laser_sub = rospy.Subscriber("/robot_{}/scan".format(i), LaserScan, self.laser_callback, i)
            odom_sub = rospy.Subscriber("/robot_{}/odom".format(i), Odometry, self.odom_callback, i)
            cmd_vel_pub = rospy.Publisher("/robot_{}/cmd_vel".format(i), Twist, queue_size=10)
            # rospy.Subscriber(f'/robot_{i}/pose', PoseStamped, self.pose_callback, i)
            self.laser_subs.append(laser_sub)
            self.odom_subs.append(odom_sub)
            self.cmd_vel_pubs.append(cmd_vel_pub)

    def laser_callback(self, msg, robot_index):
        """Callback method for processing laser scan data for a specific robot."""
        self.robot_laser_data[robot_index] = msg.ranges

    # def pose_callback(self, data, robot_id):
    #     # Update the robot pose in the dictionary
    #     self.robot_poses[robot_id] = data.pose
    #     print(f"Updated pose for robot {robot_id}: {data.pose}")  # Add print statement for debugging


    def odom_callback(self, msg, robot_index):
        """Callback method for processing odometry data for a specific robot."""
        self.robot_positions[robot_index][0] = msg.pose.pose.position.x
        self.robot_positions[robot_index][1] = msg.pose.pose.position.y
        # self.robot_positions[robot_index][2] = msg.pose.pose.position.z
        x,y = msg.pose.pose.position.x, msg.pose.pose.position.y
        self.robot_poses[robot_index] = x,y

        # Extract the orientation quaternion from the message
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]

        #Convert the quaternion to Euler angles
        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        self.robot_orientations[robot_index][0] = roll
        self.robot_orientations[robot_index][1] = pitch
        self.robot_orientations[robot_index][2] = yaw

    def step(self, actions):
        # Ensure that the correct number of actions have been provided.
        assert len(actions) == self.num_robots

        # Publish actions for each robot
        for i in range(self.num_robots):
            pub = self.cmd_vel_pubs[i]
            twist = Twist()
            twist.linear.x = actions[i][0]
            twist.angular.z = actions[i][1]
            pub.publish(twist)

        # Wait for physics to unpause and then pause again
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed! ")

        time.sleep(1)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed! ")
        
        # Calculate observations, rewards, and dones for all robots
        observations = []
        rewards = []
        dones = []
        for i in range(self.num_robots):
            observation = self.calculate_observation(i)
            reward, done = self.calculate_reward(i)
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)

        self.observations = observations
        return observations, rewards, dones, {}


    def reset(self):

        self.start_time = rospy.Time.now()
        self.robot_positions = self.generate_robot_positions(self.num_robots)

        # Resets the state of the environment and returns 
        # an Initial observation.
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed! ")
        
        # Wait for the Gazebo service to be available
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)


        # Set the initial positions of the robots in the Gazebo simulation
        for i in range(self.num_robots):
            model_state = ModelState()
            model_state.model_name = "robot_{}".format(i)
            model_state.pose.position.x = self.robot_positions[i][0]
            model_state.pose.position.y = self.robot_positions[i][1]
            model_state.pose.position.z = 0
            model_state.pose.orientation.x = 0.0
            model_state.pose.orientation.y = 0.0
            model_state.pose.orientation.z = 0.0
            model_state.pose.orientation.w = 1.0
            self.set_model_state(model_state)

        self.last_distances = self.distances()
        self.start_time = rospy.Time.now()

        new_observation = []
        for i in range(self.num_robots):
            observation = self.calculate_observation(i)
            new_observation.append(observation)
        
        self.observations = new_observation
        return new_observation
    
    # A method that calculates all the distances between the robots and the goals.
    def distances(self):
        distances = np.zeros((self.num_robots, self.num_goals))
        for i in range(self.num_robots):
            for j, goal in enumerate(self.goals):
                distances[i][j] = np.linalg.norm(self.robot_positions[i] - goal)
        return distances

    def calculate_reward(self, robot_index):
        """Compute the reward for the current timestep of the simulation."""
        reward = 0
        done = False
        
        # If all robots reached, Done
        if self.goals.size == 0:
            reward += 1000
            done = True
            return reward, done

        # Check for collisions between the specified robot and other robots
        for i in range(self.num_robots):
            if i == robot_index:
                continue
            # If the distance between the two robots is less than a threshold, penalize
            if np.linalg.norm(self.robot_positions[robot_index] - self.robot_positions[i]) < 0.1:
                reward -= 5
            
        # Check for collisions between the specified robot and obstacles
        min_obstacle_distance = np.min(self.robot_laser_data[robot_index])
        if min_obstacle_distance < 0.2:
            reward -= 50  # Huge penalty for reaching an obstacle
        else:
            reward -= 2 * (1 / min_obstacle_distance)  # Penalty for moving towards obstacles

                    
        # Check if the specified robot has reached its own unique goal
        nearest_goal_idx = np.argmin([np.linalg.norm(goal - self.robot_positions[robot_index]) for goal in self.goals])
        nearest_goal = self.goals[nearest_goal_idx]
                        
        # if np.array_equal(self.goals[robot_index], nearest_goal_idx):
        if np.linalg.norm(self.robot_positions[robot_index] - nearest_goal) < 0.5:
            reward += 1000
            print()
            print(f"AMASING!!! {nearest_goal} Reached! by {robot_index}")

        # Otherwise, give a small penalty or reward based on whether the robot is moving toward its goal
        else:
            dx = nearest_goal[0] - self.robot_positions[robot_index][0]
            dy = nearest_goal[1] - self.robot_positions[robot_index][1]
            distance_to_goal = np.linalg.norm(nearest_goal - self.robot_positions[robot_index])

            angle_to_goal = np.arctan2(dy, dx) - self.robot_orientations[robot_index][2]
            if angle_to_goal < - np.pi:
                angle_to_goal += 2*np.pi
            elif angle_to_goal > np.pi:
                angle_to_goal -= 2*np.pi
            # Penalize if the robot is not moving toward its goal

            if np.any(distance_to_goal > self.last_distances[robot_index][nearest_goal_idx]):
                # Robot is moving away from its goal, penalize
                reward -= 0.1
            else:
                # Robot is moving toward its goal, reward
                reward += 10
        
        # # Penalize if the robot is moving towards a goal that another robot is closer to
        # for j in range(self.num_robots):
        #     if j == robot_index:
        #         continue

        #     other_robot_goal_idx = np.argmin([np.linalg.norm(goal - self.robot_positions[j]) for goal in self.goals])
        #     other_robot_nearest_goal = self.goals[other_robot_goal_idx]
        #     other_robot_distance_to_goal = np.linalg.norm(other_robot_nearest_goal - self.robot_positions[j])

        #     if nearest_goal_idx == other_robot_goal_idx and distance_to_goal > other_robot_distance_to_goal:
        #         reward -= 10


        # Check if the specified robot has reached the same goal as another robot
        for j in range(self.num_robots):
            if j == robot_index:
                continue
            for goal in range(self.num_goals):
                if np.linalg.norm(self.robot_positions[robot_index] - self.robot_positions[j]) < 0.2:
                    goal_pos = self.goals[goal]
                    robot_pos_i = self.robot_positions[robot_index]
                    robot_pos_j = self.robot_positions[j]
                    if np.linalg.norm(goal_pos - robot_pos_i) <= 0.1:
                        reward -= 5
                    elif np.linalg.norm(goal_pos - robot_pos_j) <= 0.1:
                        reward -= 5


        # Check if the maximum time has been reached for the episode
        if rospy.Time.now() - self.start_time >= rospy.Duration(300):
            reward -= 100
            done = True

            
        return reward, done

    def calculate_observation(self, robot):

        robot_observation = []
        
        # calculate the relative positions of goals in the focal robot's polar coordinates
        o_t_e = np.zeros((3, self.num_goals, 2), dtype=np.float32)
        if self.observations is not None:
            o_t_e = self.observations[robot][0]

        o_t_e = np.roll(o_t_e, shift=1, axis=0)
        for i, goal in enumerate(self.goals):
            dx = goal[0] - self.robot_positions[robot][0]
            dy = goal[1] - self.robot_positions[robot][1]
            o_t_e[0, i, 0] = np.sqrt(dx**2 + dy**2)
            o_t_e[0, i, 1] = np.arctan2(dy, dx) - self.robot_orientations[robot][2]  # Correction Needed!!
        robot_observation.append(o_t_e)
        
        # calculate the relative positions of the goals in the other robots' polar coordinates
        o_t_o = np.zeros(( 3, 15, 2), dtype=np.float32)

        if self.observations is not None:
            o_t_o = self.observations[robot][1]

        o_t_o = np.roll(o_t_o, shift=1, axis=0)
        for j in range(self.num_robots):
            if robot == j:
                o_t_o[0, j*5:(j+1)*5, :] = 0   # set the focal robot's elements to zero
                continue
            start_idx = j*5
            for i, goal in enumerate(self.goals):
                dx = goal[0] - self.robot_positions[j][0]
                dy = goal[1] - self.robot_positions[j][1]
                o_t_o[0, start_idx, 0] = np.sqrt(dx**2 + dy**2)
                o_t_o[0, start_idx, 1] = np.arctan2(dy, dx) - self.robot_orientations[j][2]   # Correction
                # Normalize the heading to be between -pi and pi
                o_t_o[0, start_idx, 1] = math.atan2(math.sin(o_t_o[0, start_idx, 1]), math.cos(o_t_o[0, start_idx, 1]))
                start_idx += 1
        
        # print(o_t_o.shape)
        robot_observation.append(o_t_o)
        
        o_t_l = np.zeros((3,1, 360), dtype=np.float32)
        if self.observations is not None:
            o_t_l = self.observations[robot][2]
        # calculate the 360 degree laser scanner data of the focal robot
        o_t_l = np.roll(o_t_l, shift=1, axis=0)
        o_t_l[0] = self.robot_laser_data[robot]
        o_t_l[0] *= self.max_laser_range
        o_t_l[0] += np.array(self.robot_positions[robot][0], self.robot_positions[robot][1])

        # Perform obstacle detection using thresholding
        obstacle_mask = np.logical_or(self.robot_laser_data[robot] > self.max_laser_range, self.robot_laser_data[robot]  < self.min_laser_range)
        obstacle_mask = np.logical_or(obstacle_mask, self.robot_laser_data[robot]  < self.laser_threshold)
        o_t_l[0] = np.minimum(o_t_l[0], obstacle_mask)
        o_t_l[0] = np.linalg.norm(o_t_l[0], axis=0)

        # otl = o_t_l.reshape((3, 1, 360))

        robot_observation.append(o_t_l)
        
        return robot_observation


    # Extended part of the code.     
    def is_position_safe(self, position, obstacles, safe_distance):
        for obstacle in obstacles:
            min_x = obstacle['position'][0] - obstacle['size'][0] / 2 - safe_distance
            max_x = obstacle['position'][0] + obstacle['size'][0] / 2 + safe_distance
            min_y = obstacle['position'][1] - obstacle['size'][1] / 2 - safe_distance
            max_y = obstacle['position'][1] + obstacle['size'][1] / 2 + safe_distance

            if min_x <= position[0] <= max_x and min_y <= position[1] <= max_y:
                return False
        return True
    
    def generate_goal_positions(self, num_goals):
        obstacles = [
            {'position': np.array([-4, -4]), 'size': np.array([2, 0.2])},
            {'position': np.array([4, 4]), 'size': np.array([2, 0.2])},
            {'position': np.array([0, 0]), 'size': np.array([4, 0.2])},
        ]
        x_range = (-8, 8)
        y_range = (-8, 8)
        safe_distance = 2.0
        goal_positions = []

        while len(goal_positions) < num_goals:
            position = np.random.randint(
                low=[x_range[0], y_range[0]],
                high=[x_range[1], y_range[1]]
            )

            if self.is_position_safe(position, obstacles, safe_distance):
                goal_positions.append(position)

        return np.array(goal_positions)
    
    def spawn_goal_model(self, goal_name, position):
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

        model_pose = Pose()
        model_pose.position.x = position[0]
        model_pose.position.y = position[1]
        model_pose.position.z = 0.5

        spawn_model(goal_name, self.goal_sdf, "", model_pose, "world")



    def main(self):
        num_goals = 5
        goal_positions = self.goals

        r_name = ['one', 'two', 'three', 'four', 'five']
        for i, goal_position in enumerate(goal_positions):
            r = r_name[i]
            goal_name = f"goal_{r}"
            self.spawn_goal_model(goal_name, goal_position)


    # Create random positons of the robots 
    def generate_robot_positions(self, num_robots=3):
        x_range = (-8, 8)
        y_range = (-8, 8)
        obstacles = [
            {'position': np.array([-4, -4]), 'size': np.array([2, 0.2])},
            {'position': np.array([4, 4]), 'size': np.array([2, 0.2])},
            {'position': np.array([0, 0]), 'size': np.array([4, 0.2])},
        ]

        robot_positions = []
        safe_distance = 2.0
        while len(robot_positions) < num_robots:
            position = np.random.randint(
                low=[x_range[0], y_range[0]],
                high=[x_range[1], y_range[1]]
            )

            if self.is_position_safe(position, obstacles, safe_distance):
                robot_positions.append(position)

        return np.array(robot_positions)


    # Creating Data for Performace anlysis
    def robot_name(self):
        robot_names = {}
        for robot_index in range(self.num_robots):
            robot_names[robot_index] = f"robot_{robot_index}"
        
        return robot_names
        
    def get_robot_loc(self):
        """
        Returns the current positions of the robots.

        Returns:
            A dictionary, where the keys are the robot names and the values are the x and y coordinates of the robots.
        """

        robot_locations = {}
        for robot_index in range(self.num_robots):
            x, y = self.robot_positions[robot_index]
            robot_locations[f"robot_{robot_index}"] = (x, y)
        return robot_locations

    
    def reached_goal(self):
        robot_reached_goal = {}
        for robot_index in range(self.num_robots):
            nearest_goal_idx = np.argmin([np.linalg.norm(goal - self.robot_positions[robot_index]) for goal in self.goals])
            nearest_goal = self.goals[nearest_goal_idx]
            if np.linalg.norm(self.robot_positions[robot_index] - nearest_goal) < 0.2:
                robot_reached_goal[robot_index] = f"goal_{nearest_goal_idx}"

        return robot_reached_goal
    
    # def get_time(self):
    #     # Get the current ROS time

    #     ros_time = rospy.get_time()
    #     formatted_time = timedelta(seconds=ros_time)
    #     return str(formatted_time)



    def get_time(self):

        # Get the current ROS time
        ros_time = rospy.get_time()

        # Convert the ROS time to a datetime object
        datetime_object = datetime.datetime.fromtimestamp(ros_time)

        # Format the datetime object
        formatted_time = datetime_object.strftime('%m/%d/%Y %H:%M:%S')

        # Return the formatted datetime object
        return formatted_time

    
    def collision(self, rewards):
        collisions = {robot_index: 0 for robot_index in range(self.num_robots)}

        # If the reward is less than -10, there is a collision
        for robot_index in range(self.num_robots):
            if rewards[robot_index] <= -10:
                collisions[robot_index] += 1

        return collisions

    
    def performance_data(self, step, episode, rewards):

        # Get the data from the other methods
        robot_names = self.robot_name()
        robot_locations = self.get_robot_loc()
        goals_reached = self.reached_goal()
        current_time = self.get_time()
        collisions = self.collision(rewards)

        # Add headers to the CSV file if it doesn't exist
        file_name = f'data_generated.csv'
        if not os.path.exists(file_name):
            with open(file_name, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                headers = ['Episode', 'Step', 'Time', 'Robot', 'Reward', 'Goal Reached', 'Location', 'Collision']
                csv_writer.writerow(headers)

        # Open the CSV file in append mode
        with open(file_name, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            # Iterate through all robots and write their data to the CSV file
            for robot_index in range(self.num_robots):
                row = [
                    episode,
                    step,
                    current_time,
                    robot_names[robot_index],
                    rewards[robot_index],
                    goals_reached.get(robot_index, "None"),
                    robot_locations.get(robot_index),
                    collisions.get(robot_index, 0)
                ]
                csv_writer.writerow(row)