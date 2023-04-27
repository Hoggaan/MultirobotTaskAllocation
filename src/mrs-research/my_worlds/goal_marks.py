import rospy
import numpy as np
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, SpawnModel
from geometry_msgs.msg import Pose

rospy.init_node("set_goal_positions")

def is_position_safe(position, obstacles, safe_distance):
    for obstacle in obstacles:
        min_x = obstacle['position'][0] - obstacle['size'][0] / 2 - safe_distance
        max_x = obstacle['position'][0] + obstacle['size'][0] / 2 + safe_distance
        min_y = obstacle['position'][1] - obstacle['size'][1] / 2 - safe_distance
        max_y = obstacle['position'][1] + obstacle['size'][1] / 2 + safe_distance

        if min_x <= position[0] <= max_x and min_y <= position[1] <= max_y:
            return False
    return True

def generate_goal_positions(num_goals):
    obstacles = [
        {'position': np.array([-4, -4]), 'size': np.array([2, 0.2])},
        {'position': np.array([4, 4]), 'size': np.array([2, 0.2])},
        {'position': np.array([0, 0]), 'size': np.array([4, 0.2])},
    ]
    x_range = (-8, 8)
    y_range = (-8, 8)
    safe_distance = 0.5
    goal_positions = []

    while len(goal_positions) < num_goals:
        position = np.random.uniform(
            low=[x_range[0], y_range[0]],
            high=[x_range[1], y_range[1]]
        )

        if is_position_safe(position, obstacles, safe_distance):
            goal_positions.append(position)

    return np.array(goal_positions)

def spawn_goal_model(goal_name, position):
    spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    model_pose = Pose()
    model_pose.position.x = position[0]
    model_pose.position.y = position[1]
    model_pose.position.z = 0.5

    spawn_model(goal_name, goal_sdf, "", model_pose, "world")


if __name__ == "__main__()":
  
  # Example usage
  num_goals = 5
  goal_positions = generate_goal_positions(num_goals)

  # Define the goal model SDF string
  goal_sdf = """
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



  for i, goal_position in enumerate(goal_positions):
      goal_name = f"goal_{i + 1}"
      spawn_goal_model(goal_name, goal_position)
