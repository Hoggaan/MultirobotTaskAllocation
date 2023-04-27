import rospy 
from sensor_msgs.msg import LaserScan
import numpy as np
import torch 

def callback(msg):
    data = np.array(msg.ranges)
    data = torch.from_numpy(data)
    print(data.shape)
    #print(len(msg.ranges))

rospy.init_node('scan_values')
sub = rospy.Subscriber('/robot2/scan', LaserScan, callback)
rospy.spin()
#print(data)