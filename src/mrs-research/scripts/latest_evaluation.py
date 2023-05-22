import numpy as np
import datetime

# Load the data, skipping the first row and converting the third column to a datetime object
data = np.loadtxt("data_generated.csv", delimiter=",", skiprows=1, converters={3: lambda x: datetime.datetime.strptime(x.decode('utf-8'), '%m/%d/%Y %H:%M:%S')})

# Calculate the success rate
success_rate = np.sum(data[:, 4] == 1) / len(data)

# Calculate the completion time
completion_time = np.mean(data[:, 5])

# Calculate the collision rate
collision_rate = np.sum(data[:, 6] > 0) / len(data)

# Plot the success rate
plt.plot(data[:, 0], data[:, 4])
plt.title("Success Rate")
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.savefig("/plots/success_rate.png")

# Plot the completion time
plt.plot(data[:, 0], data[:, 5])
plt.title("Completion Time")
plt.xlabel("Episode")
plt.ylabel("Completion Time")
plt.savefig("/plots/completion_time.png")

# Plot the collision rate
plt.plot(data[:, 0], data[:, 6])
plt.title("Collision Rate")
plt.xlabel("Episode")
plt.ylabel("Collision Rate")
plt.savefig("/plots/collision_rate.png")
