import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data into a DataFrame
data = pd.read_csv("data_generated.csv")

# Calculate the overall success rate per episode
overall_success_rate = data.groupby('Episode')['Goal Reached'].apply(lambda x: (x != 'None').sum() / len(x))

# Calculate the success rate per episode for each goal
goals = ["goal_0", "goal_1", "goal_2", "goal_3", "goal_4"]
success_rate = {}
for goal in goals:
    success_rate[goal] = data.groupby('Episode')['Goal Reached'].apply(lambda x: (x == goal).sum() / len(x))

success_rate_df = pd.DataFrame(success_rate)

# Calculate the collision rate per episode
collision_rate = data.groupby('Episode')['Collision'].mean()

# Plot the overall success rate per episode
plt.plot(np.array(overall_success_rate.index), np.array(overall_success_rate))
plt.title("Overall Success Rate per Episode")
plt.xlabel("Episode")
plt.ylabel("Overall Success Rate")
plt.savefig("plots/overall_success_rate.png")
plt.clf()

# Plot the success rate for each goal
success_rate_df.plot()
plt.title("Success Rate for Each Goal")
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.legend(goals)
plt.savefig("plots/success_rate_each_goal.png")
plt.clf()

# Plot the collision rate
plt.plot(np.array(collision_rate.index), np.array(collision_rate))
plt.title("Collision Rate")
plt.xlabel("Episode")
plt.ylabel("Collision Rate")
plt.savefig("plots/collision_rate.png")
plt.clf()

# Calculate the average reward per episode
avg_reward = data.groupby('Episode')['Reward'].mean()

# Plot the average reward per episode
plt.plot(np.array(avg_reward.index), np.array(avg_reward))
plt.title("Average Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.savefig("plots/average_reward.png")
plt.clf()
