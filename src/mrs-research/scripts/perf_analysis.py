import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

def load_data(file_name):
    with open(file_name, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the header row
        data = [row for row in csv_reader]
    return data


from datetime import datetime, timedelta

def parse_time_string(time_str):
    days_str, time_str = time_str.split(', ')
    days = int(days_str.split()[0])
    time = datetime.strptime(time_str, '%H:%M:%S.%f')
    total_seconds = timedelta(days=days, hours=time.hour, minutes=time.minute, seconds=time.second, microseconds=time.microsecond).total_seconds()
    return total_seconds

def plot_average_reward_vs_time(data, file_name):
    episode_rewards = {}
    episode_times = {}
    for row in data:
        ep = int(row[0])
        if ep not in episode_rewards:
            episode_rewards[ep] = 0
            episode_times[ep] = parse_time_string(row[2])
        episode_rewards[ep] += float(row[4])

    episodes = sorted(episode_rewards.keys())
    average_rewards = [episode_rewards[ep] / (ep+1) for ep in episodes]
    time = [episode_times[ep] for ep in episodes]

    plt.figure()
    plt.plot(time, average_rewards)
    plt.xlabel('Time (s)')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Time')
    plt.grid(True)
    plt.savefig(file_name)



def plot_success_rate(data, file_name):
    total_tasks = 5
    episode_goals = {}
    
    for row in data:
        episode, goal_reached = int(row[0]), int(row[5])
        if episode not in episode_goals:
            episode_goals[episode] = 0
        if goal_reached:
            episode_goals[episode] += 1

    episodes = sorted(list(episode_goals.keys()))
    success_rate = [episode_goals[ep] / total_tasks for ep in episodes]

    plt.figure()
    plt.plot(episodes, success_rate)
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Success Rate per Episode')
    plt.grid(True)
    plt.savefig(file_name)


def plot_task_completion_time(data, file_name):
    completion_times = [parse_time_string(row[2]) for row in data if int(row[5]) == 1]

    plt.figure()
    plt.hist(completion_times, bins=20)
    plt.xlabel('Task Completion Time (s)')
    plt.ylabel('Frequency')
    plt.title('Task Completion Time Distribution')
    plt.grid(True)
    plt.savefig(file_name)


def plot_collision_rate(data, file_name):
    episode_collisions = {}
    
    for row in data:
        episode, collision = int(row[0]), int(row[7])
        if episode not in episode_collisions:
            episode_collisions[episode] = 0
        if collision:
            episode_collisions[episode] += 1

    episodes = sorted(list(episode_collisions.keys()))
    collision_rate = [episode_collisions[ep] / (ep+1) for ep in episodes]

    plt.figure()
    plt.plot(episodes, collision_rate)
    plt.xlabel('Episode')
    plt.ylabel('Collision Rate')
    plt.title('Collision Rate per Episode')
    plt.grid(True)
    plt.savefig(file_name)

def plot_robot_trajectory(data, episode, file_name):
    episode_data = [row for row in data if int(row[0]) == episode]
    robot_data = {}

    for row in episode_data:
        robot_index = int(row[3].split('_')[-1])
        if robot_index < 3:  # Only consider the first three robots
            if robot_index not in robot_data:
                robot_data[robot_index] = []
            loc = eval(row[6]) if row[6] else None
            if loc:
                robot_data[robot_index].append(loc)

    plt.figure()
    markers = ['o', 's', '^']
    colors = ['b', 'r', 'g']

    for robot_index, locations in robot_data.items():
        x = [loc[0] for loc in locations]
        y = [loc[1] for loc in locations]
        plt.plot(x, y, marker=markers[robot_index], color=colors[robot_index], linestyle='-', label=f'Robot {robot_index}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Robot Trajectory for Episode {episode}')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name)


if __name__ == '__main__':
    data_file = 'data_generated.csv'
    data = load_data(data_file)

    plot_average_reward_vs_time(data, 'plots/average_reward_vs_time.png')
    plot_success_rate(data, 'plots/success_rate.png')
    plot_task_completion_time(data, 'plots/task_completion_time.png')
    
    plot_collision_rate(data, 'plots/collision_rate.png')

    # Plot robot trajectories for the first 5 episodes
    # for i in range(1, 6):
    plot_robot_trajectory(data, 0, f'plots/robot_trajectory_episode_{0}.png')
