import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from collections import deque
import random
import time
import os

from project import MultiRobotEnv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        #states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        #return states, actions, rewards, next_states, dones
        return batch

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #o_t_l 
        self.conv1 = nn.Conv1d(1,8, 3)
        self.conv2 = nn.Conv1d(8,16, 3, 1)
        self.conv3 = nn.Conv1d(16,32, 3, 1)
        self.conv4 = nn.Conv1d(32, 32, 3, 1)
        self.conv5 = nn.Conv1d(32, 64, 3, 1)
        self.fc1_otl = nn.Linear(5568, 64)

        #o_t_o
        self.fc1_oto = nn.Linear(30, 32)
        # self.fc2_oto = nn.Linear(32, 64)
        self.fc3_oto = nn.Linear(32, 64)

        # o_t_e
        self.fc1_ote = nn.Linear(10, 8)
        self.fc2_ote = nn.Linear(8, 16)

        # o_t_th
        # self.fc_otth = nn.Linear(3,8)

        # Output layers
        self.fc1_out = nn.Linear(64 + 64 + 16, 64)
        self.fc2_out = nn.Linear(64, 32)
        # self.fc3_out_linear = nn.Linear(32, 1)
        # self.fc3_out_rotation = nn.Linear(32, 1)

        self.mean_fc = nn.Linear(32, act_dim)
        self.log_std_fc = nn.Linear(32, act_dim)
    def forward(self, obs):
        # o_t_l 
        otl = obs[2]
        otl_out = F.relu(self.conv1(otl))
        # otl_out = F.max_pool1d(otl_out, kernel_size=2)
        otl_out = F.relu(self.conv2(otl_out))
        # otl_out = F.max_pool1d(otl_out, kernel_size=2)
        otl_out = F.relu(self.conv3(otl_out))
        # otl_out = F.max_pool1d(otl_out, kernel_size=2)
        otl_out = F.relu(self.conv4(otl_out))
        otl_out = F.max_pool1d(otl_out, kernel_size=2)
        otl_out = F.relu(self.conv5(otl_out))
        otl_out = F.max_pool1d(otl_out, kernel_size=2)
        otl_out = otl_out.view(otl_out.size(0), -1)
        otl_out = self.fc1_otl(otl_out)

        # oto 
        oto = obs[1]
        flattened_oto = oto.view(oto.size(0), -1)
        out = F.relu(self.fc1_oto(flattened_oto))
        # out = F.relu(self.fc2_oto(out))
        oto_out = F.relu(self.fc3_oto(out))

        # ote
        ote = obs[0]
        batch_size = ote.size(0)
        input_size = ote.size(1) * ote.size(2)
        flattened_ote = ote.view(batch_size, input_size)
        out = F.relu(self.fc1_ote(flattened_ote))
        ote_out = F.relu(self.fc2_ote(out))

        # # otth
        # otth = torch.tensor(obs[3], dtype=torch.float32)
        # out = F.relu(self.fc_otth(otth))


        concatenated = torch.cat((otl_out, oto_out, ote_out), dim=1)

        # fully connected layers
        out = F.relu(self.fc1_out(concatenated))
        out = F.relu(self.fc2_out(out))

        # separate branches for linear and rotation velocity
        # linear_vel = nn.Linear(64, 1)(out)
        # rot_vel = nn.Linear(64, 1)(out)

        # return linear_vel, rot_vel

        mean = self.mean_fc(out)
        log_std = self.log_std_fc(out)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Constrain the logarithm of the standard deviation to a reasonable range
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.randn(mean.shape)
        action = torch.tanh(mean + std * normal)
        log_prob = self.compute_log_prob(mean, log_std, action)
        return action, log_prob
    
    # def generate_gaussian_noise(self, mean, std_dev, shape):
    #     return np.random.normal(loc=mean, scale=std_dev, size=shape)

    def compute_log_prob(self, mean, log_std, action):
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(action)
        log_prob = torch.sum(log_prob, dim=-1, keepdim=True)
        return log_prob

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        #o_t_l 
        self.conv1 = nn.Conv1d(1,8, 3)
        self.conv2 = nn.Conv1d(8,16, 3, 1)
        self.conv3 = nn.Conv1d(16,32, 3, 1)
        self.conv4 = nn.Conv1d(32, 32, 3, 1)
        self.conv5 = nn.Conv1d(32, 64, 3, 1)
        self.fc1_otl = nn.Linear(5568, 256)

        #o_t_o
        self.fc1_oto = nn.Linear(30, 64)
        # self.fc2_oto = nn.Linear(32, 128)
        self.fc3_oto = nn.Linear(64, 256)

        # o_t_e
        self.fc1_ote = nn.Linear(10, 8)
        self.fc2_ote = nn.Linear(8, 16)

        # o_t_th
        # self.fc_otth = nn.Linear(3,8)

        self.fc1 = nn.Linear(256 + 256 + 16 + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, act):
        # o_t_l 
        otl = obs[2]
        otl_out = F.relu(self.conv1(otl))
        # otl_out = F.max_pool1d(otl_out, kernel_size=2)
        otl_out = F.relu(self.conv2(otl_out))
        # otl_out = F.max_pool1d(otl_out, kernel_size=2)
        otl_out = F.relu(self.conv3(otl_out))
        # otl_out = F.max_pool1d(otl_out, kernel_size=2)
        otl_out = F.relu(self.conv4(otl_out))
        otl_out = F.max_pool1d(otl_out, kernel_size=2)
        otl_out = F.relu(self.conv5(otl_out))
        otl_out = F.max_pool1d(otl_out, kernel_size=2)
        otl_out = otl_out.view(otl_out.size(0), -1)
        otl_out = self.fc1_otl(otl_out)

        # oto 
        oto = obs[1]
        flattened_oto = oto.view(oto.size(0), -1)
        out = F.relu(self.fc1_oto(flattened_oto))
        # out = F.relu(self.fc2_oto(out))
        oto_out = F.relu(self.fc3_oto(out))

        # ote
        ote = obs[0]
        batch_size = ote.size(0)
        input_size = ote.size(1) * ote.size(2)
        flattened_ote = ote.view(batch_size, input_size)
        out = F.relu(self.fc1_ote(flattened_ote))
        ote_out = F.relu(self.fc2_ote(out))

        # # otth
        # otth = torch.tensor(obs[3], dtype=torch.float32)
        # out = F.relu(self.fc_otth(otth))
        if act.shape == torch.Size([2]):
            act_reshaped = torch.zeros((3, 2))
            act_reshaped[:, :] = act.view(1, 2).repeat(3, 1)

            act = act_reshaped

        # print(otl_out.shape, oto_out.shape, ote_out.shape, act.shape)
        x = torch.cat((otl_out, oto_out, ote_out, act), dim=1)


        # x = torch.cat([concatenated, act], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

class SACAgent:
    def __init__(self, obs_dim, act_dim, hidden_dim, buffer_size, batch_size, gamma, tau, alpha, lr, actor_update_freq):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr = lr
        self.actor_update_frequency = actor_update_freq

        self.actor = Actor(obs_dim, act_dim, hidden_dim).to(self.device)
        self.actor_optim = optim.SGD(self.actor.parameters(), lr=lr)
        
        self.q1 = Critic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.q1_optim = optim.SGD(self.q1.parameters(), lr=lr)

        self.q2 = Critic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.q2_optim = optim.SGD(self.q2.parameters(), lr=lr)

        self.target_q1 = Critic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2 = Critic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)

    def update(self, batch):
        obs, acts, rews, next_obs, dones = zip(*batch)

        acts = torch.FloatTensor(acts).to(self.device) 
        rews = torch.FloatTensor(rews).to(self.device).unsqueeze(1) 
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        target_q1_losses, target_q2_losses = [], []
        actor_losses = []
        
        for i in range(len(obs)):
            obs_i = [torch.from_numpy(o).to(self.device) for o in obs[i]]
            next_obs_i = [torch.FloatTensor(nobs).to(self.device) for nobs in next_obs[i]]

            next_acts_i, next_log_probs_i = self.actor.sample(next_obs_i)
            target_q1_values_i = self.target_q1(next_obs_i, next_acts_i)
            target_q2_values_i = self.target_q2(next_obs_i, next_acts_i)
            target_q_values_i = torch.min(target_q1_values_i, target_q2_values_i) - self.alpha * next_log_probs_i
            target_q_values_i = rews[i] + (1 - dones[i]) * self.gamma * target_q_values_i
 
            q1_values_i = self.q1(obs_i, acts[i])
            q2_values_i = self.q2(obs_i, acts[i])
            q1_loss_i = F.mse_loss(q1_values_i, target_q_values_i)
            q2_loss_i = F.mse_loss(q2_values_i, target_q_values_i)

            self.q1_optim.zero_grad()
            q1_loss_i.backward(retain_graph=True)
            self.q1_optim.step()

            self.q2_optim.zero_grad()
            q2_loss_i.backward(retain_graph=True)
            self.q2_optim.step()

            target_q1_losses.append(q1_loss_i.item())
            target_q2_losses.append(q2_loss_i.item())

            if i % self.actor_update_frequency == 0:
                sampled_acts_i, log_probs_i = self.actor.sample(obs_i)
                min_q_values_i = torch.min(self.q1(obs_i, sampled_acts_i), self.q2(obs_i, sampled_acts_i))
                actor_loss_i = (self.alpha * log_probs_i - min_q_values_i).mean()

                self.actor_optim.zero_grad()
                actor_loss_i.backward(retain_graph= True)
                self.actor_optim.step()
                actor_losses.append(actor_loss_i.item())

            for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # return target_q1_losses, target_q2_losses, actor_losses

    # Helper function for printing critic gradients. 
    def print_gradients(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.grad}")

    def select_action(self, obs):
        obs = [torch.from_numpy(o).to(self.device) for o in obs]
        with torch.no_grad():
            action, _ = self.actor.sample(obs)

        return action.cpu().numpy()[0] 

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.q1.state_dict(), '%s/%s_critic1.pth' % (directory, filename))
        torch.save(self.q2.state_dict(), '%s/%s_critic2.pth' % (directory, filename))
    
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.q1.load_state_dict(torch.load('%s/%s_critic1.pth' % (directory, filename)))
        self.q2.load_state_dict(torch.load('%s/%s_critic2.pth' % (directory, filename)))


# Define SACAgent class and its methods here (same as before)

if __name__ == "__main__":

    # Set hyperparameters
    obs_dim = 4 # observation space for each robot
    act_dim = 2 # action space
    hidden_dim = 256
    buffer_size = int(1e6)
    batch_size = 4
    gamma = 0.99
    tau = 0.05
    alpha = 0.00002
    lr = 1e-5
    updates_per_step = 1
    start_steps = 20000
    actor_update_frequency = 6

    # Create the environment
    env = MultiRobotEnv('main.launch',3,5)

    # Create the agent
    agent = SACAgent(obs_dim, act_dim, hidden_dim, buffer_size, batch_size, gamma, tau, alpha, lr, actor_update_frequency)

    # Check if there are saved model weights to load
    if os.path.exists('files/sac_actor.pth') and os.path.exists('files/sac_critic1.pth') and os.path.exists('files/sac_critic2.pth'):
        agent.load('sac', 'files')
        print("Loaded saved model weights.")

    # Create the replay buffer
    replay_buffer = ReplayBuffer(buffer_size, batch_size)

    # Create a list to store episode rewards
    episode_rewards = []

    # Set the initial state
    obs = env.reset()

    # Start the training loop
    for t in range(start_steps):
        # Sample actions from the agent
        actions = []
        for i in range(len(obs)):
            action = agent.select_action(obs[i])
            actions.append(action)
        actions = np.array(actions)

        # Take a step in the environment
        obs_list, rewards, dones, _ = env.step(actions)
        
        # Store the experience in the replay buffer
        for i in range(len(obs)):
            replay_buffer.push(obs[i], actions[i], rewards[i], obs_list[i], dones[i])
            
        # Update the state
        obs = obs_list
        # Update the agent if enough samples are available
        if len(replay_buffer) > batch_size:
            for i in range(updates_per_step):
                batch = replay_buffer.sample()
                agent.update(batch)
                
        # Store the episode rewards
        for r in rewards:
            episode_rewards.append(r)
            
        # If the episode is over, reset the environment
        if all(dones):
            obs = env.reset()

        print(rewards)
        print(f"{t}: {actions}")

        # Print the current episode reward and save the model weights
        if t % 1000 == 0:
            print("Episode: " , t , " reward: ", np.mean(episode_rewards[-1000:]))
            agent.save('sac', 'files')