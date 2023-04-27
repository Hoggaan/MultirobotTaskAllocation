import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
from buffer import ReplayBuffer
from sac_env import Gazebo_Env
import time
from utils import plot_learning_curve

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
            name, chkpt_dir='results'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # Sensor observation - O_t_l
        self.conv1 = nn.Conv1d(3, 32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.fc_otl  = nn.Linear(87*16,256)

        # O_t_o for all other robots. 
        '''
        4D Tensor(n-1,q, N, c)
        N = Number of the goals = 3
        q - Number of past frames = 3
        c = Heading and distance = 2
        n-1 - Number of the other robots = 1 
        '''
        self.fc_oto = nn.Linear(18, 32)

        #After concatination
        self.fc_cat = nn.Linear(303, 256)
        # self.fc_l = nn.Linear(256, 7)
        # self.fc_r = nn.Linear(256, 7)

        # I think this breaks if the env has a 2D state representation
        # self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        # self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        # Sensor Observation (o_t_l)
        otl = state[0]
        sensor_out= F.max_pool1d(F.relu(self.conv1(otl)), 2) # Dimention 0 of the observation space.
        sensor_out = F.max_pool1d(F.relu(self.conv2(sensor_out)), 2)
        sensor_out = sensor_out.view(87 * 16)
        otl__out = self.fc_otl(sensor_out) # Sensor observation out

        # O_t_o For other robots
        oto = state[1]
        oto = oto.view(-1, 3*3*2) # Reshape (n-1, 18)
        oto_out = self.fc_oto(oto)

        # O_t_e # For the ego_robot
        ote = state[2]
        ote_out = ote.view(-1)

        # O_t_aux
        otaux = state[3]

        # Concatination
        obs_cat = T.cat((otl__out, oto_out, ote_out, otaux), dim=0)

        #After concat
        q1 = self.fc_cat(obs_cat)

        # l = F.softmax(self.fc_l(out))
        # r = F.softmax(self.fc_r(out))

        # q1_action_value = self.fc1(T.cat([state, action], dim=1))
        # q1_action_value = F.relu(q1_action_value)
        # q1_action_value = self.fc2(q1_action_value)
        # q1_action_value = F.relu(q1_action_value)

        # q1 = self.q1(q1_action_value)

        return q1

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, max_action,
            n_actions, name, chkpt_dir='results'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.max_action = max_action
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.reparam_noise = 1e-6

        '''Our adjestments'''
        # Sensor observation - O_t_l
        self.conv1 = nn.Conv1d(3, 32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.fc_otl  = nn.Linear(87*16,256)

        # O_t_o for all other robots. 
        '''
        4D Tensor(n-1,q, N, c)
        N = Number of the goals = 3
        q - Number of past frames = 3
        c = Heading and distance = 2
        n-1 - Number of the other robots = 1 
        '''
        self.fc_oto = nn.Linear(18, 32)

        #After concatination
        self.fc_cat = nn.Linear(303, 256)
        self.fc_l = nn.Linear(256, 7)
        self.fc_r = nn.Linear(256, 7)


        # self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        # self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        # Sensor Observation (o_t_l)
        otl = state[0]
        sensor_out= F.max_pool1d(F.relu(self.conv1(otl)), 2) # Dimention 0 of the observation space.
        sensor_out = F.max_pool1d(F.relu(self.conv2(sensor_out)), 2)
        sensor_out = sensor_out.view(87 * 16)
        otl__out = self.fc_otl(sensor_out) # Sensor observation out

        # O_t_o For other robots
        oto = state[1]
        oto = oto.view(-1, 3*3*2) # Reshape (n-1, 18)
        oto_out = self.fc_oto(oto)

        # O_t_e # For the ego_robot
        ote = state[2]
        ote_out = ote.view(-1)

        # O_t_aux
        otaux = state[3]

        # Concatination
        obs_cat = T.cat((otl__out, oto_out, ote_out, otaux), dim=0)

        #After concat
        out = self.fc_cat(obs_cat)

        l = F.softmax(self.fc_l(out))
        r = F.softmax(self.fc_r(out))

        # prob = self.fc1(state)
        # prob = F.relu(prob)
        # prob = self.fc2(prob)
        # prob = F.relu(prob)

        # mu = self.mu(prob)
        # #sigma = T.sigmoid(self.sigma(prob))
        # sigma = self.sigma(prob)
        # sigma = T.clamp(sigma, min=self.reparam_noise, max=1) 
        # # authors use -20, 2 -> doesn't seem to work for my implementation

        return l,r # mu, sigma
    def sample_normal(self, state):
        # vtl = 


    # def sample_normal(self, state, reparameterize=True):
    #     mu, sigma = self.forward(state)
    #     probabilities = T.distributions.Normal(mu, sigma)

    #     if reparameterize:
    #         actions = probabilities.rsample() # reparameterizes the policy
    #     else:
    #         actions = probabilities.sample()

    #     action = T.tanh(actions)*T.tensor(self.max_action).to(self.device) 
    #     log_probs = probabilities.log_prob(actions)
    #     log_probs -= T.log(1-action.pow(2) + self.reparam_noise)
    #     log_probs = log_probs.sum(1, keepdim=True)

    #     return action, log_probs

    def sample_mvnormal(self, state, reparameterize=True):
        """
            Doesn't quite seem to work.  The agent never learns.
        """
        mu, sigma = self.forward(state)
        n_batches = sigma.size()[0]

        cov = [sigma[i] * T.eye(self.n_actions).to(self.device) for i in range(n_batches)]
        cov = T.stack(cov)
        probabilities = T.distributions.MultivariateNormal(mu, cov)

        if reparameterize:
            actions = probabilities.rsample() # reparameterizes the policy
        else:
            actions = probabilities.sample()

        action = T.tanh(actions) # enforce the action bound for (-1, 1)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.sum(T.log(1-action.pow(2) + self.reparam_noise))
        log_probs = log_probs.sum(-1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
            name, chkpt_dir='results'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env,
            env_id, gamma=0.99, 
            n_actions=2, max_size=1000000, layer1_size=256,
            layer2_size=256, batch_size=100, reward_scale=2, q=3):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.q = q

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, n_actions=n_actions,
                                  name=env_id+'_actor', 
                                  max_action=1)
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions,
                                      name=env_id+'_critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions,
                                      name=env_id+'_critic_2')
       
        self.value = ValueNetwork(beta, input_dims, layer1_size,
                                      layer2_size, name=env_id+'_value')
        self.target_value = ValueNetwork(beta, input_dims, layer1_size,
                                         layer2_size, name=env_id+'_target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        #actions, _ = self.actor.sample_mvnormal(state)
        # actions is an array of arrays due to the added dimension in state
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0
       
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        #actions, log_probs = self.actor.sample_mvnormal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * (F.mse_loss(value, value_target))
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        #actions, log_probs = self.actor.sample_mvnormal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5*F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5*F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

# Hyperparameters
# alpha=0.0003 
# beta=0.0003 
# reward_scale=2
# env_id=env_id
# input_dims=24
# tau=0.005
# env=env
# batch_size=256
# layer1_size=256
# layer2_size=256
# n_actions=2
# q = 3  # Last Number of frames


# Run the program

if __name__ == '__main__':
    env_id = 'world_name'
    env = Gazebo_Env('done.launch', 1, 1, 1)
    time.sleep(5)
    agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, env_id=env_id, 
                input_dims=24, tau=0.005,
                env=env, batch_size=40, layer1_size=256, layer2_size=256,
                n_actions=2, q = 3)
    n_games = 250

    """I don't know how this is helpful!"""
    filename = env_id + '_'+ str(n_games) + 'games_scale' + str(agent.scale) + \
                    '_clamp_on_sigma.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]   # Set up the reward system for the env
    score_history = []
    load_checkpoint = True
    if load_checkpoint:
        agent.load_models()
        # env.render(mode='human')  # There is no need!
    steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            a_in = [(action[0] + 1) / 2, action[1]]      # Added 
            observation_, reward, done, info = env.step(action)
            steps += 1
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()      # Agent Learning.
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'trailing 100 games avg %.1f' % avg_score, 
                'steps %d' % steps, env_id, 
                ' scale ', agent.scale)
    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)