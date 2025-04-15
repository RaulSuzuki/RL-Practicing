import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from shower_environment import ShowerEnv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training using device:", device)
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 32
MEMORY_SIZE = 10000
EPSILON_START = 0.5
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10
EPISODES = 500

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.model(x)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor([state], dtype=torch.float32).unsqueeze(0).to(device)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor([next_state], dtype=torch.float32).unsqueeze(0).to(device)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.cat(next_states).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)

        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
env = ShowerEnv()
state_dim = 1
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

epsilon = EPSILON_START
epsiode_rewards = []

for episode in tqdm(range(EPISODES)):
    state, _ = env.reset()
    state = torch.tensor([state], dtype=torch.float32).unsqueeze(0).to(device)
    total_reward = 0
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = policy_net(state)
                action = torch.argmax(q_values).item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.push(state, action, reward, next_state, done)
        state = torch.tensor([next_state], dtype=torch.float32).unsqueeze(0).to(device)
        total_reward += reward

        if len(memory) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
            actions = actions.unsqueeze(1)

            q_values = policy_net(states).gather(1, actions).squeeze()
            next_q_values = target_net(next_states).max(1)[0]
            expected_q = rewards + GAMMA * next_q_values * (~dones)

            loss = nn.MSELoss()(q_values, expected_q * (~dones))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsiode_rewards.append(total_reward)
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

torch.save(policy_net.state_dict(), "dqn_showerenv.pth")

plt.plot(epsiode_rewards)
plt.title("Reward per episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
plt.savefig("training_reward.png")