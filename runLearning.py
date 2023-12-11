import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
import math
import json
from readInGameValues import get_cash, get_lives, get_round
from Towers import tower_from_dict
import environment
import keyboard
import pandas as pd
import os
from collections import namedtuple, deque

# Global flag if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global flag to control training continuation
continue_training = True
# Key listener function
def on_key_press(event):
    global continue_training
    if event.name == '/':  
        continue_training = False

# Attach the key listener
keyboard.on_press(on_key_press)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# Define the Q-Network using PyTorch
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, dropout_rate=0.5):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Memory Class From Pytorch Documentation
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(state):
    q_values = policy_net(state)  
    action_mask = game_env.generate_action_mask()

    if np.random.rand() <= epsilon:
        # Exploration: Select a random action from valid actions only
        PlaceSpace = 7*25*25-1
        valid_actions = [action for action, valid in enumerate(action_mask) if valid == 1 and action > PlaceSpace]
        if valid_actions and len(valid_actions) > 1:
            if np.random.rand() > 0.5:
                action = torch.tensor(np.random.choice(valid_actions[:-1]), dtype=torch.int64).to(device)
                exploration_type = 'Random Upgrade Only'
            else:
                valid_actions = [action for action, valid in enumerate(action_mask) if valid == 1]
                action = torch.tensor(np.random.choice(valid_actions), dtype=torch.int64).to(device)
                exploration_type = 'Random Placement or Round Start, No Upgrade Available'
        else:
            valid_actions = [action for action, valid in enumerate(action_mask) if valid == 1]
            action = torch.tensor(np.random.choice(valid_actions), dtype=torch.int64).to(device)
            exploration_type = 'Random Any'

    else:
        # Exploitation: Choose the best action based on model prediction, but considering only valid actions
        with torch.no_grad():
            masked_q_values = q_values.masked_fill(action_mask == 0, -float('inf'))
            action = torch.tensor(torch.argmax(masked_q_values).item(), dtype=torch.int64).to(device)
            exploration_type = 'Model Selected'

    return action, exploration_type

# Modified Function from Pytorch Documentation
def optimize_model():
    if len(memory) < BATCH_SIZE:
            return 0
    transitions = memory.sample(BATCH_SIZE)
    #This converts batch-array of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)  
    action_batch = torch.cat([a.unsqueeze(0) for a in batch.action]).unsqueeze(-1)
    reward_batch = torch.cat([a.unsqueeze(0) for a in batch.reward])
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss

# Wait for Me to Open the Game
time.sleep(5)

game_env = environment.GameEnv()

# Hyperparameters
BATCH_SIZE = 60
GAMMA = 0.95
starting_episode = 0
NUM_EPISODES = 150
EPS_0 = 0.90
EPS_MIN = 0.1
EPS_DECAY = 0.98
TAU = 0.005
LR = 1e-3

# Define input shapes and create the PyTorch model
n_observations = game_env.get_state_size()
n_actions = game_env.get_total_action_size()

policy_net = QNetwork(n_observations, n_actions).to(device)
target_net = QNetwork(n_observations, n_actions).to(device)
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

# Comment Out if you want to start from scratch
checkpoint = torch.load('PostData/checkpoint_episode_70.pth')
policy_net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


policy_net.train()
target_net.load_state_dict(policy_net.state_dict())
target_net.train()

# Main training loop
for episode in range(starting_episode, NUM_EPISODES):
    
    print(f"Starting episode {episode + 1}")

    # Initialize state
    exploration_type = None
    game_env.reset_state()

    episode_data = []
    total_reward = 0.0
    done = False
    start_time = time.time()
    total_loss = 0
    steps = 0
    reward = 0.0
    epsilon = EPS_0*EPS_DECAY**(episode)
    if epsilon < EPS_MIN:
        epsilon = EPS_MIN

    while not done:
        if not continue_training:
            break
        if done:
            break
        state = torch.tensor(game_env.state, dtype=torch.float32, device=device).unsqueeze(0)
        action, exploration_type = select_action(state)

        next_state, reward = game_env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
    
        steps +=1

        reward = float(reward)

        # Hacky fix, for some reason the first action gives turn 1 a balooned reward value. It looks to be gone but not sure what the cause was. Leaving this in for now
        if action == 5625 and reward > 10:
            reward = 1.0

        if get_lives() <= 0:
            done = True
            reward -= 1000.0

        if get_round() >= 50:
            done = True
            reward += 1000.0
        
        if time.time() - start_time > 60*20: #Timeout condition
            done = True
            reward -= 1000.0
        total_reward += reward
        reward = torch.tensor(reward, dtype=torch.float32).to(device)

        memory.push(state, action, next_state, reward)
        loss = optimize_model()

        # Soft update of the target network's weights
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        action = action.item()
        reward = reward.item()
        if loss != 0:
            loss = loss.item()
            total_loss += loss
        episode_data.append(
                            {'steps': steps,
                             'state': state.tolist(),
                             'action': action,
                             'reward': reward,
                             'loss': loss,
                             'Exploration Type' : exploration_type,
                             'total_reward': total_reward,
                             'total_loss': total_loss,
                             'epsilon': epsilon,
                             'episode': episode + 1}
                            )

        time.sleep(0.005)
        
    episode_df = pd.DataFrame(episode_data)
    episode_df.to_csv(os.path.join('PostData', f'episode_{episode + 1}.csv'), index=False)
    
    summary_data = {'episode': episode + 1, 'total_reward': total_reward, 'total_loss': total_loss, 'steps': steps, 'epsilon': epsilon}
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(os.path.join('PostData', 'summary.csv'), mode='a', header=not os.path.exists(os.path.join('PostData', 'summary.csv')), index=False)
    
    round = get_round()
    if not continue_training:
        print('Ended Due to Keyboard Interrupt')
        break
    if get_lives() <= 0:
        game_env.reset_after_loss()
        print('Ended Due to Lives')
    if get_round() >= 50:
        time.sleep(10)  
        game_env.reset_after_win()
        print('Ended Due to Win Condition')

    if time.time() - start_time > 60*20: #Timeout condition
        game_env.reset_mid_game()
        print('Ended Due to Time')
    reward = 0
    if episode % 10 == 0:
        checkpoint = {
            'episode': episode,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

        torch.save(checkpoint, f'PostData/checkpoint_episode_{episode}.pth')

    print(f"Episode {episode + 1} finished. Highest Round: {round} Total Reward: {total_reward}.\n")

