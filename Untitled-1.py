# %%
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT,MOVEMENT
import numpy as np
import random
import numpy as np
from matplotlib import pyplot as plt

# %%
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cpu")


# %%
def statePreprocess(state):
    #the shape of the play area is from 48 to 208 in the x direction and 96 to 176 in the y direction
    state = state[48:208,96:176]
    grayscale = np.dot(state[...,:3], [0.2989, 0.5870, 0.1140])
    binary_array = grayscale.reshape(20,8,10,8).max(axis=(1,3)) > 0
    return binary_array.astype(int)

# %%
def one_hot_piece(piece):
    # Extended mapping to include variations like 'Td', 'Ld', etc.
    mapping = {'I': 0, 'O': 1, 'T': 2, 'S': 3, 'Z': 4, 'J': 5, 'L': 6,
               'Id': 7, 'Od': 8, 'Td': 9, 'Sd': 10, 'Zd': 11, 'Jd': 12, 'Ld': 13}
    vector = [0] * len(mapping)
    if piece in mapping:  # Check if the piece is recognized
        vector[mapping[piece]] = 1
    return vector

# %%
env = gym_tetris.make('TetrisA-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
state = env.reset()
##state =torch.tensor(np.array(state, copy = True), dtype=torch.float32)


# %%
state, reward, done, info = env.step(0)
current_piece = info['current_piece']
next_piece = info['next_piece']

print(current_piece)
print(next_piece)

#drwaing the state
plt.imshow(statePreprocess(state))

# %%
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# %%
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.long)  # Ensure action is a tensor
        reward = torch.tensor([reward], dtype=torch.float32)  # Ensure reward is a tensor
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)  # Ensure done is a tensor
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# %%
def process_state(grid, current_piece, next_piece):
    grid = statePreprocess(grid)
    # Flatten the grid
    flat_grid = grid.reshape(-1).astype(float)  # Convert grid to a flat, float array

    # One-hot encode the current and next pieces
    current_piece_vector = one_hot_piece(current_piece)
    next_piece_vector = one_hot_piece(next_piece)

    # Combine the flattened grid and the piece vectors into one state vector
    return torch.tensor(np.concatenate([flat_grid, current_piece_vector, next_piece_vector]), dtype=torch.float32)

# %%
# Initialize the DQN
input_dim = 200 + 14 +14  # 200 for the grid, 14 for the one-hot encoded pieces
output_dim = len(SIMPLE_MOVEMENT)  # Number of possible actions
model = DQN(input_dim, output_dim)
model.to(device)

# Initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize the replay memory
replay_memory = []
replay_memory_capacity = 10000
batch_size = 32

# %%
#number of episodes
episodes = 10000

env = gym_tetris.make('TetrisA-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

for episode in range(episodes):
    state = env.reset()
    state, reward, done, info = env.step(0)
    current_piece = info['current_piece']
    next_piece = info['next_piece']
    state = process_state(state, current_piece, next_piece)
    total_reward = 0
    while not done:
        if random.random() < 0.1:
            #select a random action from 1 to 6
            action = random.randint(0, 5)
        else:
            q_values = model(state)
            action = torch.argmax(q_values).item()
            # action = MOVEMENT[torch.argmax(q_values).item()] # Choose the action with the highest Q-value
            # #find the number of the action
            # action = MOVEMENT.index(action)
            # print(action)
        next_state, reward, done, info = env.step(action)
        env.render()
        next_current_piece = info['current_piece']
        next_next_piece = info['next_piece']
        next_state = process_state(next_state, next_current_piece, next_next_piece)
        total_reward += reward

        replay_memory.append((state, torch.tensor([action]), torch.tensor([reward], dtype=torch.float32),
                              next_state, torch.tensor([done], dtype=torch.float32)))

        if len(replay_memory) > batch_size:
            batch = random.sample(replay_memory, batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(torch.stack, zip(*batch))

            q_values = model(state_batch)
            with torch.no_grad():
                next_q_values = model(next_state_batch)
            target_q_values = reward_batch + 0.99 * torch.max(next_q_values, dim=1).values * (1 - done_batch)
            loss = nn.MSELoss()(q_values.gather(1, action_batch), target_q_values.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state
        current_piece = next_current_piece
        next_piece = next_next_piece

    print(f'Episode {episode + 1}, total reward: {total_reward}')


    



