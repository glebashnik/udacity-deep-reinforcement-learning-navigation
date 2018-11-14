from collections import namedtuple
import numpy as np
import torch

Experience = namedtuple('Experience', 'state action reward next_state done')

def get_fields_from_experiences(experiences, device):
    states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
    actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
    rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
    next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
    dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
    return states, actions, rewards, next_states, dones