# Navigation Project Report: Solving Banana Collector Environment with Deep Q-Network

## Overview
This report describes implementation of my solution to the Banana Collector environment as part of the Navigation Project from Udacity Deep Reinforcement Learning Nanodegree.

My solution implements a deep Q-network (DQN) agent with the following enhancements:

- Prioritized Experience Replay
- Fixed Q-targets
- Double DQN

It is based on the following papers:

- **DQN**: Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529.
- **Prioritized Experience Replay**: Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).
- **Double DQN**: Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep Reinforcement Learning with Double Q-Learning." AAAI. Vol. 2. 2016.

Implementation is done in Python using PyTorch. Documented code is provided in Navigation_DQN.ipynb.


## Problem environment

In this project, an agent navigates in a large, square world and collects bananas.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, an agent must get an average score of +13 over 100 consecutive episodes.

## Learning Algorithm

The main part of the solution is the DQN agent implemented as `PriorityDoubleDQNAgent` class.
This agent uses prioritized replay memory implemented as `PriorityReplayMemory` class. 
It also implements fixed Q-targets and Double DQN which require two Q-networks with the same structure, 
online Q-network for choosing actions and target Q-network used in learning to avoid harmful correlations 
and overestimation of the action values.

Training of the agent is implemented in the `train` function, which has the following flow:

1. Every timestep observation of the environment (state) is provided to the agent and the agent selects the action (`act` method).
2. Then, the next state of the environment is observed and the reward is received, e.g. +1 for collected yellow banana). 
We also get the information whether the episode is completed (done).
3. State, action, next state and the reward constitute the experience provided to the agent for learning.
The agent adds the experience to its replay memory (`percieve` method) and when there are enough experiences collected starts learning (`learn` method)
 
Learning with the prioritized experience replay, fixed Q-targets and Double DQN happens as follows:

1. A batch of experiences is sampled from the replay memory.
2. For each experience, the next action for the next state is selected to maximize the Q-values estimated from 
the online Q-network.
3. Target Q-values for the current state and action are computed as the sum of the reward and the estimated Q-value from 
the target Q-network for the next state and action selected in the previous step.
3. Expected Q-values for the current state is estimated from the online Q-network.
4. Temporal difference errors (TD-error) are computed as the difference between the target and expected Q-values.
5. Mean squares error loss is computed for the TD-errors.
6. Parameters of the online Q-network are updated by minimizing the loss through backpropagation
7. Parameters for the target Q-network are updated with the weighted sum of the target and online parameters.
9. Priorities of the experiences are updated based on the TD-errors.

This description omits some details associated with prioritized expereince replay and the use of hyperpaprameters.
which are described in the subsequent sections 

## Prioritized Experience Replay

The goal of prioritized experience replay is to learn more from difficult experiences.
It is achieved by increasing sampling probability for experiences with higher TD-error.
The original paper suggests two implementations of the replay memory: 
one based on using TD-errors as the priority directly and another on using the rank as the priority. 
Both provide similar results.
Current solution implements the rank-based one, where experiences are sampled from a sorted list with higher probability 
of sampling from the beginning of the list.

Two optimizations are applied to reduce the cost of maintaining a sorted list. 
First, a binary heap is used to maintain the list in approximately sorted order.
Second, the list is divided into a number of segments corresponding to the batch size (number of experiences to sample).
One experience per segment is selected uniformly during sampling.
Sizes of segments vary from smaller segments for the beginning of the list to larger segments toward the end.
This difference in sizes makes it so that experiences in the beginning of the list have higher chance to be sampled than towards the end.

In addition to sampled experiences, sampling also computes importance-sampling weights 
that are used in the learning process to compensate for the skew in distribution of experiences presented to the q-network.

## Q-network architecture

Q-network maps state to Q-values for each action. 
Current implementation used a dense neural network with two hidden layers where each layer has 64 nodes and uses 
rectified linear unit activation function. 
In addition, I used batch normalization, which might result in faster learning.
The complete architecture is as follows:
 
- Input: 37 nodes (state size)
- Hidden layer 1: 64 nodes
    - Batch normalization
    - Rectified linear unit activation 
- Hidden layer 2: 64 nodes
    - Batch normalization
    - Rectified linear unit activation
- Output layer: 4 nodes (number of actions)

## Hyperparameters

    replay_memory_size=int(1e5),
    replay_batch_size=32,
    replay_priority_exp=0.5,
    replay_sort_freq=int(1e3),
    replay_start=500,
    
    qnet_hidden_sizes=[64, 64]
    learn_freq=4
    learn_rate=5e-4
    discount_factor=0.99,
    target_update_freq=1,
    soft_update_factor=1e-3
    
    epsilon_start=1.0, 
    epsilon_end=0.01,
    epsilon_decay=0.995
    
    sample_exp_start=0,
    sample_exp_end=0.5,

## Future work

