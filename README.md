## Udacity Reinforcement Learning Nanodegree P2 - Continuous Control
This repo has the code for training an RL agent to move a double jointed arm to a target location in a Unity Envtt.There are 3 main files in this submission:
* Continuous_Control.ipynb - This has the code to start the environment, train the agent and then test the trained agent
* ddpg_agent.py - This file has the implementation of the DDPG Agent which is used by Continuous_Control.ipynb 
* model.py - This file has the neural network that is trained to do the funcion approximation

## About the project environment
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

There are two environments available - one with a single agent and the other with multiple (20) agents. This submission uses the multiple agent environment.
For the multiple agent environment, the scores from all the agents are averaged and envtt is considered solved when this average score is +30 over 100 consecutive episodes.

## Setting up the Python Enviroment
The following libraries are needed to run the code:
1. unityagents - ```pip install unityagents```
To see the agent in action, please download the unity environment to your local
Download the environment from one of the links below.  You need only select the environment that matches your operating system:
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
2. Separately download the headless version of the multi agent environment. I found that the training proceeded faster with this option.[this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) 
3. Pytorch - Install latest version based on your OS and machine from https://pytorch.org/get-started/locally/
4. Numpy, Matplotlib


## Training and Visualizing the trained agent
The code for training the agent is in the notebook Continuous_Control.ipynb.

### Training an agent

For training the agent, I chose the headless multi agent Linux envtt. You can load the envtt as below:
```
env = UnityEnvironment(file_name='Reacher_Linux_NoVis/Reacher.x86_64')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
```

Section 1.1 of the notebook has the code for training an agent. The command below sets up the DDPG agent. 
```
from ddpg_agent import Agent
agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)
```
The function ddepg() resets the environment, provides episodes to train the agent, gets actions from agent class and passes the results of the actions (next state, reward) to the agent class.

To train the model. Set train to True and run below:
```
train = True
if train:
    scores = ddpg()
```

### Visualizing the trained agent
Section 1.2 of the notebook has the code for playing in the Unity Environment with the trained agent. The main steps are:
1. Initialize a Multiple Agent Envtt with visualization
2. Load the trained weights into the agent
```
agent = Agent(state_size, action_size, random_seed=0)
agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
```

2. Play a game

```
env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)

while True:
    actions = agent.act(states, add_noise=False)           ## select actions from DDPG policy
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                              # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

```



