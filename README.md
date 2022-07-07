# IGLU Gridworld RL Environment

Fast and scalable reinforcement learning environment. The env represents an embodied agent with an ability to navigate, place, and break blocks of six different colors.

IGLU is a research project aimed at bridging the gap between reinforcement learning and natural language understanding in Minecraft as a collaborative environment. It provides the RL environment where the goal of an agent is to build structures within a dedicated zone. The structures are described by natural language in the game’s chat.

The main documentation is available (TODO) here.

## Installation

#### Local installation

Install env:

```
pip install git+https://github.com/iglu-contest/gridworld.git@master
```

#### Conda:

Clone the repo and build local conda env:

```
git clone https://github.com/iglu-contest/gridworld.git
cd gridworld && conda env create -f env.yml
```

#### Docker:

```
docker build -t gridworld -f ./docker/Dockerfile .
```

## Usage:

```
import gym
import gridworld
from gridworld.tasks import DUMMY_TASK

# create vector based env. Rendering is enabled by default. 
# To turn it off, use render=False.
env = gym.make('IGLUGridworld-v0')

# It is mandatory to task the environemnt. 
# For dummy loopng, you can use a DUMMY_TASK object:
env.set_task(DUMMY_TASK)
done = False
obs = env.reset()
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
```

## Working with IGLU dataset 

## Known Issues
