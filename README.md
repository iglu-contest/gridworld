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

Note that by default IGLU env runs in headless mode. To run headed do 

```
export IGLU_HEADLESS=0
```

Now, run the environment loop:

```
import gym
import gridworld
from gridworld.tasks import DUMMY_TASK

# create vector based env. Rendering is enabled by default. 
# To turn it off, use render=False.
env = gym.make('IGLUGridworld-v0')

# It is mandatory to task the environemnt. 
# For dummy looping, you can use a DUMMY_TASK object:
env.set_task(DUMMY_TASK)
done = False
obs = env.reset()
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
```

### Action spaces:

Two action spaces are available in Gridworld:

**Walking actions** allow the agent to move with gravity enabled, jump, place, break, and select blocks from inventory. 

```
env = gym.make('IGLUGridworld-v0', action_space='walking')
print(env.action_space) # Discrete(18)
```

The action space format is the following:

  * 0 - no-op
  * 1 - step forward
  * 2 - step backward
  * 3 - step left 
  * 4 - step right
  * 5 - jump
  * 6-11 - inventory select
  * 12 - move the camera left
  * 13 - move the camera right
  * 14 - move the camera up
  * 15 - move the camera down
  * 16 - break block 
  * 17 - place block

For each movement action, the agent steps for about 0.25 of one block. Camera movement changes each angle for 5 degrees. 

**Flying actions** allow the agent to fly freely within the building zone. Placement actions are the same and movement is specified by a continuous vector.

```
env = gym.make('IGLUGridworld-v0', action_space='flying')
print(env.action_space)
```

Action space format:

```
Dict(
  movement: Box(low=-1, high=1, shape=(3,)),
  camera: Box(low=-5, high=5, shape=(2,)),
  inventory: Discrete(7),
  placement: Discrete(3)
)
```

## Working with the IGLU dataset 

IGLU dataset provides a convenient loader for RL tasks. Here is an example of how to use it:

```
from gridworld.data import IGLUDataset


```

## Known Issues
