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

For each movement action, the agent steps for about 0.25 of one block. Camera movement changes each angle by 5 degrees. 

**Flying actions** allow the agent to fly freely within the building zone. Placement actions are the same and movement is specified by a continuous vector.

```
env = gym.make('IGLUGridworld-v0', action_space='flying')
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

### Observation space

Observation space format:

```
Dict(
  inventory: Box(low=0, high=20, shape=(6,)),
  compass: Box(low=-180, high=180, shape=(1,)),
  dialog: String(),
  pov: Box(low=0, high=255, shape=(64, 64, 3))
)
```

Here, `inventory` indicates the total blocks available to the agent (per color).
The `compass` component shows the angle between the agent's yaw angle and the North direction.
The `dialog` value is the full previous dialog and the most recent instruction to execute.
Finally, `pov` is an ego-centric image of the agent's observation of the world.
Note that **this space will be used during the evaluation.**
However, it is possible to access other fields of the environment, for example, during training.
The `vector_state=True` passed as keyword argument in `gym.make` will return, in addition to previous fields,

```
agentPos: Box(low=[-8, -2, -8, -90, 0], high=[8, 12, 8, 90, 360], shape=(5,)),
grid: Box(low=-1, high=7, shape=(9, 11, 11))
```

It is also possible to make a target grid a part of the observation space. To do that, pass `target_in_obs=True` to `gym.make`. This will add another key to the observation space with the same structure as the `grid` component. The name of a new component is `target_grid`. This part of the space remains fixed within an episode.

## Working with the IGLU dataset 

<video src='https://raw.githubusercontent.com/iglu-contest/gridworld/aicrowd-launch-prep/assets/c118_1_step_0.mp4' width=512></video>

By default, the environment requires a task object to run.
IGLU dataset provides a convenient loader for RL tasks. Here is an example of how to use it:

```
import gym
from gridworld.data import IGLUDataset

dataset = IGLUDataset(dataset_version='v0.1.0-rc1') 
# leave dataset_version empty to access the most recent version of the dataset.

env = gym.make('IGLUGridworld-v0')
env.set_task_generator(dataset)
```

In this example, we download the dataset of tasks for RL env. 
Internally, on each `.reset()` of the env, the dataset samples a random task (inside its own `.reset()` method) and makes it active in the env. The `Task` object is responsible for calculating the reward, providing the text part of the observation, and determining if the episode has ended.

The structure of the IGLU dataset is following. The dataset consists of structures that represent overall collaboration goals. For each structure, we have several collaboration sessions that pair architects with builders to build each particular structure. Each session consists of a sequence of "turns". Each turn represents an *atomic* instruction and corresponding changes of the blocks in the world. The structure of a `Task` object is following:

  * `target_grid` - target blocks configuration that needs to be built
  * `starting_grid` - optional, blocks for the environment to begin the episode with.
  * `chat` - full conversation between the architect and builder, including the most recent instruction
  * `last_instruction` - last utterance of the architect

Sometimes, the instructions can be ambiguous and the builder asks a clarifying question which the architect answers. In the latter case, `last_instruction` will contain three utterances: an instruction, a clarifying question, and an answer to that question. Otherwise, `last_instruction` is just one utterance of the architect.

To represent collaboration sessions, the `Subtasks` class is used. This class represents a sequence of dialog utterances and their corresponding goals (each of which is a partially completed structure). On `.reset()` call, it picks a random turn and returns a `Task` object, where starting and target grids are consecutive partial structures and the dialog contains all utterances up until the one corresponding to the target grid.

In the example above, the dataset object is structured as follows:

```
# .tasks is a dict mapping from structure to a list of sessions of interaction
dataset.tasks 
# each value contains a list corresponding to collaboration sessions.
dataset.tasks['c73']
# Each element of this list is an instance of `Subtasks` class
dataset.tasks['c73'][0]
```

The `.reset()` method of `IGLUDataset` does effectively the following:

```
def reset(dataset):
  task_id = random.choice(dataset.tasks.keys())
  session = random.choice(dataset.tasks[task_id])
  subtask = session.reset() # Task object is returned
  return subtask
```

This behavior can be customized simply by overriding the reset method in a subclass:

```
import gym
from gridworld.data import IGLUDataset

class MyDataset(IGLUDataset):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.my_task_id = 'c73'
    self.my_session = 0
  
  def reset(self):
    return self.tasks[self.my_task_id][self.my_session].reset()

env = gym.make('IGLUGridworld-v0')
my_dataset = MyDataset(dataset_version='v0.1.0-rc1')
env.set_task_generator(my_dataset)
# do training/sampling
```

On the first creation, the dataset is downloaded and parsed automatically. Below you will find the structure of the dataset:

```
dialogs.csv
builder-data/
  ...
  1-c118/ # session id - structure_id
    step-2
  ...
  9-c118/
    step-2
    step-4
    step-6
  1-c120/
    step-2
  ...
  23-c126/
    step-2
    step-4
    step-6
    step-8
```

Here, `dialog.csv` contains the utterances of architects and builders solving different tasks in 
different sessions. The `builder-data/` directory contains builder behavior recorded by the voxel.js engine. Right now we extract only the resulting grids and use them as targets.

## Known Issues
