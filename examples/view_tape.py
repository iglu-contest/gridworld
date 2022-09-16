import math
import gym
from gridworld.data import IGLUDataset
from gridworld.data.iglu_dataset import block_colour_map

dataset = IGLUDataset()

from os.path import join as pjoin
from gridworld.data.adapter.parse import ActionsParser

#
# session_id = 4
# task_id = "c96"
# turn_id = 4


task_id = "c118"
session_id = 2
turn_id = 12
from ipdb import set_trace; set_trace()

parser = ActionsParser(hits_table=pjoin(dataset.get_data_path(), dataset.DIALOGS_FILENAME))
session = parser.parse_session(pjoin(dataset.get_data_path(), 'builder-data'), f"{session_id}-{task_id}", turn_id, steps=1)

#from gridworld.data.adapter.adapter import ActionsAdapter

# import pathlib
# from gridworld.visualizer import Visualizer
# vis = Visualizer((512,512))
# vis.render_video(pathlib.Path('test'), session.events[turn_id], close=True)

# Retrieve expert actions
#actions = dataset.get_expert_actions(task_id, session_id, turn_id)

import gym
import numpy as np
import gridworld
from gridworld.tasks import DUMMY_TASK

# create vector based env. Rendering is enabled by default.
env = gym.make('IGLUGridworld-v0', action_space='flying', render=True, render_size=(512,512))
env.set_task(DUMMY_TASK)
print(f'Action space: {env.action_space}')
# print(f'Observation space: {env.observation_space}')

done = False
obs = env.reset()
#for action in expert_actions:
last_look = np.array([0, 0])
last_position = np.array([0, 0, 0])
recovering_world_state = False

init_pos = None
all_actions = []
from ipdb import set_trace; set_trace()
for i, event in enumerate(session.events[turn_id]):
    actions = []
    if env.action_space_type == 'walking':
        if env.discretize:
            # The action space format is the following:
            #    0 - no-op
            #    1 - step forward
            #    2 - step backward
            #    3 - step left
            #    4 - step right
            #    5 - jump
            #    6-11 - inventory select
            #    12 - move the camera left
            #    13 - move the camera right
            #    14 - move the camera up
            #    15 - move the camera down
            #    16 - break block
            #    17 - place block

            if event.kind == "action":
                if event.params[0] == "start_recover_world_state":
                    recovering_world_state = True
                elif event.params[0] == "finish_recover_world_state":
                    recovering_world_state = False
                elif event.params[0] == "select_and_place_block":
                    x, y, z, bid = event.grid[-1]
                    actions.append(11 + bid - 1)  # Select block
                    actions.append(17) # Place block

            elif event.kind == "set_look":
                if recovering_world_state:
                    last_look = event.camera
                else:
                    look_offset = event.camera - last_look
                    if look_offset[0] > 0:

                        actions.append()  # Select block

        else:
            self.action_space = Dict({
                'forward': Discrete(2),
                'back': Discrete(2),
                'left': Discrete(2),
                'right': Discrete(2),
                'jump': Discrete(2),
                'attack': Discrete(2),
                'use': Discrete(2),
                'camera': Box(low=-5, high=5, shape=(2,)),
                'hotbar': Discrete(7)
            })

    elif env.action_space_type == 'flying':
        #self.action_space = Dict({
        #    'movement': Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        #    'camera': Box(low=-5, high=5, shape=(2,), dtype=np.float32),
        #    'inventory': Discrete(7),
        #    'placement': Discrete(3),
        #})
        #ction: dictionary with keys:
        #   * 'movement':  Box(low=-1, high=1, shape=(3,)) - forward/backward, left/right,
        #       up/down movement
        #   * 'camera': Box(low=[-180, -90], high=[180, 90], shape=(2,)) - camera movement (yaw, pitch)
        #   * 'inventory': Discrete(7) - 0 for no-op, 1-6 for selecting block color
        #   * 'placement': Discrete(3) - 0 for no-op, 1 for placement, 2 for breaking

        if event.kind == "action":
            print("-->", event.params[0])

            if event.params[0] == "start_recover_world_state":
                recovering_world_state = True
            elif event.params[0] == "finish_recover_world_state":
                recovering_world_state = False
                init_pos = tuple(event.position) + tuple(event.camera)
                env.initialize_world(event.grid, init_pos)
            elif event.params[0] == "select_and_place_block":
                if not recovering_world_state:
                    bid = block_colour_map.get(int(event.params[1]), 1)
                    actions.append({
                        'movement': np.array((0, 0, 0), dtype=np.float32),
                        'camera': np.array((0, 0), dtype=np.float32),
                        'inventory': bid,
                        'placement': 1,  # Place block
                    })
            elif event.params[0] == "step_forward":
                last_movement = np.array((-1, 0, 0), dtype=np.float32)
            elif event.params[0] == "step_backward":
                last_movement = np.array((1, 0, 0), dtype=np.float32)
            elif event.params[0] == "step_left":
                last_movement = np.array((0, -1, 0), dtype=np.float32)
            elif event.params[0] == "step_right":
                last_movement = np.array((0, 1, 0), dtype=np.float32)

        elif event.kind == "set_look":
            if not recovering_world_state:
                actions.append({
                    'movement': np.array((0, 0, 0), dtype=np.float32),
                    'camera': event.camera - last_look,
                    'inventory': 0,  # Air block
                    'placement': 0,  # Do nothing
                })

            last_look = event.camera
            #env.initialize_world(event.grid, tuple(event.position) + tuple(event.camera))

        elif event.kind == "pos_change":
            if not recovering_world_state:
                if event.position[1] != last_position[1]:
                    print(event.__dict__)

                position_offset = (event.position - last_position)
                dx, dy, dz = position_offset

                #print(env.agent.position)
                # print(np.sqrt(np.sum(last_position - env.agent.position)**2))

                norm = np.sqrt(np.sum(position_offset**2))
                print(norm)
                if norm >= 0.14336:# and last_movement is not None and cpt >= 0:

                    norm_2d = np.sqrt(position_offset[2]**2 + position_offset[0]**2)
                    deg1 = math.degrees(math.atan2(position_offset[2]/norm_2d, position_offset[0]/norm_2d))
                    deg2 = env.agent.rotation[0]
                    #if norm >= 0.14336:# and last_movement is not None and cpt >= 0:
                    #    print(deg1, deg2, (deg1-deg2)%360, (deg2-deg1)%360, (deg1+deg2)%360)
                    rectified_motion_vector = (deg1-deg2) % 360
                    m = rectified_motion_vector

                    fwd_bwd = 0
                    left_right = 0
                    # if m < 0:
                    #     from ipdb import set_trace; set_trace()

                    if 350 <= m or m <= 10:
                        left_right = 1
                    elif 35 <= m and m <= 55:
                        left_right = 1
                        fwd_bwd = 1
                    elif 80 <= m and m <= 100:
                        fwd_bwd = 1
                    elif 125 <= m and m <= 145:
                        fwd_bwd = 1
                        left_right = -1
                    elif 170 <= m and m <= 190:
                        left_right = -1
                    elif 215 <= m and m <= 235:
                        fwd_bwd = -1
                        left_right = -1
                    elif 260 <= m and m <= 280:
                        fwd_bwd = -1
                    elif 305 <= m and m <= 325:
                        fwd_bwd = -1
                        left_right = 1
                    # else:
                        # from ipdb import set_trace; set_trace()

                #print(norm)
                if norm >= 0.14336:# and last_movement is not None and cpt >= 0:
                    if event.position[1] != last_position[1]:
                        fwd_bwd = 0
                        left_right = 0

                    actions.append({
                        'movement': (fwd_bwd, left_right, dy),
                        'camera': np.array((0, 0), dtype=np.float32),
                        'inventory': 0,  # Air block
                        'placement': 0,  # Do nothing
                    })
                # actions.append({
                #     'movement': position_offset,
                #     'camera': np.array((0, 0), dtype=np.float32),
                #     'inventory': 0,  # Air block
                #     'placement': 0,  # Do nothing
                # })

            last_position = event.position
            #env.initialize_world(event.grid, tuple(event.position) + tuple(event.camera))
            # env.initialize_world(event.grid, tuple(event.position) + tuple(env.agent.rotation))

    # from ipdb import set_trace; set_trace()
    for action in actions:
        #print(action['movement'])
        obs, reward, done, info = env.step(action)

    all_actions += actions


for i in range(240):
    # No-op
    all_actions.append({
        'movement': (0, 0, 0),
        'camera': np.array((i/100, 0), dtype=np.float32),
        'inventory': 0,  # Air block
        'placement': 0,  # Do nothing
    })

import os
from gridworld.tasks import Tasks
from cool_rotating_video import generate_multiview_video
def dump_video(subtask, actions, init_pos, folder_path="./", name="multiview"):
    #name_key = '-'.join([str(s) for s in self.task_key])
    video_name = os.path.join(folder_path, name)

    # Rerun env with bigger observation size
    env =  gym.make('IGLUGridworld-v0',
                    action_space="flying", render=True,
                    render_size=(512, 512),
                    size_reward=False, max_steps=500, vector_state=True)
    env.unwrapped.initial_position = init_pos[:3]
    env.unwrapped.initial_rotation = init_pos[-2:]
    env.set_task(subtask)
    obs = env.reset()
    #env.initialize_world(subtask.starting_grid, init_pos)

    last_instruction = subtask.last_instruction
    #starting = get_index_and_vals(subtask.starting_grid)
    target = Tasks.to_sparse(subtask.target_grid.transpose((1, 0, 2)))

    frames, grids = [], []
    frames.append(obs['pov'][..., ::-1])
    grids.append(Tasks.to_sparse(obs['grid'].transpose((1, 0, 2))))
    for action in actions:
        obs, _, done, _ = env.step(action)
        frames.append(obs['pov'][..., ::-1])
        grids.append(Tasks.to_sparse(obs['grid'].transpose((1, 0, 2))))
        #if done:
        #    break

    generate_multiview_video(subtask,
                             last_instruction,
                             frames, grids,
                             subtask.starting_grid, target,
                             video_name)

subtask = dataset.tasks['c96'][7].create_task(0, 1)
dump_video(subtask, all_actions, init_pos)
