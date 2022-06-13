import pyglet
import warnings
# pyglet.options["headless"] = True
from gridworld.world import World
from gridworld.control import Agent
from gridworld.render import Renderer, setup
from gridworld.tasks.task import Task, Tasks

from gym.spaces import Dict, Box, Discrete
from gym import Env, Wrapper
import numpy as np
from copy import copy
from math import fmod
from uuid import uuid4


class GridWorld(Env):
    def __init__(
            self, render=True, max_steps=250, select_and_place=False,
            discretize=False, right_placement_scale=1., wrong_placement_scale=0.1, 
            render_size=(64, 64), name='') -> None:
        self.world = World()
        self.agent = Agent(self.world, sustain=False)
        self.grid = np.zeros((9, 11, 11), dtype=np.int32)
        self._task = None
        self._task_generator = None
        self.step_no = 0
        self.right_placement_scale = right_placement_scale
        self.wrong_placement_scale = wrong_placement_scale
        self.max_steps = max_steps
        self.world.add_callback('on_add', self._add_block)
        self.world.add_callback('on_remove', self._remove_block)
        self.right_placement = 0
        self.wrong_placement = 0
        self.render_size = render_size
        self.select_and_place = select_and_place
        self.discretize = discretize
        self.initial_position = (0, 0, 0)
        self.initial_rotation = (0, 0)
        if discretize:
            self.parse = self.parse_low_level_action
            self.action_space = Discrete(18)
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
            self.parse = self.parse_action
        self.observation_space = {
            'agentPos': Box(
                low=np.array([-8, -2, -8, -90, 0], dtype=np.float32),
                high=np.array([8, 12, 8, 90, 360], dtype=np.float32),
                shape=(5,)),
            'inventory': Box(low=0, high=20, shape=(6,), dtype=np.float32),
            'compass': Box(low=-180, high=180, shape=(1,), dtype=np.float32),
            'grid': Box(low=-1, high=7, shape=(9, 11, 11), dtype=np.int32),
            'target_grid': Box(low=-1, high=7, shape=(9, 11, 11), dtype=np.int32)
        }
        # if render:
        #     self.observation_space['pov'] = Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.observation_space = Dict(self.observation_space)
        self.max_int = 0
        self.name = name
        self.do_render = render
        if render:
            self.renderer = Renderer(self.world, self.agent,
                                     width=self.render_size[0], height=self.render_size[1],
                                     caption='Pyglet', resizable=False)
            setup()
        else:
            self.renderer = None
            self.world._initialize()
        # self.reset()

    def enable_renderer(self):
        if self.renderer is None:
            self.reset()
            self.world.deinit()
            self.renderer = Renderer(self.world, self.agent,
                                     width=self.render_size[0], height=self.render_size[0],
                                     caption='Pyglet', resizable=False)
            setup()
            self.do_render = True
            # self.observation_space['pov'] = Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

    def _add_block(self, position, kind, build_zone=True):
        if self.world.initialized and build_zone:
            x, y, z = position
            x += 5
            z += 5
            y += 1
            self.grid[y, x, z] = kind

    def _remove_block(self, position, build_zone=True):
        if self.world.initialized and build_zone:
            x, y, z = position
            x += 5
            z += 5
            y += 1
            if self.grid[y, x, z] == 0:
                raise ValueError(f'Removal of non-existing block. address: y={y}, x={x}, z={z}; '
                                 f'grid state: {self.grid.nonzero()[0]};')
            self.grid[y, x, z] = 0

    def set_task(self, task: Task):
        """
        """
        if self._task_generator is not None:
            warnings.warn("The .set_task method has no effect with an initialized tasks generator. "
                          "Drop it using .set_tasks_generator(None) after calling .set_task")
        self._task = task
        self.reset()

    def set_task_generator(self, task_generator: Tasks):
        self._task_generator = task_generator
        self.reset()

    def initialize_world(self, starting_grid, initial_poisition):
        self.starting_grid = starting_grid
        self.initial_position = tuple(initial_poisition[:3])
        self.initial_rotation = tuple(initial_poisition[3:])
        self.reset()

    def reset(self):
        if self._task is None:
            if self._task_generator is None:
                raise ValueError('Task is not initialized! Initialize task before working with'
                                ' the environment using .set_task method OR set tasks distribution using '
                                '.set_task_generator method')
            else:
                # yield new task
                self._task = self._task_generator.reset()
                self.starting_grid = self._task.starting_grid
        self.step_no = 0
        self._task.reset()
        for block in set(self.world.placed):
            self.world.remove_block(block)
        if self.starting_grid is not None:
            for x,y,z, bid in self.starting_grid:
                self.world.add_block((x, y, z), bid)
        self.agent.position = self.initial_position
        self.agent.rotation = self.initial_rotation
        self.max_int = self._task.maximal_intersection(self.grid)
        self.prev_grid_size = len(self.grid.nonzero()[0])
        self.agent.inventory = [20 for _ in range(6)]
        if self.starting_grid is not None:
            for _, _, _, color in self.starting_grid:
                self.agent.inventory[color - 1] -= 1
        obs = {
            'agentPos': np.array([0., 0., 0., 0., 0.], dtype=np.float32),
            'inventory': np.array(self.agent.inventory, dtype=np.float32),
            'compass': np.array([0.], dtype=np.float32),
        }
        obs['grid'] = self.grid.copy().astype(np.int32)
        obs['target_grid'] = self._task.current.target_grid.copy().astype(np.int32)
        return obs

    def render(self,):
        if not self.do_render:
            raise ValueError('create env with render=True')
        return self.renderer.render()

    def parse_action(self, action):
        strafe = [0,0]
        if action['forward']:
            strafe[0] += -1
        if action['back']:
            strafe[0] += 1
        if action['left']:
            strafe[1] += -1
        if action['right']:
            strafe[1] += 1
        jump = bool(action['jump'])
        if action['hotbar'] == 0:
            inventory = None
        else:
            inventory = action['hotbar']
        camera = action['camera']
        remove = bool(action['attack'])
        add = bool(action['use'])
        return strafe, jump, inventory, camera, remove, add

    def parse_low_level_action(self, action):
        # 0 noop; 1 forward; 2 back; 3 left; 4 right; 5 jump; 6-11 hotbar; 12 camera left;
        # 13 camera right; 14 camera up; 15 camera down; 16 attack; 17 use;
        # action = list(action).index(1)
        strafe = [0, 0]
        camera = [0, 0]
        jump = False
        inventory = None
        remove = False
        add = False
        if action == 1:
            strafe[0] += -1
        elif action == 2:
            strafe[0] += 1
        elif action == 3:
            strafe[1] += -1
        elif action == 4:
            strafe[1] += 1
        elif action == 5:
            jump = True
        elif 6 <= action <= 11:
            inventory = action - 5
        elif action == 12:
            camera[0] = -5
        elif action == 13:
            camera[0] = 5
        elif action == 14:
            camera[1] = -5
        elif action == 15:
            camera[1] = 5
        elif action == 16:
            remove = True
        elif action == 17:
            add = True
        return strafe, jump, inventory, camera, remove, add

    def step(self, action):
        if self._task is None:
            if self._task_generator is None:
                raise ValueError('Task is not initialized! Initialize task before working with'
                                ' the environment using .set_task method OR set tasks distribution using '
                                '.set_task_generator method')
            else:
                raise ValueError('Task is not initialized! Run .reset() first.')
        self.step_no += 1
        strafe, jump, inventory, camera, remove, add = self.parse(action)
        if self.select_and_place and inventory is not None:
            add = True
            remove = False
        self.agent.movement(strafe=strafe, jump=jump, inventory=inventory)
        self.agent.move_camera(*camera)
        self.agent.place_or_remove_block(remove=remove, place=add)
        self.agent.update(dt=1/20.)
        x, y, z = self.agent.position
        yaw, pitch = self.agent.rotation
        while yaw > 360.:
            yaw -= 360.
        while yaw < 0.0:
            yaw += 360.0
        self.agent.rotation = (yaw, pitch)
        obs = {'agentPos': np.array([x, y, z, pitch, yaw], dtype=np.float32)}
        obs['inventory'] = np.array(copy(self.agent.inventory), dtype=np.float32)
        obs['grid'] = self.grid.copy().astype(np.int32)
        obs['compass'] = np.array([yaw - 180.,], dtype=np.float32)
        right_placement, wrong_placement, done = self._task.step_intersection(self.grid)
        done = done or (self.step_no == self.max_steps)
        if right_placement == 0:
            reward = wrong_placement * self.wrong_placement_scale
        else:
            reward = right_placement * self.right_placement_scale
        obs['target_grid'] = self._task.current.target_grid.copy().astype(np.int32)
        return obs, reward, done, {}

import cv2
import os
from collections import defaultdict

class Actions(Wrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.action_map = [
            # from new idx to old ones
            0, # noop
            1,2,3,4,
            5, # jump
            6, 7, 8, 9, 10, 11, # hotbar
            12, 13, 14, 15,
            16, # break
            # 17, # place
        ]
        self.action_space = Discrete(len(self.action_map))


    def step(self, action):
        # 0 noop; 1 forward; 2 back; 3 left; 4 right; 5 jump; 6-11 hotbar; 12 camera left;
        # 13 camera right; 14 camera up; 15 camera down; 16 attack; 17 use;
        # if action >= 6:
        #     action += 6
        return self.env.step(self.action_map[action])

class debug(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = []
        self.total_reward = 0
        self.turn = None
        self.turn_goal = None

    def step(self, action):
        self.actions.append(action)
        obs, reward, done, info = super().step(action)
        self.total_reward += reward
        if done:
            import pickle
            if self.total_reward > (18 - (self.turn - 1) * 3):
                if not os.path.exists('wrong_actions'):
                    os.makedirs('wrong_actions', exist_ok=True)
                with open(f'wrong_actions/{uuid4().hex()[:10]}.pkl', 'wb') as f:
                    pickle.dump((self.actions, self.total_reward, self.turn, self.turn_goal), f)                    
                print(f'reward of {self.total_reward} at turn {self.turn}')
        return obs, reward, done, info
    
    def reset(self):
        obs = super().reset()
        self.actions = []
        self.total_reward = 0
        self.turn = self.unwrapped.task.task_start 
        self.turn_goal = self.unwrapped.task.task_goal
        return obs

class Visual(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # ospace = dict(**self.observation_space.spaces)
        # ospace['pov'] = Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        # self.observation_space = Dict(ospace)
    
    def reset(self):
        obs = super().reset()
        obs['pov'] = self.env.render()[..., :-1]
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        pov = self.env.render()[..., :-1]
        obs['pov'] = pov
        return obs, reward, done, info

class Logged(Wrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.data = defaultdict(list)
        self.actions = []
        self.desc = ''
        self.logging = False
        self.turned_off = True
        self.glob_step = 0
        self.path = 'episodes'

    def turn_on(self):
        self.turned_off = False
        self.logging = True
    
    def set_path(self, path):
        self.path = path

    def set_desc(self, desc, glob_step):
        self.desc = desc
        self.glob_step = glob_step

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.logging:
            pov = self.env.render()[..., :-1]
            for key in obs:
                self.data[key].append(obs[key])
            self.data['reward'].append(reward)
            self.data['done'].append(done)
            self.data['pov'].append(pov[..., ::-1])
            self.actions.append(action)
        if done and self.logging and self.unwrapped.step_no != 0:
            path = f'{self.path}/step{self.glob_step}'
            # raise
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            for k in self.data.keys():
                if k != 'pov':
                    self.data[k] = np.stack(self.data[k], axis=0)
            fname = f'ep_{self.desc}_{str(uuid4().hex)[:6]}'
            np.savez_compressed(f'{path}/{fname}.npz', **self.data)
            with open(f'{path}/{fname}.csv', 'w') as f:
                for action in self.actions:
                    f.write(f'{action}\n')
            if len(self.data['pov']) > 1:
                w = cv2.VideoWriter(f'{path}/{fname}.mp4', self.fourcc, 20, (64,64))
                for pov in self.data['pov']:
                    w.write(pov)
                w.release()
            os.system(f'ffmpeg -y -hide_banner -loglevel error -i {path}/{fname}.mp4 -vcodec libx264 {path}/{fname}1.mp4 '
                      f'&& mv {path}/{fname}1.mp4 {path}/{fname}.mp4')
            self.data = defaultdict(list)
            self.actions = []
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        if not self.turned_off:
            pov = self.env.render()[..., :-1]
            self.data['pov'].append(pov[..., ::-1])
            for k in obs:
                self.data[k].append(obs[k])
        return obs

    def enable_renderer(self):
        self.env.enable_renderer()
        self.logging = True

class SizeReward(Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.size = 0

  def reset(self):
    self.size = 0
    return super().reset()

  def step(self, action):
    obs, reward, done, info = super().step(action)
    intersection = self.unwrapped.max_int
    reward = max(intersection, self.size) - self.size
    self.size = max(intersection, self.size)
    reward += min(self.unwrapped.wrong_placement * 0.02, 0)
    return obs, reward, done, info


def create_env(
        visual=True, discretize=True, size_reward=True, select_and_place=True,
        log=False, right_placement_scale=1, render_size=(64, 64),
        wrong_placement_scale=0.1, name=''
    ):
    # target = np.zeros((9, 11, 11), dtype=np.int32)

    # target[0, 5, 5] = 1
    # target[0, 6, 5] = 1
    # target[0, 7, 5] = 1
    # target[1, 7, 5] = 1
    # target[2, 7, 5] = 1

    # target[0, 4, 4] = 1
    # target[0, 6, 4] = 1
    # target[0, 4, 6] = 1
    # target[0, 6, 6] = 1
    # for i in range(4, 7):
    #     for j in range(4, 7):
    #         if i == 5 and j == 5:
    #             continue
    #         target[1, i, j] = 2
    # print(target.nonzero()[0].shape)

    steps = [
    [],
    [
        # purple
        (-3, -1, -3, 5),
        (-3, -1, -2, 5),
        (-3, 0, -3, 5),
    ],
    [
        # blue
        (-2, -1, -2, 1),
        (-2, -1, -1, 1),
        (-2, 0, -2, 1),
    ],
    [
        # green
        (-1, -1, -1, 3),
        (-1, -1, 0, 3),
        (-1, 0, -1, 3),
    ],
    [
        # yellow
        (0, -1, 0, 2),
        (0, -1, 1, 2),
        (0, 0, 0, 2),
    ],
    [
        # orange
        (1, -1, 1, 4),
        (1, -1, 2, 4),
        (1, 0, 1, 4),
    ],
    [
        # red
        (2, -1, 2, 6),
        (2, -1, 3, 6),
        (2, 0, 2, 6),
    ]
    ]
    target = [sum(steps[:i], []) for i in range(1, len(steps) + 1)]
    print(len(target))
    # print('steps!')
    env = GridWorld(
        render=visual, select_and_place=select_and_place,
        discretize=discretize, right_placement_scale=right_placement_scale,
        wrong_placement_scale=wrong_placement_scale, name=name,
        render_size=render_size
    )
    # if visual:
    #     env = Visual(env)
    if size_reward:
        env = SizeReward(env)
    if log:
        env = Logged(env)
    env = Actions(env)
    # env = debug(env)
    # print(env.action_space)
    return env
