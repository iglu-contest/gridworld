from pandas import NA
import pyglet
pyglet.options["headless"] = True
from gridworld.world import World
from gridworld.control import Agent
from gridworld.render import Renderer, setup
from gridworld.task import Task

from gym.spaces import Dict, Box, Discrete
from gym import Env, Wrapper
import numpy as np
from copy import copy
from math import fmod
from uuid import uuid4


class GridWorld(Env):
    def __init__(self, target, render=True, max_steps=250, select_and_place=False, discretize=False) -> None:
        self.world = World()
        self.agent = Agent(self.world, sustain=False)
        self.grid = np.zeros((9, 11, 11), dtype=np.int32)
        self.task = Task('', target)
        self.step_no = 0
        self.max_steps = max_steps
        self.world.add_callback('on_add', self.add_block)
        self.world.add_callback('on_remove', self.remove_block)
        self.right_placement = 0
        self.wrong_placement = 0
        self.select_and_place = select_and_place
        self.discretize = discretize
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
            'grid': Box(low=-1, high=7, shape=(9, 11, 11), dtype=np.float32)
        }
        # if render:
        #     self.observation_space['pov'] = Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.observation_space = Dict(self.observation_space)
        self.max_int = 0
        self.do_render = render
        if render:
            self.renderer = Renderer(self.world, self.agent,
                                     width=64, height=64,
                                     caption='Pyglet', resizable=False)
            setup()
        else:
            self.renderer = None
            self.world._initialize()
        self.reset()

    def enable_renderer(self):
        if self.renderer is None:
            self.reset()
            self.world.deinit()
            self.renderer = Renderer(self.world, self.agent,
                                     width=64, height=64,
                                     caption='Pyglet', resizable=False)
            setup()
            self.do_render = True
            # self.observation_space['pov'] = Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

    def add_block(self, position, kind, build_zone=True):
        if self.world.initialized and build_zone:
            x, y, z = position
            x += 5
            z += 5
            y += 1
            self.grid[y, x, z] = kind

    def remove_block(self, position, build_zone=True):
        if self.world.initialized and build_zone:
            # import pdb
            # pdb.set_trace()
            x, y, z = position
            x += 5
            z += 5
            y += 1
            if self.grid[y, x, z] == 0:
                raise ValueError(f'Removal of non-existing block. address: y={y}, x={x}, z={z}; '
                                 f'grid state: {self.grid.nonzero()[0]};')
            self.grid[y, x, z] = 0

    def reset(self):
        self.prev_grid_size = 0
        self.max_int = 0
        self.step_no = 0
        for block in set(self.world.placed):
            self.world.remove_block(block)
        self.agent.position = (0, 0, 0)
        self.agent.prev_position = (0, 0, 0)
        self.agent.rotation = (0, 0)
        self.agent.inventory = [20 for _ in range(6)]
        obs = {
            'agentPos': np.array([0., 0., 0., 0., 0.], dtype=np.float32),
            'inventory': np.array([20. for _ in range(6)], dtype=np.float32),
            'compass': np.array([0.], dtype=np.float32),
        }
        obs['grid'] = self.grid.copy().astype(np.float32)
        # print('>>>>>>>.', obs['grid'].nonzero())
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
        # print(self.agent.position, self.agent.rotation, action)
        # print('>>>>>>>>>>')
        self.step_no += 1
        self.agent.prev_position = self.agent.position
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
        obs['grid'] = self.grid.copy().astype(np.float32)
        obs['compass'] = np.array([yaw - 180.,], dtype=np.float32)
        # print('>>>>>>>.', obs['grid'].nonzero())

        # done = (self.step_no == self.max_steps)
        # reward = 0
        grid_size = (self.grid != 0).sum().item()
        wrong_placement = (self.prev_grid_size - grid_size) * 0.1
        max_int = self.task.maximal_intersection(self.grid) if wrong_placement != 0 else self.max_int
        done = max_int == self.task.target_size
        self.prev_grid_size = grid_size
        right_placement = (max_int - self.max_int) * 1
        self.max_int = max_int
        if right_placement == 0:
            reward = wrong_placement
        else:
            reward = right_placement
        self.right_placement = right_placement
        self.wrong_placement = wrong_placement
        done = done or (self.step_no == self.max_steps)
        # done = self.step_no == self.max_steps
        # reward = x - self.agent.prev_position[0] + z - self.agent.prev_position[2]
        return obs, reward, done, {'target_grid': self.task.target_grid}

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
            6, 7, #8, 9, 10, 11, # hotbar
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

class Visual(Wrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.c = None
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.w = None
        self.data = defaultdict(list)
        self.logging = False
        self.turned_off = True
        self.glob_step = 0
        self.observation_space['obs'] = Box(low=0, high=1, shape=(64, 64, 3), dtype=np.float32)

    def turn_on(self):
        self.turned_off = False

    def set_idx(self, ix, glob_step):
        self.c = ix
        self.glob_step = glob_step
        self.w = cv2.VideoWriter(f'episodes/step{self.glob_step}_ep{self.c}.mp4', self.fourcc, 20, (64,64))

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # pov = self.env.render()
        # self.w.write(pov)
        pov = self.env.render()[..., :-1]
        obs['obs'] = pov.astype(np.float32) / 255.
        if not self.turned_off:

            if self.logging:
                for key in obs:
                    self.data[key].append(obs[key])
                self.data['reward'].append(reward)
                self.data['done'].append(done)
                self.w.write(pov)
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        if not self.turned_off:
            if self.logging:
                if not os.path.exists('episodes'):
                    os.makedirs('episodes', exist_ok=True)
                for k in self.data.keys():
                    self.data[k] = np.stack(self.data[k], axis=0)
                np.savez_compressed(f'episodes/step{self.glob_step}_ep{self.c}.npz', **self.data)
                self.data = defaultdict(list)
                self.w.release()
                fname = f'step{self.glob_step}_ep{self.c}'
                os.system(f'ffmpeg -y -hide_banner -loglevel error -i episodes/{fname}.mp4 -vcodec libx264 episodes/{fname}1.mp4 '
                          f'&& mv episodes/{fname}1.mp4 episodes/{fname}.mp4')
                self.w = None
                self.c += 1000
                self.w = cv2.VideoWriter(f'episodes/step{self.glob_step}_ep{self.c}.mp4', self.fourcc, 20, (64,64))
        obs['obs'] = self.env.render()[..., :-1].astype(np.float32) / 255.

        return obs

    def enable_renderer(self):
        self.env.enable_renderer()
        self.logging = True


import atexit
class ActionsSaver(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = []
        self.path = f'episodes/actions/{str(uuid4().hex)}.csv'
        self.f = open(self.path, 'w')
        self.f.write('action\n')
        atexit.register(self.reset)

    def reset(self):
        obs = super().reset()
        self.f.close()
        os.makedirs('episodes/actions', exist_ok=True)
        self.path = f'episodes/actions/{str(uuid4().hex)}.csv'
        self.f = open(self.path, 'w')
        self.f.write('action\n')
        return obs

    def step(self, action):
        self.f.write(f'{action}\n')
        return super().step(action)


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


def create_env(visual=True, discretize=True, size_reward=True, select_and_place=True, log_actions=False):
    target = np.zeros((9, 11, 11), dtype=np.int32)
    # target[0, 5, 5] = 1
    # target[0, 6, 5] = 1
    # target[0, 7, 5] = 1
    # target[1, 7, 5] = 1
    # target[2, 7, 5] = 1
    target[0, 4, 4] = 1
    target[0, 6, 4] = 1
    target[0, 4, 6] = 1
    target[0, 6, 6] = 1
    for i in range(4, 7):
        for j in range(4, 7):
            if i == 5 and j == 5:
                continue
            target[1, i, j] = 2
    print(target.nonzero()[0].shape)
    env = GridWorld(target, render=visual, select_and_place=select_and_place, discretize=discretize)
    if visual:
        env = Visual(env)
    if log_actions:
        env = ActionsSaver(env)
    if size_reward:
        env = SizeReward(env)

    # env = Actions(env)
    print(env.action_space)
    return env
