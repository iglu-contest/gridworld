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


class GridWorld(Env):
    def __init__(self, target, render=True, max_steps=500, discretize=False) -> None:
        self.world = World()
        self.agent = Agent(self.world, sustain=False)
        self.grid = np.zeros((9, 11, 11), dtype=np.int32)
        self.task = Task('', target)
        self.step_no = 0
        self.max_steps = max_steps
        self.world.add_callback('on_add', self.add_block)
        self.world.add_callback('on_remove', self.remove_block)
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
            'agentPos': Box(low=-1000, high=1000, shape=(5,)),
            'inventory': Box(low=0, high=20, shape=(6,)),
            'grid': Box(low=-1, high=7, shape=(9, 11, 11))
        }
        if render:
            self.observation_space['pov'] = Box(low=0, high=255, shape=(64, 64, 3))
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

    def add_block(self, position, kind ):
        if self.world.initialized:
            x, y, z = position
            x += 5
            z += 5
            y += 1
            self.grid[y, x, z] = kind

    def remove_block(self, position):
        if self.world.initialized:
            x, y, z = position
            self.grid[y, x, z] = 0

    def reset(self):
        self.prev_grid_size = 0
        self.max_int = 0
        self.step_no = 0
        for block in set(self.world.placed):
            self.world.remove_block(block)
        self.agent.position = (0, 0, 0)
        self.agent.rotation = (0, 0)
        self.agent.inventory = [20 for _ in range(6)]
        obs = {
            'agentPos': np.array([0., 0., 0., 0., 0.], dtype=np.float32),
            'inventory': np.array([20. for _ in range(6)], dtype=np.float32)
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
        strafe, jump, inventory, camera, remove, add = self.parse(action)
        self.agent.movement(strafe=strafe, jump=jump, inventory=inventory)
        self.agent.move_camera(*camera)
        self.agent.place_or_remove_block(remove=remove, place=add)
        self.agent.update(dt=1/20.)
        x, y, z = self.agent.position
        pitch, yaw = self.agent.rotation
        yaw = fmod(yaw, 360)
        obs = {'agentPos': np.array([x, y, z, pitch, yaw], dtype=np.float32)}
        obs['inventory'] = np.array(copy(self.agent.inventory), dtype=np.float32)
        obs['grid'] = self.grid.copy().astype(np.float32)
        # print('>>>>>>>.', obs['grid'].nonzero())
        
        grid_size = (self.grid != 0).sum().item()
        wrong_placement = (self.prev_grid_size - grid_size) * 1
        max_int = self.task.maximal_intersection(self.grid) if wrong_placement != 0 else self.max_int
        done = max_int == self.task.target_size
        self.prev_grid_size = grid_size
        right_placement = (max_int - self.max_int) * 2
        self.max_int = max_int
        if right_placement == 0:
            reward = wrong_placement
        else:
            reward = right_placement
        done = done or (self.step_no == self.max_steps)
        return obs, reward, done, {'target_grid': self.task.target_grid}

import cv2

class Visual(Wrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        # self.c = 0
        # self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # self.w = cv2.VideoWriter(f'test{self.c}.mp4', self.fourcc, 20, (640,640))

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # pov = self.env.render()
        # self.w.write(pov)
        obs['pov'] = self.env.render()
        return obs, reward, done, info
    
    def reset(self):
        obs = super().reset()
        # self.w = cv2.VideoWriter('test{self.c}.mp4',self.fourcc, 20, (640,640))
        # self.c += 1
        # pov = self.env.render()
        # self.w.write(pov)
        obs['pov'] = self.env.render()
        return obs



def create_env(visual=True, discretize=True):
    target = np.zeros((9, 11, 11), dtype=np.int32)
    target[0, 5, 5] = 1
    target[0, 6, 5] = 2
    target[0, 7, 5] = 1
    target[1, 7, 5] = 2
    target[2, 7, 5] = 1
    env = GridWorld(target, render=visual, discretize=discretize)
    if visual:
        env = Visual(env)
    return env