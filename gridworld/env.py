import pyglet
pyglet.options["headless"] = True
from gridworld.world import World
from gridworld.control import Agent
from gridworld.render import Renderer, setup
from gridworld.task import Task

from gym.spaces import Dict, Box, Discrete
from gym import Env
import numpy as np
from copy import copy
import os


class GridWorld(Env):
    def __init__(self, target, render=True) -> None:
        self.world = World()
        self.agent = Agent(self.world, sustain=False)
        self.grid = np.zeros((9, 11, 11), dtype=np.int32)
        self.task = Task('', target)
        self.world.add_callback('on_add', self.add_block)
        self.world.add_callback('on_remove', self.remove_block)
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
        self.observation_space = Dict({
            'agentPos': Box(low=0, high=360, shape=(5,)),
            'inventory': Box(low=0, high=20, shape=(6,))
        })
        self.max_int = 0
        self.do_render = render
        if render:
            self.renderer = Renderer(self.world, self.agent,
                                     width=64, height=64, caption='Pyglet', resizable=False)
            setup()
        else:
            self.renderer = None

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
        for block in self.world.placed:
            self.world.remove_block(block)
        self.agent.position = (0, 0, 0)
        self.agent.rotation = (0, 0)
        self.agent.inventory = [20 for _ in range(6)]
        obs = {
            'agentPos': np.array([0, 0, 0, 0, 0]),
            'inventory': [20 for _ in range(6)]
        }
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

    def step(self, action):
        strafe, jump, inventory, camera, remove, add = self.parse_action(action)
        self.agent.movement(strafe=strafe, jump=jump, inventory=inventory)
        self.agent.move_camera(*camera)
        self.agent.place_or_remove_block(remove=remove, place=add)
        self.agent.update(dt=1/20.)
        x, y, z = self.agent.position
        pitch, yaw = self.agent.rotation
        obs = {'agentPos': np.array([x, y, z, pitch, yaw])}
        obs['inventory'] = copy(self.agent.inventory)
        obs['grid'] = self.grid.copy()
        
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
        return obs, reward, done, {'target_grid': self.task.target_grid}


def create_env_example():
    target = np.zeros((9, 11, 11), dtype=np.int32)
    target[0, 0, 0] = 1
    target[0, 1, 0] = 2
    target[0, 2, 0] = 1
    target[1, 2, 0] = 2
    target[2, 2, 0] = 1
    env = GridWorld(target, render=True)
    return env