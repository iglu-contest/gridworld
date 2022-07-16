import warnings
import os
from gridworld.core.world import Agent, World
from gridworld.tasks.task import Task, Tasks

from gym.spaces import Dict, Box, Discrete, Space
from gym import Env, Wrapper as gymWrapper
import gym
import numpy as np
from copy import copy

class String(Space):
    def __init__(self, ):
        super().__init__(shape=(), dtype=np.object_)

    def sample(self):
        return ''

    def contains(self, obj):
        return isinstance(obj, str)


class Wrapper(gymWrapper):
    def __getattr__(self, name):
        return getattr(self.env, name)


class GridWorld(Env):
    def __init__(
            self, render=True, max_steps=250, select_and_place=False,
            discretize=False, right_placement_scale=1., wrong_placement_scale=0.1,
            render_size=(64, 64), target_in_obs=False, action_space='walking', 
            vector_state=True, name='') -> None:
        self.agent = Agent(sustain=False)
        self.world = World()
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
        self.target_in_obs = target_in_obs
        self.vector_state = vector_state
        self.discretize = discretize
        self.action_space_type = action_space
        self.starting_grid = None
        self._overwrite_starting_grid = None
        self.initial_position = (0, 0, 0)
        self.initial_rotation = (0, 0)
        if action_space == 'walking':
            if discretize:
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
        elif action_space == 'flying':
            self.action_space = Dict({
                'movement': Box(low=-1, high=1, shape=(3,), dtype=np.float32),
                'camera': Box(low=-5, high=5, shape=(2,), dtype=np.float32),
                'inventory': Discrete(7),
                'placement': Discrete(3),
            })
            self.agent.flying = True
        self.observation_space = {
            'inventory': Box(low=0, high=20, shape=(6,), dtype=np.float32),
            'compass': Box(low=-180, high=180, shape=(1,), dtype=np.float32),
            'dialog': String()
        }
        if vector_state:
            self.observation_space['agentPos'] = Box(
                low=np.array([-8, -2, -8, -90, 0], dtype=np.float32),
                high=np.array([8, 12, 8, 90, 360], dtype=np.float32),
                shape=(5,)
            )
            self.observation_space['grid'] = Box(low=-1, high=7, shape=(9, 11, 11), dtype=np.int32)
        if target_in_obs:
            self.observation_space['target_grid'] = Box(low=-1, high=7, shape=(9, 11, 11), dtype=np.int32)
        if render:
            self.observation_space['pov'] = Box(low=0, high=255, shape=(*self.render_size, 3), dtype=np.uint8)
        self.observation_space = Dict(self.observation_space)
        self.max_int = 0
        self.name = name
        self.do_render = render

        if render:
            from gridworld.render import Renderer, setup
            self.renderer = Renderer(self.world, self.agent,
                                     width=self.render_size[0], height=self.render_size[1],
                                     caption='Pyglet', resizable=False)
            setup()
        else:
            self.renderer = None
            self.world._initialize()

    def enable_renderer(self):
        if self.renderer is None:
            from gridworld.render import Renderer, setup
            self.reset()
            self.world.deinit()
            from gridworld.render import Renderer, setup
            self.renderer = Renderer(self.world, self.agent,
                                     width=self.render_size[0], height=self.render_size[0],
                                     caption='Pyglet', resizable=False)
            setup()
            self.do_render = True

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
        Assigns provided task into the environment. On each .reset, the env
        Queries the .reset method for the task object. This method should drop
        the task state to the initial one.
        Note that the env can only work with non-None task or task generator.
        """
        if self._task_generator is not None:
            warnings.warn("The .set_task method has no effect with an initialized tasks generator. "
                          "Drop it using .set_tasks_generator(None) after calling .set_task")
        self._task = task
        self.reset()

    def set_task_generator(self, task_generator: Tasks):
        """
        Sets task generator for the current environment. On each .reset, the environment
        queries the .reset method of generator which returns the next task according to the generator.
        Note that the env can only work with non-None task or task generator.
        """
        self._task_generator = task_generator
        self.reset()

    def initialize_world(self, starting_grid, initial_poisition):
        """
        """
        self._overwrite_starting_grid = starting_grid
        warnings.warn(
            'Default task starting grid is overwritten using .initialize_world method. '
            'Use .deinitialize_world to restore the original state.'
        )
        self.initial_position = tuple(initial_poisition[:3])
        self.initial_rotation = tuple(initial_poisition[3:])
        self.reset()

    def deinitialize_world(self):
        self._overwrite_starting_grid = None
        self.initial_position = (0, 0, 0)
        self.initial_rotation = (0, 0)
        self.reset()

    @property
    def task(self):
        if self._task is None:
            if self._task_generator is None:
                raise ValueError('Task is not initialized! Initialize task before working with'
                                ' the environment using .set_task method OR set tasks distribution using '
                                '.set_task_generator method')
            self._task = self._task_generator.reset()
            self.starting_grid = self._task.starting_grid
        return self._task

    def reset(self):
        if self._task is None:
            if self._task_generator is None:
                raise ValueError('Task is not initialized! Initialize task before working with'
                                ' the environment using .set_task method OR set tasks distribution using '
                                '.set_task_generator method')
            else:
                # yield new task
                self._task = self._task_generator.reset()
        elif self._task_generator is not None:
            self._task = self._task_generator.reset()
        self.step_no = 0
        self._task.reset()
        if self._overwrite_starting_grid is not None:
            self.starting_grid = self._overwrite_starting_grid
        else:
            self.starting_grid = self._task.starting_grid
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
            'inventory': np.array(self.agent.inventory, dtype=np.float32),
            'compass': np.array([0.], dtype=np.float32),
            'dialog': self._task.chat
        }
        if self.vector_state:
            obs['grid'] = self.grid.copy().astype(np.int32)
            obs['agentPos'] = np.array([0., 0., 0., 0., 0.], dtype=np.float32)
        if self.target_in_obs:
            obs['target_grid'] = self._task.target_grid.copy().astype(np.int32)
        if self.do_render:
            obs['pov'] = self.render()[..., :-1]
        return obs

    def render(self,):
        if not self.do_render:
            raise ValueError('create env with render=True')
        return self.renderer.render()

    def step(self, action):
        if self._task is None:
            if self._task_generator is None:
                raise ValueError('Task is not initialized! Initialize task before working with'
                                ' the environment using .set_task method OR set tasks distribution using '
                                '.set_task_generator method')
            else:
                raise ValueError('Task is not initialized! Run .reset() first.')
        self.step_no += 1
        self.world.step(
            self.agent, action, select_and_place=self.select_and_place,
            action_space=self.action_space_type, discretize=self.discretize
        )
        x, y, z = self.agent.position
        yaw, pitch = self.agent.rotation
        obs = {}
        obs['inventory'] = np.array(copy(self.agent.inventory), dtype=np.float32)
        obs['compass'] = np.array([yaw - 180.,], dtype=np.float32)
        obs['dialog'] = self._task.chat
        if self.vector_state:
            obs['grid'] = self.grid.copy().astype(np.int32)
            obs['agentPos'] = np.array([x, y, z, pitch, yaw], dtype=np.float32)
        right_placement, wrong_placement, done = self._task.step_intersection(self.grid)
        done = done or (self.step_no == self.max_steps)
        if right_placement == 0:
            reward = wrong_placement * self.wrong_placement_scale
        else:
            reward = right_placement * self.right_placement_scale
        if self.target_in_obs:
            obs['target_grid'] = self._task.target_grid.copy().astype(np.int32)
        if self.do_render:
            obs['pov'] = self.render()[..., :-1]
        return obs, reward, done, {}


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
        render=True, discretize=True, size_reward=True, select_and_place=True,
        right_placement_scale=1, render_size=(64, 64), target_in_obs=False,
        vector_state=False, max_steps=250, action_space='walking',
        wrong_placement_scale=0.1, name=''
    ):
    env = GridWorld(
        render=render, select_and_place=select_and_place,
        discretize=discretize, right_placement_scale=right_placement_scale,
        wrong_placement_scale=wrong_placement_scale, name=name,
        render_size=render_size, target_in_obs=target_in_obs,
        vector_state=vector_state, max_steps=max_steps,
        action_space=action_space
    )
    if size_reward:
        env = SizeReward(env)
    # env = Actions(env)
    return env

gym.envs.register(
     id='IGLUGridworld-v0',
     entry_point='gridworld.env:create_env',
     kwargs={}
)

gym.envs.register(
     id='IGLUGridworldVector-v0',
     entry_point='gridworld.env:create_env',
     kwargs={'vector_state': True, 'render': False}
)
