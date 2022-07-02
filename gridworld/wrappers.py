import cv2
import os
from collections import defaultdict

from gym.spaces import Discrete
from gym import Env, Wrapper
import numpy as np
from uuid import uuid4


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

