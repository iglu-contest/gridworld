import os
import re
import json
import shutil
import sys
import pickle
import uuid
from zipfile import ZipFile
import pandas as pd
from collections import defaultdict
from typing import Tuple, List

import numpy as np

from .task import Task, Tasks


BUILD_ZONE_SIZE_X = 11
BUILD_ZONE_SIZE_Z = 11
BUILD_ZONE_SIZE = 9, 11, 11


class CustomTasks(Tasks):
    """ TaskSet that consists of user-defined goal structures

    Args:
        goals (List[Tuple[str, np.ndarray]]): list of tasks.
            Each task is represented by a pair
            (string conversation, 3d numpy grid)
    """
    def __init__(self, goals: List[Tuple[str, np.ndarray]]):
        super().__init__()
        self.tasks = {
            str(uuid.uuid4().hex): Task(conversation, grid)
            for conversation, grid in goals
        }
        self.task_ids = list(self.tasks.keys())


class RandomTasks(Tasks):
    """
    TaskSet that consists of number of randomly generated tasks

    Args:
        max_blocks (``int``): The maximal number of blocks in each task. Defaults to 3.
        height_levels (``int``): How many height levels random blocks can occupy. Defaults to 1.
        allow_float (``bool``): Whether to allow blocks to have the air below. Defaults to False.
        max_dist (``int``): Maximum (Chebyshev) distance between two blocks. Defaults to 2.
        num_colors (``int``): Maximum number of unique colors. Defaults to 1.
        max_cache (``int``): If zero, each `.sample_task` will generate new task. Otherwise, the number of random tasks to cache. Defaults to 0.

    """
    def __init__(
        self, max_blocks=4,
        height_levels=1, allow_float=False, max_dist=2,
        num_colors=1, max_cache=0,
    ):
        self.height_levels = height_levels
        self.max_blocks = max_blocks
        self.allow_float = allow_float
        self.max_dist = max_dist
        self.num_colors = num_colors
        self.max_cache = max_cache
        self.tasks = {}
        self.current = None
        for _ in range(self.max_cache):
            uid = str(uuid.uuid4().hex)
            self.tasks[uid] = self.sample_task()
        self.sample()

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump({uid: t.target_grid for uid, t in self.tasks.items()}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            grids = pickle.load(f)
        self.tasks = {uid: Task('', g) for uid, g in grids.items()}

    def __repr__(self):
        hps = dict(
            max_blocks=self.max_blocks,
            height_levels=self.height_levels,
            allow_float=self.allow_float,
            max_dist=self.max_dist,
            num_colors=self.num_colors,
            max_cache=self.max_cache,
        )
        hp_str = ', '.join(f'{k}={v}' for k, v in hps.items())
        return f'RandomTasks({hp_str})'

    def sample(self):
        if self.max_cache > 0:
            sample = np.random.choice(list(self.tasks.keys()))
            self.current = self.tasks[sample]
            self.current_id = sample
            return self.current
        else:
            self.current = self.sample_task()
            return self.current

    def set_task(self, task_id):
        self.current = self.tasks[task_id]
        return self.current

    def sample_task(self):
        chat = ''
        target_grid = np.zeros(BUILD_ZONE_SIZE, dtype=np.int32)
        for height in range(self.height_levels):
            shape = target_grid[height].shape
            block_x = np.random.choice(BUILD_ZONE_SIZE_X)
            block_z = np.random.choice(BUILD_ZONE_SIZE_Z)
            color = np.random.choice(self.num_colors) + 1
            target_grid[height, block_x, block_z] = color
            for _ in range(self.max_blocks - 1):
                block_delta_x, block_delta_z = 0, 0
                while block_delta_x == 0 and block_delta_z == 0 \
                        or block_x + block_delta_x >= BUILD_ZONE_SIZE_X \
                        or block_z + block_delta_z >= BUILD_ZONE_SIZE_Z \
                        or block_x + block_delta_x < 0 \
                        or block_z + block_delta_z < 0 \
                        or target_grid[height, block_x + block_delta_x, block_z + block_delta_z] != 0:
                    block_delta_x = np.random.choice(2 * self.max_dist + 1) - self.max_dist
                    block_delta_z = np.random.choice(2 * self.max_dist + 1) - self.max_dist
                    color = np.random.choice(self.num_colors) + 1
                target_grid[height, block_x + block_delta_x, block_z + block_delta_z] = color

        return Task(chat, target_grid)

# to initialize task descriptions
# _ = CDMDataset(preset=[f'C{j}' for j in range(1, 158)], update_task_dict=True)

# ALL_TASKS = CDMDataset.subset([f'C{k}' for k in range(1, 158)])
# SIMPLEST_TASKS = CDMDataset.subset(['C3', 'C8', 'C12', 'C14', 'C32', 'C17'])
