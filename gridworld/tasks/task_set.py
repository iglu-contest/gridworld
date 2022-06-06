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

from .load import download
from .task import Task

BUILD_ZONE_SIZE_X = 11
BUILD_ZONE_SIZE_Z = 11
BUILD_ZONE_SIZE = 9, 11, 11


if 'IGLU_DATA_PATH' in os.environ:
    DATA_PREFIX = os.path.join(os.environ['IGLU_DATA_PATH'], 'data')
else:
    DATA_PREFIX = os.path.join(os.environ['HOME'], '.iglu', 'data')

class CDMDataset:
    """
    Dataset from paper Collaborative dialogue in Minecraft [1]. 

    Contains 156 structures of blocks, 509 game sessions (several game sessions per
    structure), 15k utterances. 

    Note that this dataset cannot split the collaboration into instructions since 
    the invariant (of instruction/grid sequence) align does not hold for this dataset.


    [1] Anjali Narayan-Chen, Prashant Jayannavar, and Julia Hockenmaier. 2019. 
    Collaborative Dialogue in Minecraft. In Proceedings of the 57th Annual Meeting 
    of the Association for Computational Linguistics, pages 5405-5415, Florence, 
    Italy. Association for Computational Linguistics.
    """
    ALL = {}
    URL = "https://storage.googleapis.com/iglu_dataset/cdm.zip"
    block_map = {
        'air': 0,
        'cwc_minecraft_blue_rn': 1,
        'cwc_minecraft_yellow_rn': 2,
        'cwc_minecraft_green_rn': 3,
        'cwc_minecraft_orange_rn': 4,
        'cwc_minecraft_purple_rn': 5,
        'cwc_minecraft_red_rn': 6,
    }
    def __init__(self, update_task_dict=False):
        self.index = None
        self._load_data(
            force_download=os.environ.get('IGLU_FORCE_DOWNLOAD', '0') == '1',
            update_task_dict=update_task_dict)
        self.tasks = defaultdict(list)
        self.current = None
        for task_id, task_sessions in self.index.groupby('structure_id'):
            if len(task_sessions) == 0:
                continue
            for _, session in task_sessions.iterrows():
                task_path = os.path.join(DATA_PREFIX, session.group, 'logs', session.session_id)
                task = Task(*self._parse_task(task_path, task_id, update_task_dict=update_task_dict))
                self.tasks[task_id].append(task)

    def reset(self):
        sample = np.random.choice(len(self.tasks))
        sess_id = np.random.choice(len(self.tasks[sample]))
        self.current = self.tasks[sample][sess_id]
        return self.current

    def set_task(self, task_id):
        self.current = self.tasks[task_id]
        return self.current

    def _load_data(self, force_download=False, update_task_dict=False):
        if update_task_dict:
            path = sys.modules[__name__].__file__
            path_dir, _ = os.path.split(path)
            tasks = pd.read_csv(os.path.join(path_dir, 'task_names.txt'), sep='\t', names=['task_id', 'name'])
            CDMDataset.ALL = dict(tasks.to_records(index=False))
        if not os.path.exists(DATA_PREFIX):
            os.makedirs(DATA_PREFIX, exist_ok=True)
        path = os.path.join(DATA_PREFIX, 'data.zip')
        done = len(list(filter(lambda x: x.startswith('data-'), os.listdir(DATA_PREFIX)))) == 16
        if done and not force_download:
            return
        if force_download:
            for dir_ in os.listdir(DATA_PREFIX):
                if dir_.startswith('data-'):
                    shutil.rmtree(os.path.join(DATA_PREFIX, dir_), ignore_errors=True)
        if not os.path.exists(path) or force_download:
            download(
                url=CDMDataset.URL,
                destination=path,
                data_prefix=DATA_PREFIX
            )
            with ZipFile(path) as zfile:
                zfile.extractall(DATA_PREFIX)
        self.task_index = pd.read_csv(os.path.join(DATA_PREFIX, 'index.csv'))
        shutil.rmtree(path, ignore_errors=True)

    def _parse_task(self, path, task_id, update_task_dict=False):
        if not os.path.exists(path):
            # try to unzip logs.zip
            path_prefix, top = path, ''
            while top != 'logs':
                path_prefix, top = os.path.split(path_prefix)
            with ZipFile(os.path.join(path_prefix, 'logs.zip')) as zfile:
                zfile.extractall(path_prefix)
        with open(os.path.join(path, 'postprocessed-observations.json'), 'r') as f:
            data = json.load(f)
        data = data['WorldStates'][-1]
        chat = '\n'.join(data['ChatHistory'])
        target_grid = np.zeros(BUILD_ZONE_SIZE, dtype=np.int32)
        total_blocks = 0
        for block in data['BlocksInGrid']:
            coord = block['AbsoluteCoordinates']
            x, y, z = coord['X'], coord['Y'], coord['Z']
            if not (-5 <= x <= 5 and -5 <= z <= 5 and 0 <= y <= 8):
                continue
            target_grid[
                coord['Y'] - 1,
                coord['X'] + 5,
                coord['Z'] + 5
            ] = CDMDataset.block_map[block['Type']]
            total_blocks += 1
        if update_task_dict:
            colors = len(np.unique([b['Type'] for b in data['BlocksInGrid']]))
            CDMDataset.ALL[task_id] = f'{CDMDataset.ALL[task_id]} ({total_blocks} blocks, {colors} colors)'
        return chat, target_grid

    def __repr__(self):
        tasks = ", ".join(f'"{t}"' for t in self.task_ids)
        return f'TaskSet({tasks})'

    @staticmethod
    def subset(task_set):
        return {k: v for k, v in CDMDataset.ALL.items() if k in task_set}


class CustomTasks(CDMDataset):
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


class RandomTasks(CDMDataset):
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
_ = CDMDataset(preset=[f'C{j}' for j in range(1, 158)], update_task_dict=True)

ALL_TASKS = CDMDataset.subset([f'C{k}' for k in range(1, 158)])
SIMPLEST_TASKS = CDMDataset.subset(['C3', 'C8', 'C12', 'C14', 'C32', 'C17'])
