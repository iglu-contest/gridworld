from collections import defaultdict
import numpy as np
import pandas as pd
import os
import sys
import shutil
import json
from zipfile import ZipFile

from .load import download
from ..tasks.task import Task

BUILD_ZONE_SIZE_X = 11
BUILD_ZONE_SIZE_Z = 11
BUILD_ZONE_SIZE = 9, 11, 11


if 'IGLU_DATA_PATH' in os.environ:
    DATA_PREFIX = os.path.join(os.environ['IGLU_DATA_PATH'], 'data', 'cdm')
else:
    DATA_PREFIX = os.path.join(os.environ['HOME'], '.iglu', 'data', 'cdm')


class CDMDataset:
    """
    Dataset from paper Collaborative dialogue in Minecraft [1]. 

    Contains 156 structures of blocks, ~550 game sessions (several game sessions per
    structure), 15k utterances. 

    Note that this dataset cannot split the collaboration into instructions since 
    the invariant (of instruction/grid sequence) align does not hold for this dataset.


    [1] Anjali Narayan-Chen, Prashant Jayannavar, and Julia Hockenmaier. 2019. 
    Collaborative Dialogue in Minecraft. In Proceedings of the 57th Annual Meeting 
    of the Association for Computational Linguistics, pages 5405-5415, Florence, 
    Italy. Association for Computational Linguistics.
    """
    ALL = {}
    URL = "https://iglumturkstorage.blob.core.windows.net/public-data/cdm_dataset.zip"
    block_map = {
        'air': 0,
        'cwc_minecraft_blue_rn': 1,
        'cwc_minecraft_yellow_rn': 2,
        'cwc_minecraft_green_rn': 3,
        'cwc_minecraft_orange_rn': 4,
        'cwc_minecraft_purple_rn': 5,
        'cwc_minecraft_red_rn': 6,
    }
    def __init__(self, task_kwargs=None):
        self.task_index = None
        if task_kwargs is None:
            task_kwargs = {}
        self._load_data(
            force_download=os.environ.get('IGLU_FORCE_DOWNLOAD', '0') == '1',
        )
        self.task_kwargs = task_kwargs
        self.tasks = defaultdict(list)
        self.current = None
        for task_id, task_sessions in self.task_index.groupby('structure_id'):
            if len(task_sessions) == 0:
                continue
            for _, session in task_sessions.iterrows():
                task_path = os.path.join(DATA_PREFIX, session.group, 'logs', session.session_id)
                task = Task(*self._parse_task(task_path, task_id), **self.task_kwargs)
                self.tasks[task_id.lower()].append(task)

    def reset(self):
        sample = np.random.choice(list(self.tasks.keys()))
        sess_id = np.random.choice(len(self.tasks[sample]))
        self.current = self.tasks[sample][sess_id]
        return self.current

    def __len__(self):
        return len(t for ts in self.tasks.values() for t in ts)
    
    def __iter__(self):
        for ts in self.tasks.values():
            for t in ts:
                yield t

    def set_task(self, task_id):
        self.current = self.tasks[task_id]
        return self.current

    def _load_data(self, force_download=False):
        path = sys.modules[__name__].__file__
        path_dir, _ = os.path.split(path)
        tasks = pd.read_csv(os.path.join(path_dir, 'task_names.txt'), sep='\t', names=['task_id', 'name'])
        CDMDataset.ALL = dict(tasks.to_records(index=False))
        if not os.path.exists(DATA_PREFIX):
            os.makedirs(DATA_PREFIX, exist_ok=True)
        path = os.path.join(DATA_PREFIX, 'data.zip')
        done = len(list(filter(lambda x: x.startswith('data-'), os.listdir(DATA_PREFIX)))) == 16
        if done and not force_download:
            self.task_index = pd.read_csv(os.path.join(DATA_PREFIX, 'index.csv'))
            shutil.rmtree(path, ignore_errors=True)
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
