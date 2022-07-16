import os
import json
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from ..tasks.task import Subtasks, Task, Tasks
from .load import download

from zipfile import ZipFile


if 'IGLU_DATA_PATH' in os.environ:
    DATA_PREFIX = os.path.join(os.environ['IGLU_DATA_PATH'], 'data', 'iglu')
elif 'HOME' in os.environ:
    DATA_PREFIX = os.path.join(os.environ['HOME'], '.iglu', 'data', 'iglu')
else:
    DATA_PREFIX = os.path.join(
        os.path.expanduser('~'), '.iglu', 'data', 'iglu')

VOXELWORLD_GROUND_LEVEL = 63

block_colour_map = {
    # voxelworld's colour id : iglu colour id
    0: 0,  # air
    57: 1,  # blue
    50: 6,  # yellow
    59: 2,  # green
    47: 4,  # orange
    56: 5,  # purple
    60: 3  # red
}


def fix_xyz(x, y, z):
    XMAX = 11
    YMAX = 9
    ZMAX = 11
    COORD_SHIFT = [5, -63, 5]

    x += COORD_SHIFT[0]
    y += COORD_SHIFT[1]
    z += COORD_SHIFT[2]

    index = z + y * YMAX + x * YMAX * ZMAX
    new_x = index // (YMAX * ZMAX)
    index %= (YMAX * ZMAX)
    new_y = index // ZMAX
    index %= ZMAX
    new_z = index % ZMAX

    new_x -= COORD_SHIFT[0]
    new_y -= COORD_SHIFT[1]
    new_z -= COORD_SHIFT[2]

    return new_x, new_y, new_z


def fix_log(log_string):
    """
    log_string: str
        log_string should be a string of the full log.
        It should be multiple lines, each corresponded to a timestamp,
        and should be separated by newline character.
    """

    lines = []

    for line in log_string.splitlines():

        if "block_change" in line:
            line_splits = line.split(" ", 2)
            try:
                info = eval(line_splits[2])
            except:
                lines.append(line)
                continue
            x, y, z = info[0], info[1], info[2]
            new_x, new_y, new_z = fix_xyz(x, y, z)
            new_info = (new_x, new_y, new_z, info[3], info[4])
            line_splits[2] = str(new_info)
            fixed_line = " ".join(line_splits)
            # logging.info(f"Fixed {line} to {fixed_line}")

            lines.append(fixed_line)
        else:
            lines.append(line)

    return "\n".join(lines)


class IGLUDataset(Tasks):
    DATASET_URL = {
        "v0.1.0-rc1": 'https://iglumturkstorage.blob.core.windows.net/public-data/iglu_dataset.zip'
    }  # Dictionary holding dataset version to dataset URI mapping

    def __init__(self, dataset_version="v0.1.0-rc1", task_kwargs=None, force_download=False) -> None:
        """
        Collaborative dataset for the IGLU competition.

        Current version of the dataset covers 31 structures in 128 staged game sessions 
        resulting in 608 tasks.

        Args:
            dataset_version: Which dataset version to use. 
            task_kwargs: Task-class specific kwargs. For reference see gridworld.task.Task class
            force_download: Whether to force dataset downloading
        """
        self.dataset_version = dataset_version
        if dataset_version not in IGLUDataset.DATASET_URL.keys():
            raise Exception(
                "Unknown dataset_version:{} provided.".format(dataset_version))
        if task_kwargs is None:
            task_kwargs = {}
        self.task_kwargs = task_kwargs
        path = f'{DATA_PREFIX}/iglu_dataset.zip'
        if not os.path.exists(f'{DATA_PREFIX}/dialogs.csv') or force_download:
            download(
                url=IGLUDataset.DATASET_URL[self.dataset_version],
                destination=path,
                data_prefix=DATA_PREFIX
            )
            with ZipFile(path) as zfile:
                zfile.extractall(DATA_PREFIX)
        dialogs = pd.read_csv(f'{DATA_PREFIX}/dialogs.csv')
        self.tasks = defaultdict(list)
        self.parse_tasks(dialogs, DATA_PREFIX)

    def process(self, s):
        return re.sub(r'\$+', '\n', s)

    def parse_tasks(self, dialogs, path):
        for sess_id, gr in dialogs.groupby('PartitionKey'):
            utt_seq = []
            blocks = []
            if not os.path.exists(f'{path}/builder-data/{sess_id}'):
                continue
            assert len(gr.structureId.unique()) == 1
            structure_id = gr.structureId.values[0]
            for i, row in gr.sort_values('StepId').reset_index(drop=True).iterrows():
                if row.StepId % 2 == 1:
                    if isinstance(row.instruction, str):
                        utt_seq.append([])
                        utt_seq[-1].append(
                            f'<Architect> {self.process(row.instruction)}')
                    elif isinstance(row.Answer4ClarifyingQuestion, str):
                        utt_seq[-1].append(
                            f'<Architect> {self.process(row.Answer4ClarifyingQuestion)}')
                else:
                    if not row.IsHITQualified:
                        continue
                    if isinstance(row.ClarifyingQuestion, str):
                        utt_seq[-1].append(
                            f'<Builder> {self.process(row.ClarifyingQuestion)}')
                        continue
                    blocks.append([])
                    curr_step = f'{path}/builder-data/{sess_id}/step-{row.StepId}'
                    if not os.path.exists(curr_step):
                        break
                        # TODO: in this case the multiturn collection was likely
                        # "reset" so we need to stop parsing this session. Need to check that.
                    with open(curr_step) as f:
                        step_data = json.load(f)
                    for x, y, z, bid in step_data['worldEndingState']['blocks']:
                        y = y - VOXELWORLD_GROUND_LEVEL - 1
                        # TODO: some blocks have id 1, check why
                        bid = block_colour_map.get(bid, 5)
                        blocks[-1].append((x, y, z, bid))
            i = 0
            while i < len(blocks):
                if len(blocks[i]) == 0:
                    if i == len(blocks) - 1:
                        blocks = blocks[:i]
                        utt_seq = utt_seq[:i]
                    else:
                        blocks = blocks[:i] + blocks[i + 1:]
                        utt_seq[i] = utt_seq[i] + utt_seq[i + 1]
                        utt_seq = utt_seq[:i + 1] + utt_seq[i + 2:]
                i += 1
            if len(blocks) > 0:
                task = Subtasks(utt_seq, blocks, **self.task_kwargs)
                self.tasks[structure_id].append(task)

    def reset(self):
        sample = np.random.choice(list(self.tasks.keys()))
        sess_id = np.random.choice(len(self.tasks[sample]))
        self.current = self.tasks[sample][sess_id]
        return self.current.reset()

    def __len__(self):
        return sum(len(sess.structure_seq) for sess in sum(self.tasks.values(), []))

    def __iter__(self):
        for task_id, tasks in self.tasks.items():
            for j, task in enumerate(tasks):
                for k, subtask in enumerate(task):
                    yield task_id, j, k, subtask
