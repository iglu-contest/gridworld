import os
import json
import re
import pandas as pd
import numpy as np
from collections import defaultdict

from ..tasks.task import Subtasks, Task, Tasks
from .load import download

from zipfile import ZipFile

VOXELWORLD_GROUND_LEVEL = 63

block_colour_map = {
    # voxelworld's colour id : iglu colour id
    0: 0,  # air
    57: 1, # blue
    50: 6, # yellow
    59: 2, # green
    47: 4, # orange
    56: 5, # purple
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
    """
    Collaborative dataset for the IGLU competition.

    Current version of the dataset covers 31 structures in 128 staged game sessions
    resulting in 608 tasks.
    """
    URL = 'https://iglumturkstorage.blob.core.windows.net/public-data/iglu_dataset.zip'
    DIALOGS_FILENAME = 'dialogs.csv'

    def __init__(self, task_kwargs=None, force_download=False) -> None:
        if task_kwargs is None:
            task_kwargs = {}
        self.task_kwargs = task_kwargs
        data_path = self.get_data_path()
        self.download_dataset(data_path, force_download)
        dialogs = self.get_instructions(data_path)
        self.tasks = defaultdict(list)
        self.parse_tasks(dialogs, data_path)

    def get_instructions(self, data_path):
        return pd.read_csv(os.path.join(data_path, self.DIALOGS_FILENAME))

    @classmethod
    def get_data_path(cls):
        """Returns the path where iglu dataset will be downloaded and cached.

        It can be set using the environment variable IGLU_DATA_PATH. Otherwise,
        it will be `~/.iglu/data/iglu`.

        Returns
        -------
        str
            The absolute path to data folder.
        """
        if 'IGLU_DATA_PATH' in os.environ:
            data_path = os.path.join(
                os.environ['IGLU_DATA_PATH'], 'data', 'iglu')
        elif 'HOME' in os.environ:
            data_path = os.path.join(
                os.environ['HOME'], '.iglu', 'data', 'iglu')
        else:
            data_path = os.path.join(
                os.path.expanduser('~'), '.iglu', 'data', 'iglu')
        return data_path

    def download_dataset(self, data_path, force_download):
        path = f'{data_path}/iglu_dataset.zip'
        if (not os.path.exists(os.path.join(data_path, self.DIALOGS_FILENAME))
                or force_download):
            download(
                url=IGLUDataset.URL,
                destination=path,
                data_prefix=data_path
            )
            with ZipFile(path) as zfile:
                zfile.extractall(data_path)

    def process(self, s):
        return re.sub(r'\$+', '\n', s)

    @classmethod
    def transform_block(cls, block):
        """Adjust block coordinates and replace id."""
        x, y, z, bid = block
        y = y - VOXELWORLD_GROUND_LEVEL - 1
        bid = block_colour_map.get(bid, 5) # TODO: some blocks have id 1, check why
        return x, y, z, bid

    def parse_tasks(self, dialogs, path):
        """Fills attribute `self.tasks` with utterances from `dialogs` and
        VoxelWorld states for each step.

        Parameters
        ----------
        dialogs : pandas.DataFrame
            Contains information of each turn in the session, originally stored
            in database tables. The information includes:
                - PartitionKey: corresponds to Game attempt or session. It is
                  constructed following the pattern `{attemptId}-{taskId}`
                - structureId: task id of the session.
                - StepId: number of step in the session. For multi-turn IGLU
                  data, all odd steps have type architect and even steps
                  have type builder. Depending on the task type, different
                  columns will be used to fill the task.
                - IsHITQualified: boolean indicating if the step is valid.

        path : _type_
            Path with the state of the VoxelWorld grid after each session.
            Each session should have an associated directory named with the
            session id, with json files that describe the world state after
            each step.

        """
        # Partition key
        for sess_id, gr in dialogs.groupby('PartitionKey'):
            # This corresponds to the entire dialog between steps with
            # changes to the blocks
            utt_seq = []
            blocks = []
            if not os.path.exists(f'{path}/builder-data/{sess_id}'):
                continue
            # Each session should have a single taskId associated.
            assert len(gr.structureId.unique()) == 1
            structure_id = gr.structureId.values[0]
            # Read the utterances and block end positions for each step.
            for i, row in gr.sort_values('StepId').reset_index(drop=True).iterrows():
                if row.StepId % 2 == 1:
                    # Architext step
                    if isinstance(row.instruction, str):
                        utt_seq.append([])
                        utt_seq[-1].append(f'<Architect> {self.process(row.instruction)}')
                    elif isinstance(row.Answer4ClarifyingQuestion, str):
                        utt_seq[-1].append(f'<Architect> {self.process(row.Answer4ClarifyingQuestion)}')
                else:
                    # Builder step
                    if not row.IsHITQualified:
                        continue
                    if isinstance(row.ClarifyingQuestion, str):
                        utt_seq[-1].append(f'<Builder> {self.process(row.ClarifyingQuestion)}')
                        continue
                    blocks.append([])
                    curr_step = f'{path}/builder-data/{sess_id}/step-{row.StepId}'
                    if not os.path.exists(curr_step):
                        break
                        # TODO: in this case the multiturn collection was likely
                        # "reset" so we need to stop parsing this session. Need to check that.
                    with open(curr_step) as f:
                        step_data = json.load(f)
                    for block in step_data['worldEndingState']['blocks']:
                        x, y, z, bid = self.transform_block(block)
                        blocks[-1].append((x, y, z, bid))
            # Aggregate all previous blocks into each step
            i = 0
            while i < len(blocks):
                # Collapse steps where there are no block changes.
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
                # Create random subtasks from the sequence of dialogs and blocks
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


class SingleTurnIGLUDataset(IGLUDataset):
    SINGLE_TURN_INSTRUCTION_FILENAME = 'HitsTableSingleTurn.csv'
    MULTI_TURN_DIRNAME = 'mturk-multi-turn'

    def get_instructions(self, data_path):
        return pd.read_csv(os.path.join(
            data_path, self.SINGLE_TURN_INSTRUCTION_FILENAME))

    @classmethod
    def get_data_path(cls):
        """Returns the path where iglu dataset will be downloaded and cached.

        It can be set using the environment variable IGLU_DATA_PATH. Otherwise,
        it will be `~/.iglu/data/single_turn`.

        Returns
        -------
        str
            The absolute path to data folder.
        """
        if 'IGLU_DATA_PATH' in os.environ:
            data_path = os.path.join(
                os.environ['IGLU_DATA_PATH'], 'data', 'single_turn')
        elif 'HOME' in os.environ:
            data_path = os.path.join(
                os.environ['HOME'], '.iglu', 'data', 'single_turn')
        else:
            data_path = os.path.join(
                os.path.expanduser('~'), '.iglu', 'data', 'single_turn')
        return data_path

    def download_dataset(self, data_path, force_download):
        # TODO include all data in the same .zip file
        # Download multi-turn 2021 data
        super().download_dataset(
            os.path.join(data_path, self.MULTI_TURN_DIRNAME), force_download)
        instruction_filepath = os.path.join(
            data_path, self.SINGLE_TURN_INSTRUCTION_FILENAME)
        if not os.path.exists(instruction_filepath):
            # TODO download 2022 zip file and uncompress
            raise IOError(f'File {instruction_filepath} does not exists')

    def create_task(self, instruction, initial_grid, target_grid):
        task = Task(
            chat='',
            target_grid=Tasks.to_dense(target_grid),
            starting_grid=Tasks.to_sparse(initial_grid),
            full_grid=Tasks.to_dense(target_grid),
            last_instruction=instruction
        )
        # To properly init max_int and prev_grid_size fields
        task.reset()
        return task

    def parse_tasks(self, dialogs, path):
        """Fills attribute `self.tasks` with instances of Task.

        A Task contains an initial world state, a target world state and a
        single instruction.

        Parameters
        ----------
        dialogs : pandas.DataFrame
            Contains information of each session, originally stored
            in database tables. The information includes:
                - InitializedWorldStructureId or InitializedWorldGameId:
                  Original target structure id of the initial world.
                - InitializedWorldPath: Path to a json file that contains the
                  initial blocks of the world.
                - ActionDataPath: Path relative to dataset location with the
                  target world.
                - InputInstruction: Session instruction
                - IsHITQualified: boolean indicating if the step is valid.

        path : _type_
            Path with the state of the VoxelWorld grid after each session.
            Each session should have an associated directory named with the
            session id, with json files that describe the world state after
            each step.

        """
        # TODO apply these transformations before creating final .zip file
        dialogs = dialogs[
            (dialogs.IsHITQualified == True) &
            (dialogs.InitializedWorldPath.notna())]
        # Use value in InitializedWorldGameId if InitializedWorldStructureId
        # is null
        dialogs.loc[:,'initial_structure'] = \
            dialogs.InitializedWorldStructureId.fillna(
                dialogs.InitializedWorldGameId)
        dialogs.loc[:,'intial_world_dirname'] = \
            dialogs.InitializedWorldPath.apply(
                lambda path: path.replace('mturk-vw', self.MULTI_TURN_DIRNAME))
        # Mistake in data, some rows where saved with incorrect path
        dialogs.loc[:,'target_world_dirname'] = \
            dialogs.ActionDataPath.apply(
                lambda path: path.replace('game-game', 'game'))
        for _, row in dialogs.iterrows():
            # Read initial structure
            initial_world_path = os.path.join(path, row.intial_world_dirname)
            if not os.path.exists(initial_world_path):
                print(f'File {initial_world_path} not found')
                continue
            with open(initial_world_path, 'r') as initial_file:
                initial_step = json.load(initial_file)
            initial_world_blocks = [
                self.transform_block(block)
                for block in initial_step['worldEndingState']['blocks']
            ]

            # Read target structure
            target_world_filepath = os.path.join(
                path, row.target_world_dirname,
                f'{row.PartitionKey}-step-action')
            if not os.path.exists(target_world_filepath):
                print(f'File {target_world_filepath} not found')
                continue
            with open(target_world_filepath, 'r') as target_file:
                target_step = json.load(target_file)
            target_world_blocks = [
                self.transform_block(block)
                for block in target_step['worldEndingState']['blocks']
            ]
            if len(target_world_blocks) == 0:
                print(f'No target blocks for structure {row.PartitionKey}')
                continue

            # Construct task
            task = self.create_task(
                row.InputInstruction, initial_world_blocks, target_world_blocks)

            assert row.initial_structure is not None
            self.tasks[row.initial_structure].append(task)

    def __iter__(self):
        for task_id, tasks in self.tasks.items():
            for j, task in enumerate(tasks):
                yield task_id, j, 1, task