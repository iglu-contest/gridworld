import os
import json
import re
import pandas as pd
import numpy as np
import pickle
import bz2
import itertools
from collections import defaultdict

from ..tasks.task import Subtasks, Task, Tasks
from .load import download

from zipfile import ZipFile
from tqdm import tqdm


VOXELWORLD_GROUND_LEVEL = 63


class IGLUDataset(Tasks):
    DATASET_URL = {
        "v0.1.0-rc1": 'https://iglumturkstorage.blob.core.windows.net/public-data/iglu_dataset.zip',
        "v0.1.0-rc2": (
            'https://iglumturkstorage.blob.core.windows.net/public-data/iglu_dataset.zip',
            'https://iglumturkstorage.blob.core.windows.net/public-data/parsed_tasks_multi_turn_dataset.tar.bz2'
        ),
        "v0.1.0-rc3": (
            'https://iglumturkstorage.blob.core.windows.net/public-data/iglu_dataset.zip',
            'https://iglumturkstorage.blob.core.windows.net/public-data/parsed_tasks_multi_turn_dataset.rc3.tar.bz2'
        )
    }  # Dictionary holding dataset version to dataset URI mapping
    DIALOGS_FILENAME = 'dialogs.csv'
    BLOCK_MAP = {  # voxelworld's colour id : iglu colour id
        00: 0,  # air
        57: 1,  # blue
        50: 6,  # yellow
        59: 2,  # green
        47: 4,  # orange
        56: 5,  # purple
        60: 3,  # red
    }

    def __init__(self, dataset_version="v0.1.0-rc2", task_kwargs=None,
                 force_download=False, force_parsing=False) -> None:
        """
        Collaborative dataset for the IGLU competition.

        Current version of the dataset covers 31 structures in 128 staged game sessions
        resulting in 608 tasks.

        Args:
            dataset_version: Which dataset version to use.
            task_kwargs: Task-class specific kwargs. For reference see gridworld.task.Task class
            force_download: Whether to force dataset downloading
            force_parsing: Whether to ignore cached dataset and force parsing it.
        """
        self.dataset_version = dataset_version
        if dataset_version not in self.DATASET_URL.keys():
            raise Exception(
                "Unknown dataset_version:{} provided.".format(dataset_version))
        if task_kwargs is None:
            task_kwargs = {}
        self.task_kwargs = task_kwargs
        data_path, custom = self.get_data_path()
        if isinstance(self.DATASET_URL[self.dataset_version], tuple):
            filename = self.DATASET_URL[self.dataset_version][1].split('/')[-1]
        else:
            filename = self.DATASET_URL[self.dataset_version].split('/')[-1]
        if custom:
            filename = f'cached_{filename}'
        parse = False
        if not custom and not force_parsing:
            try:
                # first, try downloading the lightweight parsed dataset
                self.download_parsed(data_path=data_path, file_name=filename, force_download=force_download)
                self.load_tasks_dataset(os.path.join(data_path, filename))
            except Exception as e:
                print(e)
                parse = True
        if custom or parse or force_parsing:
            print('Loading parsed dataset failed. Downloading full dataset.')
            # if it fails, download it manually and cache it
            self.download_dataset(data_path, force_download)
            dialogs = self.get_instructions(data_path)
            self.tasks = defaultdict(list)
            self.parse_tasks(dialogs, data_path)
            self.dump_tasks_dataset(os.path.join(data_path, filename))

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
            custom = True
        elif 'HOME' in os.environ:
            data_path = os.path.join(
                os.environ['HOME'], '.iglu', 'data', 'iglu')
            custom = False
        else:
            data_path = os.path.join(
                os.path.expanduser('~'), '.iglu', 'data', 'iglu')
            custom = False
        return data_path, custom

    def download_dataset(self, data_path, force_download):
        path = os.path.join(data_path, 'iglu_dataset.zip')
        if (not os.path.exists(os.path.join(data_path, self.DIALOGS_FILENAME))
                or force_download):
            url = self.DATASET_URL[self.dataset_version]
            if not isinstance(url, str):
                url = url[0]
            download(
                url=url,
                destination=path,
                data_prefix=data_path,
                description='downloading multiturn dataset'
            )
            with ZipFile(path) as zfile:
                zfile.extractall(data_path)

    def download_parsed(self, data_path, file_name='parsed_tasks_multiturn_dataset.tar.bz2',
                        force_download=False):
        path = os.path.join(data_path, file_name)
        if (not os.path.exists(path) or force_download):
            url = self.DATASET_URL[self.dataset_version]
            if isinstance(url, str):
                raise ValueError('this dataset version does not support parsed data!')
            url = url[1]
            download(
                url=url,
                destination=path,
                data_prefix=data_path,
                description='downloading task dataset'
            )

    def dump_tasks_dataset(self, path):
        print('caching tasks dataset... ', end='')
        pickled = pickle.dumps(self.tasks)
        compressed = bz2.compress(pickled)
        with open(path, 'wb') as f:
            f.write(compressed)
        print('done')

    def load_tasks_dataset(self, path):
        with open(path, 'rb') as f:
            data = f.read()
        data = bz2.decompress(data)
        self.tasks = pickle.loads(data)

    def process(self, s):
        return re.sub(r'\$+', '\n', s)

    @classmethod
    def transform_block(cls, block):
        """Adjust block coordinates and replace id."""
        x, y, z, bid = block
        y = y - VOXELWORLD_GROUND_LEVEL - 1
        bid = cls.BLOCK_MAP[bid]
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

        path : str
            Path with the state of the VoxelWorld grid after each session.
            Each session should have an associated directory named with the
            session id, with json files that describe the world state after
            each step.

        """
        # Partition key
        groups = dialogs.groupby('PartitionKey')
        for sess_id, gr in tqdm(groups, total=len(groups), desc='parsing dataset'):
            # This corresponds to the entire dialog between steps with
            # changes to the blocks
            utt_seq = []
            blocks = []
            block_changes = []
            if not os.path.exists(f'{path}/builder-data/{sess_id}'):
                continue

            # Each session should have a single taskId associated.
            assert len(gr.structureId.unique()) == 1
            structure_id = gr.structureId.values[0]

            # Read the utterances and block end positions for each step.
            rows = list(gr.sort_values('StepId').reset_index(drop=True).iterrows())
            for i, row in rows:
                if not row.IsHITQualified:
                    continue
                if row.StepId % 2 == 1:
                    # Architect step
                    if isinstance(row.instruction, str):
                        blocks.append([])
                        block_changes.append([])
                        utt_seq.append([])
                        utt_seq[-1].append(
                            f'<Architect> {self.process(row.instruction)}')
                    elif isinstance(row.Answer4ClarifyingQuestion, str):
                        utt_seq[-1].append(
                            f'<Architect> {self.process(row.Answer4ClarifyingQuestion)}')
                else:
                    # Builder step
                    if isinstance(row.ClarifyingQuestion, str):
                        utt_seq[-1].append(
                            f'<Builder> {self.process(row.ClarifyingQuestion)}')

                    curr_step = f'{path}/builder-data/{sess_id}/step-{row.StepId}'
                    if not os.path.exists(curr_step):
                        break
                        # TODO: in this case the multiturn collection was likely
                        # "reset" so we need to stop parsing this session. Need to check that.
                    with open(curr_step) as f:
                        step_data = json.load(f)

                    blocks[-1] = []
                    for block in step_data['worldEndingState']['blocks']:
                        x, y, z, bid = self.transform_block(block)
                        blocks[-1].append((x, y, z, bid))

                    # Read the tape and look for 'select_and_place_block' and 'break' action events.
                    lines_with_block_change = [line for line in step_data["tape"].split("\n")
                                               if "action break" in line or "action select_and_place_block" in line]

                    # Keep track of the grid values while reading block changes.
                    starting_grid = list(itertools.chain(*blocks[-2:-1]))
                    grid = {tuple((x,y,z)): bid for x,y,z,bid in starting_grid}

                    # Replay what we have collected so far.
                    for x, y, z, bid in block_changes[-1]:
                        grid[tuple((x, y, z))] = bid

                    # Extract block changes.
                    for line in lines_with_block_change:
                        if "break" in line:
                            bid = 0
                            x, y, z = map(int, line.split("break ")[-1].split()[:3])
                        elif "select_and_place_block" in line:
                            bid, x, y, z = map(int, line.split("select_and_place_block ")[-1].split()[:4])
                        else:
                            raise NotImplementedError("Should not happen.")

                        y = y - VOXELWORLD_GROUND_LEVEL - 1

                        if bid not in self.BLOCK_MAP:
                            print(f"**Warning block ID {bid} not found while parsing '{line}'.")

                        bid = self.BLOCK_MAP.get(bid, 3)  # Why?

                        if grid.get(tuple((x, y, z)), 0) == bid:
                            # Skip redundant block changes. Usually happens during world state recovery at the tape's beginning.
                            continue

                        grid[tuple((x, y, z))] = bid
                        block_changes[-1].append((x, y, z, bid))

                    # Make sure we can reconstruct the target grid from the block changes.
                    assert sorted((cell + (value,)) for cell, value in grid.items() if value) == sorted(blocks[-1])

            # To be considered, a session needs to end without a pending ClarifyingQuestion.
            if isinstance(rows[i-1][1].ClarifyingQuestion, str):
                blocks = blocks[:-1]
                utt_seq = utt_seq[:-1]
                block_changes = block_changes[:-1]

            if len(blocks) > 1:
                if len(block_changes[-1]) == 0 and set(blocks[-2]) == set(blocks[-1]):
                    blocks = blocks[:-1]
                    utt_seq = utt_seq[:-1]
                    block_changes = block_changes[:-1]

            # Aggregate all previous blocks into each step
            if len(blocks) < len(utt_seq):
                # handle the case of missing of the last blocks record
                utt_seq = utt_seq[:len(blocks)]

            i = 0
            while i < len(blocks):
                # Collapse steps where there are no block changes.
                if len(blocks[i]) == 0:
                    if i == len(blocks) - 1:
                        blocks = blocks[:i]
                        utt_seq = utt_seq[:i]
                        block_changes = block_changes[:i]
                    else:
                        blocks = blocks[:i] + blocks[i + 1:]
                        utt_seq[i] = utt_seq[i] + utt_seq[i + 1]
                        utt_seq = utt_seq[:i + 1] + utt_seq[i + 2:]
                        block_changes = block_changes[:i] + block_changes[i + 1:]

                else:
                    i += 1

            if len(blocks) > 0:
                # Create random subtasks from the sequence of dialogs and blocks
                task = Subtasks(utt_seq, blocks, block_changes, **self.task_kwargs)
                assert len(utt_seq) == len(blocks)
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
    SINGLE_TURN_INSTRUCTION_FILENAME = 'single_turn_instructions.csv'
    MULTI_TURN_INSTRUCTION_FILENAME = 'multi_turn_dialogs.csv'
    DATASET_URL = {
        "v0.1.0-rc1": 'https://iglumturkstorage.blob.core.windows.net/public-data/single_turn_dataset.zip',
        "v0.1.0-rc2": (
            'https://iglumturkstorage.blob.core.windows.net/public-data/single_turn_dataset.zip',
            'https://iglumturkstorage.blob.core.windows.net/public-data/parsed_tasks_single_turn_dataset.tar.bz2'
        ),
        "v0.1.0-rc3": (
            'https://iglumturkstorage.blob.core.windows.net/public-data/single_turn_dataset.zip',
            'https://iglumturkstorage.blob.core.windows.net/public-data/parsed_tasks_single_turn_dataset.rc3.tar.bz2'
        )
    }
    BLOCK_MAP = {
        # voxelworld's colour id : iglu colour id
        00: 0,  # air
        57: 1,  # blue
        50: 6,  # yellow
        59: 2,  # green
        47: 4,  # orange
        56: 5,  # purple
        60: 3,  # red
        # voxelworld (freeze version)'s colour id : iglu colour id
        86: 1,  # blue
        87: 6,  # yellow
        88: 2,  # green
        89: 4,  # orange
        90: 5,  # purple
        91: 3,  # red
    }

    def __init__(self, dataset_version='v0.1.0-rc3', task_kwargs=None,
            force_download=False, force_parsing=False, limit=None) -> None:
        self.limit = limit
        super().__init__(dataset_version=dataset_version,
            task_kwargs=task_kwargs, force_download=force_download, force_parsing=force_parsing)

    def get_instructions(self, data_path):
        single_turn_df = pd.read_csv(os.path.join(
            data_path, self.SINGLE_TURN_INSTRUCTION_FILENAME))
        if self.limit is not None:
            return single_turn_df[:self.limit]
        return single_turn_df

    def get_multiturn_dialogs(self, data_path):
        return pd.read_csv(os.path.join(
            data_path, self.MULTI_TURN_INSTRUCTION_FILENAME))

    @classmethod
    def get_data_path(cls):
        """Returns the path where iglu dataset will be downloaded and cached.

        It can be set using the environment variable IGLU_DATA_PATH. Otherwise,
        it will be `~/.iglu/data/single_turn_dataset`.

        Returns
        -------
        str
            The absolute path to data folder.
        """
        if 'IGLU_DATA_PATH' in os.environ:
            data_path = os.environ['IGLU_DATA_PATH']
            custom = True
        elif 'HOME' in os.environ:
            data_path = os.path.join(
                os.environ['HOME'], '.iglu', 'data', 'single_turn_dataset')
            custom = False
        else:
            data_path = os.path.join(
                os.path.expanduser('~'), '.iglu', 'data', 'single_turn_dataset')
            custom = False
        return data_path, custom

    def download_dataset(self, data_path, force_download):
        instruction_filepath = os.path.join(
            data_path, self.SINGLE_TURN_INSTRUCTION_FILENAME)
        path = os.path.join(data_path, 'single_turn_dataset.zip')
        if os.path.exists(instruction_filepath) and not force_download:
            print("Using cached dataset")
            return
        url = self.DATASET_URL[self.dataset_version]
        if not isinstance(url, str):
            url = url[0]
        print(f"Downloading dataset from {url}")
        download(
            url=url,
            destination=path,
            data_prefix=data_path
        )
        with ZipFile(path) as zfile:
            zfile.extractall(data_path)

    def create_task(self, previous_chat, initial_grid, target_grid,
                    last_instruction):
        task = Task(
            chat=previous_chat,
            target_grid=Tasks.to_dense(target_grid),
            starting_grid=Tasks.to_sparse(initial_grid),
            full_grid=Tasks.to_dense(target_grid),
            last_instruction=last_instruction
        )
        # To properly init max_int and prev_grid_size fields
        task.reset()
        return task

    def get_previous_dialogs(self, single_turn_row, multiturn_dialogs):
        # Filter multiturn rows with this game id and previous to step
        utterances = []
        mturn_data_path = single_turn_row.InitializedWorldPath.split('/')[-2:]
        if len(mturn_data_path) != 2 or '-' not in mturn_data_path[1]:
            print(f"Error with initial data path {single_turn_row.InitializedWorldPath}."
                  "Could not parse data path to get previous dialogs.")
            return utterances
        mturn_game_id = mturn_data_path[0]
        try:
            mturn_last_step = int(mturn_data_path[1].replace("step-", ""))
        except Exception as e:
            print(f"Error with initial data path {single_turn_row.InitializedWorldPath}."
                  "Could not parse step id to get previous dialogs.")
            return utterances
        dialog_rows = multiturn_dialogs[
            (multiturn_dialogs.PartitionKey == mturn_game_id) &
            (multiturn_dialogs.StepId < mturn_last_step) &
            (multiturn_dialogs.IsHITQualified == True)]

        for _, row in dialog_rows.sort_values('StepId')\
                .reset_index(drop=True).iterrows():
            if row.StepId % 2 == 1:
                # Architect step
                if isinstance(row.instruction, str):
                    utterance = row.instruction
                    utterances.append(
                        f'<Architect> {self.process(utterance)}')
                elif isinstance(row.Answer4ClarifyingQuestion, str):
                    utterance = row.Answer4ClarifyingQuestion
                    utterances.append(
                        f'<Architect> {self.process(utterance)}')
            elif isinstance(row.ClarifyingQuestion, str):
                utterances.append(
                    f'<Builder> {self.process(row.ClarifyingQuestion)}')

        return utterances

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
        dialogs = dialogs[dialogs.InitializedWorldPath.notna()]
        dialogs['InitializedWorldPath'] = dialogs['InitializedWorldPath'] \
            .apply(lambda x: x.replace('\\', os.path.sep))
        dialogs['InitializedWorldPath'] = dialogs['InitializedWorldPath'] \
            .apply(lambda x: x.replace('/', os.path.sep))

        # Get the list of games for which the instructions were clear.
        turns = dialogs[dialogs.GameId.str.match("CQ-*")]

        # Util function to read structure from disk.
        def _load_structure(structure_path):
            filepath = os.path.join(path, structure_path)
            if not os.path.exists(filepath):
                return None

            with open(filepath) as structure_file:
                structure_data = json.load(structure_file)
                blocks = structure_data['worldEndingState']['blocks']
                structure = [self.transform_block(block) for block in blocks]

            return structure

        multiturn_dialogs = self.get_multiturn_dialogs(path)

        tasks_count = 0
        pbar = tqdm(turns.iterrows(), total=len(turns), desc='parsing dataset')
        for _, row in pbar:
            pbar.set_postfix_str(f"{tasks_count} tasks")
            assert row.InitializedWorldStructureId is not None

            # Read initial structure
            initial_world_blocks = _load_structure(row.InitializedWorldPath)
            if initial_world_blocks is None:
                pbar.write(f"Skipping '{row.GameId}'. Can't load starting structure from '{row.InitializedWorldPath}'.")
                continue

            target_world_blocks = _load_structure(row.TargetWorldPath)
            if target_world_blocks is None:
                pbar.write(f"Skipping '{row.GameId}'. Can't load target structure from '{row.TargetWorldPath}'.")
                continue

            # Check if target structure matches the initial structure.
            if sorted(initial_world_blocks) == sorted(target_world_blocks):
                pbar.write(f"Skipping '{row.GameId}'. Target structure is the same as the initial one.")
                continue

            # Get the original game.
            orig = dialogs[dialogs.GameId == row.GameId[len("CQ-"):]]
            if len(orig) == 0:
                pbar.write(f"Skipping '{row.GameId}'. Can't find its original game '{row.GameId[len('CQ-'):]}'.")
                continue

            assert len(orig) == 1
            orig = orig.iloc[0]

            # Load original structure.
            orig_target_world_blocks = _load_structure(orig.TargetWorldPath)
            if orig_target_world_blocks is None:
                pbar.write(f"Skipping '{row.GameId}'. Can't load original target structure from '{orig.TargetWorldPath}'.")
                continue

            # Check if original structure matches the rebuilt one.
            if sorted(orig_target_world_blocks) != sorted(target_world_blocks):
                pbar.write(f"Skipping '{row.GameId}'. Target structure doesn't match the one in '{orig.GameId}'.")
                continue

            last_instruction = '<Architect> ' + self.process(row.InputInstruction)
            # Read utterances
            utterances = self.get_previous_dialogs(row, multiturn_dialogs)
            utterances.append(last_instruction)
            utterances = '\n'.join(utterances)
            # Construct task
            task = self.create_task(
                utterances, initial_world_blocks, target_world_blocks,
                last_instruction=last_instruction)

            # e.g. initial_world_states\builder-data/8-c92/step-4 -> 8-c92/step-4
            task_id, step_id = row.InitializedWorldPath.split("/")[-2:]
            #self.tasks[row.InitializedWorldStructureId].append(task)
            self.tasks[f"{task_id}/{step_id}"].append(task)
            tasks_count += 1

    def __iter__(self):
        for task_id, tasks in self.tasks.items():
            for j, task in enumerate(tasks):
                yield task_id, j, 1, task

    def __len__(self):
        return len(sum(self.tasks.values(), []))
