from collections import defaultdict
import re
import os
import bz2
import pickle
import pathlib
import json
import warnings
from time import time
from functools import partial
from argparse import ArgumentParser
import multiprocessing

from gridworld.visualizer import Visualizer

warnings.filterwarnings('ignore', '.*box bound precision lowered.*', category=UserWarning)
warnings.filterwarnings('ignore', '.*minerl_patched.utils.process_watcher.*', category=RuntimeWarning)

import logging
import pandas as pd
import numpy as np
import gym
from tqdm import tqdm

# from minerl_patched.data.util.constants import ACTIONABLE_KEY, HANDLER_TYPE_SEPERATOR, MONITOR_KEY, OBSERVABLE_KEY, REWARD_KEY

from gridworld.tasks.task import BUILD_ZONE_SIZE
from ..iglu_dataset import block_colour_map, DATA_PREFIX


from .parse import ActionsParser
# from .render import Renderer
from .common import GameSession, DEFAULT_POS, VOXELWORLD_GROUND_LEVEL
from gridworld.data.iglu_dataset import block_colour_map


logger = logging.getLogger(__name__)


class ActionsAdapter:
    def __init__(self, hits_table=None, ):
        self.renderer = None
        self.parser = None
        self.hits_table = pd.read_csv(hits_table)

    def action_space(self):
        env = gym.make('IGLUGridworldVector-v0')
        action_space = env.action_space
        del env
        return action_space

    def dialog_step(self, session, start=0, steps=-1):
        df = self.hits_table
        df = df[df.PartitionKey == session].sort_values('StepId')
        n_turns = 0
        result = []
        for _, row in df[df.StepId >= start].iterrows():
            if row.StepId % 2 == 1:
                if row.Role == 'architect-normal':
                    if n_turns == steps and steps != -1:
                        break
                    result.append(f'A: {row.instruction}')
                    n_turns += 1
                else:
                    result.append(f'A: {row.Answer4ClarifyingQuestion}')
            else:
                if row.Role == 'builder-normal' and isinstance(row.ClarifyingQuestion, str):
                    result.append(f'B: {row.ClarifyingQuestion}')
                else:
                    result.append(None)
        return result
    
    def parse_session(self, path, session, start_step=0, steps=-1, init_from_end=False, verbose=False, position=None):
        from gridworld.core import World, Agent
        vis = Visualizer((512, 512))
        # world = World()
        # world._initialize()
        # agent = Agent()
        world = vis.world
        agent = vis.agent
        agent.flying = True
        self.parser = ActionsParser(world, agent)
        game_session = GameSession()
        logger.info(f'parsing session {session}')
        logs_steps = []
        path = pathlib.Path(path)
        p = re.compile(r'[-a-zA-Z]*(\d+)$')
        for f in os.listdir(path / session):
            match = p.match(f)
            if match is not None:
                n = int(match.group(1))
                if n >= start_step:
                    logs_steps.append(n)
        logs_steps = sorted(logs_steps)
        if steps != -1:
            logs_steps = logs_steps[:steps]
        data = None
        for j in tqdm(logs_steps, disable=not verbose):
            with open(path/session/f'step-{j}') as f:
                data = json.load(f)
                self.parser.data_sequence.append(data)
            if init_from_end and not init:
                init = True
                start_position, initial_blocks = self.parser.parse_init_conds(
                    data, position=position)
            else:
                start_position = DEFAULT_POS
                initial_blocks = None
            events = self.parser.parse(data, j//2 - 1)
            game_session.init_conds[j] = (start_position, initial_blocks)
            game_session.events[j] = events
        self.parser.reset()
        target_blocks = []
        target_block_ids = []
        if data is not None:
            for block in data['worldEndingState']['blocks']:
                block[0] += 5
                block[1] -= VOXELWORLD_GROUND_LEVEL - 1
                block[2] += 5
                block[0], block[1] = block[1], block[0]
                # TODO: check for block[3] not in block_colour_map
                block[3] = block_colour_map.get(block[3], 1)
                target_blocks.append(np.array(block[:3]))
                target_block_ids.append(block[3])
        target = np.zeros(BUILD_ZONE_SIZE, dtype=np.int)
        if len(target_blocks) != 0:
            target[tuple(np.array(target_blocks).T.tolist())] = target_block_ids
        game_session.target =  target
        game_session.dialogs = self.dialog_step(session=session, steps=steps)
        game_session.name = session
        logger.info(f'session {session} parsed. Episode length: {game_session.episode_steps()} steps.')
        return game_session

    def has_buffer(self):
        return self._dir_non_emtpy('buffer')

    def _dir_non_emtpy(self, subdir):
        path = pathlib.Path(DATA_PREFIX)
        path = path / subdir
        return path.exists() and len(list(path.glob('*'))) != 0

    def parse_sessions(self, path, verbose=False):
        path = pathlib.Path(path)
        sessions = []
        for sess_dir in tqdm(path.glob('*-c*/'), disable=not verbose):
            game_session = self.parse_session(sess_dir.parent, sess_dir.name)
            sessions.append(game_session)
        return sessions

    def save_session(self, session):
        path = pathlib.Path(DATA_PREFIX).parent
        sessions_path = path / 'buffer'
        sessions_path.mkdir(exist_ok=True)
        with open(sessions_path / f'{session.name}_session.pkl', 'wb') as f:
            f.write(bz2.compress(pickle.dumps(session)))
        return session

    def load_session(self, session_name):
        path = pathlib.Path(DATA_PREFIX).parent
        session_path = path / 'buffer' / session_name
        with open(session_path, 'rb') as f:
            compressed_session = f.read()
        return pickle.loads(bz2.decompress(compressed_session))
    
    def render_session_video(self, session, visualize=False, debug=False, postprocess=True, step_callback=None, verbose=False):
        """

        Args:
            session (GameSession): 
            visualize (bool, optional): whether to render a high-resolution human friendly visualization
            debug (bool, optional): whether to add a debug info on each screen. Defaults to True.
            postprocess (bool, optional): whether to re-encode video using h264 codec. !!! REQUIRES THE LATEST FFMPEG !!!
        """
        path = pathlib.Path(DATA_PREFIX)
        if visualize:
            session_path = path / 'session_videos'
            height, width = 1000, 1000
        else:
            session_path = path / 'buffer'
            height, width = 512, 512
        writer = None
        data = defaultdict(list)
        logger.info(f'rendering session {session.name}')
        callback = partial(self._write_observation_vec, data, step_callback)
        from gridworld.visualizer import Visualizer
        visualizer = Visualizer((width, height))#, h264=postprocess, fps=60)
        # renderer.create_env(task_target=session.target)
        for i in range(2, len(session.dialogs) + 1, 2):
            if i not in session.events:
                break
            writer = visualizer.render_video(session_path / f'{session.name}_{i // 2 - 1}',
                event_sequence=session.events[i], render_text=visualize,
                text=session.dialogs[:i], reset=i == 2,
                render_debug=debug, render_crosshair=visualize,  
                # close=(i + 1) >= len(session.dialogs), 
                close=True,
                writer=writer,
                on_env_step=callback
            )
            logger.info(f'session {session.name}: rendering step {i // 2 + 1}/{len(session.dialogs) // 2}')
            visualizer.postproc_video(session_path / f'{session.name}_{i // 2 - 1}')
        # if not visualize:
        #     obs_file = session_path / f'{session.name}.npz'
        #     data = {k: v for k, v in data.items()}
        #     for key in ['agentPos', 'inventory', 'compass.angle', 'reward']:
        #         if key not in data:
        #             continue
        #         if isinstance(data[key], np.ndarray):
        #             data[key] = np.row_stack(data[key])
        #         else:
        #             data[key] = np.array(data[key])
        #     if 'grid' in data:
        #         data['grid'] = np.array(data['grid'], dtype=np.object)
        #     data_ = {
        #         f'{OBSERVABLE_KEY}{HANDLER_TYPE_SEPERATOR}{k}': v
        #         for k, v in data.items() if k != 'reward'
        #     }
        #     if 'reward' in data:
        #         data_['reward'] = data['reward']
        #     actions = [action for k in session.events.keys() for event in session.events[k] for action in event.actions]
        #     if len(actions) != 0:
        #         actions_t = {k: [] for k in actions[0].keys()}
        #         for a in actions:
        #             for k in actions_t:
        #                 act = a[k]
        #                 if isinstance(act, np.ndarray) and act.shape == ():
        #                     act = act.item()
        #                 actions_t[k].append(act)
        #         for k in actions_t:
        #             if isinstance(actions_t[k][0], np.ndarray):
        #                 actions_t[k] = np.row_stack(actions_t[k])
        #             else:
        #                 actions_t[k] = np.array(actions_t[k])
        #         actions = {
        #             f'{ACTIONABLE_KEY}{HANDLER_TYPE_SEPERATOR}{k}': v
        #             for k, v in actions_t.items()
        #         }
        #         data_.update(actions)
        #     np.savez_compressed(str(obs_file), **data_)        
        
    def _write_observation_vec(self, data, callback, event, i, k, output, obs, reward, done, info):
        data['grid'].append([(y, z, x, obs['grid'][y, z, x]) for (y, z, x) in zip(*obs['grid'].nonzero())])
        data['agentPos'].append(obs['agentPos'])
        data['compass.angle'].append(obs['compass']['angle'].item())
        data['inventory'].append(obs['inventory'])
        data['reward'].append(reward)
        if callable(callback):
            callback(event, i, k, output, obs, reward, done, info)


def run(session_id=None, overwrite=False, adapt=True, render=True, visualize=False, step_callback=None):
    if render and visualize:
        raise ValueError('render and visualize are mutually exclusive. '
            'Use the first to build a replay bufffer and the second for human visualization.')
    path = pathlib.Path(DATA_PREFIX)
    if not (path / 'dialogs.csv').exists():
        raise ValueError(f'make sure the hits table is present under {str(path / "dialogs.csv")}')
    adapter = ActionsAdapter(hits_table=path / 'dialogs.csv')
    if not (path / 'builder-data').exists():
        raise ValueError(f'make sure logs are present under {str(path / "builder-data")}')
    if session_id is None:
        dirs = list((path / 'builder-data').glob('*-c*/'))
    else:
        dirs = [path / 'builder-data' / session_id]
    for data_dir in tqdm(dirs):
        if adapt or render or visualize:
            if True: #overwrite or not adapter.has_sessions():
                try:
                    session = adapter.parse_session(data_dir.parent, session=data_dir.name)
                except:
                    logger.critical(f'while parsing session {data_dir.name}')
                    raise
                if len(session.events) == 0 or session.episode_states() <= 32:
                    logger.info(f'session {session.name} is empty. Skipping.')
                    return
                session = adapter.save_session(session)
        if render or visualize:
            if True: #overwrite or not adapter.has_buffer():
                try:
                    adapter.render_session_video(session, visualize=visualize, step_callback=step_callback)
                except EOFError:
                    logger.critical(f'while rendering session {session.name}')
                    raise


def run_multiprocess(session_id=None, num_workers=-1, overwrite=False, adapt=True, 
        render=True, visualize=False, step_callback=None):
    run_one = partial(run, overwrite=overwrite, adapt=adapt, render=render, visualize=visualize, step_callback=step_callback)
    path = pathlib.Path(DATA_PREFIX)
    if not (path / 'dialogs.csv').exists():
        raise ValueError(f'make sure the hits table is present under {str(path / "dialogs.csv")}')
    if not (path / 'builder-data').exists():
        raise ValueError(f'make sure logs are present under {str(path / "builder-data")}')
    if session_id is None:
        dirs = list((path / 'builder-data').glob('*-c*/'))
    else:
        dirs = [path / 'builder-data' / session_id]
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(run_one, [d.name for d in dirs])


if __name__ == '__main__':
    start = time()
    logger.setLevel(logging.INFO)
    logging.getLogger('minerl_patched').setLevel(logging.CRITICAL)
    parser = ArgumentParser()
    parser.add_argument('--adapt', action='store_true', default=False)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--path', type=str, default=None, 
        help='path to store processed data. by default specified by $IGLU_DATA_PATH')
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--session', type=str, default=None, help='session id to process. Only used with num_workers=1')

    args = parser.parse_args()
    if args.num_workers > 1:
        runner = partial(run_multiprocess, num_workers=args.num_workers)
    else:
        runner = run
        if args.session is not None:
            runner = partial(run, session_id=args.session)
    runner(overwrite=args.overwrite, adapt=args.adapt, render=args.render, visualize=args.visualize)
    print(f'data processing done! Spent {(time() - start) / 60:.1f} minutes.')