import bz2
import pickle
import pathlib
import warnings
from functools import partial
import multiprocessing

warnings.filterwarnings('ignore', '.*box bound precision lowered.*', category=UserWarning)
warnings.filterwarnings('ignore', '.*minerl_patched.utils.process_watcher.*', category=RuntimeWarning)

import logging
import gym
from tqdm import tqdm

from .parse import ActionsParser
from gridworld.data.iglu_dataset import IGLUDataset


logger = logging.getLogger(__name__)


class ActionsAdapter:
    def __init__(self):
        self.renderer = None
        self.parser = None
        self.dataset = IGLUDataset()

    def action_space(self):
        env = gym.make('IGLUGridworldVector-v0')
        action_space = env.action_space
        del env
        return action_space

    def has_buffer(self):
        return self._dir_non_emtpy('buffer')

    def _dir_non_emtpy(self, subdir):
        path = pathlib.Path(self.dataset.get_data_path())
        path = path / subdir
        return path.exists() and len(list(path.glob('*'))) != 0

    def parse_sessions(self, path, verbose=False):
        path = pathlib.Path(path)
        sessions = []
        for sess_dir in tqdm(path.glob('*-c*/'), disable=not verbose):
            game_session = self.parse_session(sess_dir.parent, sess_dir.name)
            sessions.append(game_session)
        return sessions

    def save_session(self, session, save_path=None):
        if save_path is None:
            path = pathlib.Path(self.dataset.get_data_path()).parent
            sessions_path = path / 'buffer'
            sessions_path.mkdir(exist_ok=True)
        else:
            sessions_path = pathlib.Path(save_path)
            sessions_path.mkdir(exist_ok=True)
        with open(sessions_path / f'{session.name}_session.pkl', 'wb') as f:
            f.write(bz2.compress(pickle.dumps(session)))
        return session

    def load_session(self, session_name, load_path=None):
        if load_path is None:
            path = pathlib.Path(self.dataset.get_data_path()).parent
            session_path = path / 'buffer' / session_name
        else:
            session_path = pathlib.Path(load_path)
        with open(session_path, 'rb') as f:
            compressed_session = f.read()
        return pickle.loads(bz2.decompress(compressed_session))

    def render_session_video(self, session,
            visualize=False, postprocess=True,
            render_size=(64, 64), outpath=None,
            single_turn=False
        ):
        """

        Args:
            session (GameSession):
            visualize (bool, optional): whether to render a high-resolution human friendly visualization
            debug (bool, optional): whether to add a debug info on each screen. Defaults to True.
            postprocess (bool, optional): whether to re-encode video using h264 codec. !!! REQUIRES THE LATEST FFMPEG !!!
        """
        path = pathlib.Path(outpath or self.dataset.get_data_path())
        if visualize:
            raise ValueError('this mode does not work yet')
            session_path = path / 'session_videos'
            height, width = 1000, 1000
        else:
            session_path = path
        logger.info(f'rendering session {session.name}')
        from gridworld.visualizer import Visualizer
        visualizer = Visualizer((render_size, render_size))
        if single_turn:
            visualizer.render_video(session_path,
                event_sequence=session.events[0], close=True
            )
            if postprocess:
                visualizer.postproc_video(session_path)
        else:
            for i in range(2, len(session.dialogs) + 1, 2):
                if i not in session.events:
                    break
                visualizer.render_video(session_path / f'{session.name}_{i // 2 - 1}',
                    event_sequence=session.events[i], close=True
                )
                logger.info(f'session {session.name}: rendering step {i // 2 + 1}/{len(session.dialogs) // 2}')
                if postprocess:
                    visualizer.postproc_video(session_path / f'{session.name}_{i // 2 - 1}')

def run(
        session_id=None,
        path=None,
        outpath=None,
        dialogs_path=None,
        single_turn=False,
        #FLAGS
        overwrite=False, adapt=True, render=True, visualize=False, overlay=False, step_callback=None, render_size=64):
    if render and visualize:
        raise ValueError('render and visualize are mutually exclusive. '
            'Use the first to build a replay bufffer and the second for human visualization.')
    path_ = path
    path = pathlib.Path(path or DATA_PREFIX)
    dialogs_path = pathlib.Path(dialogs_path or str(path / 'dialogs.csv'))
    if not dialogs_path.exists():
        raise ValueError(f'make sure the hits table is present under {dialogs_path}')
    adapter = ActionsAdapter()
    parser = ActionsParser(hits_table=dialogs_path)
    if 'builder-data' not in str(path):
        path = path / 'builder-data'
    if not path.exists():
        raise ValueError(f'make sure logs are present under {path}')
    if session_id is None:
        dirs = list(path.glob('*-c*/'))
        if len(dirs) == 0: # To handle single-turn
            dirs = list(path.glob('*game-*/'))
    else:
        # dirs = [path / session_id / f'{session_id}-step-action']
        dirs = [path / session_id]
    for data_dir in tqdm(dirs):
        if single_turn:
            session_name = data_dir.name
        else:
            session_name = data_dir.glob('*')
        if adapt or render or visualize:
            if True: #overwrite or not adapter.has_sessions():
                try:
                    if not single_turn:
                        session = parser.parse_session(data_dir.parent, session=data_dir.name)
                    # logger.info(f'session {session} parsed. Episode length: {game_session.episode_steps()} steps.')
                    else:
                        session = parser.parse_single_turn_session(data_dir, session=data_dir.name)
                except:
                    logger.critical(f'while parsing session {data_dir.name}')
                    raise
                if len(session.events) == 0 or session.episode_states() <= 32:
                    logger.info(f'session {session.name} is empty. Skipping.')
                    return
                session = adapter.save_session(session, save_path=outpath)
        if render or visualize:
            if True: #overwrite or not adapter.has_buffer():
                try:
                    adapter.render_session_video(
                        session, visualize=visualize, outpath=outpath,
                        single_turn=single_turn, render_size=render_size
                    )
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
