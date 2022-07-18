from time import time
import logging
logger = logging.getLogger(__name__)

from argparse import ArgumentParser
from functools import partial
from .adapter import run, run_multiprocess

if __name__ == '__main__':
    start = time()
    # logger.setLevel(logging.INFO)
    parser = ArgumentParser()
    parser.add_argument('--adapt', action='store_true', default=False)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--path', type=str, default=None, 
        help='path to take the raw logs from. by default specified by $IGLU_DATA_PATH')
    parser.add_argument('--outpath', type=str, default=None, 
        help='path to store visualized data. by default specified by $IGLU_DATA_PATH')
    parser.add_argument('--dialogs_path', type=str, default=None, 
        help='path to dialogs.csv. by default specified by $IGLU_DATA_PATH/data/')
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--session', type=str, default=None, help='session id to process. Only used with num_workers=1')
    parser.add_argument('--single_turn', action='store_true', default=False, help='whether to parse a single turn data or a multiturn one.')
    parser.add_argument('--render_size', type=int, default=256, help='width and height of a rendered video.')

    args = parser.parse_args()
    if args.num_workers > 1:
        runner = partial(run_multiprocess, num_workers=args.num_workers)
    else:
        runner = run
        if args.session is not None:
            runner = partial(run, session_id=args.session)
    runner(overwrite=args.overwrite, adapt=args.adapt, render=args.render, visualize=args.visualize,
        path=args.path, outpath=args.outpath, dialogs_path=args.dialogs_path,
        single_turn=args.single_turn, render_size=args.render_size
    )

    print(f'data processing done! Spent {(time() - start) / 60:.1f} minutes.')