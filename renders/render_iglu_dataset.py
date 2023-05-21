import os
import sys

sys.path.insert(0, '../')

from gridworld import GridWorld
from gridworld.data.iglu_dataset import IGLUDataset
from gridworld.tasks import DUMMY_TASK
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
import pickle
import math
import textwrap
import cv2
import gym
render_size = (640, 640)

os.environ['IGLU_HEADLESS'] = '0'

rendered_texts = {}
def put_multiline_text(lines, height, width, text_frac=0.6):
    global rendered_texts
    lines = lines.split('\n')
    if tuple(lines) not in rendered_texts:
        width = int(math.ceil(width * text_frac))
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(canvas)
        fnt = ImageFont.truetype("arial.ttf", 18)
        char_width = np.mean([fnt.getsize(char)[0] for char in set(' '.join([l for l in lines if l is not None]))])
        chars = int(0.9 * canvas.size[0] / char_width)
        text = []
        for line in lines:
            if line is None: continue
            lined = textwrap.fill(line, width=chars)
            text.append(lined)
        text = '\n'.join(text)
        pos = (int(0.05 * canvas.size[0]), int(0.03 * canvas.size[1]))
        draw.multiline_text(pos, text, font=fnt, fill=(0,0,0,0), stroke_width=0)
        rendered_texts[tuple(lines)] = np.array(canvas)
    return np.copy(rendered_texts[tuple(lines)])

env = gym.make('IGLUGridworld-v0', render_size=render_size)
done = False

DATASET_VERSION = "v0.1.0-rc1"
tasks = IGLUDataset(dataset_version=DATASET_VERSION)
# env.set_task_generator(None)
env.set_task(DUMMY_TASK)
os.makedirs('task_renders', exist_ok=True)
SIZE = 512
sqrt2 = np.sqrt(2)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
i = 0
tq = tqdm(total=len(tasks) * 180 * 2)
for task_id, n, m, subtask in tasks:
    target = tasks.to_sparse(np.transpose(subtask.target_grid, (1, 0, 2)))
    starting = subtask.starting_grid
    if starting is None:
        starting = []
    delta = [b for b in target if b not in starting]
    cache = []
    for j, grid in enumerate([target, delta]):
        i += 1
        suffix = 'target' if j == 0 else 'delta'
        os.makedirs(f'task_renders/{task_id}_{n}', exist_ok=True)
        writer = cv2.VideoWriter(f'task_renders/{task_id}_{n}/{task_id}_{n}_step_{m}.mp4', fourcc, 20, (1664, 640))
        for frame_no, view_angle in enumerate(range(0, 360, 2)):
            done = False
            mean_pos = np.array([
                np.mean([ss[j] for ss in grid])
                for j in range(3)
            ]) if len(grid) != 0 else np.zeros(3)
            max_pos = np.array([
                np.max([ss[j] for ss in grid])
                for j in range(3)
            ]) + 0.5 if len(grid) != 0 else np.zeros(3)
            min_pos = np.array([
                np.min([ss[j] for ss in grid])
                for j in range(3)
            ]) + 0.5 if len(grid) != 0 else np.zeros(3)
            dx, dy, dz = (max_pos - min_pos)
            dxz = np.sqrt(dx ** 2 + dz ** 2) / 1. # 1.5 is given by the fov
            dy = dy / 1.
            dist = max(1, dxz, dy)
            ddx = np.sin(view_angle / 180 * np.pi) * sqrt2
            ddz = np.cos(view_angle / 180 * np.pi) * sqrt2
            init_pos = np.array([ddx * dist + mean_pos[0], mean_pos[1] + 1, ddz * dist + mean_pos[2]])
            init_pos1 = init_pos.copy()
            init_pos[1] += 1.75
            vec = mean_pos - init_pos
            vec = vec / np.linalg.norm(vec)
            pitch = 90 - np.arccos(vec[1]) * 180 / np.pi
            xz = np.sqrt(vec[0] ** 2 + vec[2] ** 2)
            yaw = 90 + np.arctan2(vec[2] / xz, vec[0] / xz) * 180 / np.pi
            init_pos = init_pos.tolist() + [yaw, pitch]
            env.initialize_world(grid, init_pos)
            obs = env.reset()
            img = env.unwrapped.render()
            if j == 0:
                cache.append(img[..., :-1][..., ::-1])
            else:
                left = cache[frame_no]
                right = img[..., :-1][..., ::-1]
                text = put_multiline_text(subtask.last_instruction, 640, 640)
                frame = np.concatenate([left, text, right], axis=1)
                writer.write(frame)
            tq.update(1)
        if j == 1:
            writer.release()
            os.system(f'ffmpeg -y -hide_banner -loglevel error -i task_renders/{task_id}_{n}/{task_id}_{n}_step_{m}.mp4 '
                      f'-vcodec libx264 task_renders/{task_id}_{n}/{task_id}_{n}_step_{m}1.mp4 '
                      f'&& mv task_renders/{task_id}_{n}/{task_id}_{n}_step_{m}1.mp4 '
                      f'task_renders/{task_id}_{n}/{task_id}_{n}_step_{m}.mp4')
tq.close()
