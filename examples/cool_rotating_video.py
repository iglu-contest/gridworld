import sys

from gridworld import GridWorld
from gridworld.data.iglu_dataset import IGLUDataset
from gridworld.tasks.task import Subtasks
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
import os
import pickle
import math
import textwrap
import cv2
import gym


class TextRenderer:
    def __init__(self):
        self.rendered_texts = {}

    def put_multiline_text(self, lines, width, height=None, bold=False, size=40):
        lines = lines.split('\n')
        if tuple(lines) not in self.rendered_texts:
            fnt = ImageFont.truetype("FreeMono.ttf" if not bold else "FreeMonoBold.ttf", size)
            char_width = np.mean([fnt.getsize(char)[0] for char in set(' '.join([l for l in lines if l is not None]))])
            char_height = np.mean([fnt.getsize(char)[1] for char in set(' '.join([l for l in lines if l is not None]))])
            chars = int(0.9 * width / char_width)
            text = []
            n_lines = 0
            for line in lines:
                if line is None: continue
                lined = textwrap.fill(line, width=chars)
                n_lines += len(lined.split('\n'))
                text.append(lined)
            text = '\n'.join(text)

            # width = int(math.ceil(width * text_frac))
            if height is None:
                height = int(math.ceil(1.1 * (n_lines + 1) * char_height))
            canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
            canvas = Image.fromarray(canvas)
            draw = ImageDraw.Draw(canvas)

            pos = (int(0.05 * canvas.size[0]), int(0.03 * canvas.size[1]))
            draw.multiline_text(pos, text, font=fnt, fill=(0,0,0,0), stroke_width=0)
            self.rendered_texts[tuple(lines)] = np.array(canvas)
        return np.copy(self.rendered_texts[tuple(lines)])

def generate_multiview_video(task_id, last_instruction,
                             beh_frames, beh_grids,
                             starting, target, video_name):
    render_size = (512, 512)

    env = gym.make('IGLUGridworld-v0', render_size=render_size)
    done = False

    env.set_task(task_id)
    obs = env.reset()
    img = env.unwrapped.render()
    os.makedirs('task_renders', exist_ok=True)
    SIZE = render_size[0]
    sqrt2 = np.sqrt(2)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    i = 0
    tq = tqdm(total=1 * 360 * 2)

    textrenderer = TextRenderer()
    cache = []
    for j, grid in enumerate([starting, target]):
        i += 1
        writer_size = (1280, 640)
        writer = cv2.VideoWriter(f'task_renders/{video_name}vis.mp4', fourcc, 30, writer_size)
        focus_grid = target
        for frame_no, view_angle in enumerate(range(0, 360, 1)):
            done = False
            mean_pos = np.array([
                np.mean([ss[j] for ss in focus_grid])
                for j in range(3)
            ]) if len(focus_grid) != 0 else np.zeros(3)
            max_pos = np.array([
                np.max([ss[j] for ss in focus_grid])
                for j in range(3)
            ]) + 0.5 if len(focus_grid) != 0 else np.zeros(3)
            min_pos = np.array([
                np.min([ss[j] for ss in focus_grid])
                for j in range(3)
            ]) + 0.5 if len(focus_grid) != 0 else np.zeros(3)
            dx, dy, dz = (max_pos - min_pos)
            dxz = np.sqrt(dx ** 2 + dz ** 2) / 1. # 1.5 is given by the fov
            dy = dy / 1.
            dist = max(1, dxz, dy)
            ddx = np.sin(view_angle / 180 * np.pi - np.pi/2 - np.pi / 4) * sqrt2
            ddz = np.cos(view_angle / 180 * np.pi - np.pi/2 - np.pi / 4) * sqrt2
            init_pos = np.array([ddx * dist + mean_pos[0], mean_pos[1] + 1, ddz * dist + mean_pos[2]])
            init_pos1 = init_pos.copy()
            init_pos[1] += 1.75
            vec = mean_pos - init_pos
            vec = vec / np.linalg.norm(vec)
            pitch = 90 - np.arccos(vec[1]) * 180 / np.pi
            xz = np.sqrt(vec[0] ** 2 + vec[2] ** 2)
            yaw = 90 + np.arctan2(vec[2] / xz, vec[0] / xz) * 180 / np.pi
            init_pos = init_pos.tolist() + [yaw, pitch]
            if j == 0:
                #grid1 = beh_grids[(frame_no // 6) % (len(beh_frames))]
                grid1 = beh_grids[int(frame_no / 360 * (len(beh_frames)-1))]
            else:
                grid1 = grid
            env.initialize_world(grid1, init_pos)
            obs = env.reset()
            img = env.unwrapped.render()

            if j == 0:
                cache.append(img[..., :-1][..., ::-1])
            else:
                start = cache[frame_no]
                beh = beh_frames[int(frame_no / 360 * (len(beh_frames)-1))]
                targ = img[..., :-1][..., ::-1]
                text = textrenderer.put_multiline_text(last_instruction, 512, 512)
                builder_goal = textrenderer.put_multiline_text('Builder goal', 512, 512, bold=True, size=30)
                builder_prog = textrenderer.put_multiline_text('Builder progress', 512, 512, bold=True, size=30)
                builder_pov = textrenderer.put_multiline_text('Builder POV', 512, 512, bold=True, size=30)
                frame1 = np.concatenate([(targ * (builder_goal / 255.)).astype(np.uint8), (start * (builder_prog / 255.)).astype(np.uint8)], axis=0)
                frame2 = np.concatenate([text, (beh * (builder_pov / 255.)).astype(np.uint8)], axis=0)
                frame = np.concatenate([frame1, frame2], axis=1)
                if frame.shape[0] != writer_size[1] or frame.shape[1] != writer_size[0]:
                    writer.release()
                    writer_size = tuple(frame.shape)[:-1][::-1]
                    writer = cv2.VideoWriter(f'{video_name}.mp4', fourcc, 60, writer_size)
                writer.write(frame)
            tq.update(1)
        if j == 1:
            writer.release()
            os.system(f'ffmpeg -y -hide_banner -loglevel error -i {video_name}.mp4 '
                      f'-vcodec libx264 {video_name}1.mp4 '
                      f'&& mv {video_name}1.mp4 '
                      f'{video_name}.mp4')
    tq.close()
