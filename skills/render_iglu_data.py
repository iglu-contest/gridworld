import sys
sys.path.insert(0, '../')
from gridworld.env import create_env
from gridworld.data.iglu_dataset import IGLUDataset
from tqdm import tqdm
from PIL import Image
import numpy as np
import os
import pickle
import cv2
render_size = (640, 640)
env = create_env(
    visual=True, discretize=True, size_reward=False, 
    render_size=render_size, select_and_place=True
)
done = False
tasks = IGLUDataset()
env.set_task_generator(tasks)
obs = env.reset()
img = env.unwrapped.render()
os.makedirs('task_renders', exist_ok=True)
SIZE = 512
sqrt2 = np.sqrt(2)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
i = 0
tq = tqdm(total=len(tasks) * 180)
for task_id, n, subtask in tasks:
    target = tasks.to_sparse(np.transpose(subtask.target_grid, (1, 0, 2)))
    starting = subtask.starting_grid
    ddy = 1
    for j, grid in enumerate([starting, target]):
        i += 1
        writer = cv2.VideoWriter(f'task_renders/{task_id}_{n}_{j}.mp4', fourcc, 20, (1280, 1280))
        for view_angle in range(0, 360, 2):
            done = False
            mean_pos = np.array([
                np.mean([ss[j] for ss in target])
                for j in range(3)
            ])
            max_pos = np.array([
                np.max([ss[j] for ss in target])
                for j in range(3)
            ]) + 0.5
            min_pos = np.array([
                np.min([ss[j] for ss in target])
                for j in range(3)
            ]) + 0.5
            dx, dy, dz = (max_pos - min_pos)
            dxz = np.sqrt(dx ** 2 + dz ** 2) / 1. # 1.5 is given by the fov
            dy = dy / 1.
            dist = max(1, dxz, dy)
            y_coord = -1 if ddy == 0 else mean_pos[1] + 3
            ddx = np.sin(view_angle / 180 * np.pi) * sqrt2
            ddz = np.cos(view_angle / 180 * np.pi) * sqrt2
            init_pos = np.array([ddx * dist + mean_pos[0], y_coord, ddz * dist + mean_pos[2]])
            init_pos1 = init_pos.copy()
            init_pos[1] += 1.75
            vec = mean_pos - init_pos
            vec = vec / np.linalg.norm(vec)
            pitch = 90 - np.arccos(vec[1]) * 180 / np.pi
            xz = np.sqrt(vec[0] ** 2 + vec[2] ** 2)
            yaw = 90 + np.arctan2(vec[2] / xz, vec[0] / xz) * 180 / np.pi
            init_pos = init_pos.tolist() + [yaw, pitch]
            env.initialize_world(target, init_pos)
            obs = env.reset()
            img = env.unwrapped.render() 
            jx = int(ddx / 2 + 0.5) * 2 + int(ddz / 2 + 0.5)
            jy = ddy
            writer.write(img[..., :-1][..., ::-1])
            tq.update(1)
        writer.release()
        # os.system(f'ffmpeg -y -hide_banner -loglevel error -i task_renders/{i}_{ddy}.mp4 -vcodec libx264 task_renders/{i}_{ddy}1.mp4 '
        #           f'&& mv task_renders/{i}_{ddy}1.mp4 task_renders/{i}_{ddy}.mp4')
tq.close()