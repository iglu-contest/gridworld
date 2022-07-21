import sys
sys.path.insert(0, '../')
from gridworld.visualizer import Visualizer
from tqdm import tqdm
from PIL import Image
import numpy as np
import os
import pickle

seq = []
c = 0
d = 0.
os.makedirs('labeling', exist_ok=True)
q = 0
SIZE = 512
vis = Visualizer(render_size=(512, 512))
with open('goals.pkl', 'rb') as fl:
    stru = pickle.load(fl)
for task_id, s in tqdm(list(stru.items())):
    idx = 0
    quad = np.zeros((2 * SIZE, 4 * SIZE, 3), dtype=np.uint8)
    q = 0
    vis.clear()
    for ddy in [0, 1]:
        for ddx, ddz in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:
            mean_pos = np.array([
                np.mean([ss[j] for ss in s])
                for j in range(3)
            ])
            max_pos = np.array([
                np.max([ss[j] for ss in s])
                for j in range(3)
            ]) + 0.5
            min_pos = np.array([
                np.min([ss[j] for ss in s])
                for j in range(3)
            ]) + 0.5
            dx, dy, dz = (max_pos - min_pos)
            dxz = np.sqrt(dx ** 2 + dz ** 2) / 1.5 # 2 is given by the fov
            dy = dy / 1.5
            dist = max(1, dxz, dy)
            y_coord = -1 if ddy == 0 else mean_pos[1] + 3
            init_pos = np.array([ddx * dist, y_coord, ddz * dist])
            init_pos1 = init_pos.copy()
            init_pos[1] += 1.75
            vec = mean_pos - init_pos
            vec = vec / np.linalg.norm(vec)
            pitch = 90 - np.arccos(vec[1]) * 180 / np.pi
            xz = np.sqrt(vec[0] ** 2 + vec[2] ** 2)
            yaw = 90 + np.arctan2(vec[2] / xz, vec[0] / xz) * 180 / np.pi
            vis.set_agent_state(init_pos.tolist(), [yaw, pitch])
            vis.set_world_state(s)
            img = vis.render()
            jx = int(ddx / 2 + 0.5) * 2 + int(ddz / 2 + 0.5)
            jy = ddy
            quad[jy * SIZE: (jy + 1) * SIZE,
                 jx * SIZE: (jx + 1) * SIZE] = img
            q += 1
            Image.fromarray(img).save(f'labeling/{task_id}_{q}.png')
    Image.fromarray(quad).save(f'labeling/{task_id}.png')
