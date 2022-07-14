import re
import math
import logging
import os
from copy import deepcopy as copy

from .common import DEFAULT_POS, VWEvent, \
    AIR_TYPE, VOXELWORLD_GROUND_LEVEL, NORTH_YAW
import numpy as np

from ..iglu_dataset import block_colour_map, fix_xyz, fix_log


class ActionsParser:
    def __init__(self, world, agent):
        self.init_pos = DEFAULT_POS
        self.world = world
        self.agent = agent
        self.ground_level = 63
        # voxel world coordinates
        self.camera = np.array([0., 0.]) # pitch yaw
        self.position = np.array([0., 0., 0.]) # x y z
        self.prev_line = None
        self.last_grid = []
        
        self.data_sequence = []
        self.global_vw_step = 0

    def reset(self):
        self.camera = np.array([0., 0.]) # pitch yaw
        self.position = np.array([0., 0., 0.]) # x y z
        self.last_grid = []
        self.global_vw_step = 0

    def new_event(
            self, kind, params, grid=None, gridUpdate=None, 
            camera=None, position=None, step=None, turn=None
        ):
        if grid is not None and gridUpdate is not None:
            raise ValueError('Either grid or gridUpdate should be None')
        if grid is not None:
            self.last_grid = copy(grid)
        if gridUpdate is not None:
            grid = sorted(copy(self.last_grid) + gridUpdate)
            self.last_grid = grid
        if grid is None and gridUpdate is None:
            grid = self.last_grid
        if camera is None:
            camera = self.camera
        else:
            self.camera = copy(camera)
        if position is None:
            position = self.position
        else:
            self.position = copy(position)
        return VWEvent(
            kind=kind, params=params,
            grid=grid, camera=camera, position=position, 
            step=step, turn=turn
        )

    def set_look(self, *args, n=0, g=0, **kwargs):
        camera_vec = eval(' '.join(args))
        camera_vec = np.array(camera_vec).astype(np.float32)
        camera_vec *= 180 / np.pi
        # camera_vec = [pitch, yaw]
        # camera_vec[0] *= -1
        camera_vec[1] *= -1
        camera_vec = camera_vec[::-1]
        # if camera_vec[0] > 0:
        #     # player in the voxelworld is smaller than in malmo.
        #     # therefore, we need to adjust the vertical camera
        #     # angle to make to point the same direction
        #     pitch = np.pi / 2 - camera_vec[0] * np.pi / 180
        #     malmo_h = 1.8
        #     # vw_h = 1.22
        #     vw_h = 1.25
        #     y = self.position[1]
        #     frac = (vw_h + y) / (malmo_h + y)
        #     adjusted_pitch = math.atan(math.tan(pitch) * frac)
        #     adjusted_pitch = (np.pi / 2 - adjusted_pitch) * 180 / np.pi
        #     camera_vec[0] = min(90, max(adjusted_pitch, 0))
        return self.new_event(
            kind='set_look', params=args, 
            camera=camera_vec, step=n, turn=g
        )

    def block_change(self, *args, n=0, g=0, **kwargs):
        # handle multiblock case
        args = re.sub('\)', '),', ' '.join(args))
        args = f'({args})'
        for (x, y, z, prev_type, new_type) in eval(args):
            if new_type != AIR_TYPE:
                new_grid = None
            else:
                new_grid = copy(self.last_grid)
                for j in range(len(new_grid)):
                    if new_grid[j] == (x + 5, y - VOXELWORLD_GROUND_LEVEL - 1, z + 5):
                        del new_grid[j]
                        break
        return self.new_event(
            kind='block_change', params=args,
            grid=new_grid, step=n, turn=g
        )

    def pos_change(self, *args, n=0, g=0, **kwargs):
        x, y, z = eval(' '.join(args))
        y -= VOXELWORLD_GROUND_LEVEL + 0.25
        move = np.array([x, y, z])
        return self.new_event(
            kind='pos_change', params=args,
            position=move, step=n, turn=g
        )

    def action(self, action_type, *args, n=0, g=0, **kwargs):
        if action_type == 'select_and_place_block':
            # TODO: check for the new data format
            self.agent.position = self.position.tolist()
            self.agent.rotation = self.camera.tolist()
            vector = self.world.get_sight_vector(self.agent)
            block, prev = self.world.hit_test(self.agent.position, vector, max_distance=10)
            new_block = list(map(int, args[:4]))
            # new_block[1] += 5
            # new_block[3] += 5
            new_block[2] -= VOXELWORLD_GROUND_LEVEL + 1
            new_block[0] = block_colour_map.get(new_block[0], 1) # todo: check why some have broken id.
            gridUpdate = [tuple(new_block)]
        else:
            gridUpdate = None
        return self.new_event(
            kind='action', params=(action_type, *args),
            gridUpdate=gridUpdate, step=n, turn=g
        )

    def parse_one(self, line, n=0, g=0):
        _, event_type, *args = line.strip().split()
        if not hasattr(self, event_type):
            return
        handler = getattr(self, event_type)
        event = handler(*args, n=n, g=g)
        return event

    def parse_init_conds(self, data, position=None):
        if position is None:
            position = {}
        if 'avatarInfo' not in data:
            return DEFAULT_POS, []
        x, y, z = data['avatarInfo']['pos']
        pitch, yaw = data['avatarInfo']['look']
        yaw += 4 * np.pi
        yaw += NORTH_YAW / 180 * np.pi
        yaw = np.fmod(yaw, 2 * np.pi) / np.pi * 180
        yaw = 360 - yaw if yaw > 180 else yaw
        y -= VOXELWORLD_GROUND_LEVEL + 1
        if 'x' in position:
            x = position['x']
        if 'y' in position:
            y = position['y']
        if 'z' in position:
            z = position['z']
        if 'pitch' in position:
            pitch = position['pitch']
        if 'yaw' in position:
            yaw = position['yaw']
        start_position = (x, y, z, pitch, yaw)

        initial_blocks = [
            (x, y - VOXELWORLD_GROUND_LEVEL - 1, z, block_colour_map[bid]) 
            for (x, y, z, bid) in data['worldEndingState']['blocks']
        ]
        return start_position, initial_blocks

    def render_callback(self, event, i, k, output, obs, reward, done, info):
        malmo_grid = sorted(list(zip(*np.transpose(obs['grid'], (2,0,1)).nonzero())))
        try:
            vw_grid = sorted(event.grid)
        except:
            vw_grid = malmo_grid
        if k == len(event.actions) - 1:
            if any(g1 != g2 for g1, g2 in zip(malmo_grid, vw_grid)):
                outdir, _ = os.path.split(output)
                os.makedirs(outdir, exist_ok=True)
                name = os.path.join(
                    outdir, 
                    f'mismatch_turn_{event.turn}_step_{i}_'
                    f'action_{k}_y_{event.position[1].round(2).item()}.png'
                )
                image = obs['pov']
                d = int(math.floor(0.01 * (image.shape[0] + image.shape[1]) / 2))
                image[image.shape[0]//2-d:image.shape[0]//2+d+1,image.shape[1]//2] = 0
                image[image.shape[0]//2, image.shape[1]//2-d:image.shape[1]//2+d+1] = 0
                import matplotlib.pyplot as plt
                plt.imsave(name, obs['pov'])
                print('grid mismatch!')

    def parse(self, sourcefile, g=0, last=False):
        if isinstance(sourcefile, str) and os.path.exists(sourcefile):
            with open(sourcefile) as f:
                for i, line in enumerate(f.readlines()):
                    self.parse_one(line, n=i, g=g)
        else:
            data = sourcefile
            tape = fix_log(data['tape'].strip()).split('\n')
            
            j = 0
            if g > 0 and 'prevWorldEndingState' in data:
                # skip state restoring actions
                prev_world = data['prevWorldEndingState']['blocks']
                # todo: why there is some strange moving actions at random? 
                if len(prev_world) != 0:
                    if 'action step_' in tape[j]:
                        j += 1
                    # first go placement actions
                    while 'action select_and_place_block' in tape[j]:
                        j += 1
                    
                    assert 'block_change' in tape[j]
                    blks = re.sub('\)', '),', ' '.join(tape[j].split(' ')[2:]))
                    blks = eval(f'({blks})')
                    assert len(blks) == len(prev_world)
                    j += 1
                    assert 'pos_change' in tape[j]
                    j += 1
                    assert 'set_look' in tape[j]
                    j += 1
            self.global_vw_step += j
            event_sequence = []
            for i, line in enumerate(tape[j:], start=j):
                event_sequence.append(self.parse_one(line, n=self.global_vw_step, g=g))
                self.global_vw_step += 1
            return event_sequence

if __name__ == '__main__':
    parser = ActionsParser()
    parser.parse('/Users/admin/workspace/gridworld2/examples/data/iglu/builder-data/17-c163/step-2')