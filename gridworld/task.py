from posix import listdir

import numpy as np

BUILD_ZONE_SIZE_X = 11
BUILD_ZONE_SIZE_Z = 11
BUILD_ZONE_SIZE = 9, 11, 11


class Task:
    def __init__(self, chat, target_grid, starting_grid=None, full_grid=None):
        self.chat = chat
        self.starting_grid = starting_grid
        self.full_grid = full_grid
        self.admissible = [[] for _ in range(4)]
        self.target_size = (target_grid != 0).sum().item()
        self.full_size = self.target_size
        if full_grid is not None:
            self.full_size = (full_grid != 0).sum().item()
        self.target_grid = target_grid
        self.target_grids = [target_grid]
        full_grids = [full_grid]
        self.max_int = 0
        self.prev_grid_size = 0
        self.right_placement = 0
        self.wrong_placement = 0
        # fill self.target_grids with four rotations of the original grid around the vertical axis
        for _ in range(3):
            self.target_grids.append(np.zeros(target_grid.shape, dtype=np.int32))
            full_grids.append(np.zeros(target_grid.shape, dtype=np.int32))
            for x in range(BUILD_ZONE_SIZE_X):
                for z in range(BUILD_ZONE_SIZE_Z):
                    self.target_grids[-1][:, z, BUILD_ZONE_SIZE_X - x - 1] \
                        = self.target_grids[-2][:, x, z]
                    if full_grid is not None:
                        full_grids[-1][:, z, BUILD_ZONE_SIZE_X - x - 1] \
                            = full_grids[-2][:, x, z]
        # (dx, dz) is admissible iff the translation of target grid by (dx, dz) preserve (== doesn't cut)
        # target structure within original (unshifted) target grid
        self.admissible = [[(0,0)]]
        return
        for i in range(4):
            if full_grid is not None:
                grid = full_grids[i]
            else:
                grid = self.target_grids[i]
            for dx in range(-BUILD_ZONE_SIZE_X + 1, BUILD_ZONE_SIZE_X):
                for dz in range(-BUILD_ZONE_SIZE_Z + 1, BUILD_ZONE_SIZE_Z):
                    sls_target = grid[:, max(dx, 0):BUILD_ZONE_SIZE_X + min(dx, 0),
                                         max(dz, 0):BUILD_ZONE_SIZE_Z + min(dz, 0):]
                    if (sls_target != 0).sum().item() == self.full_size:
                        self.admissible[i].append((dx, dz))

    def sample(self):
        if self.starting_grid is not None:
            self.max_int = self.maximal_intersection(Tasks.to_dense(self.starting_grid))
        else:
            self.max_int = 0
        self.prev_grid_size = len(self.starting_grid) if self.starting_grid is not None else 0
        self.right_placement = 0
        self.wrong_placement = 0
        return self

    def calc_reward(self, grid):
        grid_size = (grid != 0).sum().item()
        wrong_placement = (self.prev_grid_size - grid_size)
        max_int = self.maximal_intersection(grid) if wrong_placement != 0 else self.max_int
        done = max_int == self.target_size
        self.prev_grid_size = grid_size
        right_placement = (max_int - self.max_int)
        self.max_int = max_int
        self.right_placement = right_placement
        self.wrong_placement = wrong_placement
        return right_placement, wrong_placement, done

    def maximal_intersection(self, grid):
        max_int = 0
        for i, admissible in enumerate(self.admissible):
            for dx, dz in admissible:
                x_sls = slice(max(dx, 0), BUILD_ZONE_SIZE_X + min(dx, 0))
                z_sls = slice(max(dz, 0), BUILD_ZONE_SIZE_Z + min(dz, 0))
                sls_target = self.target_grids[i][:, x_sls, z_sls]

                x_sls = slice(max(-dx, 0), BUILD_ZONE_SIZE_X + min(-dx, 0))
                z_sls = slice(max(-dz, 0), BUILD_ZONE_SIZE_Z + min(-dz, 0))
                sls_grid = grid[:, x_sls, z_sls]
                intersection = ((sls_target == sls_grid) & (sls_target != 0)).sum().item()
                if intersection > max_int:
                    max_int = intersection
        return max_int

class Tasks:
    """
    Represents many tasks where one can be active
    """
    @classmethod
    def to_dense(cls, blocks):
        if isinstance(blocks, (list, tuple)):
            if all(isinstance(b, (list, tuple)) for b in blocks):
                grid = np.zeros(BUILD_ZONE_SIZE, dtype=np.int)
                for x, y, z, block_id in blocks:
                    grid[y + 1, x + BUILD_ZONE_SIZE_X // 2, z + BUILD_ZONE_SIZE_Z // 2] = block_id
                blocks = grid
        return blocks

    @classmethod
    def to_sparse(cls, blocks):
        if isinstance(blocks, np.ndarray):
            idx = blocks.nonzero()
            types = [blocks[i] for i in zip(*idx)]
            blocks = [(*i, t) for i, t in zip(idx, types)]
            new_blocks = []
            for x, y, z, bid in blocks:
                new_blocks.append((x - BUILD_ZONE_SIZE_X // 2, y - 1, z - BUILD_ZONE_SIZE_Z // 2, bid))
            blocks = new_blocks
        return blocks

    def sample(self) -> Task:
        return NotImplemented

    def set_task(self, task_id):
        return NotImplemented

    def set_task_obj(self, task: Task):
        return NotImplemented


class Subtasks(Tasks):
    """ Subtasks object represents a staged task where subtasks have separate segments
    """
    def __init__(self, dialog, structure_seq) -> None:
        self.dialog = dialog
        self.structure_seq = structure_seq
        self.next = None
        self.full = False
        self.full_structure = self.to_dense(self.structure_seq[-1])
        self.current = self.sample()

    def sample(self):
        if self.next is None:
            turn = np.random.choice(len(self.structure_seq) - 1) + 1
        else:
            turn = self.next
        self.task_id = turn
        self.current = self.create_task(self.task_id)
        return self.current

    def create_task(self, turn):
        dialog = '\n'.join([utt for utt in self.dialog[:turn] if utt is not None])
        initial_blocks = self.structure_seq[turn - 1]
        tid = min(turn + 1, len(self.structure_seq) - 1) if not self.full else -1
        target_grid = self.structure_seq[tid]
        task = Task(
            dialog, target_grid=self.to_dense(target_grid),
            starting_grid=self.to_sparse(initial_blocks),
            full_grid=self.full_structure
        )
        # To properly init max_int and prev_grid_size fields
        task.sample()
        return task

    def calc_reward(self, grid):
        return self.current.calc_reward(grid)

    def set_task(self, task_id):
        self.task_id = task_id
        self.current = self.create_task(task_id)
        return self.current

    def set_task_obj(self, task: Task):
        self.task_id = None
        self.current = task
        return self.current
