import numpy as np

BUILD_ZONE_SIZE_X = 11
BUILD_ZONE_SIZE_Z = 11


class Task:
    def __init__(self, chat, target_grid):
        self.chat = chat
        self.admissible = [[] for _ in range(4)]
        self.target_size = (target_grid != 0).sum().item()
        self.target_grid = target_grid
        self.target_grids = [target_grid]
        # fill self.target_grids with four rotations of the original grid around the vertical axis
        for _ in range(3):
            self.target_grids.append(np.zeros(target_grid.shape, dtype=np.int32))
            for x in range(BUILD_ZONE_SIZE_X):
                for z in range(BUILD_ZONE_SIZE_Z):
                    self.target_grids[-1][:, z, BUILD_ZONE_SIZE_X - x - 1] \
                        = self.target_grids[-2][:, x, z]
        #self.admissible = [[(0, 0)], [(0, 0)], [(0, 0)], [(0, 0)]]
        #return
        # (dx, dz) is admissible iff the translation of target grid by (dx, dz) preserve (== doesn't cut)
        # target structure within original (unshifted) target grid
        for i in range(4):
            for dx in range(-BUILD_ZONE_SIZE_X + 1, BUILD_ZONE_SIZE_X):
                for dz in range(-BUILD_ZONE_SIZE_Z + 1, BUILD_ZONE_SIZE_Z):
                    sls_target = self.target_grids[i][:, max(dx, 0):BUILD_ZONE_SIZE_X + min(dx, 0),
                                                         max(dz, 0):BUILD_ZONE_SIZE_Z + min(dz, 0):]
                    if (sls_target != 0).sum().item() == self.target_size:
                        self.admissible[i].append((dx, dz))

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
