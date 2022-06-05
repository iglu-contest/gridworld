from pyglet.gl import *
from numba import jit

from .utils import WHITE, GREY
from .utils import FACES
from .utils import normalize, sectorize

class World(object):
    def __init__(self):
        self.world = {}
        self.shown = {}
        self.placed = set()

        self.callbacks = {
            'on_add': [],
            'on_remove': []
        }
        self.initialized = False

    def add_callback(self, name, func):
        self.callbacks[name].append(func)

    def deinit(self):
        for block in list(self.placed):
            self.remove_block(block)
        self.initialized = False
        for block in list(self.world.keys()):
            self.remove_block(block)
        self.world = {}
        self.shown = {}

        self.placed = set()

    def build_zone(self, x, y, z, pad=0):
        return -5 - pad <= x <= 5 + pad and -5 - pad <= z <= 5 + pad and -1 - pad <= y < 8 + pad

    def _initialize(self):
        """ Initialize the world by placing all the blocks.

        """
        n = 18  # 1/2 width and height of world
        s = 1  # step size
        y = 0  # initial y height
        for x in range(-n, n + 1, s):
            for z in range(-n, n + 1, s):
                # create a layer stone an grass everywhere.
                color = GREY if not self.build_zone(x, y, z) else WHITE
                self.add_block((x, y - 2, z), color)
        self.initialized = True

    def clear(self):
        pass

    def hit_test(self, position, vector, max_distance=8):
        """ Line of sight search from current position. If a block is
        intersected it is returned, along with the block previously in the line
        of sight. If no block is found, return None, None.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position to check visibility from.
        vector : tuple of len 3
            The line of sight vector.
        max_distance : int
            How many blocks away to search for a hit.

        """
        m = 5
        x, y, z = position
        dx, dy, dz = vector
        previous = None
        for _ in range(max_distance * m):
            key = normalize((x, y, z))
            if key != previous and key in self.world:
                return key, previous
            previous = key
            x, y, z = x + dx / m, y + dy / m, z + dz / m
        return None, None

    def exposed(self, position):
        """ Returns False is given `position` is surrounded on all 6 sides by
        blocks, True otherwise.

        """
        x, y, z = position
        for dx, dy, dz in FACES:
            if (x + dx, y + dy, z + dz) not in self.world:
                return True
        return False

    def add_block(self, position, texture):
        """ Add a block with the given `texture` and `position` to the world.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to add.
        texture : list of len 3
            The coordinates of the texture squares. Use `tex_coords()` to
            generate.
        immediate : bool
            Whether or not to draw the block immediately.

        """
        if position in self.world:
            self.remove_block(position)
        self.world[position] = texture
        if self.exposed(position):
            self.show_block(position)
        if self.initialized:
            self.placed.add(position)

    def remove_block(self, position):
        """ Remove the block at the given `position`.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to remove.
        immediate : bool
            Whether or not to immediately remove block from canvas.

        """
        del self.world[position]
        if position in self.shown:
            self.hide_block(position)
        if self.initialized:
            self.placed.remove(position)

    def show_block(self, position):
        """ Show the block at the given `position`. This method assumes the
        block has already been added with add_block()

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to show.

        """
        texture = self.world[position]
        self.shown[position] = texture
        for cb in self.callbacks['on_add']:
            cb(position, texture, build_zone=self.build_zone(*position))

    def hide_block(self, position):
        """ Hide the block at the given `position`. Hiding does not remove the
        block from the world.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position of the block to hide.
        immediate : bool
            Whether or not to immediately remove the block from the canvas.

        """
        self.shown.pop(position)
        for cb in self.callbacks['on_remove']:
            cb(position, build_zone=self.build_zone(*position))