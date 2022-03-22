import math
import numba

TICKS_PER_SEC = 60000

# Size of sectors used to ease block loading.
SECTOR_SIZE = 16

WALKING_SPEED = 5
FLYING_SPEED = 15

GRAVITY = 20.0
MAX_JUMP_HEIGHT = 1.0 # About the height of a block.
# To derive the formula for calculating jump speed, first solve
#    v_t = v_0 + a * t
# for the time at which you achieve maximum height, where a is the acceleration
# due to gravity and v_t = 0. This gives:
#    t = - v_0 / a
# Use t and the desired MAX_JUMP_HEIGHT to solve for v_0 (jump speed) in
#    s = s_0 + v_0 * t + (a * t^2) / 2
JUMP_SPEED = math.sqrt(2 * GRAVITY * MAX_JUMP_HEIGHT)
TERMINAL_VELOCITY = 50

PLAYER_HEIGHT = 2

def cube_vertices(x, y, z, n, top_only=False):
    """ Return the vertices of the cube at position x, y, z with size 2*n.

    """
    if top_only:
        return [
            x-n,y+n,z-n, x-n,y+n,z+n, x+n,y+n,z+n, x+n,y+n,z-n,  # top
        ]
    return [
        x-n,y+n,z-n, x-n,y+n,z+n, x+n,y+n,z+n, x+n,y+n,z-n,  # top
        x-n,y-n,z-n, x+n,y-n,z-n, x+n,y-n,z+n, x-n,y-n,z+n,  # bottom
        x-n,y-n,z-n, x-n,y-n,z+n, x-n,y+n,z+n, x-n,y+n,z-n,  # left
        x+n,y-n,z+n, x+n,y-n,z-n, x+n,y+n,z-n, x+n,y+n,z+n,  # right
        x-n,y-n,z+n, x+n,y-n,z+n, x+n,y+n,z+n, x-n,y+n,z+n,  # front
        x+n,y-n,z-n, x-n,y-n,z-n, x-n,y+n,z-n, x+n,y+n,z-n,  # back
    ]

def cube_normals(top_only=False):
    if top_only:
        return [
            0, 1,0, 0, 1,0, 0, 1,0, 0, 1,0, # top
        ]
    return [
        0, 1,0, 0, 1,0, 0, 1,0, 0, 1,0, # top
        0,-1,0, 0,-1,0, 0,-1,0, 0,-1,0, # bottom
        -1,0,0, -1,0,0, -1,0,0, -1,0,0, # left
         1,0,0,  1,0,0,  1,0,0,  1,0,0, # right
        0,0, 1, 0,0, 1, 0,0, 1, 0,0, 1, # front
        0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1, # back
    ]

@numba.jit
def normalize(position):
    """ Accepts `position` of arbitrary precision and returns the block
    containing that position.

    Parameters
    ----------
    position : tuple of len 3

    Returns
    -------
    block_position : tuple of ints of len 3

    """
    x, y, z = position
    x, y, z = (int(round(x)), int(round(y)), int(round(z)))
    return (x, y, z)


def sectorize(position):
    """ Returns a tuple representing the sector for the given `position`.

    Parameters
    ----------
    position : tuple of len 3

    Returns
    -------
    sector : tuple of len 3

    """
    x, y, z = normalize(position)
    x, y, z = x // SECTOR_SIZE, y // SECTOR_SIZE, z // SECTOR_SIZE
    return (x, 0, z)

def tex_coord(x, y, n=4):
    """ Return the bounding vertices of the texture square.

    """
    m = 1.0 / n
    dx = x * m
    dy = y * m
    return dx, dy, dx + m, dy, dx + m, dy + m, dx, dy + m


def tex_coords(*side, top_only=False):
    """ Return a list of the texture squares for the top, bottom and side.

    """
    side = tex_coord(*side)
    result = []
    for _ in range(1 if top_only else 6):
        result.extend(side)
    return result


WHITE = -1
GREY = 0
BLUE = 1
GREEN = 2
RED = 3
ORANGE = 4
PURPLE = 5
YELLOW = 6

id2texture = {
    WHITE: tex_coords(0, 0),
    GREY: tex_coords(1, 0),
    BLUE: tex_coords(2, 0),
    GREEN: tex_coords(3, 0),
    RED: tex_coords(0, 1),
    ORANGE: tex_coords(1, 1),
    PURPLE: tex_coords(2, 1),
    YELLOW: tex_coords(3, 1)
}

id2top_texture = {
    WHITE: tex_coords(0, 0, top_only=True),
    GREY: tex_coords(1, 0, top_only=True),
    BLUE: tex_coords(2, 0, top_only=True),
    GREEN: tex_coords(3, 0, top_only=True),
    RED: tex_coords(0, 1, top_only=True),
    ORANGE: tex_coords(1, 1, top_only=True),
    PURPLE: tex_coords(2, 1, top_only=True),
    YELLOW: tex_coords(3, 1, top_only=True)
}

FACES = [
    ( 0, 1, 0),
    ( 0,-1, 0),
    (-1, 0, 0),
    ( 1, 0, 0),
    ( 0, 0, 1),
    ( 0, 0,-1),
]
