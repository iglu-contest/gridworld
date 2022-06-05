import math
from .utils import WHITE, GREY, BLUE, FACES, id2texture
from .utils import FLYING_SPEED, WALKING_SPEED, GRAVITY, TERMINAL_VELOCITY, PLAYER_HEIGHT, JUMP_SPEED
from .utils import normalize, PLAYER_HEIGHT

from typing import Optional


class Agent(object):
    PAD = 0.25
    def __init__(self, world, sustain=False) -> None:
        # When flying gravity has no effect and speed is increased.
        self.flying = False

        # Strafing is moving lateral to the direction you are facing,
        # e.g. moving to the left or right while continuing to face forward.
        #
        # First element is -1 when moving forward, 1 when moving back, and 0
        # otherwise. The second element is -1 when moving left, 1 when moving
        # right, and 0 otherwise.
        self.strafe = [0, 0]

        # Current (x, y, z) position in the world, specified with floats. Note
        # that, perhaps unlike in math class, the y-axis is the vertical axis.
        self.position = (0, 0, 0)

        # First element is rotation of the player in the x-z plane (ground
        # plane) measured from the z-axis down. The second is the rotation
        # angle from the ground plane up. Rotation is in degrees.
        #
        # The vertical plane rotation ranges from -90 (looking straight down) to
        # 90 (looking straight up). The horizontal rotation range is unbounded.
        self.rotation = (0, 0)
        # The crosshairs at the center of the screen.
        self.reticle = None
        # actions are long-lasting switches
        self.sustain = sustain

        # Velocity in the y (upward) direction.
        self.dy = 0
        self.time_int_steps = 2

        # A list of blocks the player can place. Hit num keys to cycle.
        self.inventory = [
            20, 20, 20, 20, 20, 20
        ]
        # The current block the user can place. Hit num keys to cycle.
        self.active_block = BLUE
        self.world = world

        # Convenience list of num keys.
        # self.num_keys = [
        #     key._1, key._2, key._3, key._4, key._5,
        #     key._6, key._7, key._8, key._9, key._0]
    
    def get_sight_vector(self):
        """ Returns the current line of sight vector indicating the direction
        the player is looking.

        """
        x, y = self.rotation
        # y ranges from -90 to 90, or -pi/2 to pi/2, so m ranges from 0 to 1 and
        # is 1 when looking ahead parallel to the ground and 0 when looking
        # straight up or down.
        m = math.cos(math.radians(y))
        # dy ranges from -1 to 1 and is -1 when looking straight down and 1 when
        # looking straight up.
        dy = math.sin(math.radians(y))
        dx = math.cos(math.radians(x - 90)) * m
        dz = math.sin(math.radians(x - 90)) * m
        return (dx, dy, dz)

    def get_motion_vector(self):
        """ Returns the current motion vector indicating the velocity of the
        player.

        Returns
        -------
        vector : tuple of len 3
            Tuple containing the velocity in x, y, and z respectively.

        """
        if any(self.strafe):
            x, y = self.rotation
            strafe = math.degrees(math.atan2(*self.strafe))
            y_angle = math.radians(y)
            x_angle = math.radians(x + strafe)
            if self.flying:
                m = math.cos(y_angle)
                dy = math.sin(y_angle)
                if self.strafe[1]:
                    # Moving left or right.
                    dy = 0.0
                    m = 1
                if self.strafe[0] > 0:
                    # Moving backwards.
                    dy *= -1
                # When you are flying up or down, you have less left and right
                # motion.
                dx = math.cos(x_angle) * m
                dz = math.sin(x_angle) * m
            else:
                dy = 0.0
                dx = math.cos(x_angle)
                dz = math.sin(x_angle)
        else:
            dy = 0.0
            dx = 0.0
            dz = 0.0
        return (dx, dy, dz)
    
    def update(self, dt=1.0/5):
        """ This method is scheduled to be called repeatedly by the pyglet
        clock.

        Parameters
        ----------
        dt : float
            The change in time since the last call.

        """
        m = self.time_int_steps
        dt = min(dt, 0.2)
        for _ in range(m):
            self._update(dt / m)
        if not self.sustain:
            self.strafe = [0, 0]

    def _update(self, dt):
        """ Private implementation of the `update()` method. This is where most
        of the motion logic lives, along with gravity and collision detection.

        Parameters
        ----------
        dt : float
            The change in time since the last call.

        """
        # walking
        speed = FLYING_SPEED if self.flying else WALKING_SPEED
        d = dt * speed # distance covered this tick.
        dx, dy, dz = self.get_motion_vector()
        # New position in space, before accounting for gravity.
        dx, dy, dz = dx * d, dy * d, dz * d
        # gravity
        if not self.flying:
            # Update your vertical speed: if you are falling, speed up until you
            # hit terminal velocity; if you are jumping, slow down until you
            # start falling.
            self.dy -= dt * GRAVITY
            if self.dy < -14:
                self.time_int_steps = 12
            elif self.dy < -10:
                self.time_int_steps = 8
            elif self.dy < -5:
                self.time_int_steps = 4
            else:
                self.time_int_steps = 2
            self.dy = max(self.dy, -TERMINAL_VELOCITY)
            dy += self.dy * dt
        # collisions
        x, y, z = self.position
        cand = (x + dx, y + dy, z + dz)
        if self.world.build_zone(*cand, pad=2):
            x, y, z = self.collide(cand, PLAYER_HEIGHT)
        elif not self.flying:
            x, y, z = self.collide((x, y + dy, z), PLAYER_HEIGHT)
        self.position = (x, y, z)
    
    def collide(self, position, height, new_blocks=None):
        """ Checks to see if the player at the given `position` and `height`
        is colliding with any blocks in the world.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position to check for collisions at.
        height : int or float
            The height of the player.

        Returns
        -------
        position : tuple of len 3
            The new position of the player taking into account collisions.

        """
        # How much overlap with a dimension of a surrounding block you need to
        # have to count as a collision. If 0, touching terrain at all counts as
        # a collision. If .49, you sink into the ground, as if walking through
        # tall grass. If >= .5, you'll fall through the ground.
        pad = Agent.PAD
        p = list(position)
        np = normalize(position)
        for face in FACES:  # check all surrounding blocks
            for i in range(3):  # check each dimension independently
                if not face[i]:
                    continue
                # How much overlap you have with this dimension.
                d = (p[i] - np[i]) * face[i]
                if d < pad:
                    continue
                for dy in range(height):  # check each height
                    op = list(np)
                    op[1] -= dy
                    op[i] += face[i]
                    if tuple(op) not in self.world.world \
                       and (new_blocks is None or tuple(op) not in new_blocks):
                        continue
                    p[i] -= (d - pad) * face[i]
                    if face == (0, -1, 0) or face == (0, 1, 0):
                        # You are colliding with the ground or ceiling, so stop
                        # falling / rising.
                        self.dy = 0
                    break
        return tuple(p)
    
    def place_or_remove_block(self, remove: bool, place: bool):
        if place and remove: return
        vector = self.get_sight_vector()
        block, previous = self.world.hit_test(self.position, vector)
        if place:
            # ON OSX, control + left click = right click.
            if previous:
                if self.inventory[self.active_block - 1] > 0 and self.world.build_zone(*previous): 
                    # # block under themself
                    x, y, z = self.position
                    y = y - (PLAYER_HEIGHT - 1) + Agent.PAD
                    bx, by, bz = previous
                    bx -= 0.5
                    bz -= 0.5
                    if not (bx <= x <= bx + 1 and bz <= z <= bz + 1 
                       and (by <= y <= by + 1 or by <= (y + 1) <= by + 1)):
                        self.world.add_block(previous, self.active_block)
                        self.inventory[self.active_block - 1] -= 1
        if remove and block:
            texture = self.world.world[block]
            if texture != GREY and texture != WHITE:
                self.world.remove_block(block)
                # TODO: avoid direct comparisons
                # self.inventory[self.active_block] += 1
    

    def get_focused_block(self,):
        vector = self.get_sight_vector()
        return self.world.hit_test(self.position, vector)[0]

    def move_camera(self, dx, dy):
        x, y = self.rotation
        x, y = x + dx, y + dy
        y = max(-90, min(90, y))
        self.rotation = (x, y)

    def movement(self, strafe: list, jump: bool, inventory: Optional[int]):
        self.strafe[0] += strafe[0]
        self.strafe[1] += strafe[1]
        if jump and self.dy == 0:
            self.dy = JUMP_SPEED
        if inventory is not None:
            if inventory < 1 or inventory > 6:
                raise ValueError(f'Bad inventory id: {inventory}')
            self.active_block = inventory