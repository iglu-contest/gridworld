from pyglet.window import key, mouse
from pyglet.graphics import vertex_list
from pyglet.gl import *

from .render import Renderer

from gridworld.core.world import Agent, World


class Viewer(Renderer):
    def __init__(self, *args, overlay=True, **kwargs) -> None:
        self.exclusive = False
        self.world = World()
        self.agent = Agent(sustain=True)
        super().__init__(model=self.world, agent=self.agent, *args, **kwargs)
        self.overlay = overlay
        self.num_keys = [
            key._1, key._2, key._3, key._4, key._5,
            key._6, key._7, key._8, key._9, key._0]
        x, y = self.width // 2, self.height // 2
        n = 10
        self.reticle = vertex_list(4,
            ('v2i', (x - n, y, x + n, y, x, y - n, x, y + n))
        )

    def on_mouse_press(self, x, y, button, modifiers):
        """ Called when a mouse button is pressed. See pyglet docs for button
        amd modifier mappings.

        Parameters
        ----------
        x, y : int
            The coordinates of the mouse click. Always center of the screen if
            the mouse is captured.
        button : int
            Number representing mouse button that was clicked. 1 = left button,
            4 = right button.
        modifiers : int
            Number representing any modifying keys that were pressed when the
            mouse button was clicked.

        """
        if self.exclusive:
            right_click = (button == mouse.RIGHT) or ((button == mouse.LEFT) and (modifiers & key.MOD_CTRL))
            left_click = button == mouse.LEFT
            if right_click or left_click:
                self.world.place_or_remove_block(self.agent, remove=left_click, place=right_click)
        else:
            self.set_exclusive_mouse(True)
    
    def on_mouse_motion(self, x, y, dx, dy):
        """ Called when the player moves the mouse.

        Parameters
        ----------
        x, y : int
            The coordinates of the mouse click. Always center of the screen if
            the mouse is captured.
        dx, dy : float
            The movement of the mouse.

        """
        if self.exclusive:
            m = 0.15
            self.world.move_camera(self.agent, dx * m, dy * m)

    def set_exclusive_mouse(self, exclusive):
        """ If `exclusive` is True, the game will capture the mouse, if False
        the game will ignore the mouse.

        """
        super().set_exclusive_mouse(exclusive)
        self.exclusive = exclusive

    def on_key_press(self, symbol, modifiers):
        """ Called when the player presses a key. See pyglet docs for key
        mappings.

        Parameters
        ----------
        symbol : int
            Number representing the key that was pressed.
        modifiers : int
            Number representing any modifying keys that were pressed.

        """
        strafe = [0, 0]
        dy = 0
        inventory = None
        if symbol == key.W:
            strafe[0] -= 1
        elif symbol == key.S:
            strafe[0] += 1
        elif symbol == key.A:
            strafe[1] -= 1
        elif symbol == key.D:
            strafe[1] += 1
        elif symbol == key.SPACE:
            dy = 1
        elif symbol == key.ESCAPE:
            self.set_exclusive_mouse(False)
        elif symbol == key.TAB:
            self.agent.flying = not self.agent.flying
        elif symbol == key.Z and self.agent.flying:
            dy = -1
        elif symbol in self.num_keys:
            index = (symbol - self.num_keys[0]) % len(self.agent.inventory) + 1
            inventory = index
        self.world.movement(self.agent, strafe, dy, inventory)

    def on_key_release(self, symbol, modifiers):
        """ Called when the player releases a key. See pyglet docs for key
        mappings.

        Parameters
        ----------
        symbol : int
            Number representing the key that was pressed.
        modifiers : int
            Number representing any modifying keys that were pressed.

        """
        strafe = [0, 0]
        if symbol == key.W:
            strafe[0] += 1
        elif symbol == key.S:
            strafe[0] -= 1
        elif symbol == key.A:
            strafe[1] += 1
        elif symbol == key.D:
            strafe[1] -= 1
        self.world.movement(self.agent, strafe, dy=0, inventory=None)
    
    def on_resize(self, width, height):
        """ Called when the window is resized to a new `width` and `height`.

        """
        # label
        self.label.y = height - 10
        # reticle
        if self.reticle:
            self.reticle.delete()
        x, y = self.width // 2, self.height // 2
        n = 10
        self.reticle = vertex_list(4,
            ('v2i', (x - n, y, x + n, y, x, y - n, x, y + n))
        )
