from pyglet.window import Window
from pyglet.gl import *
from pyglet.graphics import Batch, TextureGroup
from scipy.ndimage import gaussian_filter
from pyglet import image
import pyglet
from filelock import FileLock
import math
import numpy as np
import os
from PIL import Image
import gridworld

from .utils import WHITE, GREY, cube_vertices, cube_normals, id2texture, id2top_texture


def setup_fog():
    """ Configure the OpenGL fog properties.
    """
    glEnable(GL_FOG)
    glFogfv(GL_FOG_COLOR, (GLfloat * 4)(0.5, 0.69, 1.0, 1))
    glHint(GL_FOG_HINT, GL_DONT_CARE)
    glFogi(GL_FOG_MODE, GL_LINEAR)
    glFogf(GL_FOG_START, 25.0)
    glFogf(GL_FOG_END, 30.0)


def setup():
    """ Basic OpenGL configuration.

    """
    glClearColor(0.5, 0.69, 1.0, 1)
    glEnable(GL_CULL_FACE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    # setup_fog()


class Renderer(Window):
    TEXTURE_PATH = 'shades_texture.png'


    def __init__(self, model, agent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.agent = agent
        self.model.add_callback('on_add', self.add_block)
        self.model.add_callback('on_remove', self.remove_block)
        self.batch = Batch()
        dir_path = os.path.dirname(gridworld.__file__)
        TEXTURE_PATH = os.path.join(dir_path, Renderer.TEXTURE_PATH)
        np_texture = np.asarray(Image.open(TEXTURE_PATH)).copy()
        path = TEXTURE_PATH
        # if kwargs['width'] <= 128:
        #     s = 1.5
        #     if kwargs['width'] <= 64:
        #         s = 2
        #     # TODO: create separate textures for low-res
        #     # for edge lines to look better in low resolution
        #     for i in range(4):
        #         for j in range(4):
        #             np_texture[i * 64: (i + 1) * 64, j * 64: (j + 1) * 64] = gaussian_filter(
        #                 np_texture[i * 64: (i + 1) * 64, j * 64: (j + 1) * 64], 
        #                 (s, s, 0), mode='nearest'
        #             )
        #     path = os.path.join(os.path.dirname(TEXTURE_PATH), 'some.png')
        #     with FileLock(f'/tmp/mylock'):
        #         Image.fromarray(np_texture).save(path)
        # else:
        #     path = TEXTURE_PATH
        with FileLock(f'/tmp/mylock'):
            self.texture_group = TextureGroup(image.load(path).get_texture())
        self.overlay = False
        self._shown = {}
        self.label = pyglet.text.Label('', font_name='Arial', font_size=18,
            x=10, y=self.height - 10, anchor_x='left', anchor_y='top',
            color=(0, 0, 0, 255))
        self.model._initialize()
        self.buffer_manager = pyglet.image.get_buffer_manager()

    def set_2d(self):
        """ Configure OpenGL to draw in 2d.

        """
        width, height = self.get_size()
        glDisable(GL_DEPTH_TEST)
        viewport = self.get_viewport_size()
        glViewport(0, 0, max(1, viewport[0]), max(1, viewport[1]))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, max(1, width), 0, max(1, height), -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def set_3d(self):
        """ Configure OpenGL to draw in 3d.

        """
        width, height = self.get_size()
        glEnable(GL_DEPTH_TEST)
        # glEnable(GL_LIGHTING)
        # glEnable(GL_LIGHT0)
        viewport = self.get_viewport_size()
        glViewport(0, 0, max(1, viewport[0]), max(1, viewport[1]))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(120.0, width / float(height), 0.1, 30.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        x, y = self.agent.rotation
        glRotatef(x, 0, 1, 0)
        glRotatef(-y, math.cos(math.radians(x)), 0, math.sin(math.radians(x)))
        x, y, z = self.agent.position
        glTranslatef(-x, -y, -z)
        # glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat*4)(0.0,9,0.0,1))
        # glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat*4)(1,1,1,1))
        # glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat*4)(1.0,1.0,1.0,1))
        

    def on_draw(self):
        """ Called by pyglet to draw the canvas.

        """
        self.clear()
        self.set_3d()
        glColor3d(1, 1, 1)
        self.batch.draw()
        
        if self.overlay:
            self.draw_focused_block()
            self.set_2d()
            self.draw_label()
            self.draw_reticle()
    
    def render(self):
        self.on_draw()
        # width, height = self.get_size()
        return np.asanyarray(
            self.buffer_manager
            .get_color_buffer()
            .get_image_data()
            .get_data()
        ).reshape((64, 64, 4))[::-1]

    def add_block(self, position, texture_id, **kwargs):
        x, y, z = position
        top_only = texture_id in [WHITE, GREY]
        texture = (id2top_texture if top_only else id2texture)[texture_id]
        vertex_data = cube_vertices(x, y, z, 0.5, top_only=top_only)
        normal_data = cube_normals(top_only=top_only)
        texture_data = list(texture)
        # create vertex list
        # FIXME Maybe `add_indexed()` should be used instead
        self._shown[position] = self.batch.add(4 if top_only else 24, GL_QUADS, self.texture_group,
            ('v3f/static', vertex_data),
            ('t2f/static', texture_data),
            ('n3f/static', normal_data)
        )
    
    def remove_block(self, position, **kwargs):
        if position in self._shown:
            self._shown.pop(position).delete()

    def draw_focused_block(self):
        """ Draw black edges around the block that is currently under the
        crosshairs.

        """
        block = self.agent.get_focused_block()
        if block:
            x, y, z = block
            vertex_data = cube_vertices(x, y, z, 0.51)
            glColor3d(0, 0, 0)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            pyglet.graphics.draw(24, GL_QUADS, ('v3f/static', vertex_data))
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def draw_label(self):
        """ Draw the label in the top left of the screen.

        """
        x, y, z = self.agent.position
        self.label.text = '%02d (%.2f, %.2f, %.2f) %d / %d' % (
            pyglet.clock.get_fps(), x, y, z,
            len(self._shown), len(self.model.world))
        self.label.draw()

    def draw_reticle(self):
        """ Draw the crosshairs in the center of the screen.

        """
        glColor3d(0, 0, 0)
        self.reticle.draw(GL_LINES)