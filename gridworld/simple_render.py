import os
import ctypes
import pyglet
pyglet.options['shadow_window'] = False
if os.environ.get('IGLU_HEADLESS', '1') == '1':
    import pyglet
    pyglet.options["headless"] = True
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices is not None and devices != '':
        pyglet.options['headless_device'] = int(devices.split(',')[0])
from pyglet.window import Window
from pyglet.gl import *
from pyglet.graphics import Batch, TextureGroup
from pyglet import image, resource, graphics
from filelock import FileLock
from copy import deepcopy
import math
import numpy as np
import os
import time
import gridworld

_60FPS = 1./60

def vec(args):
    return (GLfloat * len(args))(*args)

def gl_setup():
    """ Basic OpenGL configuration.

    """
    glClearColor(0.5, 0.69, 1.0, 1)
    glEnable(GL_CULL_FACE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    glEnable(GL_LIGHT0)
    fourfv = ctypes.c_float * 4        
#    glLightfv(GL_LIGHT0, GL_POSITION, fourfv(10, 10, 10, 1))
    glLightfv(GL_LIGHT0, GL_POSITION, fourfv(300, 200, 5000, 0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, fourfv(0.1, 0.1,  0.1, 1.0))
#        glLightfv(GL_LIGHT0, GL_AMBIENT, fourfv(.2, .2, .2, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, fourfv(1.0, 1.0, 1.0, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, fourfv(1.0, 1.0, 1.0, 1.0))
    
    glEnable(GL_MULTISAMPLE_ARB)
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)


class Renderer(Window):
    def __init__(self, agent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gl_setup()
        self.entities = {}
        self.agent = agent
        self.batch = Batch()
        dir_path = os.path.dirname(gridworld.__file__)
        self.entity_assets = {}
        assets_path = os.path.join(dir_path, "entity_assets")
        assets = os.listdir(assets_path)
        with FileLock(f'/tmp/mylock'):
            for asset in assets:
                if asset.endswith(".obj"):
                    name = asset[:-4]
                    o = OBJ(os.path.join(assets_path, asset))
                    self.entity_assets[name] = o
        self.playing_field = Checker()
        self.playing_field.add_to(self.batch)
        self.buffer_manager = pyglet.image.get_buffer_manager()
        
    def on_draw(self, camera=None):
        """ 
        camera is an ((x,y,z), (tx, ty, tz)) tuple where
        x, y, z is the location in space of the camera and (tx, ty, tz) is the target
        direction
        """
        self.clear()
        viewport = self.get_viewport_size()
        glViewport(0, 0, max(1, viewport[0]), max(1, viewport[1]))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        width, height = self.get_size()
        gluPerspective(60.0, width / float(height), 0.1, 30.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        if not camera:
            camera = ((10, -5, 0), (0, 0, 0))
        x, y, z = camera[0]
        tx, ty, tz = camera[1]
        gluLookAt(x, y, z, tx, ty, tz, 0, 1, 0)
        self.batch.draw()

    def render(self, camera=None):
        self.on_draw(camera=camera)
        width, height = self.get_size()
        new_shape = (height, width, 4)
        rendered = np.asanyarray(
            self.buffer_manager
            .get_color_buffer()
            .get_image_data()
            .get_data()
        ).reshape(new_shape)[::-1]
        return rendered

    def add_entity(self, entity):
        """
        each entity is an Entity object, its relevant OBJ, and a vertex group
        if
        """
        if entity.asset_name is None:
            self.entities[entity.entityId] = [entity, None, None]
        else:
            # instantiate the obj, link to entity.entityId
            obj = deepcopy(self.entity_assets[entity.asset_name])
            V = obj.add_to(self.batch)
            self.entities[entity.entityId] = [entity, obj, V]

    def remove_entity(self, entity):
        if not self.entities.get(entity.entityId):
            return
        entity, obj, V = self.entities.get[entity.entityId]
        if V:
            for v in V:
                v.delete()
        del self.entities.get[entity.entityId]

    def update_entity(self, entity):
        if not self.entities.get(entity.entityId):
            self.add_entity(entity)
        _, obj, _ = self.entities.get(entity.entityId)
        if obj is not None:
            obj.translate(*entity.position)
#            yaw = entity.rotation[0]
#            obj.rotate(yaw, 0, 1, 0)

    def update_entities(self, entities):
        for e in entities:
            self.update_entity(e)

##############################################################################
# rectanguloids...
############################################################################


class Cube:
    def __init__(self, color=(.9, .9, .9, 1.0), center=(0,0,0), radius=.5):
        """
        color is an r,g,b,a tuple
        center is an (x, y, z) tuple
        radius is a float giving distance from center to nearest point on a face
        """
        self.color = color
        self.center = center
        self.radius = radius
        self.shifter = [center[0], center[1], center[2]]
        self.material = Material("cube_surface", shifter=self.shifter)
        self.material.diffuse = list(color[:3])
        self.material.opacity = color[3]

    def translate(self, x, y, z):
        v = self.shifter
        v[0] = x
        v[1] = y
        v[2] = z

    def _vertices(self):
        r = self.radius
        return [
            -r,+r,-r, -r,+r,+r, +r,+r,+r, +r,+r,-r,  # top
            -r,-r,-r, +r,-r,-r, +r,-r,+r, -r,-r,+r,  # bottom
            -r,-r,-r, -r,-r,+r, -r,+r,+r, -r,+r,-r,  # left
            +r,-r,+r, +r,-r,-r, +r,+r,-r, +r,+r,+r,  # right
            -r,-r,+r, +r,-r,+r, +r,+r,+r, -r,+r,+r,  # front
            +r,-r,-r, -r,-r,-r, -r,+r,-r, +r,+r,-r,  # back
        ]

    def _normals(self):
        return [
            0, 1,0, 0, 1,0, 0, 1,0, 0, 1,0, # top
            0,-1,0, 0,-1,0, 0,-1,0, 0,-1,0, # bottom
            -1,0,0, -1,0,0, -1,0,0, -1,0,0, # left
            1,0,0,  1,0,0,  1,0,0,  1,0,0, # right
            0,0, 1, 0,0, 1, 0,0, 1, 0,0, 1, # front
            0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1, # back
        ]

    def add_to(self, batch):
        b = batch.add(24, GL_QUADS, self.material,
            ('v3f/static', self._vertices()),
            ('n3f/static', self._normals()),
        )
        return b

class Square:
    def __init__(self, color=(.9, .9, .9, 1.0), center=(0,0,0), radius=.5):
        """
        color is an r,g,b,a tuple
        center is an (x, y, z) tuple
        radius is a float giving distance from center to nearest point on a face
        """
        self.color = color
        self.center = center
        self.radius = radius
        self.shifter = [center[0], center[1], center[2]]
        self.material = Material("square_surface", shifter=self.shifter)
        self.material.diffuse = list(color[:3])
        self.material.opacity = color[3]

    def translate(self, x, y, z):
        v = self.shifter
        v[0] = x
        v[1] = y
        v[2] = z

    def _vertices(self):
        r = self.radius
        return [
            -r,+r,-r, -r,+r,+r, +r,+r,+r, +r,+r,-r,  # top
        ]

    def _normals(self):
        return [
            0, 1,0, 0, 1,0, 0, 1,0, 0, 1,0, # top
        ]

    def add_to(self, batch):
        b = batch.add(4, GL_QUADS, self.material,
            ('v3f/static', self._vertices()),
            ('n3f/static', self._normals()),
        )
        return b
    
class Axes:
    def __init__(self, n=10):
        N = 2*n + 1
        self.cubes = []
        for i in range(3):
            for j in range(-n, n):
                if j != 0:
                    color = [0, 0, 0, 1.0]
                    color[i] = (j + n)/N
                    center = [0, 0, 0]
                    center[i] = j
                    self.cubes.append(Cube(color=color, center=center, radius=.5))
    def add_to(self, batch):
        for c in self.cubes:
            c.add_to(batch)

class Checker:
    def __init__(self, color=(.8, .8, .8, 1.0), center=(0,0,0), radius=5, tile_radius=.5):
        self.cubes = []
        N = 2*radius + 1
        for i in range(N):
            for j in range(N):
                c = list(color)
                if (i+j) % 2 == 0:
                    for s in range(3):
                        c[s] = max(c[s] + .1, 1.0)
                cube_center = list(center)
                cube_center[0] = 2*tile_radius*(i - radius)
                cube_center[2] = 2*tile_radius*(j - radius)
                self.cubes.append(Cube(color=c, center=cube_center, radius=tile_radius))
                    
    def add_to(self, batch):
        for c in self.cubes:
            c.add_to(batch)
                
        
    
########################################################################
# .obj loaders etc.:
# Wavefront OBJ renderer using pyglet's Batch class.

# Based on pyglet/contrib/model/model/obj.py
# and on the code of 
# Juan J. Martinez <jjm@usebox.net>
# from https://github.com/reidrac/pyglet-obj-batch/blob/master/obj_batch.py
##########################################################################

class Material(graphics.Group):
    diffuse = [.8, .8, .8]
    ambient = [.2, .2, .2]
    specular = [0., 0., 0.]
    emission = [0., 0., 0.]
    shininess = 0.
    opacity = 1.

    def __init__(self, name, shifter=[0, 0, 0], **kwargs):
        self.name = name
        super().__init__(**kwargs)
        self.shifter = shifter
#        self.rotator = rotator

    def set_state(self):
        face = GL_FRONT_AND_BACK
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(self.diffuse + [self.opacity]))
        x, y, z = self.shifter
        glTranslatef(x, y, z)
#        glMaterialfv(face, GL_DIFFUSE, vec(self.diffuse + [self.opacity]))
#        glMaterialfv(face, GL_AMBIENT, vec(self.ambient + [self.opacity]))
        #glMaterialfv(face, GL_SPECULAR,
        #    (GLfloat * 4)(*(self.specular + [self.opacity])))
        #glMaterialfv(face, GL_EMISSION,
        #    (GLfloat * 4)(*(self.emission + [self.opacity])))
        #glMaterialf(face, GL_SHININESS, self.shininess)

    def unset_state(self):
        x, y, z = self.shifter
        glTranslatef(-x, -y, -z)
        glDisable(GL_COLOR_MATERIAL)

class MaterialGroup(object):
    def __init__(self, material):
        self.material = material

        # Interleaved array of floats in GL_N3F_V3F format
        self.vertices = []
        self.normals = []
        # not currently adding textures to batches!!!!
        self.tex_coords = []
        self.array = None

class Mesh(object):
    def __init__(self, name):
        self.name = name
        self.groups = []

class OBJ(object):
    def __init__(self, filename):
        self.materials = {}
        self.material_tforms = {}
        self.meshes = {}        # Name mapping
        self.mesh_list = []     # Also includes anonymous meshes

        self._load_file(filename)
        
    def translate(self, x, y, z):
        for k, v in self.material_tforms.items():
            v[0] = x
            v[1] = y
            v[2] = z
#
#    def rotate(self, angle, x, y, z):
#        self.transforms.rotate_axis(math.pi*angle/180.0, euclid.Vector3(x, y, z))
#
#    def scale(self, x, y, z):
#        self.transforms.scale(x, y, z)
#        self.normalize = True

    def add_to(self, batch):
        '''
        Add the meshes to a batch, 
        the vertices are normalized so that the total mesh is 0-centered, and has 
        max cardinal diameter 1
        and the normals are scaled to have norm 1
        '''
        V = []
        N = []
        M = []
        for mesh in self.mesh_list:
            for group in mesh.groups:
                n = len(group.vertices)//3
                V.append(np.array(group.vertices).reshape(n, 3))
                N.append(np.array(group.normals).reshape(n, 3))
                M.append(group.material)
        X = np.concatenate(V)
        X = X - X.mean(0)
        diams = X.max(0) - X.min(0)
        diams[1] = 0 # we don't want to squish if the height is large
        D = max(diams)/2 #/2 so object is a little bigger than a tile
        X = X/D
        N = np.concatenate(N)
        N = N/np.linalg.norm(N, 2, 1).reshape(-1,1)
        count = 0
        for i, g in enumerate(V):
            n = g.shape[0]
            vertices = X[count: count + n].reshape(-1).tolist()
            normals = N[count: count + n].reshape(-1).tolist()
            batch.add(n, GL_TRIANGLES, M[i],
                      ('v3f/static', tuple(vertices)),
                      ('n3f/static', tuple(normals)),
                )
            count = count + n

    def open_material_file(self, filename):
        '''Override for loading from archive/network etc.'''
        return open(os.path.join(self.path, filename), 'r')

    def load_material_library(self, filename):
        material = None
        file = self.open_material_file(filename)

        for line in file:
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue

            if values[0] == 'newmtl':
                shifter = [0, 0, 0]
                material = Material(values[1], shifter=shifter)
                self.materials[material.name] = material
                self.material_tforms[material.name] = shifter
            elif material is None:
                logging.warn('Expected "newmtl" in %s' % filename)
                continue
            
            try:
                if values[0] == 'Kd':
                    material.diffuse = list(map(float, values[1:]))
                elif values[0] == 'Ka':
                    material.ambient = list(map(float, values[1:]))
                elif values[0] == 'Ks':
                    material.specular = list(map(float, values[1:]))
                elif values[0] == 'Ke':
                    material.emissive = list(map(float, values[1:]))
                elif values[0] == 'Ns':
                    material.shininess = float(values[1])
                elif values[0] == 'd':
                    material.opacity = float(values[1])
                elif values[0] == 'map_Kd':
                    try:
                        material.texture = resource.image(values[1]).texture
                    except BaseException as ex:
                        logging.warn('Could not load texture %s: %s' % (values[1], ex))
            except BaseException as ex:
                logging.warn('Parse error in %s.' % (filename, ex))

    def _load_file(self, filename):
        file = open(filename, 'r')
        path = os.path.dirname(filename)
        self.path = path

        mesh = None
        group = None
        material = None

        vertices = [[0., 0., 0.]]
        normals = [[0., 0., 0.]]
        tex_coords = [[0., 0.]]

        for line in file:
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue

            if values[0] == 'v':
                vertices.append(list(map(float, values[1:4])))
            elif values[0] == 'vn':
                normals.append(list(map(float, values[1:4])))
            elif values[0] == 'vt':
                tex_coords.append(list(map(float, values[1:3])))
            elif values[0] == 'mtllib':
                self.load_material_library(values[1])
            elif values[0] in ('usemtl', 'usemat'):
                material = self.materials.get(values[1], None)
                if material is None:
                    logging.warn('Unknown material: %s' % values[1])
                if mesh is not None:
                    group = MaterialGroup(material)
                    mesh.groups.append(group)
            elif values[0] == 'o':
                mesh = Mesh(values[1])
                self.meshes[mesh.name] = mesh
                self.mesh_list.append(mesh)
                group = None
            elif values[0] == 'f':
                if mesh is None:
                    mesh = Mesh('')
                    self.mesh_list.append(mesh)
                if material is None:
                    # FIXME
                    material = Material("<unknown>")
                if group is None:
                    group = MaterialGroup(material)
                    mesh.groups.append(group)

                # For fan triangulation, remember first and latest vertices
                n1 = None
                nlast = None
                t1 = None
                tlast = None
                v1 = None
                vlast = None
                #points = []
                for i, v in enumerate(values[1:]):
                    v_index, t_index, n_index = \
                        (list(map(int, [j or 0 for j in v.split('/')])) + [0, 0])[:3]
                    if v_index < 0:
                        v_index += len(vertices) - 1
                    if t_index < 0:
                        t_index += len(tex_coords) - 1
                    if n_index < 0:
                        n_index += len(normals) - 1
                    #vertex = tex_coords[t_index] + \
                    #         normals[n_index] + \
                    #         vertices[v_index]

                    group.normals += normals[n_index]
                    group.tex_coords += tex_coords[t_index]
                    group.vertices += vertices[v_index]

                    if i >= 3:
                        # Triangulate
                        group.normals += n1 + nlast
                        group.tex_coords += t1 + tlast
                        group.vertices += v1 + vlast

                    if i == 0:
                        n1 = normals[n_index]
                        t1 = tex_coords[t_index]
                        v1 = vertices[v_index]
                    nlast = normals[n_index]
                    tlast = tex_coords[t_index]
                    vlast = vertices[v_index]



                
if __name__ == "__main__":
    from gridworld.core.entity import Entity, Agent
    import visdom
    vis = visdom.Visdom()
    agent = Agent(sustain=False, asset_name=None, agent_fpv=False)
    r = Renderer(agent, width=256, height=256, resizable=False)
    E2 = Entity(asset_name="Snake")
    E = Entity(asset_name="Robot")
    r.add_entity(E)
    r.add_entity(E2)
    def image_xyz(r, xyz=(3, 0, 3), txyz=(0,0,0)):
        im = r.render(camera=(xyz, txyz))
        vis.image(im.transpose((2, 0, 1)))

    for i in range(9):
        E.move(i-5, 1, 0)
        E2.move(i, 1, i-5)
        r.update_entities([E, E2])
#            r.update_entity(E2)
#            image_xyzyp(r, 0, 3, -8, 40*j, -40 + 10*i)
        image_xyz(r, (4, 4, -8), (0,0,0))
