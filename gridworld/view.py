from gridworld.viewer import Viewer
from gridworld.render import setup
from functools import partial
import pyglet

if __name__ == '__main__':
    viewer = Viewer(width=800, height=600, overlay=True, 
                    caption='Pyglet', resizable=True)
    pyglet.clock.schedule_interval(partial(viewer.world.update, viewer.agent), 1.0 / 200)
    viewer.set_exclusive_mouse(True)
    setup()
    pyglet.app.run()
