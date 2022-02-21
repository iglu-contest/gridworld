from gridworld.viewer import Viewer
from gridworld.render import setup
import pyglet

if __name__ == '__main__':
    viewer = Viewer(width=800, height=600, 
                    caption='Pyglet', resizable=True)
    pyglet.clock.schedule_interval(viewer.agent.update, 1.0 / 200)
    viewer.set_exclusive_mouse(True)
    setup()
    pyglet.app.run()