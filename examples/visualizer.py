from gridworld.visualizer import Visualizer
from PIL import Image

vis = Visualizer(render_size=(128, 128))


# let's create one block at the center of the world:
vis.set_world_state(blocks=[(0, -1, 2, 3)], add=True)
# we added one block at coord X=0, Y=-1, Z=2, with id=3 (red). id color mapping can be found here: https://github.com/iglu-contest/gridworld/blob/master/gridworld/utils.py#L127

# next set agent's position and rotation:
# position is [x, y, z]; rotation is [yaw, pitch]
vis.set_agent_state(position=[0, 0, 3], rotation=[0, -35])

img = vis.render()
Image.fromarray(img).save('myimg.png')
vis.clear() # reset blocks in the world and the position of the agent
