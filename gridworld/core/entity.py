import uuid
from ..utils import BLUE

class Entity:
    def __init__(self,
                 passable=True,
                 pickable=False,
                 position=(0,0,0),
                 rotation=(0,0),
                 asset_name=None,
                 entityId=None,
                 render_info=None):
        self.passable = passable
        self.pickable = pickable
        self.position = position
        self.rotation = rotation
        self.asset_name = asset_name
        self.render_info = render_info
        self.entityId = entityId or uuid.uuid4().hex

    def move(self, x, y, z):
        self.position = (x, y, z)

    def rotate(self, yaw, pitch):
        self.rotation = (yaw, pitch)
        
    
class Agent(Entity):
    PAD = 0.25
    __slots__ = 'flying', 'strafe', 'position', 'rotation', 'reticle', 'sustain', 'dy', 'time_int_steps', \
        'inventory', 'active_block'
    def __init__(self,
                 passable=True,
                 pickable=False,
                 position=(0,0,0),
                 rotation=(0,0),
                 asset_name=None,
                 entityId=None,
                 agent_fpv=False,
                 sustain=False) -> None:
        super().__init__(
            passable=passable,
            pickable=pickable,
            position=position,
            rotation=rotation,
            asset_name=asset_name,
            entityId=entityId
            )
        # When flying gravity has no effect and speed is increased.
        self.flying = False
        self.strafe = [0, 0]
        self.position = (0, 0, 0)
        self.rotation = (0, 0)
        self.reticle = None

        # if not fpv, then we need to draw the agent (and keep camera fixed)
        self.agent_fpv = agent_fpv

        # actions are long-lasting state switches
        self.sustain = sustain

        # Velocity in the y (upward) direction.
        self.dy = 0
        self.time_int_steps = 2
        self.inventory = [
            20, 20, 20, 20, 20, 20
        ]
        self.active_block = BLUE

