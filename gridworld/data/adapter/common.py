from dataclasses import dataclass, field
import numpy as np
import logging

from gridworld.tasks.task import BUILD_ZONE_SIZE

RESOLUTION = (64, 64)

AIR_TYPE = 0
NORTH_YAW = -90.0
VOXELWORLD_GROUND_LEVEL = 63
DEFAULT_POS = DEFAULT_X, DEFAULT_Y, DEFAULT_Z, DEFAULT_PITCH, DEFAULT_YAW = \
    (0., 0., 0., 0., -180.)


class VWEvent:
    def __init__(
            self, kind=None, params=None, actions=None, grid=None, 
            camera=None, position=None, step=None, turn=None
        ):
        self.kind = kind
        self.params = params
        if actions is None:
            actions = []
        assert isinstance(actions, (list, tuple))
        self.actions = list(actions)
        if grid is None:
            grid = []
        assert isinstance(grid, (list, tuple))
        self.grid = list(grid) # grid in sparse format
        if camera is None:
            camera = np.array([0., 0.])
        if isinstance(camera, (list, tuple)):
            camera = np.array(camera)
        assert isinstance(camera, np.ndarray)
        self.camera = camera
        if position is None:
            position = np.array([0., 0., 0.])
        if isinstance(position, (list, tuple)):
            position = np.array(position)
        assert isinstance(position, np.ndarray)
        self.position = position
        self.step = step
        self.turn = turn

@dataclass
class GameSession:
    events: dict = field(default_factory=dict, repr=False)
    dialogs: list = field(default_factory=list, repr=False)
    init_conds: dict = field(default_factory=dict, repr=False)
    name: str = None
    target: np.ndarray = field(default_factory=lambda: np.zeros(BUILD_ZONE_SIZE, dtype=np.int), repr=False)

    def episode_steps(self):
        return sum([sum([len(ev.actions) for ev in event_step]) for event_step in self.events.values()])

    def episode_states(self):
        return sum([len(event_step) for event_step in self.events.values()])