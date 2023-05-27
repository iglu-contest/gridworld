import os
import math

import numpy as np
from tqdm import tqdm

from .render import Renderer, setup
from .core import World, Agent


class Visualizer:
    def __init__(self, render_size=(64, 64)) -> None:
        self.agent = Agent(sustain=False)
        self.agent.flying = True
        self.world = World()
        self.render_size = render_size
        self.renderer = Renderer(self.world, self.agent,
            width=self.render_size[0], height=self.render_size[1],
            caption='Pyglet', resizable=False)
        self.renderer.overlay = False
        setup()

    def set_agent_state(self, position=None, rotation=None):
        """
        Specifies new position and rotation for the agent.

        Args:
            position:
            rotation: 
        """
        self.agent.position = position
        self.agent.rotation = rotation

    def set_world_state(self, blocks, add=True):
        """
        Adds or removes blocks to/from the world.

        Args:
            blocks: list of blocks to add/remove
            add: whether to add or remove blocks
        """
        if add:
            for x,y,z, bid in blocks:
                self.world.add_block((x, y, z), bid)
        else:
            for x,y,z, bid in blocks:
                self.world.remove_block((x, y, z))
    
    def clear(self):
        for x,y,z in list(self.world.placed):
            self.world.remove_block((x,y,z))

    def render(self, position=None, rotation=None, blocks=None):
        """
        Args:
            position (list of len 3): optional, x,y,z position of agent to make a picture from
            rotation (list of len 2): optional, yaw,pitch rotation of agent to make a picture from
            blocks (list of tuples (x,y,z,block_id)): optional, list of blocks to initialize the world
        """
        if position is None:
            position = self.agent.position
        if rotation is None:
            rotation = self.agent.rotation
        self.agent.position = position
        self.agent.rotation = rotation
        if blocks is not None:
            for block in set(self.world.placed):
                self.world.remove_block(block)
            for x,y,z,bid in blocks:
                self.world.add_block((x, y - 1, z), bid)
        return self.renderer.render()[..., :-1]

    def render_video(self, output, 
            event_sequence, init_conds=None,
            overlay=False, render_debug=False, close=True,
            verbose=False, fps=60
        ):
        import cv2
        if init_conds is not None:
            self.set_agent_state(*init_conds)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output.parent.mkdir(exist_ok=True)
        image = self.render()
        writer = cv2.VideoWriter(f'{output}.mp4', fourcc, fps, image.shape[:2])
        writer.write(image[..., ::-1])
        tq = tqdm(total=len(event_sequence), disable=not verbose)
        for _, event in enumerate(event_sequence):
            if event.kind != 'action':
                self.set_agent_state(position=event.position.tolist(), 
                                     rotation=event.camera.tolist())
                if len(self.world.placed) != len(event.grid):
                    # for now just remove everything and build again
                    self.set_world_state(event.grid)
                image = self.render()
                writer.write(image[..., ::-1])
            tq.update(1)
        tq.close()
        if close:
            writer.release()
        else:
            return writer

    def postproc_video(self, output):
        code = os.system(f"ffmpeg -hide_banner -loglevel error -i {output}.mp4 -vcodec libx264 {output}2.mp4")
        if code == 0:
            os.system(f"mv {output}2.mp4 {output}.mp4")
        else:
            raise ValueError('Install the latest version of ffmpeg')
    
