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
        self.renderer.overlay = True
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
            for bid,x,y,z in blocks:
                self.world.add_block((x, y, z), bid)
        else:
            for bid,x,y,z in blocks:
                self.world.remove_block((x, y, z))

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
            event_sequence, init_conds=None, on_env_step=None,
            render_text=False, text=None, render_crosshair=False, 
            render_debug=False, reset=True, close=True, writer=None,
            verbose=False, fps=60
        ):
        import cv2
        # TODO: set current task for the env
        if init_conds is not None:
            self.set_agent_state(*init_conds)
        done = False
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output.parent.mkdir(exist_ok=True)
            writer = cv2.VideoWriter(f'{output}.mp4', fourcc, fps, self.render_size)
            # if reset:
                # image = self.postprocess_frame(
                #     obs['pov'][:], render_text=render_text, render_crosshair=render_crosshair,
                #     render_debug=render_debug, text=text,
                #     g=None, n=None, last_action='None'
                # )
                # if callable(on_env_step):
                #     on_env_step(None, -1, 0, output, obs, 0, done, None)
        image = self.render()
        writer.write(np.transpose(image, (0, 1, 2))[..., ::-1])
        tq = tqdm(total=len(event_sequence), disable=not verbose)
        last_key_action = 'None'
        for i, event in enumerate(event_sequence):
            # if event.kind == 'action':
            #     last_key_action = event.params[0]
            # for k, action in enumerate(event.actions):
                # obs, reward, done, info = self.env.step(action)
                # image = self.postprocess_frame(
                #     obs['pov'][:], render_text=render_text, render_crosshair=render_crosshair,
                #     render_debug=render_debug, text=text,
                #     g=event.turn, n=event.step, last_action=last_key_action
                # )
                # if callable(on_env_step):
                #     on_env_step(event, i, k, output, obs, reward, done, info)
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
            # self.env.close()
            # self.postproc_video(output)
        else:
            return writer

    def postproc_video(self, output):
        code = os.system(f"ffmpeg -hide_banner -loglevel error -i {output}.mp4 -vcodec libx264 {output}2.mp4")
        if code == 0:
            os.system(f"mv {output}2.mp4 {output}.mp4")
        else:
            raise ValueError('Install the latest version of ffmpeg')
    