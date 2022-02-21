# Use the official image as a parent image
#FROM tensorflow/tensorflow:2.0.0-gpu-py3
FROM nvidia/opengl:1.2-glvnd-devel-ubuntu18.04

ENV MJLIB_PATH=/tf/mujoco200_linux/bin/libmujoco200.so
ENV MJKEY_PATH=/tf/mujoco200_linux/bin/mjkey.txt
ENV LD_LIBRARY_PATH=/tf/mujoco200_linux/bin/

# i need this for ffmpeg
ENV PATH="/tf/.local/bin/:${PATH}"

RUN apt-cache search mesa
RUN apt-get update && apt-get install -y libgl1-mesa-glx libosmesa6 freeglut3-dev python3 python3-pip libfreetype6

# Run the command inside your image filesystem
RUN pip3 install --upgrade pip && \
    pip3 install gym==0.18.3 PyOpenGL==3.1.5 tqdm PILLOW opencv-python pyglet numba
ENV PYOPENGL_PLATFORM=egl
ENV MUJOCO_GL=egl
