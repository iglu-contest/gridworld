FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get install -y --no-install-recommends \
    libxau6 libxau6:i386 \
    libxdmcp6 libxdmcp6:i386 \
    libxcb1 libxcb1:i386 \
    libxext6 libxext6:i386 \
    libx11-6 libx11-6:i386 \
    libfreetype6:i386
    # rm -rf /var/lib/apt/lists/* # buildkit
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \     
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf # buildkit
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglvnd0 libglvnd0:i386 \
    libgl1 libgl1:i386 \
    libglx0 libglx0:i386 \
    libegl1 libegl1:i386 \
    libgles2 libgles2:i386 
    # rm -rf /var/lib/apt/lists/* # buildkit
ADD ./docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd-dev libglvnd-dev:i386 \
    libgl1-mesa-dev libgl1-mesa-dev:i386 \
    libegl1-mesa-dev libegl1-mesa-dev:i386 \
    libgles2-mesa-dev libgles2-mesa-dev:i386
# i need this for ffmpeg
# ENV PATH="/tf/.local/bin/:${PATH}"

ENV ROOT="/root"

RUN apt-get update && apt-get install -y \ 
    libgl1-mesa-glx \
    libosmesa6 \
    freeglut3-dev \
    python3 \
    python3-pip \
    libfreetype6 \
    curl \
    git \
    vim

# Run the command inside your image filesystem
RUN pip3 install --upgrade pip && \
    pip3 install \ 
    gym==0.21.0 \
    PyOpenGL==3.1.5 \
    tqdm==4.62.3 \ 
    PILLOW==9.0.1 \
    opencv-python==4.5.5.64 \
    pyglet==1.5.21 \
    numba==0.55.2 \
    scipy==1.7.3 \
    numpy==1.21.5 \
    pandas==1.3.5 \
    filelock==3.7.0 \
    requests==2.27.1
ENV PYOPENGL_PLATFORM=egl
ENV MUJOCO_GL=egl
ADD . /gridworld
RUN cd /gridworld && python setup.py install && cd / && rm -rf gridworld
