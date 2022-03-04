# Use the official image as a parent image
#FROM tensorflow/tensorflow:2.0.0-gpu-py3
FROM nvidia/opengl:1.2-glvnd-devel-ubuntu18.04

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
    curl

ENV CONDALINK="https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh"
RUN curl -so "$ROOT/miniconda.sh" $CONDALINK \
 && chmod +x "$ROOT/miniconda.sh" \
 && "$ROOT/miniconda.sh" -b -p "$ROOT/.conda/" \
 && rm "$ROOT/miniconda.sh"
ENV PATH="$ROOT/.conda/bin:$PATH"

ADD ./env.yml "$ROOT/env.yml"

RUN conda install -y conda-build \
 && conda env create -f "$ROOT/env.yml" \
 && conda clean -ya && conda init
ENV CONDA_DEFAULT_ENV=environ
ENV CONDA_PREFIX="$ROOT/.conda/envs/$CONDA_DEFAULT_ENV"
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Run the command inside your image filesystem
RUN pip3 install --upgrade pip && \
    pip3 install \ 
    gym==0.21.0 \
    PyOpenGL==3.1.5 \
    tqdm==4.62.3 \ 
    PILLOW==9.0.1 \
    opencv-python \
    pyglet==1.5.21 \
    numba \
    scipy==1.7.3 \
    numpy==1.21.5
ENV PYOPENGL_PLATFORM=egl
ENV MUJOCO_GL=egl