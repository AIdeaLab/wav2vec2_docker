FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ARG PYTHON_VERSION=3.8
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  git \
  curl \
  ca-certificates \
  libjpeg-dev \
  libpng-dev && \
  rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
  chmod +x ~/miniconda.sh && \
  ~/miniconda.sh -b -p /opt/conda && \
  rm ~/miniconda.sh && \
  /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
  /opt/conda/bin/conda install -y pytorch==1.6.0 torchvision torchaudio==0.6.0 cudatoolkit=10.2 -c pytorch
ENV PATH /opt/conda/bin:$PATH


WORKDIR /
RUN mkdir -p /opt/program
WORKDIR /opt/program

RUN mkdir libs
RUN pip install soundfile sentencepiece gdown
WORKDIR /opt/program/libs

RUN git clone https://github.com/pytorch/fairseq.git
WORKDIR fairseq
RUN git checkout c8a0659be5cdc15caa102d5bbf72b872567c4859
RUN pip install --editable ./


RUN apt-get update && apt-get upgrade -y && apt-get install -y && apt-get -y install apt-utils gcc libpq-dev libsndfile-dev
RUN apt install -y build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
RUN apt-get install -y libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev


WORKDIR /opt/program/libs
RUN git clone https://github.com/kpu/kenlm.git
WORKDIR /opt/program/libs/kenlm
RUN mkdir -p build
WORKDIR build
RUN cmake ..
RUN make -j 4

RUN pip install packaging
WORKDIR /opt/program/libs
RUN git clone -b v0.2 https://github.com/facebookresearch/wav2letter.git
WORKDIR /opt/program/libs/wav2letter/bindings/python
ENV KENLM_ROOT_DIR=/opt/program/libs/kenlm
RUN rm -rf build
RUN pip install -e .
RUN pip uninstall -y numpy
RUN pip install numpy
RUN pip install wandb editdistance gdown sklearn deepspeed awscli urllib3

RUN apt-get install -y tmux vim unzip wget

RUN conda install -c conda-forge jupyterlab

WORKDIR /opt/program/

ENV HYDRA_FULL_ERROR=1
ENV LANG=C.UTF-8
ENV LANGUAGE=en_US:
ENV WANDB_API_KEY=xxxx

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Set up the program in the image
COPY train /opt/program/
COPY copy_checkpoints.sh /opt/program/ 
COPY copy_checkpoints.sh /opt/program/
RUN mkdir -p config/pretraining
COPY wav2vec2_base_librispeech.yaml /opt/program/config/pretraining/
ENV PATH="/opt/program:${PATH}"
