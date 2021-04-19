# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
# NCCL 2.7.8

USER root:root

##############################################################################
# Environment variables
##############################################################################

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH "/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

ENV STAGE_DIR=/root/gpu/install \
    CUDA_HOME=/usr/local/cuda \
    CUDNN_HOME=/usr/lib/x86_64-linux-gnu \
    CUDACXX=/usr/local/cuda/bin/nvcc

RUN mkdir -p $STAGE_DIR

RUN apt-get -y update && \
    apt-get --assume-yes --no-install-recommends install \
    build-essential \
    autotools-dev \
    curl \
    wget \
    openssh-server \
    openssh-client \
    tmux \
    vim \
    sudo \
    g++ \
    gcc \
    git \
    bc \
    tar \
    bash \
    pbzip2 \
    pv bzip2 \
    cabextract \
    dos2unix \
    less \
    unzip \
    pdsh \
    pssh \
    nfs-common \
    libfuse-dev \
    htop iftop iotop rsync iputils-ping \
    net-tools && \
    rm -rf /var/lib/apt/lists/*

# # https://github.com/lhelontra/tensorflow-on-arm/issues/13
# RUN apt-get -y upgrade libstdc++6

# Labels for the docker
LABEL description="This docker sets up the environment to run Turing with ORT Training"

##############################################################################
# ORT requirement
# Conda Environment
##############################################################################
ARG PYTHON_INSTALL_VERSION=3.7.7

ENV MINICONDA_VERSION ${MINICONDA_VERSION}
ENV PATH /opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh    && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

##############################################################################
# Generic ENV
##############################################################################
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/usr/local/lib:/usr/lib/x86_64-linux-gnu"
ENV STAGE_DIR "/root/gpu/install"
RUN mkdir -p $STAGE_DIR

##############################################################################
# Java Development Kit
##############################################################################
ENV JDK_VERSION=8u121
RUN mkdir -p /usr/jdk && \
    cd /usr/jdk && \
    wget -q -O - https://indexserve.blob.core.windows.net/java/jdk-${JDK_VERSION}-linux-x64.tar.gz | tar xzf -
ENV JAVA_HOME=/usr/jdk/jdk1.8.0_121
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${JAVA_HOME}/jre/lib/amd64/server
ENV PATH=${PATH}:${JAVA_HOME}/bin

##############################################################################
# Hadoop
##############################################################################
ENV HADOOP_VERSION=2.9.2
RUN cd /usr/local && \
    wget -q -O - https://indexserve.blob.core.windows.net/hadoop/hadoop.tar.gz | tar xzf - && \
    find /usr/local/hadoop -type f -print0 | xargs -0 dos2unix -q && \
    chmod -R +x /usr/local/hadoop/bin
ENV HADOOP_HOME=/usr/local/hadoop
ENV PATH=${PATH}:${HADOOP_HOME}/bin

#############################################################################
# IB user space libs  Mellanox driver should be installed on kernel space already
#############################################################################

RUN apt-get update && apt-get install -y --no-install-recommends  libnuma-dev  libmlx4-1      libmlx5-1      librdmacm1     libmthca1      libdapl2      dapl2-utils      openssh-client      openssh-server      iproute2 && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends cpio libmlx4-1 libmlx5-1 librdmacm1 libmthca1 libdapl2 dapl2-utils pciutils ibutils ibverbs-utils rdmacm-utils infiniband-diags perftest librdmacm-dev && rm -rf /var/lib/apt/lists/*


##############################################################################
# nv_peer_mem
##############################################################################
RUN apt-get -y update && apt-get install -y --no-install-recommends debhelper dkms && rm -rf /var/lib/apt/lists/*
RUN mkdir -p ${STAGE_DIR} && \
    git clone https://github.com/Mellanox/nv_peer_memory.git ${STAGE_DIR}/nv_peer_memory && \
    cd ${STAGE_DIR}/nv_peer_memory && \
    git checkout 4ed7715d62edf1cbcbb522a9f9a0efb13e43b0d0 && \
    ./build_module.sh && \
    cd /tmp && \
    tar xzf /tmp/nvidia-peer-memory_1.1.orig.tar.gz && \
    cd nvidia-peer-memory-1.1 && \
    dpkg-buildpackage -us -uc && \
    dpkg -i ../nvidia-peer-memory_1.1-0_all.deb
WORKDIR $STAGE_DIR

##############################################################################
# OPENMPI
##############################################################################
ENV OPENMPI_BASEVERSION=4.1
ENV OPENMPI_VERSION_STRING=${OPENMPI_BASEVERSION}.0
RUN cd ${STAGE_DIR} && \
    wget -q -O - https://download.open-mpi.org/release/open-mpi/v${OPENMPI_BASEVERSION}/openmpi-${OPENMPI_VERSION_STRING}.tar.gz | tar xzf - && \
    cd openmpi-${OPENMPI_VERSION_STRING} && \
    ./configure  --enable-orterun-prefix-by-default && \
    make uninstall && \
    make -j"$(nproc)" install && \
    # Sanity check:
    test -f /usr/local/bin/mpic++ && \
    ldconfig && \
    cd ${STAGE_DIR} && \
    rm -r ${STAGE_DIR}/openmpi-${OPENMPI_VERSION_STRING}
ENV PATH=/usr/local/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}


ENV CMAKE_VERSION=3.16.4
RUN cd /usr/local && \
    wget -q -O - https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz | tar zxf -
ENV PATH=/usr/local/cmake-${CMAKE_VERSION}-Linux-x86_64/bin:${PATH}

WORKDIR /workspace

##############################################################################
# Some Packages
##############################################################################
RUN apt-get -y update && apt-get -y install --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && apt-get install -y --allow-change-held-packages --allow-downgrades --no-install-recommends \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libsndfile-dev \
    libcupti-dev \
    libjpeg-dev \
    libpng-dev \
    screen \
    libxml2-dev \
    libxslt-dev &&\
    rm -rf /var/lib/apt/lists/*


##############################################################################
# Install Deep Learning packges: tensorflow 1.4
##############################################################################
RUN conda install -y python=$PYTHON_INSTALL_VERSION pyyaml scipy ipython cython typing mkl mkl-include setuptools
RUN conda install -y -c conda-forge ruamel.yaml

RUN echo /usr/lib/x86_64-linux-gnu >>/etc/ld.so.conf.d/cuda-10-2.conf
RUN ldconfig

##############################################################################
# update pip
##############################################################################

RUN python -m pip install --upgrade pip

##############################################################################
# PyTorch
##############################################################################

RUN pip install --no-cache-dir torch torchvision torchaudio

##############################################################################
# Tensorflow tensorboard tensorboardX
##############################################################################

RUN pip install --no-cache-dir tensorflow tensorboard tensorboardX

##############################################################################
# install horovod
##############################################################################

# RUN HOROVOD_GPU_OPERATIONS=NCCL pip install --no-cache-dir horovod

################################################################################################
# Python Packages including AML-defaults, dataprep, transformers, onnx
################################################################################################
RUN pip install --no-cache-dir --use-feature=2020-resolver \
    psutil \
    yappi \
    cffi \
    ipdb \
    pandas \
    matplotlib \
    py3nvml \
    pyarrow \
    graphviz \
    astor \
    boto3 \
    tqdm \
    ipdb \
    sentencepiece \
    msgpack \
    requests \
    pandas \
    sphinx \
    sphinx_rtd_theme \
    scipy \
    numpy \
    GPUtil \
    sklearn \
    scikit-learn \
    nvidia-ml-py3 \
    h5py \
    py3nvml \
    ipdb \
    mpi4py \
    onnx \
    # transformers \
    cerberus \
    azureml-defaults==1.20.0 \
    azureml-telemetry==1.20.0 \
    azureml-dataprep==2.7.3 \
    azureml-train==1.20.0 \
    pytorch-lightning \
    sympy \
    ninja \
    datasets

# AGI required packages
RUN pip install --no-cache-dir nltk spacy==2.3.2 &&  \
    python -m nltk.downloader -d /usr/share/nltk_data punkt && \
    python -m spacy download en

##############################################################################
# packages for TNLRv3
##############################################################################

RUN pip install --no-cache-dir \
    six \
    path.py \
    lmdb \
    py-lz4framed \
    methodtools \
    langid \
    toolz

#######################################################
# Build onnxruntime training
#######################################################
#RUN git clone --depth 1 https://github.com/microsoft/onnxruntime.git &&\
#    cd onnxruntime &&\
#    git submodule update --init --recursive && \
#    python tools/ci_build/build.py \
#        --cmake_extra_defines \
#            ONNXRUNTIME_VERSION=`cat ./VERSION_NUMBER` \
#        --config Release \
#        --enable_training \
#        --use_cuda \
#        --cuda_home /usr/local/cuda \
#        --cudnn_home /usr/lib/x86_64-linux-gnu/ \
#        --update \
#        --parallel \
#        --build_dir build \
#        --build \
#        --build_wheel \
#        --skip_tests &&\
#    pip install build/Release/dist/*.whl &&\
#    cd .. &&\
#    rm -rf onnxruntime /opt/cmake

############################################
# Install DeepSpeed
############################################

RUN git clone https://github.com/microsoft/DeepSpeed.git /DeepSpeed
RUN cd /DeepSpeed && \
    git config pull.ff only && \
    git pull && \
    git checkout master && \
    pip install -v . && \
    ds_report

############################################
# Install apex
############################################
#
#RUN git clone https://github.com/NVIDIA/apex &&\
#    cd apex &&\
#    export TORCH_CUDA_ARCH_LIST="Volta;Turing;Kepler+Tesla" &&\
#    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

############################################
# Install transfomers from source
############################################

RUN git clone https://github.com/huggingface/transformers.git &&\
    cd transformers &&\
    pip install -e .

