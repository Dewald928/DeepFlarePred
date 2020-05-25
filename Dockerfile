# Our base image
# docker run -it --gpus all dvd928/deep_flare_pred:1

FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
#FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

WORKDIR /home/fuzzy/work/DeepFlarePred
ENV CUDA_HOME=/usr/local/cuda

# Some common environmenta variables that Python uses
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install lower level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    python3-pip \
    libjpeg-dev \
    libpng-dev \
 && rm -rf /var/lib/apt/lists/*

#RUN apt-get update -y
#RUN apt-get install -y git curl ca-certificates bzip2 cmake tree htop bmon iotop sox libsox-dev libsox-fmt-all vim
#RUN apt-get install ocl-icd-opencl-dev
#RUN apt-get update --fix-missing && \
#    apt-get -y update && \
#    apt-get install -y python 3.7 && \
#    apt-get install -y python3-pip && \
##    apt -y update && \
##    apt install -y git && \
#    update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
#    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
#    apt-get clean && \
#    apt-get autoremove && \
#    rm -rf /var/lib/apt/lists/*

# Install a specific version of TensorFlow
# You may also install anything else from pip like this
RUN pip3 install pandas && \
    pip3 install numpy && \
    pip3 install scipy && \
    pip3 install scikit-learn && \
    pip3 install argparse && \
    pip3 install wandb && \
    pip3 install matplotlib && \
    pip3 install torch==1.3.1+cu100 torchvision==0.4.2+cu100 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install captum && \
    pip3 install -U skorch

ENV WANDB_API_KEY=07796e7d3a148a6feceafdfef8c37e21a3f4e7c2



