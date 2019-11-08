# Our base image
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

WORKDIR /home/fuzzy/work/DeepFlarePred

# Some common environmenta variables that Python uses
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install lower level dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y curl python3.7 python3-pip && \
#    apt -y update && \
#    apt install -y git && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Install a specific version of TensorFlow
# You may also install anything else from pip like this
RUN pip install pandas && \
    pip install numpy && \
    pip install scipy && \
    pip install scikit-learn && \
    pip install argparse && \
    pip install wandb && \
    pip install matplotlib && \
    pip install torch torchvision && \
    pip install captum

ENV WANDB_API_KEY=07796e7d3a148a6feceafdfef8c37e21a3f4e7c2



