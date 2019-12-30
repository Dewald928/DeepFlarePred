# Our base image
FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

WORKDIR /home/fuzzy/work/DeepFlarePred
ENV CUDA_HOME=/usr/local/cuda

# Some common environmenta variables that Python uses
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install lower level dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
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
RUN pip install pandas && \
    pip install numpy && \
    pip install scipy && \
    pip install scikit-learn && \
    pip install argparse && \
    pip install wandb && \
    pip install matplotlib && \
    pip install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install captum && \
    pip install -U skorch

ENV WANDB_API_KEY=07796e7d3a148a6feceafdfef8c37e21a3f4e7c2



