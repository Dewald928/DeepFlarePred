# Our base image
# docker run -it --gpus all dvd928/deep_flare_pred:1

FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
FROM continuumio/miniconda3

ENV CUDA_HOME=/usr/local/cuda
ENV WANDB_API_KEY=07796e7d3a148a6feceafdfef8c37e21a3f4e7c2

# Some common environmenta variables that Python uses
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH=$PATH:/opt/conda/bin/
ENV USER FuZzy

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

# create environment
COPY archive/deepflarepred.yml .
RUN conda env create -f deepflarepred.yml

WORKDIR /DeepFlarePred
# Activate Source
CMD source activate DeepFlarePred
CMD source ~/.bashrc

RUN chmod -R a+w /DeepFlarePred
WORKDIR /DeepFlarePred

# Clone repo
RUN git clone https://github.com/Dewald928/DeepFlarePred.git
COPY run.sh /run.sh

CMD ["/run.sh"]






