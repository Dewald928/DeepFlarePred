#FROM continuumio/miniconda3
#RUN conda create -n env python=3.6
#RUN echo "source activate env" > ~/.bashrc
#ENV PATH /opt/conda/envs/env/bin:$PATH

FROM continuumio/miniconda3
ADD environment.yml environment.yml
RUN conda env create -f environment.yml
RUN pip install git+https://github.com/skorch-dev/skorch.git
# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH