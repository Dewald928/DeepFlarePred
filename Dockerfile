FROM continuumio/miniconda3

WORKDIR /home/fuzzy/work/DeepFlarePred

COPY brood.yaml deepflarepred.yaml

RUN conda config --add channels conda-forge \
	&& conda env create -f deepflarepred.yaml
# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 deepflarepred.yaml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH ~/local/miniconda3/envs/$(head -1 deepflarepred.yaml | cut -d' ' -f2)/bin:$PATH
#RUN conda init bash
#RUN echo ". ~/local/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
#RUN echo "conda activate $(head -1 deepflarepred.yaml | cut -d' ' -f2)" >> ~/.bashrc