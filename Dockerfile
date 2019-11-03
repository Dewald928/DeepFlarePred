FROM continuumio/miniconda3

WORKDIR /home/fuzzy/work/DeepFlarePred

COPY brood.yaml deepflarepred.yaml

RUN conda config --add channels conda-forge \
	&& conda env create -f deepflarepred.yaml
# Pull the environment name out of the environment.yml
#RUN conda init
#
#CMD exec bash && \
#    conda activate DeepFlarePred

RUN echo "source activate $(head -1 deepflarepred.yaml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH ~/local/miniconda3/envs/$(head -1 deepflarepred.yaml | cut -d' ' -f2)/bin:$PATH
#CMD source activate DeepFlarePred
#CMD source ~/.bashrc
#
#COPY init.sh /init.sh
#CMD ["/init.sh"]
