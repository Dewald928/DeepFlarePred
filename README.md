# DeepFlarePred
## Predicting the likelihood that a flare occur
By using the [Liu dataset](https://github.com/JasonTLWang/LSTM-flare-prediction), that contains SHARP parameters
(not images), we train a DNN to predict the likelihood of a flare erupting from a sunspot.

## Installation
### Conda install
```wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh```
```bash Miniconda3-latest-Linux-x86_64.sh```
### Git clone
```git clone ```
### Create virtual environment
Install conda and install environment:

```conda env create -f deepflarepred.yml```

Activate environment:

```conda activate DeepFlarePred```

### Download and prepare data
The Liu data should be downloaded from [Liu dataset](https://github.com/JasonTLWang/LSTM-flare-prediction).
For the ```main_LSTM_Liu.py``` and ```main_TCN_Liu.py``` the Liu dataset needs to be downloaded and extracted to the
 "/Data/Liu" folder such that:
 ```bash
./Data/
├── Liu
│   ├── C
│   ├── M
│   └── M5
 ```
### Run script
To run the script, you can either do a Weight & Biases sweep, or just simply run:

```python main_TCN_Liu.py```

The default args can be changed or passed inline e.g.

```python main_TCN_Liu.py --learning_rate 0.001```


## Docker cloud gpu setup
### Build container
Follow instructions [here](https://docs.paperspace.com/gradient/notebooks/notebook-containers/building-a-custom-container)
* Install docker
* Install GPU drivers 
* Install nvidia docker [Quickstart](https://github.com/NVIDIA/nvidia-docker)
* Then build dockerfile and push image to hub (useful
 [Link1](https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile)
 [Link2](https://github.com/anibali/docker-pytorch/blob/master/cuda-10.0/Dockerfile))

### Dockerhub container
`dvd928/deep_flare_pred:1`

### Example of runstring on paperspace
`paperspace jobs create --container dvd928/deep_flare_pred:1 --machineType
 P4000 --command wandb agent 5ks0xbql --ports 5000:5000 --project Liu_pytorch`
 


## Plans for the Project
### Preliminary tests
* [x] Copy Liu's code to pytorch somewhat.
* [ ] Copy Liu architecture completely
* [x] Cross-validation: [Skorch library](https://skorch.readthedocs.io/en/stable/user/dataset.html)
* [x] Regularization: L2 + Dropout
* [ ] Shuffled vs. Unshuffled
* [x] GPU integration
* [x] GPU optimization, just use larger batch sizes
* [x] Implement Weight and Biases
* [x] W&B sweeps check if it work
* [x] W&B multiple gpu sweep. [How to](https://www.wandb.com/articles/multi-gpu-sweeps) by using ```tmux```
* [x] Pytorch bottleneck test: ***Inconclusive, revisit***
* [x] [Attention models](https://medium.com/intel-student-ambassadors/implementing-attention-models-in-pytorch-f947034b3e66)
Tested, but not sweeped
* [x] Understand LSTM + TCN better
* [x] GRU, LSTM and RNN switchable between
* [x] hdf5 test script: Chen data uses hdf5, but unable to read the data
* [x] MLP skorch test: RNN and custom logs not well supported
* [x] [TCN networks](https://github.com/locuslab/TCN) : better so far, slightly
* [x] Early stopping and checkpointing on best validation TSS (LSTM only, so far)
* [ ] Test data? best wat to test network?
* [ ] LR scheduler

### Main Objectives
* [ ] ~~Create MLP that is equivalent to Nishzuka et al paper~~
* [ ] ~~Establish a baseline MLP~~
* [x] LSTM vs. TCN?
* [ ] TCN baseline (Liu dataset (20/40 features?))
* [x] ROC + Precision Recall curves, with AUC (train, val & test set)
* [ ] Find best features out of the 40. ([captum](https://captum.ai/))
* [ ] What do these best features mean? (fits with other literature?)
* [ ] SHARP only TCN
* [ ] Case studies
* [ ] What does TSS mean in this context?
* [ ] How to interpret W&B parameters?



### Future plans
* [ ] SHARP query and infer model pipeline
* [ ] Incorporate SHARP magnetogram images like Chen article
* [ ] Use GAN for detecting anomalies
* [ ] MLP/LSTM attention models
* [ ] See a regression problem?

### Questions
* The sliding window, how to know how many hours ahead is predicted
* Best way to test?

### Data sets
* [Chen et al. 2019 data (some of it)](https://deepblue.lib.umich.edu/data/concern/data_sets/0r967377q?locale=en)
* [Liu dataset](https://github.com/JasonTLWang/LSTM-flare-prediction)



