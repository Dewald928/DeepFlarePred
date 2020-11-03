# DeepFlarePred
## Predicting the likelihood that a flare occur
By using the [Liu dataset](https://github.com/JasonTLWang/LSTM-flare-prediction), that contains SHARP parameters
(not images), we train a DNN to predict the likelihood of a flare erupting from a sunspot.

## Installation
### Conda install
```wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh```

```bash Miniconda3-latest-Linux-x86_64.sh```

```echo ". /home/<user>/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc```

```conda init```
### Git clone
```git clone https://github.com/Dewald928/DeepFlarePred.git```
### Create virtual environment
Install conda and install environment:

```conda env create -f environment.yml```

Activate environment:

```conda activate DeepFlarePred```

If you have errors with the predefined validation set do:

```pip install git+https://github.com/skorch-dev/skorch.git```
```pip install git+https://github.com/skorch-dev/skorch.git@6508d88c4044bcdb2d944081d437c706e346354a```

### Download and prepare data
The Liu data should be downloaded from [Liu dataset](https://github.com/JasonTLWang/LSTM-flare-prediction).
For the ```main_LSTM_Liu.py``` and ```main_TCN_Liu.py``` the Liu dataset needs to be downloaded and extracted to the
 "/Data/Liu" folder such that:
 ```
./Data/
├── Liu
│   ├── C
│   ├── M
│   └── M5

 ```
To fix the incorrect column names used, run the following (Not needed if
 repo is cloned, dataset already the fixed version):
```python liu_data_fix```

To create the normalized dataset run:
* for z_train: ```python normalize_z_train.py```
* for z_minmax_train: ```python normalize_z_minmax_train.py```
* fot z_minmax_all: ```cp Data/Liu/M5/ Data/Liu/z_minmax_all -rf```

To create the power transformed dataset run:
```python normality_check.py```

### Run script
To run the script, you can either do a Weight & Biases sweep, or just simply run:

```python main_TCN_Liu.py```

The default configurations can be changed in ```config-defaults.yaml```.



## Docker cloud gpu setup (not working atm)
### Build container
Follow instructions [here](https://docs.paperspace.com/gradient/notebooks/notebook-containers/building-a-custom-container)
* Install docker
* Install GPU drivers 
* Install nvidia docker [Quickstart](https://github.com/NVIDIA/nvidia-docker)
* Then build dockerfile and push image to hub (useful
 [Link1](https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile)
 [Link2](https://github.com/anibali/docker-pytorch/blob/master/cuda-10.0/Dockerfile))

### Dockerhub container
`dvd928/deep_flare_pred:latest`

### Example of runstring on paperspace
`paperspace jobs create --container dvd928/deep_flare_pred:latest --machineType
 P4000 --command wandb agent 5ks0xbql --ports 5000:5000 --project Liu_pytorch`
 

## Test/Analysis Scripts
| Script | Description |
|---|---|
|```cme_svm_updated_for_pyastro.ipynb``` | Example notebook of Bobra's CME SVM
|```data_aquisition_pipeline.ipynb```| Notebook for generating Liu et al. data (WIP)
|`feature_selection.py`| For Univariate Feature selection and RFE
|`inspectdata.py`| Basic data analysis & Pair plot generation
|`nested_crossval.py`| Example script for nested crossval
|`plot_classifier_comparison.py`| Sklearn script
|`plot_cv_indices.py`| Sklearn script
|`regression.py`| Synthetic LSTM regression testing
|`roc_test.py`| ROC vs. PR for imbalanced dataset
|`skorchCV.py`| Used for generating [Toy Unbalanced Classification](https://app.wandb.ai/dewald123/skorchcv/reports/Toy-Unbalanced-Classification--Vmlldzo3NTkxMA)
|`test_tcn.py`| For analysis of TCN and 1D convolution using sequences
|`Titanic_Basic_Interpret.py`| Captum Example
|`moving_std_protocol.py`| Protocol for downloading wandb runs and model selection, based on smooth training
|`WNBtestscript.py`| wandb setup script
|`workers_test.py`| Pytorch optimal workers test





## Plans for the Project
### Preliminary tests
* [x] Copy Liu's code to pytorch somewhat.
* [ ] Copy Liu architecture completely
* [x] Cross-validation: [Skorch library](https://skorch.readthedocs.io/en/stable/user/dataset.html)
* [x] Regularization: L2 + Dropout
* [X] Shuffled vs. Unshuffled. Shuffling is not very advers.
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
* [x] Establish a baseline MLP
* [x] Understand TCN operation
* [ ] Synthetic dataset
* [x] Change sequence length with TCN
* [x] LSTM vs. TCN?
* [ ] TCN baseline (Liu dataset (20/40 features?))
* [x] ROC + Precision Recall curves, with AUC (train, val & test set)
* [ ] Find best features out of the 40. ([captum](https://captum.ai/))
* [ ] Occlusions method compare.
* [ ] What do these best features mean? (fits with other literature?)
* [ ] SHARP only TCN
* [ ] Case studies
* [ ] What does TSS mean in this context?
* [x] How to interpret W&B parameters?



### Future plans
* [ ] SHARP query and infer model pipeline (un-normalize data)
* [ ] Incorporate SHARP magnetogram images like Chen article
* [ ] Use GAN for detecting anomalies
* [ ] MLP/LSTM attention models
* [x] See a regression problem? LSTM regression is possible.

### Questions


### Data sets
* [Chen et al. 2019 data (some of it)](https://deepblue.lib.umich.edu/data/concern/data_sets/0r967377q?locale=en)
* [Liu dataset](https://github.com/JasonTLWang/LSTM-flare-prediction)



