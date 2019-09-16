# DeepFlarePred
## Predicting the likelihood that a flare occur
By using the [Liu dataset](https://github.com/JasonTLWang/LSTM-flare-prediction), that contains SHARP parameters
(not images), we train a DNN to predict the likelihood of a flare erupting from a sunspot.

Only the ```main.py``` script is functional, the rest is still broken

## Plans for the Neural Network
### Preliminary tests
* [x] Copy Liu's code to pytorch somewhat.
* [ ] Copy Liu architecture completely
* [ ] Cross-validation: [Skorch library](https://skorch.readthedocs.io/en/stable/user/dataset.html)
* [x] Regularization: L2 + Dropout
* [ ] Shuffled vs. Unshuffled
* [x] GPU integration
* [ ] GPU optimization
* [x] Implement Weight and Biases
* [x] W&B sweeps check if it work
* [x] W&B multiple gpu sweep. [How to](https://www.wandb.com/articles/multi-gpu-sweeps) by using ```tmux```
* [x] Pytorch bottleneck test: ***Inconclusive, revisit***
* [x] [Attention models](https://medium.com/intel-student-ambassadors/implementing-attention-models-in-pytorch-f947034b3e66)
Tested, but not implemented on dataset
* [ ] Understand LSTM 
* [x] GRU, LSTM and RNN switchable between
* [x] hdf5 test script: Chen data uses hdf5, but unable to read the data
* [x] MLP skorch test: RNN and custom logs not well supported

### Main Objectives
* [ ] Create MLP that is equivalent to Nishzuka et al paper
* [ ] Establish a baseline MLP
* [ ] Find best features


### Future plans
* [ ] Incorporate SHARP magnetogram images like Chen article
* [ ] Use GAN for detecting anomalies
* [ ] MLP/LSTM attention models

### Data sets
* [Chen et al. 2019 data (some of it)](https://deepblue.lib.umich.edu/data/concern/data_sets/0r967377q?locale=en)