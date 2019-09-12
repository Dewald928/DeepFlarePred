# DeepFlarePred
## Predicting the likelihood that a flare occur
By using the [Liu dataset](https://github.com/JasonTLWang/LSTM-flare-prediction), that contains SHARP parameters
(not images), we train a DNN to predict the likelihood of a flare erupting from a sunspot.

Only the ```main.py``` script is functional, the rest is still broken

## Plans for the Neural Network
### Preliminary tests
* [x] Copy Liu's code to pytorch somewhat.
* [ ] Copy Liu architecture completely
* [ ] Cross-validation
* [x] Regularization: L2 + Dropout
* [ ] Shuffled vs. Unshuffled
* [x] GPU integration
* [ ] GPU optimization
* [x] Implement Weight and Biases
* [x] W&B sweeps check if it work
* [x] W&B multiple gpu sweep. [How to](https://www.wandb.com/articles/multi-gpu-sweeps) by using ```tmux```
* [x] Pytorch bottleneck test: ***Inconclusive, revisit***
* [ ] [Attention models](https://medium.com/intel-student-ambassadors/implementing-attention-models-in-pytorch-f947034b3e66)

### Main Objectives
* [ ] Create MLP that is equivalent to Nishzuka et al paper
* [ ] Establish a baseline MLP
* [ ] Find best features


### Future plans
* [ ] Incorporate SHARP magnetogram images like Chen article
* [ ] Use GAN for detecting anomalies