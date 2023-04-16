**Score-Based Generative Models on MNIST Dataset using Pytorch Lightning**


This project aims to implement score-based generative model paper(https://arxiv.org/pdf/2011.13456.pdf) on the MNIST dataset using the PyTorch Lightning framework. 


The implementation involves the following steps:

1. Implementing the variance-preserving SDE.
2. Using a simple U-Net for the Score Model.
3. Trained and evaluated the model based on the BPD metric
4. Wandb sweep functionality was used to run a grid of hyperparameters and results that were obtained are given in the results file.
5. Pytorch ligthning Checkpoint framework saves the model with the min BPD


**Requirements**

To run the code in this project, you need to have the following libraries installed:

1. PyTorch - 2.0.0+cu118
2. PyTorch Lightning - 2.0.1.post0
3. Wandb - 0.14.2
4. NumPy - 1.24.1

** For tuning hyper-parameters** 
W&B Sweeps can be defined in multiple ways:

1. with a YAML file - best for distributed sweeps and runs from command line
2. with a Python object - best for notebooks


In this project we use a YAML file. You can refer to W&B documentation(https://docs.wandb.com/library/integrations/lightning) for more Pytorch-Lightning examples.
```
wandb sweep config.yaml
```
```
wandb agent <sweep_id> where <sweep_id> is given by previous command
```

