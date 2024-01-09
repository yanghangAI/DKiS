# DKiS: Decay weight invertible image steganography with private key
This repo is the official code for

* **DKiS: Decay weight invertible image steganography with private key**



 
## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux)).
- [PyTorch = 1.11.0](https://pytorch.org/) .
- See requirements.txt for other dependencies.


## Get Started
- Run `python main.py` for training.


## Dataset
- Before you start your training, please make sure the dataset path is correct.

- Check the dataset path in `config.py`:

    `DIV_TRAIN_PATH = '' ` 

    `DIV_VAL_PATH = '' `

    `pub_TRAIN_PATH = '' `

    `pub_VAL_PATH = '' `


## Demo
- Here we provide a [Demo](http://47.94.105.69/hidekey/).

- You can hide a secret image into a host image with private key in our Demo by clicking your mouse.

