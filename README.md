# Laplace-HDC

This project is corresponding to the paper [Laplace-HDC: Understanding the geometry of binary hyperdimensional computing](https://doi.org/10.48550/arXiv.2404.10759).

A basic plain example is provided in the file "*basic_example.ipynb*"

## Instructions
####Prerequisite Python Libraries:
* torch
* torchvision
* numpy
* tqdm

####Encoder Modes:

The default encoder mode in the example is simple 1-dimensional cyclic shift (`shift_1d`). Other options:
* `shift_2d`
* `block_diag_shift_1d`
* `block_diag_shift_2d`

####Classifier Modes:

The default classifier in the example is Binary Stochastic Gradient Descent (`binary_sgd`). Other options:
* `float_sgd`
* `binary_majority`
* `float_majority`

####Corresponding Authors:
* Saeid Pourmand*
* Wyatt D. Whiting*
* Alireza Aghasi
* Nicholas F. Marshall

*: Equal Contribution
