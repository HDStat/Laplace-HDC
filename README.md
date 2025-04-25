# Laplace-HDC

This project corresponds to the published paper "Laplace-HDC: Understanding the geometry of binary hyperdimensional computing"
[arXiv.2404.10759](https://doi.org/10.48550/arXiv.2404.10759) In Journal of Artificial Intelligence Research.

```
@article{pourmand2024laplacehdc,
      title={Laplace-HDC: Understanding the geometry of binary hyperdimensional computing}, 
      author={Saeid Pourmand and Wyatt D. Whiting and Alireza Aghasi and Nicholas F. Marshall},
      journal={arXiv:2404.10759},
      year={2024},
}
```

A basic example is provided in the file "*basic_example.ipynb*"

## Instructions
**Prerequisite Python Libraries**:
* torch
* torchvision
* numpy
* tqdm

**Encoder Modes**:

The default encoder mode in the example is simple `shift_1d` (1-dimensional cyclic shift). Other options:
* `shift_2d` (2-dimensional cyclic shift)
* `block_diag_shift_1d` (1-dimensional block diagonal shift)
* `block_diag_shift_2d` (2-dimensional block diagonal shift)
* `rand_permutation` (Random Permutation)

**Classifier Modes**:

The default classifier in the example is Binary Stochastic Gradient Descent (`binary_sgd`). Other options:
* `float_sgd` (Float Stochastic Gradient Descent)
* `binary_majority` (Binary Majority Vote)
* `float_majority` (Float Majority Vote)

**Authors**:
* Saeid Pourmand*
* Wyatt D. Whiting*
* Alireza Aghasi
* Nicholas F. Marshall

*: Equal Contribution
