# Sparse Coding

Reference sparse coding implementations for efficient learning and inference implemented in PyTorch with GPU support.

## Features

### Dictionary Learning

* Repo currently includes classic patch-wise sparse coding dictionary learning.

### Implemented Inference Methods

* Locally Competative Algorithm (LCA)
* Gradient Descent with Euler's method on Laplace Prior (Vanilla)
* Laplacian Scale Mixture (LSM)
* Iterative Shrinkage-threshold Algorithm (ISTA)
* Generic PyTorch minimization of arbitrary loss function (PyTorchOptimizer)

## Setup

1. Clone the repo.
2. Navigate to the directory containing the repo directory.
3. Run `pip install -e ".[all]"`
4. Install the natural images dataset from this link: https://rctn.org/bruno/sparsenet/IMAGES.mat
5. Try running the demo notebook: `examples/sparse_coding.ipynb`

Note: If you are using a Jupyter notebook and change a source file, you can either: 1) restart the Jupyter kernel, or 2) follow instructions [here](https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html#autoreload).

## Contributing

See the [contributing](docs/contributing.md) document!
