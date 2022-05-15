# Sparsecoding
Reference sparse coding implementations for efficient learning and inference.

## Dictionary Learning
* Classic sparse coding dictionary learning implemented, formulated as $\mathbf{x} = \mathbf{\Phi a} + \mathbf{n}$ where inner loop of coefficient inference occurs for $\mathbf{a}$, with outer loop of dictionary updates on $\mathbf{\Phi}$.

## Implemented inference methods
* Locally Competative Algorithm (LCA)
* Gradient Descent with Euler's method on Laplace Prior (Vanilla)
* Laplacian Scale Mixture (LSM)
* Iterative Shrinkage-threshold Algorithm (ISTA)
* Genertic PyTorch minimization of arbitraty loss function (PyTorchOptimizer)

# Setup
1. Clone the repo.
2. Navigate to the directory containing the repo directory.
3. Run `pip install -e sparsecoding`
4. Navigate into the repo and install the requirements using `pip install -r requirements.txt`
5. Install the natural images dataset from this link: https://rctn.org/bruno/sparsenet/IMAGES.mat
6. Try running the demo notebook: `examples/sparse_coding.ipynb`

# Contributing
See the [contributing](https://github.com/rctn/sparsecoding/blob/formatting/docs/contributing.md) document!