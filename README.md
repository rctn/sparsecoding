# RCTN SparseCoding Library

`sparsecoding` is a Python library developed by UC Berkeley's [Redwood Center for Theoretical Neuroscience (RCTN)](https://redwood.berkeley.edu). It provides efficient, batched, and GPU-compatible [PyTorch](https://github.com/pytorch/pytorch) implementations for sparse coding related-algorithms, including dictionary learning, inference, and data processing.

Historically, sparse coding has been largely focused on learning sparse representations of images and we provide visualization and transformation tools to work with such data. However, we’ve tried to structure the transformation, dictionary learning methods, and inference methods in a manner that is data-agnostic, making them applicable to a wide range of use cases.

We believe that sharing code within the scientific community is an important part of science and we hope that the research community finds this library useful. 


## Features

- Check out our [Quickstart Guide](https://sparsecoding.readthedocs.io/en/latest/quickstart.html) for an overview and setup instructions.
- Refer to the [API Reference](https://sparsecoding.readthedocs.io/en/latest/api.html) for detailed usage of the library's features.


## Setup

To install the library, follow these steps:

```bash
git clone https://github.com/rctn/sparsecoding.git
cd sparsecoding
pip install -e ".[all]"
```

Try running the demo notebook: `examples/sparse_coding.ipynb`.

For more detailed instructions, see our [Installation Guide](https://sparsecoding.readthedocs.io/en/latest/install.html).

Note: If you're using a Jupyter notebook and make changes to the source files, you can either:
* Restart the Jupyter kernel, or
* Use the autoreload extension as explained [here](https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html#autoreload).


# Contributing
We welcome contributions! Please see our [contributing](https://sparsecoding.readthedocs.io/en/latest/contributing.html) for details on how to get involved.
