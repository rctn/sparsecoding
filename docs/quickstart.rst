==========
Quickstart
==========

Overview
--------

.. _sparsecoding: https://github.com/rctn/sparsecoding/

`sparsecoding`_ is a Python package that provides tools for implementing sparse coding algorithms. 
Traditionally, sparse coding has been primarily used for learning sparse representations of images. 
To support this, we include tools for visualization and data transformation specific to image data. 
However, we have designed the dictionary learning and inference methods to be data-agnostic, 
allowing for broader applications.

The `sparsecoding`_ library is built largely on PyTorch, enabling it to inherit several 
performance benefits, such as:

- GPU support
- Batched operations
- Auto-grad optimizers

Structure of the Library
-------------------------

The functionalities of `sparsecoding`_ are organized into several modules:

- ``sparsecoding.models``: Contains dictionary learning models (e.g., SparseCoding).
- ``sparsecoding.inference``: Includes algorithms for computing latent coefficients.
- ``sparsecoding.visualization``: Provides tools for visualizing image dictionaries and data.
- ``sparsecoding.priors``: Offers methods for sampling from various sparse coding priors.
- ``sparsecoding.datasets``: Contains utilities for loading datasets.
- ``sparsecoding.transforms``: Includes methods for working with data, such as whitening and 
  extracting patches from images.

Getting Started
---------------

Explore our `example notebooks <https://github.com/rctn/sparsecoding/tree/main/examples>`_ 
to get started.
