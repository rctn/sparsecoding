==========
Quickstart
==========

Overview
--------

.. _sparsecoding: https://github.com/rctn/sparsecoding/

`sparsecoding`_ is a python package that provides tools for using sparse coding algorithms. 
Historically, sparse coding has been largely focused on learning sparse representations 
of images and we provide visualization and transformation tools to work with such data.
However, we've tried to structure the dictionary learning methods and inference methods 
in a manner that is data-agnostic.

The `sparsecoding`_ library is largely built using PyTorch which enables it to inheret 
many performance benifits. These include:

- GPU support

- Batched operations

- Auto-grad optimizers



Structure of library
--------------------

The functionalities of `sparsecoding`_ are broken into differnent modules.

- ``sparsecoding.models`` contains dictionary learning models (e.g. SparseCoding).

- ``sparsecoding.inference`` contains algorithms for computing latent coefficients.

- ``sparsecoding.visualization`` contains tools for visualizing image dictionaries and data.

- ``sparsecoding.priors`` contains methods for sampling from various sparse coding priors.

- ``sparsecoding.datasets`` contains methods for loading datasets.

- ``sparsecoding.transforms`` contains methods working with data, such as whitening and extracting patches from images.


Getting started
---------------

`See example notebooks <https://github.com/rctn/sparsecoding/tree/main/examples>`_.