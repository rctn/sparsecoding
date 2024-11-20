============
Contributing
============

All contributions are welcome!

Bug Reporting
-------------

If you find a bug, submit a bug report on GitHub Issues. 

Adding Features/Fixing Bugs
---------------------------

If you have identified a new feature or bug that you can fix yourself, please follow the following procedure.

#. Clone ``main`` branch.
#. Create a new branch to contain your changes. 
#. ``add``, ``commit``, and ``push`` your changes to this branch. 
#. Create a pull request (PR). See more information on submitting a PR request below.

Submitting a Pull Request
-------------------------

#. If necessary, please **write your own unit tests** and add them to `the tests directory <https://github.com/rctn/sparsecoding/blob/main/tests>`_. 
#. Verify that all tests are passed by running `python -m unittest discover tests -vvv`.
#. Be sure that your PR follows formatting guidelines, `PEP8 <https://peps.python.org/pep-0008/>`_ and `flake8 <https://flake8.pycqa.org/en/latest/>`_. 
#. Make sure the title of your PR summarizes the features/issues resolved in your branch. 
#. Submit your pull request and add reviewers. 

Coding Style Guidelines
-----------------------
We adhere to the `NumPy documentation standards <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

#. Format code in accordance with `flake8 <https://flake8.pycqa.org/en/latest/>`_ standard.
#. Use underscores to separate words in non-class names: ``n_samples`` rather than ``nsamples``.
#. Avoid single-character variable names.