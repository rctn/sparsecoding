============
Contributing
============

We welcome all contributions!

Bug Reporting
-------------

If you encounter a bug, please report it by creating an issue on GitHub.

Adding Features or Fixing Bugs
------------------------------

If you’ve identified a new feature to add or a bug you can fix, follow these steps:

#. Clone the ``main`` branch.
#. Create a new branch to work on your changes.
#. Use ``add``, ``commit``, and ``push`` to save your changes to the new branch.
#. Create a pull request (PR). See the "Submitting a Pull Request" section for more details.

Submitting a Pull Request
-------------------------

#. If applicable, write unit tests for your changes and add them to the 
   `tests directory <https://github.com/rctn/sparsecoding/blob/main/tests>`_.
#. Verify that all tests pass by running ``python -m unittest discover tests -vvv``.
#. Ensure your code adheres to the formatting guidelines specified in 
   `PEP8 <https://peps.python.org/pep-0008/>`_ and validated by 
   `flake8 <https://flake8.pycqa.org/en/latest/>`_.
#. Provide a concise and descriptive title for your PR that summarizes the changes made in your branch.
#. Submit your PR and assign reviewers as necessary.

Coding Style Guidelines
------------------------

We follow the `NumPy documentation standards <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

1. Format your code according to the `flake8 <https://flake8.pycqa.org/en/latest/>`_ standard.
2. Use underscores to separate words in non-class names (e.g., ``n_samples`` instead of ``nsamples``).
3. Avoid single-character variable names.
