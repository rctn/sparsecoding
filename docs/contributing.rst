============
Contributing
============

We welcome all contributions to this project! Whether it’s reporting bugs, suggesting features, 
fixing issues, or improving documentation, your input is invaluable.

Bug Reporting
-------------

If you encounter a bug, please report it by creating an issue on GitHub. Include as much detail as 
possible to help us reproduce and fix the issue.

Adding Features or Fixing Bugs
------------------------------

If you’ve identified a new feature to add or a bug you can fix, follow these steps:

#. Clone the ``main`` branch.
#. Create a new branch to work on your changes. Use a descriptive name for your branch, such as 
   ``fix-issue-123`` or ``feature-add-logging``.
#. Use ``add``, ``commit``, and ``push`` to save your changes to the new branch.
#. Create a pull request (PR). See the "Submitting a Pull Request" section for more details.

Submitting a Pull Request
-------------------------
To ensure a smooth review process and maintain high code quality, follow these guidelines when 
submitting a PR:

#. If applicable, write unit tests for your changes. We use the 
   `pytest <https://pytest.readthedocs.io/>`_ framework. Every Python module, extension module, 
   or subpackage in the sparsecoding package directory should have a corresponding ``test_<name>.py`` 
   file. Pytest examines these files for test methods (named ``test*``) and test classes (named 
   ``Test*``). Add your tests to the appropriate ``test_*.py`` (create this file if it doesn't 
   already exist).
#. Verify that all tests pass by running ``pytest sparsecoding/`` from the base repository directory.
#. Ensure your code adheres to the formatting guidelines specified in 
   `PEP8 <https://peps.python.org/pep-0008/>`_ and validated by 
   `flake8 <https://flake8.pycqa.org/en/latest/>`_.
#. Prepare a detailed and clear PR description: 
   
   * Summarize the purpose of the PR and the changes made.
   
   * Include any relevant context, such as links to related issues or discussions.
   
   * Specify testing steps or considerations for reviewers.

#. Submit your PR and assign reviewers as necessary.
#. Reviewers: Use squash and merge when merging the PR.

   * Set the merge description to match the PR description.

   * Squash commits into a single commit to maintain a clean project history.


Coding Style Guidelines
-----------------------

We follow the `NumPy documentation standards <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

1. Format your code according to the `flake8 <https://flake8.pycqa.org/en/latest/>`_ standard.
2. Use underscores to separate words in non-class names (e.g., ``n_samples`` instead of ``nsamples``).
3. Avoid single-character variable names.
