# Contributing

All contributions are welcome!

## Bug Reporting

If you find a bug, submit a bug report on GitHub Issues.

## Adding Features/Fixing Bugs

If you have identified a new feature or bug that you can fix yourself, please follow the following procedure.

1. Fork `main` branch.
2. Create a new branch to contain your changes.
3. `add`, `commit`, and `push` your changes to this branch.
4. Create a pull request (PR). See more information on submitting a PR request below.

### Submitting a Pull Request

1. If necessary, please **write your own unit tests** and place them near the code being tested. High-level tests, such as integration or example tests can be placed in the top-level "tests" folder.
2. Verify that all tests are passed by running `python -m pytest .`.
3. Be sure that your PR follows formatting guidelines, [PEP8](https://peps.python.org/pep-0008/) and [flake8](https://flake8.pycqa.org/en/latest/).
4. Make sure the title of your PR summarizes the features/issues resolved in your branch.
5. Submit your pull request and add reviewers.

## Coding Style Guidelines

The following are some guidelines on how new code should be written. Of course, there are special cases, and there will be exceptions to these rules.

1. Format code in accordance with [flake8](https://flake8.pycqa.org/en/latest/) standard.
2. Use underscores to separate words in non-class names: `n_samples` rather than `nsamples`.
3. Avoid single-character variable names.

## Docstrings

When writing docstrings, please follow the following example.

```py
def count_beans(self, baz, use_gpu=False, foo="vector"
                bar=None):
   """Write a one-line summary for the method.

   Parameters
   ----------
   baz : array-like, shape [..., dim]
      Write a short description of parameter baz.
   use_gpu : bool, default=False
      Write a short description of parameter use_gpu.
   foo : str, {"vector", "matrix"}, default="vector"
      Write a short description of parameter foo.
   bar : array-like, shape [...,], optional
      Write a short description of parameter bar.

   Returns
   -------
   n_beans : array-like, shape [..., dim, dim]
      Write a short description of the result returned by the method.

   Notes
   -----
   If relevant, provide equations with (:math:)
   describing computations performed in the method.

   Example
   -------
   Provide code snippets showing how the method is used.
   You can link to scripts of the examples/ directory.

   Reference
   ---------
   If relevant, provide a reference with associated pdf or
   wikipedia page.
   ex: 
   [1] Einstein, A., Podolsky, B., & Rosen, N. (1935). Can 
   quantum-mechanical description of physical reality be 
   considered complete?. Physical review, 47(10), 777.
   """
```
