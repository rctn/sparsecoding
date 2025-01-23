import torch


class InferenceMethod:
    """Base class for inference method."""

    def __init__(self, solver):
        """
        Parameters
        ----------
        """
        self.solver = solver

    def initialize(self, a):
        """Define initial coefficients.

        Parameters
        ----------

        Returns
        -------
        """
        raise NotImplementedError

    def grad(self):
        """Compute the gradient step.

        Parameters
        ----------

        Returns
        -------
        """
        raise NotImplementedError

    def infer(self, dictionary, data, coeff_0=None, use_checknan=False):
        """Infer the coefficients given a dataset and dictionary.

        Parameters
        ----------
        dictionary : array-like, shape [n_features,n_basis]

        data : array-like, shape [n_samples,n_features]

        coeff_0 : array-like, shape [n_samples,n_basis], optional
            Initial coefficient values.
        use_checknan : bool, default=False
            Check for nans in coefficients on each iteration

        Returns
        -------
        coefficients : array-like, shape [n_samples,n_basis]
        """
        raise NotImplementedError

    @staticmethod
    def checknan(data=torch.tensor(0), name="data"):
        """Check for nan values in data.

        Parameters
        ----------
        data : array-like, optional
            Data to check for nans
        name : str, default="data"
            Name to add to error, if one is thrown

        Raises
        ------
        ValueError
            If the nan found in data
        """
        if torch.isnan(data).any():
            raise ValueError("InferenceMethod error: nan in %s." % (name))
