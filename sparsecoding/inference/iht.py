import numpy as np
import torch

from .inference_method import InferenceMethod


class IHT(InferenceMethod):
    """
    Infer coefficients for each image in data using elements dictionary.
    Method description can be traced to
    "Iterative Hard Thresholding for Compressed Sensing" (T. Blumensath & M. E. Davies, 2009)
    """

    def __init__(self, sparsity, n_iter=10, solver=None, return_all_coefficients=False):
        '''

        Parameters
        ----------
        sparsity : scalar (1,)
            Sparsity of the solution. The number of active coefficients will be set
            to ceil(sparsity * data_dim) at the end of each iterative update.
        n_iter : scalar (1,) default=100
            number of iterations to run for an inference method
        return_all_coefficients : string (1,) default=False
            returns all coefficients during inference procedure if True
            user beware: if n_iter is large, setting this parameter to True
            can result in large memory usage/potential exhaustion. This function typically used for
            debugging
        solver : default=None
        '''
        super().__init__(solver)
        self.n_iter = n_iter
        self.sparsity = sparsity
        self.return_all_coefficients = return_all_coefficients

    def infer(self, data, dictionary):
        """
        Infer coefficients for each image in data using dict elements dictionary using Iterative Hard Thresholding (IHT)

        Parameters
        ----------
        data : array-like (batch_size, n_features)
            data to be used in sparse coding
        dictionary : array-like, (n_features, n_basis)
            dictionary to be used to get the coefficients

        Returns
        -------
        coefficients : array-like (batch_size, n_basis)
        """
        # Get input characteristics
        batch_size, n_features = data.shape
        n_features, n_basis = dictionary.shape
        device = dictionary.device

        # Define signal sparsity
        K = np.ceil(self.sparsity*n_basis).astype(int)

        # Initialize coefficients for the whole batch
        coefficients = torch.zeros(
            batch_size, n_basis, requires_grad=False, device=device)

        for _ in range(self.n_iter):
            # Compute the prediction given the current coefficients
            preds = coefficients @ dictionary.T  # [batch_size, n_features]

            # Compute the residual
            delta = data - preds  # [batch_size, n_features]

            # Compute the similarity between the residual and the atoms in the dictionary
            update = delta @ dictionary  # [batch_size, n_basis]
            coefficients = coefficients + update  # [batch_size, n_basis]

            # Apply kWTA nonlinearity
            topK_values, indices = torch.topk(torch.abs(coefficients), K, dim=1)

            # Reconstruct coefficients using the output of torch.topk
            coefficients = (
                torch.sign(coefficients)
                * torch.zeros(batch_size, n_basis, device=device).scatter_(1, indices, topK_values)
            )

        return coefficients.detach()
