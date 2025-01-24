import numpy as np
import torch

from .inference_method import InferenceMethod


class MP(InferenceMethod):
    """
    Infer coefficients for each image in data using elements dictionary.
    Method description can be traced
    to "Matching Pursuits with Time-Frequency Dictionaries" (S. G. Mallat & Z. Zhang, 1993)

    Parameters
    ----------
    sparsity : scalar (1,)
        sparsity of the solution
    return_all_coefficients : string (1,) default=False
        returns all coefficients during inference procedure if True
        user beware: if n_iter is large, setting this parameter to True
        can result in large memory usage/potential exhaustion. This function typically used for
        debugging
    solver : default=None
    """

    def __init__(self, sparsity, solver=None, return_all_coefficients=False):
        super().__init__(solver)
        self.sparsity = sparsity
        self.return_all_coefficients = return_all_coefficients

    def infer(self, data, dictionary):
        """
        Infer coefficients for each image in data using dict elements dictionary using Matching Pursuit (MP)

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
        K = np.ceil(self.sparsity * n_basis).astype(int)

        # Get dictionary norms in case atoms are not normalized
        dictionary_norms = torch.norm(dictionary, p=2, dim=0, keepdim=True)

        # Initialize coefficients for the whole batch
        coefficients = torch.zeros(batch_size, n_basis, requires_grad=False, device=device)

        residual = data.clone()  # [batch_size, n_features]

        for _ in range(K):
            # Select which (coefficient, basis function) pair to update using the inner product.
            candidate_coefs = residual @ dictionary  # [batch_size, n_basis]
            top_coef_idxs = torch.argmax(torch.abs(candidate_coefs) / dictionary_norms, dim=1)  # [batch_size]

            # Update the coefficient.
            top_coefs = candidate_coefs[torch.arange(batch_size), top_coef_idxs]  # [batch_size]
            coefficients[torch.arange(batch_size), top_coef_idxs] = top_coefs

            # Explain away/subtract the chosen coefficient and corresponding basis from the residual.
            top_coef_bases = dictionary[..., top_coef_idxs]  # [n_features, batch_size]
            residual = residual - top_coefs.reshape(batch_size, 1) * top_coef_bases.T  # [batch_size, n_features]

        return coefficients.detach()
