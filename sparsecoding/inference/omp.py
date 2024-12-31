import numpy as np
import torch

from .inference_method import InferenceMethod


class OMP(InferenceMethod):
    """
    Infer coefficients for each image in data using elements dictionary.
    Method description can be traced to:
        "Orthogonal Matching Pursuit: Recursive Function Approximation with Application to Wavelet Decomposition"
        (Y. Pati & R. Rezaiifar & P. Krishnaprasad, 1993)
    """

    def __init__(self, sparsity, solver=None, return_all_coefficients=False):
        """

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
        super().__init__(solver)
        self.sparsity = sparsity
        self.return_all_coefficients = return_all_coefficients

    def infer(self, data, dictionary):
        """
        Infer coefficients for each image in data using dict elements dictionary using Orthogonal Matching Pursuit (OMP)

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

        # The basis functions that are used to infer the coefficients will be updated each time.
        used_basis_fns = torch.zeros((batch_size, n_basis), dtype=torch.bool)

        for t in range(K):
            # Select which (coefficient, basis function) pair to update using the inner product.
            candidate_coefs = residual @ dictionary  # [batch_size, n_basis]
            top_coef_idxs = torch.argmax(torch.abs(candidate_coefs) / dictionary_norms, dim=1)  # [batch_size]
            used_basis_fns[torch.arange(batch_size), top_coef_idxs] = True

            # Update the coefficients
            used_dictionary = dictionary[..., used_basis_fns.nonzero()[:, 1]].reshape((n_features, batch_size, t + 1))

            (used_coefficients, _, _, _) = torch.linalg.lstsq(
                used_dictionary.permute((1, 0, 2)),  # [batch_size, n_features, t + 1]
                data.reshape(batch_size, n_features, 1),
            )  # [batch_size, t + 1, 1]
            coefficients[used_basis_fns] = used_coefficients.reshape(-1)

            # Update the residual.
            residual = data.clone() - coefficients @ dictionary.T  # [batch_size, n_features]

        return coefficients.detach()
