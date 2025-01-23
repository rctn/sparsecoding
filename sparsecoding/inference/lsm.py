import torch

from .inference_method import InferenceMethod


class LSM(InferenceMethod):
    def __init__(
        self,
        n_iter=100,
        n_iter_LSM=6,
        beta=0.01,
        alpha=80.0,
        sigma=0.005,
        sparse_threshold=10**-2,
        solver=None,
        return_all_coefficients=False,
    ):
        """Infer latent coefficients generating data given dictionary.
        Method implemented according to "Group Sparse Coding with a Laplacian
        Scale Mixture Prior" (P. J. Garrigues & B. A. Olshausen, 2010)

        Parameters
        ----------
        n_iter : int, default=100
            Number of iterations to run for an optimizer
        n_iter_LSM : int, default=6
            Number of iterations to run the outer loop of  LSM
        beta : float, default=0.01
            LSM parameter used to update lambdas
        alpha : float, default=80.0
            LSM parameter used to update lambdas
        sigma : float, default=0.005
            LSM parameter used to compute the loss function
        sparse_threshold : float, default=10**-2
            Threshold used to discard smallest coefficients in the final
            solution SM parameter used to compute the loss function
        return_all_coefficients : bool, default=False
            Returns all coefficients during inference procedure if True
            User beware: If n_iter is large, setting this parameter to True
            can result in large memory usage/potential exhaustion. This
            function typically used for debugging.
        solver : default=None

        References
        ----------
        [1] Garrigues, P., & Olshausen, B. (2010). Group sparse coding with
        a laplacian scale mixture prior. Advances in neural information
        processing systems, 23.
        """
        super().__init__(solver)
        self.n_iter = n_iter
        self.n_iter_LSM = n_iter_LSM
        self.beta = beta
        self.alpha = alpha
        self.sigma = sigma
        self.sparse_threshold = sparse_threshold
        self.return_all_coefficients = return_all_coefficients

    def lsm_Loss(self, data, dictionary, coefficients, lambdas, sigma):
        """Compute LSM loss according to equation (7) in (P. J. Garrigues &
        B. A. Olshausen, 2010)

        Parameters
        ----------
        data : array-like, shape [batch_size, n_features]
            Data to be used in sparse coding
        dictionary : array-like, shape [n_features, n_basis]
            Dictionary to be used
        coefficients : array-like, shape [batch_size, n_basis]
            The current values of coefficients
        lambdas : array-like, shape [batch_size, n_basis]
            The current values of regularization coefficient for all basis
        sigma : float, default=0.005
            LSM parameter used to compute the loss functions

        Returns
        -------
        loss : array-like, shape [batch_size, 1]
            Loss values for each data sample
        """

        # Compute loss
        preds = torch.mm(dictionary, coefficients.t()).t()
        mse_loss = (1 / (2 * (sigma**2))) * torch.sum(torch.square(data - preds), dim=1, keepdim=True)
        sparse_loss = torch.sum(lambdas * torch.abs(coefficients), dim=1, keepdim=True)
        loss = mse_loss + sparse_loss
        return loss

    def infer(self, data, dictionary):
        """Infer coefficients for each image in data using dict elements
        dictionary using Laplacian Scale Mixture (LSM)

        Parameters
        ----------
        data : array-like, shape [batch_size, n_features]
            Data to be used in sparse coding
        dictionary : array-like, shape [n_features, n_basis]
            Dictionary to be used to get the coefficients

        Returns
        -------
        coefficients : array-like, shape [batch_size, n_basis]
        """
        # Get input characteristics
        batch_size, n_features = data.shape
        n_features, n_basis = dictionary.shape
        device = dictionary.device

        # Initialize coefficients for the whole batch
        coefficients = torch.zeros(batch_size, n_basis, device=device, requires_grad=True)

        # Set up optimizer
        optimizer = torch.optim.Adam([coefficients], lr=1e-1)

        # Outer loop, set sparsity penalties (lambdas).
        for i in range(self.n_iter_LSM):
            # Compute the initial values of lambdas
            lambdas = (self.alpha + 1) / (self.beta + torch.abs(coefficients.detach()))

            # Inner loop, optimize coefficients w/ current sparsity penalties.
            # Exits early if converged before `n_iter`s.
            last_loss = None
            for t in range(self.n_iter):
                # compute LSM loss for the current iteration
                loss = self.lsm_Loss(
                    data=data,
                    dictionary=dictionary,
                    coefficients=coefficients,
                    lambdas=lambdas,
                    sigma=self.sigma,
                )
                loss = torch.sum(loss)

                # Backward pass: compute gradient and update model parameters.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Break if coefficients have converged.
                if last_loss is not None and loss > 1.05 * last_loss:
                    break

                last_loss = loss

        # Sparsify the final solution by discarding the small coefficients
        coefficients.data[torch.abs(coefficients.data) < self.sparse_threshold] = 0

        return coefficients.detach()
