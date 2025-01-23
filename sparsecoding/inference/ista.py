import torch

from .inference_method import InferenceMethod


class ISTA(InferenceMethod):
    def __init__(
        self,
        n_iter=100,
        sparsity_penalty=1e-2,
        stop_early=False,
        epsilon=1e-2,
        solver=None,
        return_all_coefficients=False,
    ):
        """Iterative shrinkage-thresholding algorithm for solving LASSO problems.

        Parameters
        ----------
        n_iter : int, default=100
            Number of iterations to run
        sparsity_penalty : float, default=0.2

        stop_early : bool, default=False
            Stops dynamics early based on change in coefficents
        epsilon : float, default=1e-2
            Only used if stop_early True, specifies criteria to stop dynamics
        return_all_coefficients : str, default=False
            Returns all coefficients during inference procedure if True
            User beware: if n_iter is large, setting this parameter to True
            can result in large memory usage/potential exhaustion. This
            function typically used for debugging.
        solver : default=None

        References
        ----------
        [1] Beck, A., & Teboulle, M. (2009). A fast iterative
        shrinkage-thresholding algorithm for linear inverse problems.
        SIAM journal on imaging sciences, 2(1), 183-202.
        """
        super().__init__(solver)
        self.n_iter = n_iter
        self.sparsity_penalty = sparsity_penalty
        self.stop_early = stop_early
        self.epsilon = epsilon
        self.coefficients = None
        self.return_all_coefficients = return_all_coefficients

    def threshold_nonlinearity(self, u):
        """Soft threshhold function

        Parameters
        ----------
        u : array-likes, shape [batch_size, n_basis]
            Membrane potentials

        Returns
        -------
        a : array-like, shape [batch_size, n_basis]
            activations
        """
        a = (torch.abs(u) - self.threshold).clamp(min=0.0)
        a = torch.sign(u) * a
        return a

    def infer(self, data, dictionary, coeff_0=None, use_checknan=False):
        """Infer coefficients for each image in data using dictionary elements.
        Uses ISTA (Beck & Taboulle 2009), equations 1.4 and 1.5.

        Parameters
        ----------
        data : array-like, shape [batch_size, n_features]

        dictionary : array-like, shape [n_features, n_basis]

        coeff_0 : array-like, shape [n_samples, n_basis], optional
            Initial coefficient values
        use_checknan : bool, default=False
            Check for nans in coefficients on each iteration. Setting this to
            False can speed up inference time.
        Returns
        -------
        coefficients : array-like, shape [n_samples, n_basis] OR [n_samples, n_iter+1, n_basis]
           First case occurs if return_all_coefficients == "none". If
           return_all_coefficients != "none", returned shape is second case.
           Returned dimension along dim 1 can be less than n_iter when
           stop_early==True and stopping criteria met.
        """
        batch_size = data.shape[0]
        n_basis = dictionary.shape[1]
        device = dictionary.device

        # Calculate stepsize based on largest eigenvalue of
        # dictionary.T @ dictionary.
        lipschitz_constant = torch.linalg.eigvalsh(torch.mm(dictionary.T, dictionary))[-1]
        stepsize = 1.0 / lipschitz_constant
        self.threshold = stepsize * self.sparsity_penalty

        # Initialize coefficients.
        if coeff_0 is not None:
            u = coeff_0.to(device)
        else:
            u = torch.zeros((batch_size, n_basis)).to(device)
        coefficients = torch.zeros((batch_size, 0, n_basis)).to(device)
        self.coefficients = self.threshold_nonlinearity(u)
        residual = torch.mm(dictionary, self.coefficients.T).T - data

        for _ in range(self.n_iter):
            if self.stop_early:
                old_u = u.clone().detach()

            if self.return_all_coefficients:
                coefficients = torch.concat(
                    [coefficients, self.threshold_nonlinearity(u).clone().unsqueeze(1)],
                    dim=1,
                )

            u -= stepsize * torch.mm(residual, dictionary)
            self.coefficients = self.threshold_nonlinearity(u)

            if self.stop_early:
                # Stopping condition is function of change of the coefficients.
                a_change = torch.mean(torch.abs(old_u - u) / stepsize)
                if a_change < self.epsilon:
                    break

            residual = torch.mm(dictionary, self.coefficients.T).T - data
            u = self.coefficients

            if use_checknan:
                self.checknan(u, "coefficients")

        coefficients = torch.concat([coefficients, self.coefficients.clone().unsqueeze(1)], dim=1)
        return torch.squeeze(coefficients)
