import torch

from .inference_method import InferenceMethod


class Vanilla(InferenceMethod):
    def __init__(self, n_iter=100, coeff_lr=1e-3, sparsity_penalty=0.2,
                 stop_early=False, epsilon=1e-2, solver=None,
                 return_all_coefficients=False):
        """Gradient descent with Euler's method on model in Olshausen & Field
        (1997) with laplace prior over coefficients (corresponding to l-1 norm
        penalty).

        Parameters
        ----------
        n_iter : int, default=100
            Number of iterations to run
        coeff_lr : float, default=1e-3
            Update rate of coefficient dynamics
        sparsity_penalty : float, default=0.2

        stop_early : bool, default=False
            Stops dynamics early based on change in coefficents
        epsilon : float, default=1e-2
            Only used if stop_early True, specifies criteria to stop dynamics
        return_all_coefficients : str, default=False
            Returns all coefficients during inference procedure if True
            User beware: If n_iter is large, setting this parameter to True
            Can result in large memory usage/potential exhaustion. This
            function typically used for debugging.
        solver : default=None

        References
        ----------
        [1] Olshausen, B. A., & Field, D. J. (1997). Sparse coding with an
        overcomplete basis set: A strategy employed by V1?. Vision research,
        37(23), 3311-3325.
        """
        super().__init__(solver)
        self.coeff_lr = coeff_lr
        self.sparsity_penalty = sparsity_penalty
        self.stop_early = stop_early
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.return_all_coefficients = return_all_coefficients

    def grad(self, residual, dictionary, a):
        """Compute the gradient step on coefficients

        Parameters
        ----------
        residual : array-like, shape [batch_size, n_features]
            Residual between reconstructed image and original
        dictionary : array-like, shape [n_features,n_coefficients]
            Dictionary
        a : array-like, shape [batch_size, n_coefficients]
            Coefficients

        Returns
        -------
        da : array-like, shape [batch_size, n_coefficients]
            Gradient of membrane potentials
        """
        da = (dictionary.t()@residual.t()).t() - \
            self.sparsity_penalty*torch.sign(a)
        return da

    def infer(self, data, dictionary, coeff_0=None, use_checknan=False):
        """Infer coefficients using provided dictionary

        Parameters
        ----------
        dictionary : array-like, shape [n_features, n_basis]
            Dictionary
        data : array like, shape [n_samples, n_features]

        coeff_0 : array-like, shape [n_samples, n_basis], optional
            Initial coefficient values
        use_checknan : bool, default=False
            check for nans in coefficients on each iteration. Setting this to
            False can speed up inference time

        Returns
        -------
        coefficients : array-like, shape [n_samples, n_basis] OR [n_samples, n_iter+1, n_basis]
           First case occurs if return_all_coefficients == "none". If
           return_all_coefficients != "none", returned shape is second case.
           Returned dimension along dim 1 can be less than n_iter when
           stop_early==True and stopping criteria met.
        """
        batch_size, n_features = data.shape
        n_features, n_basis = dictionary.shape
        device = dictionary.device

        # initialize
        if coeff_0 is not None:
            a = coeff_0.to(device)
        else:
            a = torch.rand((batch_size, n_basis)).to(device)-0.5

        coefficients = torch.zeros((batch_size, 0, n_basis)).to(device)

        residual = data - (dictionary@a.t()).t()
        for i in range(self.n_iter):

            if self.return_all_coefficients:
                coefficients = torch.concat([coefficients, a.clone().unsqueeze(1)], dim=1)

            if self.stop_early:
                old_a = a.clone().detach()

            da = self.grad(residual, dictionary, a)
            a = a + self.coeff_lr*da

            if self.stop_early:
                if torch.linalg.norm(old_a - a)/torch.linalg.norm(old_a) < self.epsilon:
                    break

            residual = data - (dictionary@a.t()).t()

            if use_checknan:
                self.checknan(a, "coefficients")

        coefficients = torch.concat([coefficients, a.clone().unsqueeze(1)], dim=1)
        return torch.squeeze(coefficients)
