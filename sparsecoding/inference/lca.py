import torch

from .inference_method import InferenceMethod


class LCA(InferenceMethod):
    def __init__(
        self,
        n_iter=100,
        coeff_lr=1e-3,
        threshold=0.1,
        stop_early=False,
        epsilon=1e-2,
        solver=None,
        return_all_coefficients="none",
        nonnegative=False,
    ):
        """Method implemented according locally competative algorithm (LCA)
        with the ideal soft thresholding function.

        Parameters
        ----------
        n_iter : int, default=100
            Number of iterations to run
        coeff_lr : float, default=1e-3
            Update rate of coefficient dynamics
        threshold : float, default=0.1
            Threshold for non-linearity
        stop_early : bool, default=False
            Stops dynamics early based on change in coefficents
        epsilon : float, default=1e-2
            Only used if stop_early True, specifies criteria to stop dynamics
        nonnegative : bool, default=False
            Constrain coefficients to be nonnegative
        return_all_coefficients : str, {"none", "membrane", "active"}, default="none"
            Returns all coefficients during inference procedure if not equal
            to "none". If return_all_coefficients=="membrane", membrane
            potentials (u) returned. If return_all_coefficients=="active",
            active units (a) (output of thresholding function over u) returned.
            User beware: if n_iter is large, setting this parameter to True
            can result in large memory usage/potential exhaustion. This
            function typically used for debugging.
        solver : default=None

        References
        ----------
        [1] Rozell, C. J., Johnson, D. H., Baraniuk, R. G., & Olshausen,
        B. A. (2008). Sparse coding via thresholding and local competition
        in neural circuits. Neural computation, 20(10), 2526-2563.
        """
        super().__init__(solver)
        self.threshold = threshold
        self.coeff_lr = coeff_lr
        self.stop_early = stop_early
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.nonnegative = nonnegative
        if return_all_coefficients not in ["none", "membrane", "active"]:
            raise ValueError(
                "Invalid input for return_all_coefficients. Valid" 'inputs are: "none", "membrane", "active".'
            )
        self.return_all_coefficients = return_all_coefficients

    def threshold_nonlinearity(self, u):
        """Soft threshhold function

        Parameters
        ----------
        u : array-like, shape [batch_size, n_basis]
            Membrane potentials

        Returns
        -------
        a : array-like, shape [batch_size, n_basis]
            Activations
        """
        if self.nonnegative:
            a = (u - self.threshold).clamp(min=0.0)
        else:
            a = (torch.abs(u) - self.threshold).clamp(min=0.0)
            a = torch.sign(u) * a
        return a

    def grad(self, b, G, u, a):
        """Compute the gradient step on membrane potentials

        Parameters
        ----------
        b : array-like, shape [batch_size, n_coefficients]
            Driver signal for coefficients
        G : array-like, shape [n_coefficients, n_coefficients]
            Inhibition matrix
        a : array-like, shape [batch_size, n_coefficients]
            Currently active coefficients

        Returns
        -------
        du : array-like, shape [batch_size, n_coefficients]
            Gradient of membrane potentials
        """
        du = b - u - (G @ a.t()).t()
        return du

    def infer(self, data, dictionary, coeff_0=None, use_checknan=False):
        """Infer coefficients using provided dictionary

        Parameters
        ----------
        dictionary : array-like, shape [n_features, n_basis]

        data : array-like, shape [n_samples, n_features]

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
        batch_size, n_features = data.shape
        n_features, n_basis = dictionary.shape
        device = dictionary.device

        # initialize
        if coeff_0 is not None:
            u = coeff_0.to(device)
        else:
            u = torch.zeros((batch_size, n_basis)).to(device)

        coefficients = torch.zeros((batch_size, 0, n_basis)).to(device)

        b = (dictionary.t() @ data.t()).t()
        G = dictionary.t() @ dictionary - torch.eye(n_basis).to(device)
        for i in range(self.n_iter):
            # store old membrane potentials to evalute stop early condition
            if self.stop_early:
                old_u = u.clone().detach()

            # check return all
            if self.return_all_coefficients != "none":
                if self.return_all_coefficients == "active":
                    coefficients = torch.concat(
                        [
                            coefficients,
                            self.threshold_nonlinearity(u).clone().unsqueeze(1),
                        ],
                        dim=1,
                    )
                else:
                    coefficients = torch.concat([coefficients, u.clone().unsqueeze(1)], dim=1)

            # compute new
            a = self.threshold_nonlinearity(u)
            du = self.grad(b, G, u, a)
            u = u + self.coeff_lr * du

            # check stopping condition
            if self.stop_early:
                relative_change_in_coeff = torch.linalg.norm(old_u - u) / torch.linalg.norm(old_u)
                if relative_change_in_coeff < self.epsilon:
                    break

            if use_checknan:
                self.checknan(u, "coefficients")

        # return active units if return_all_coefficients in ["none", "active"]
        if self.return_all_coefficients == "membrane":
            coefficients = torch.concat([coefficients, u.clone().unsqueeze(1)], dim=1)
        else:
            final_coefficients = self.threshold_nonlinearity(u)
            coefficients = torch.concat([coefficients, final_coefficients.clone().unsqueeze(1)], dim=1)

        return coefficients.squeeze()
