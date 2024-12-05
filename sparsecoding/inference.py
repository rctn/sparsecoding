import numpy as np
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


class LCA(InferenceMethod):
    def __init__(self, n_iter=100, coeff_lr=1e-3, threshold=0.1,
                 stop_early=False, epsilon=1e-2, solver=None,
                 return_all_coefficients="none", nonnegative=False):
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
            raise ValueError("Invalid input for return_all_coefficients. Valid"
                             "inputs are: \"none\", \"membrane\", \"active\".")
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
            a = (u - self.threshold).clamp(min=0.)
        else:
            a = (torch.abs(u) - self.threshold).clamp(min=0.)
            a = torch.sign(u)*a
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
        du = b-u-(G@a.t()).t()
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

        b = (dictionary.t()@data.t()).t()
        G = dictionary.t()@dictionary-torch.eye(n_basis).to(device)
        for i in range(self.n_iter):
            # store old membrane potentials to evalute stop early condition
            if self.stop_early:
                old_u = u.clone().detach()

            # check return all
            if self.return_all_coefficients != "none":
                if self.return_all_coefficients == "active":
                    coefficients = torch.concat(
                        [coefficients, self.threshold_nonlinearity(u).clone().unsqueeze(1)], dim=1)
                else:
                    coefficients = torch.concat(
                        [coefficients, u.clone().unsqueeze(1)], dim=1)

            # compute new
            a = self.threshold_nonlinearity(u)
            du = self.grad(b, G, u, a)
            u = u + self.coeff_lr*du

            # check stopping condition
            if self.stop_early:
                relative_change_in_coeff = torch.linalg.norm(old_u - u)/torch.linalg.norm(old_u)
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


class ISTA(InferenceMethod):
    def __init__(self, n_iter=100, sparsity_penalty=1e-2, stop_early=False,
                 epsilon=1e-2, solver=None, return_all_coefficients=False):
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
        a = (torch.abs(u) - self.threshold).clamp(min=0.)
        a = torch.sign(u)*a
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
        lipschitz_constant = torch.linalg.eigvalsh(
            torch.mm(dictionary.T, dictionary))[-1]
        stepsize = 1. / lipschitz_constant
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
                coefficients = torch.concat([coefficients,
                                             self.threshold_nonlinearity(u).clone().unsqueeze(1)], dim=1)

            u -= stepsize * torch.mm(residual, dictionary)
            self.coefficients = self.threshold_nonlinearity(u)

            if self.stop_early:
                # Stopping condition is function of change of the coefficients.
                a_change = torch.mean(
                    torch.abs(old_u - u) / stepsize)
                if a_change < self.epsilon:
                    break

            residual = torch.mm(dictionary, self.coefficients.T).T - data
            u = self.coefficients

            if use_checknan:
                self.checknan(u, "coefficients")

        coefficients = torch.concat([coefficients, self.coefficients.clone().unsqueeze(1)], dim=1)
        return torch.squeeze(coefficients)


class LSM(InferenceMethod):
    def __init__(self, n_iter=100, n_iter_LSM=6, beta=0.01, alpha=80.0,
                 sigma=0.005, sparse_threshold=10**-2, solver=None,
                 return_all_coefficients=False):
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
        mse_loss = (1/(2*(sigma**2))) * torch.sum(torch.square(data - preds), dim=1, keepdim=True)
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
            lambdas = (
                (self.alpha + 1)
                / (self.beta + torch.abs(coefficients.detach()))
            )

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
                if (
                    last_loss is not None
                    and loss > 1.05 * last_loss
                ):
                    break

                last_loss = loss

        # Sparsify the final solution by discarding the small coefficients
        coefficients.data[torch.abs(coefficients.data)
                          < self.sparse_threshold] = 0

        return coefficients.detach()


class PyTorchOptimizer(InferenceMethod):
    def __init__(self, optimizer_f, loss_f, n_iter=100, solver=None):
        """Infer coefficients using provided loss functional and optimizer

        Parameters
        ----------
        optimizer : function handle
            Pytorch optimizer handle have single parameter:
                (coefficients)
            where coefficients is of shape [batch_size, n_basis]
        loss_f : function handle
            Must have parameters:
                 (data, dictionary, coefficients)
            where data is of shape [batch_size, n_features]
            and loss_f must return tensor of size [batch_size,]
        n_iter : int, default=100
            Number of iterations to run for an optimizer
        solver : default=None
        """
        super().__init__(solver)
        self.optimizer_f = optimizer_f
        self.loss_f = loss_f
        self.n_iter = n_iter

    def infer(self, data, dictionary, coeff_0=None):
        """Infer coefficients for each image in data using dict elements
        dictionary by minimizing provided loss function with provided
        optimizer.

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

        # initialize
        if coeff_0 is not None:
            coefficients = coeff_0.requires_grad_(True)
        else:
            coefficients = torch.zeros((batch_size, n_basis), requires_grad=True, device=device)

        optimizer = self.optimizer_f([coefficients])

        for i in range(self.n_iter):

            # compute LSM loss for the current iteration
            loss = self.loss_f(
                data=data,
                dictionary=dictionary,
                coefficients=coefficients,
            )

            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to
            # model parameters
            loss.backward(torch.ones((batch_size,), device=device))

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()

        return coefficients.detach()


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


class MP(InferenceMethod):
    """
    Infer coefficients for each image in data using elements dictionary.
    Method description can be traced
    to "Matching Pursuits with Time-Frequency Dictionaries" (S. G. Mallat & Z. Zhang, 1993)
    """

    def __init__(self, sparsity, solver=None, return_all_coefficients=False):
        '''

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
        '''
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
        K = np.ceil(self.sparsity*n_basis).astype(int)

        # Get dictionary norms in case atoms are not normalized
        dictionary_norms = torch.norm(dictionary, p=2, dim=0, keepdim=True)

        # Initialize coefficients for the whole batch
        coefficients = torch.zeros(
            batch_size, n_basis, requires_grad=False, device=device)

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


class OMP(InferenceMethod):
    """
    Infer coefficients for each image in data using elements dictionary.
    Method description can be traced to:
        "Orthogonal Matching Pursuit: Recursive Function Approximation with Application to Wavelet Decomposition"
        (Y. Pati & R. Rezaiifar & P. Krishnaprasad, 1993)
    """

    def __init__(self, sparsity, solver=None, return_all_coefficients=False):
        '''

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
        '''
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
        K = np.ceil(self.sparsity*n_basis).astype(int)

        # Get dictionary norms in case atoms are not normalized
        dictionary_norms = torch.norm(dictionary, p=2, dim=0, keepdim=True)

        # Initialize coefficients for the whole batch
        coefficients = torch.zeros(
            batch_size, n_basis, requires_grad=False, device=device)

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


class CEL0(InferenceMethod):
    def __init__(self, n_iter=100, coeff_lr=1e-3, threshold=1e-2, return_all_coefficients="none", solver=None):
        """
        Parameters
        ----------
        n_iter : int, default=100
            Number of iterations to run
        coeff_lr : float, default=1e-3
            Update rate of coefficient dynamics
        threshold : float, default=1e-2
            Threshold for non-linearity
        return_all_coefficients : str, {"none", "active"}, default="none"
            Returns all coefficients during inference procedure if not equal
            to "none". If return_all_coefficients=="active",
            active units (a) (output of thresholding function over u) returned.
            User beware: if n_iter is large, setting this parameter to True
            can result in large memory usage/potential exhaustion. This
            function typically used for debugging.
        solver : default=None

        References
        ----------
        [1] https://arxiv.org/abs/2301.10002
        """
        super().__init__(solver)
        self.threshold = threshold
        self.coeff_lr = coeff_lr
        self.n_iter = n_iter
        self.return_all_coefficients = return_all_coefficients
        self.dictionary_norms = None

    def threshold_nonlinearity(self, u, a=1):
        '''
        CEL0 thresholding function: A continuous exact l0 penalty

        Note: It is assumed that the dictionary is normalized

        Parameters
        ----------
        u : array-like, shape [batch_size, n_basis]
        a : the norm of the column of the dictionary, default=1

        Returns
        -------
        re : array-like, shape [batch_size, n_basis]

        '''
        if a * self.coeff_lr < 1:
            num = (np.abs(u) - torch.sqrt(2 * self.threshold) * a * self.coeff_lr)
            num[num < 0] = 0
            den = 1 - a ** 2 * self.coeff_lr
            re = np.sign(u) * np.minimum(np.abs(u), np.divide(num, den))    # * (a ** 2 * self.coeff_lr < 1)
            return re
        else:
            # TODO: This is not the same as the paper
            larger = u[np.abs(u) < torch.sqrt(2 * self.threshold * self.coeff_lr)]
            equal = u[np.abs(u) == torch.sqrt(2 * self.threshold * self.coeff_lr)]
            re = larger + equal
            return re

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

        self.dictionary_norms = torch.norm(dictionary, dim=0, keepdim=True).squeeze()[0]
        assert self.dictionary_norms == 1, "Dictionary must be normalized"

        for i in range(self.n_iter):
            # check return all
            if self.return_all_coefficients != "none":
                if self.return_all_coefficients == "active":
                    coefficients = torch.concat(
                        [coefficients, self.CEL0Thresholding(u).clone().unsqueeze(1)], dim=1)
                else:
                    coefficients = torch.concat(
                        [coefficients, u.clone().unsqueeze(1)], dim=1)

            # compute new
            # Step 1: Gradient descent on u
            recon = u @ dictionary.T
            residual = data - recon
            dLda = residual @ dictionary
            u = u + self.coeff_lr * dLda

            # Step 2: Thresholding
            u = self.threshold_nonlinearity(u)

            if use_checknan:
                self.checknan(u, "coefficients")

        # return active units if return_all_coefficients in ["none", "active"]
        if self.return_all_coefficients == "active":
            coefficients = torch.concat([coefficients, u.clone().unsqueeze(1)], dim=1)
        else:
            final_coefficients = u
            coefficients = torch.concat([coefficients, final_coefficients.clone().unsqueeze(1)], dim=1)

        return coefficients.squeeze()
