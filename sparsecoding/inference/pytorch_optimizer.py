import torch

from .inference_method import InferenceMethod


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
