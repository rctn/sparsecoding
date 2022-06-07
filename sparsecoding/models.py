from typing import List, Optional

import numpy as np
import pickle as pkl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sparsecoding.priors.common import Prior


class SparseCoding(torch.nn.Module):

    def __init__(self, inference_method, n_basis, n_features,
                 sparsity_penalty=0.2, device=None, **kwargs):
        """Class for learning a sparse code via dictionary learning

        Parameters
        ----------
        inference_method : sparsecoding.InferenceMethod
            Method for inferring coefficients for each image given the
            dictionary
        n_basis : int
            Number of basis functions in dictionary
        n_features : int
            Number of features in data
        sparsity_penalty : float, default=0.2
            Sparsity penalty
        dictionary_lr : float, default=1e-2
            Learning rate of dictionary update
        device : torch.device, default=torch.device("cpu")
            Which device to utilize
        """
        super(SparseCoding, self).__init__()
        self.inference_method = inference_method
        self.n_basis = n_basis
        self.n_features = n_features
        self.device = torch.device("cpu") if device is None else device
        self.dictionary_lr = torch.tensor(np.float32(kwargs.pop("dictionary_lr", 1e-2))).to(self.device)
        self.sparsity_penalty = torch.tensor(np.float32(sparsity_penalty)).to(self.device)
        self.dictionary = torch.randn((self.n_features, self.n_basis)).to(self.device)
        self.normalize_dictionary()

    def compute_grad_dict(self, data, a):
        """Compute gradient of loss function w.r.t. dictionary elements

        Parameters
        ----------
        data : array-like, shape [batch_size, n_features]
            input data
        a : array-like, shape [batch_size, n_basis]
            already-inferred coefficients

        Returns
        -------
        dictionary_grad : array-like, shape [n_features, n_basis]
            gradient of dictionary
        """
        residual = data - torch.mm(self.dictionary, a.t()).t()
        dictionary_grad = torch.mm(residual.t(), a)
        return dictionary_grad

    def update_dictionary(self, data, a):
        """Compute gradient of loss function w.r.t. dictionary elements, and
        update

        Parameters
        ----------
        data : array-like, shape [batch_size,n_features]
            Input data
        a : array-like, shape [batch_size, n_basis]
            Already-inferred coefficients
        """
        dictionary_grad = self.compute_grad_dict(data, a)
        self.dictionary = torch.add(self.dictionary,
                                    self.dictionary_lr*dictionary_grad)
        self.checknan()

    def normalize_dictionary(self):
        """Normalize columns of dictionary matrix to unit norm."""
        self.dictionary = self.dictionary.div_(self.dictionary.norm(p=2, dim=0))
        self.checknan()

    def learn_dictionary(self, dataset, n_epoch, batch_size):
        """Learn dictionary for n_epoch epochs

        Parameters
        ----------
        dataset : torch.utils.data.Dataset or array-like, shape [n_samples, n_features]
            Input dataset
        n_epoch : int
            Iumber of iterations to learn dictionary
        batch_size : int
            Batch size to do dictionary updates over

        Returns
        -------
        losses : array-like, shape [nepoch,]
            Model losses (i.e., energy for the Boltzmann enthusiasts) after
            each dictionary update
        """
        losses = []

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        iterloader = iter(dataloader)
        for i in range(n_epoch):
            try:
                batch = next(iterloader)
            except StopIteration:
                dataloader = DataLoader(dataset, batch_size=batch_size,
                                        shuffle=True)
                iterloader = iter(dataloader)
                batch = next(iterloader)

            # infer coefficients
            a = self.inference_method.infer(batch, self.dictionary)

            # update dictionary
            self.update_dictionary(batch, a)

            # normalize dictionary
            self.normalize_dictionary()

            # compute current loss
            loss = self.compute_loss(batch, a)

            losses.append(loss)
        return np.asarray(losses)

    def compute_loss(self, data, a):
        """Compute loss given data and inferred coefficients

        Parameters
        ----------
        data : array-like, shape [batch_size, n_features]

        a : array-like, shape [batch_size, n_basis]
            inferred coefficients

        Returns
        -------
        float
            loss
        """
        batch_size, _ = data.shape

        MSE_loss = torch.square(torch.linalg.vector_norm(data-torch.mm(self.dictionary, a.t()).t(), dim=1))
        sparsity_loss = self.sparsity_penalty*torch.abs(a).sum(dim=1)
        total_loss = torch.sum(MSE_loss + sparsity_loss)
        return total_loss.item()/batch_size

    def get_numpy_dictionary(self):
        """Returns dictionary as numpy array

        Returns
        -------
        dictionary : array-like, shape [n_features,n_basis]
            numpy array dictionary
        """
        return self.dictionary.cpu().detach().numpy()

    def checknan(self, data=torch.tensor(0), name="data"):
        """Check for nan values in dictinary, or data

        Parameters
        ----------
        data : array-like, optional
            Data to check for nans
        name : str, optional
            Name to add to error, if one is thrown

        Raises
        ------
        ValueError
            If nan is found in data
        """
        if torch.isnan(data).any():
            raise ValueError("sparsecoding error: nan in %s." % (name))
        if torch.isnan(self.dictionary).any():
            raise ValueError("sparsecoding error: nan in dictionary.")

    def set_dictionary(self, dictionary):
        """Set model dictionary to passed dictionary

        Parameters
        ----------
        dictionary : array-like, shape [n_features, n_basis]
            Dictionary to set default dictionary to
        """
        self.dictionary = dictionary.to(self.device)
        self.n_features = dictionary.shape[0]
        self.n_basis = dictionary.shape[1]

    def load_dictionary(self, filename):
        """Load dictionary from pkl dump

        Parameters
        ----------
        filename : str
            File to load dictionary from
        """
        file = open(filename, "rb")
        dictionary = pkl.load(file)
        dictionary = torch.tensor(dictionary.astype(np.float32))
        file.close()
        self.set_dictionary(dictionary)

    def save_dictionary(self, filename):
        """Save dictionary to pkl dump

        Parameters
        ----------
        filename : str
            File to save current dictionary to
        """
        filehandler = open(filename, "wb")
        pkl.dump(self.get_numpy_dictionary(), filehandler)
        filehandler.close()


class Hierarchical(torch.nn.Module):
    """Class for hierarchical sparse coding.

    Layer x_{n+1} is recursively defined as:
        x_{n+1} := Phi_n x_n + a_n,
    where:
        Phi_n is a basis set (with unit norm),
        a_n has a sparse prior.

    The `a_n`s can be thought of the errors or residuals in a predictive coding
        model, or the sparse weights at each layer in a generative model.

    Parameters
    ----------
    priors : List[Prior]
        Prior on weights for each layer.
    """

    def __init__(
        self,
        priors: List[Prior],
    ):
        self.priors = priors

        self.dims = [prior.D for prior in priors]

        self.bases = [
            torch.normal(
                mean=torch.zeros((self.dims[n], self.dims[n + 1]), dtype=torch.float32),
                std=torch.ones((self.dims[n], self.dims[n + 1]), dtype=torch.float32),
            )
            for n in range(self.L - 1)
        ]
        # Normalize / project onto unit sphere
        self.bases = list(map(
            lambda basis: basis / torch.norm(basis, dim=1, keepdim=True),
            self.bases
        ))
        for basis in self.bases:
            basis.requires_grad = True

    @property
    def L(self):
        """Number of layers in the generative model.
        """
        return len(self.dims)

    def generate(
        bases: List[torch.Tensor],
        weights: List[torch.Tensor],
    ):
        """Run the generative model forward.

        Parameters
        ----------
        bases : List[Tensor], length L - 1, shape [D_i, D_{i+1}]
            Basis functions to transform between layers.
        weights : List[Tensor], length L, shape [N, D_i]
            Weights at each layer.

        Returns
        -------
        data : Tensor, shape [N, D_L]
            Generated data from the given weights and bases.
        """
        Hierarchical._check_bases_weights(bases, weights)

        x_i = weights[0]
        for (basis, weight) in zip(bases, weights[1:]):
            x_i = torch.einsum(
                "ni,ij->nj",
                x_i,
                basis,
            ) + weight

        return x_i

    def sample(
        n_samples: int,
        priors: List[Prior],
        bases: List[torch.Tensor],
    ):
        """Sample from the generative model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        bases : List[Tensor], length L - 1, shape [D_i, D_{i+1}]
            Basis functions to transform between layers.
        priors : List[Prior], length L
            Priors for the weights at each layer.

        Returns
        -------
        data : Tensor, shape [N, D_L]
            Sampled data using the given priors and bases.
        """
        if n_samples < 0:
            raise ValueError(f"`n_samples` must be non-negative, got {n_samples}.")
        Hierarchical._check_bases_priors(bases, priors)

        weights = list(map(
            lambda prior: prior.sample(n_samples),
            priors,
        ))
        return Hierarchical.generate(bases, weights)

    def log_prob(
        data: torch.Tensor,
        bases: List[torch.Tensor],
        priors: List[Prior],
        weights: List[torch.Tensor],
    ):
        """Compute the log-probability of the `data` under the generative model.

        Parameters
        ----------
        data : Tensor, shape [N, D_L]
            Data to get the log-probability of.
        bases : List[Tensor], length L - 1, shape [D_i, D_{i+1}]
            Basis functions to transform between layers.
        priors: List[Prior], length L
            Priors on the weights at each layer.
        weights : List[Tensor], length L - 1, shape [N, D_i]
            Weights for the basis functions at each layer,
            EXCEPT for the bottom layer, where the weights are
            implicitly defined as the difference between the data and the
            generated predictions from the previous layers.

        Returns
        -------
        log_prob : Tensor, shape [N]
            Log-probabilities of the data under the generative model.
        """
        Hierarchical._check_bases_priors(bases, priors)
        Hierarchical._check_bases_weights(
            bases,
            # Need to add dummy weights since last layer weights
            # are not specified in the input.
            weights + [torch.zeros((weights[0].shape[0], bases[-1].shape[1]))]
        )

        # First layer, no basis
        x_i = weights[0]
        log_prob = priors[0].log_prob(weights[0])

        # Middle layers
        for (prior, basis, weight) in zip(priors[1:-1], bases[:-1], weights[1:]):
            x_i = torch.einsum(
                "ni,ij->nj",
                x_i,  # [N, D_i]
                basis,  # [D_i, D_{i+1}]
            ) + weight
            log_prob = log_prob + prior.log_prob(weight)

        # Last layer, implicit weights calculated from the data
        x_l = torch.einsum(
            "ni,ij->nj",
            x_i,  # [N, D_i]
            bases[-1],  # [D_i, D_{i+1}]
        )
        weight_l = data - x_l
        log_prob = log_prob + priors[-1].log_prob(weight_l)

        return log_prob

    def infer_weights(
        self,
        data: torch.Tensor,
        n_iter: int = 1000,
        learning_rate: float = 0.1,
        return_history: bool = False,
        initial_weights: Optional[List[torch.Tensor]] = None,
    ):
        """Infer weights for the input `data` to maximize the log-likelihood.

        Performs gradient descent with Adam.

        Parameters
        ----------
        data : Tensor, shape [N, D_L]
            Data to be generated.
        n_iter : int
            Number of iterations of gradient descent to perform.
        learning_rate : float
            Learning rate for the optimizer.
        return_history : bool
            Flag to return the history of the inferred weights during inference.
        initial_weights : optional, List[Tensor], length L - 1, shape [N, D_i]
            If provided, the initial weights to start inference from.
            Otherwise, weights are set to 0.

        Returns
        -------
        weights : List[Tensor], length L, shape [N, D_i]
            Inferred weights for each layer.
        weights_history : optional, List[Tensor], length L, shape [n_iter + 1, N, D_i]
            Returned if `return_history`. The inferred weights for each layer
            throughout inference.
        """
        N = data.shape[0]

        if initial_weights is None:
            top_weights = [
                torch.zeros((N, self.dims[i]), dtype=torch.float32, requires_grad=True)
                for i in range(self.L - 1)
            ]
        else:
            top_weights = initial_weights
            for weight in top_weights:
                weight.requires_grad = True

        bases = list(map(lambda basis: basis.detach(), self.bases))

        if return_history:
            with torch.no_grad():
                bottom_weights = Hierarchical._compute_bottom_weights(data, bases, top_weights)
                weights = top_weights + [bottom_weights]
                weights_history = list(map(
                    lambda weight: [weight.detach().clone()],
                    weights,
                ))

        optimizer = torch.optim.Adam(top_weights, lr=learning_rate)
        for _ in range(n_iter):
            log_prob = Hierarchical.log_prob(data, bases, self.priors, top_weights)
            optimizer.zero_grad()
            (-torch.mean(log_prob)).backward()
            optimizer.step()

            if return_history:
                with torch.no_grad():
                    bottom_weights = Hierarchical._compute_bottom_weights(data, bases, top_weights)
                    weights = top_weights + [bottom_weights]
                    for (weight_history, weight) in zip(weights_history, weights):
                        weight_history.append(weight.detach().clone())

        top_weights = list(map(lambda weight: weight.detach(), top_weights))
        bottom_weights = Hierarchical._compute_bottom_weights(data, bases, top_weights)
        weights = top_weights + [bottom_weights]

        if not return_history:
            return weights
        else:
            for layer in range(self.L):
                weight_history[layer] = torch.stack(weight_history[layer], dim=0)
            return weights, weights_history

    def infer_weights_local(
        self,
        data: torch.Tensor,
        n_iter: int = 100000,
        learning_rate: float = 0.0001,
        return_history_interval: Optional[int] = None,
        initial_weights: Optional[List[torch.Tensor]] = None,
    ):
        """Infer weights for the input `data` to maximize the log-likelihood.
        However, information flow is constrained to be between adjacent layers.

        Performs gradient descent with Adam.

        Parameters
        ----------
        data : Tensor, shape [N, D_L]
            Data to be generated.
        n_iter : int
            Number of iterations of gradient descent to perform.
        learning_rate : float
            Learning rate for the optimizer.
        return_history_interval : optional, int
            If set, inferred weights during inference will be saved
            at this frequency.
        initial_weights : optional, List[Tensor], length L - 1, shape [N, D_i]
            If provided, the initial weights to start inference from.
            Otherwise, weights are set to 0.

        Returns
        -------
        weights : List[Tensor], length L, shape [N, D_i]
            Inferred weights for each layer.
        weights_history : optional, List[Tensor], length L, shape [n_iter + 1, N, D_i]
            Returned if `return_history`. The inferred weights for each layer
            throughout inference.
        """
        N = data.shape[0]

        if initial_weights is None:
            top_weights = [
                torch.zeros((N, self.dims[i]), dtype=torch.float32, requires_grad=True)
                for i in range(self.L - 1)
            ]
        else:
            top_weights = initial_weights
            for weight in top_weights:
                weight.requires_grad = True

        bases = list(map(lambda basis: basis.detach(), self.bases))

        if return_history_interval:
            with torch.no_grad():
                bottom_weights = Hierarchical._compute_bottom_weights(data, bases, top_weights)
                weights = top_weights + [bottom_weights]
                weights_history = [[weight.detach()] for weight in weights]

        optimizer = torch.optim.Adam(top_weights, lr=learning_rate)
        for it in range(n_iter):
            # Generate data under current weights (with no gradient) to get targets.
            xs_ng = []
            with torch.no_grad():
                xs_ng.append(top_weights[0].detach())
                for (basis, weight) in zip(bases[:-1], top_weights[1:]):
                    xs_ng.append(xs_ng[-1] @ basis + weight.detach())
                xs_ng.append(data)
            
            # Get log-probability for Layer 1.
            weight_1 = top_weights[0]
            x_ng_below = xs_ng[1]
            basis_to_below = bases[0]
            prior = self.priors[0]
            prior_below = self.priors[1]
            log_prob = (
                prior.log_prob(weight_1)
                + prior_below.log_prob(x_ng_below - weight_1 @ basis_to_below)
            )

            # Get log-probabilities for Layers 2 through L.
            for layer in range(2, self.L):
                weight = top_weights[layer - 1]
                
                x_ng_above = xs_ng[layer - 2]
                basis_from_above = bases[layer - 2]
                
                x_ng_below = xs_ng[layer]
                basis_to_below = bases[layer - 1]
                
                prior = self.priors[layer - 1]
                prior_below = self.priors[layer]
                log_prob += (
                    prior.log_prob(weight)
                    + prior_below.log_prob(x_ng_below - (x_ng_above @ basis_from_above + weight) @ basis_to_below)
                )
            
            optimizer.zero_grad()
            (-torch.mean(log_prob)).backward()
            optimizer.step()
            
            if return_history_interval and it % return_history_interval == 0:
                with torch.no_grad():
                    bottom_weights = Hierarchical._compute_bottom_weights(data, bases, top_weights)
                    weights = top_weights + [bottom_weights]
                    for (weight_history, weight) in zip(weights_history, weights):
                        weight_history.append(weight.detach().clone())

        top_weights = list(map(lambda weight: weight.detach(), top_weights))
        bottom_weights = Hierarchical._compute_bottom_weights(data, bases, top_weights)
        weights = top_weights + [bottom_weights]

        if not return_history_interval:
            return weights
        else:
            for layer in range(self.L):
                weights_history[layer] = torch.stack(weights_history[layer], dim=0)
            return weights, weights_history

    def learn_bases(
        self,
        data: torch.Tensor,
        n_iter: int = 125,
        learning_rate: float = 0.01,
        inference_n_iter: int = 25,
        inference_learning_rate: float = 0.01,
        return_history: bool = False,
    ):
        """Update the bases to maximize the log-likelihood of `data`.

        In each iteration, we first infer the weights under the current basis functions,
        and then we update the bases with those weights fixed.

        Uses gradient descent with Adam.

        Parameters
        ----------
        data : Tensor, shape [N, D_L]
            Data to be generated.
        n_iter : int
            Number of iterations of gradient descent to perform.
        learning_rate : float
            Step-size for learning the bases.
        inference_n_iter : int
            Number of iterations of gradient descent
            to perform during weight inference.
        inference_learning_rate : float
            Step-size for inferring the weights.
        return_history : bool
            Flag to return the history of the learned bases during inference.

        Returns
        -------
        bases_history : optional, List[Tensor], length L - 1, shape [n_iter + 1, D_i, D_{i+1}]
            Returned if `return_history`. The learned bases throughout training.
        """    
        N = data.shape[0]

        if return_history:
            bases_history = list(map(
                lambda basis: [basis.detach().clone()],
                self.bases,
            ))

        bases_optimizer = torch.optim.Adam(self.bases, lr=learning_rate)

        top_weights = [
            torch.zeros((N, self.dims[i]), dtype=torch.float32, requires_grad=True)
            for i in range(self.L - 1)
        ]
        weights_optimizer = torch.optim.Adam(top_weights, lr=inference_learning_rate)
        
        for _ in tqdm(range(n_iter)):
            # Infer weights under the current bases.
            bases = list(map(lambda basis: basis.detach(), self.bases))
            for _ in range(inference_n_iter):
                log_prob = Hierarchical.log_prob(data, bases, self.priors, top_weights)
                weights_optimizer.zero_grad()
                (-torch.mean(log_prob)).backward()
                weights_optimizer.step()

            # Update bases from the current weights.
            weights = list(map(lambda weight: weight.detach(), top_weights))
            log_prob = Hierarchical.log_prob(data, self.bases, self.priors, weights)
            bases_optimizer.zero_grad()
            (-torch.mean(log_prob)).backward()
            bases_optimizer.step()

            # Normalize basis elements (project them back onto the unit sphere).
            with torch.no_grad():
                for basis in self.bases:
                    basis /= torch.norm(basis, dim=1, keepdim=True)

                if return_history:
                    for (basis_history, basis) in zip(bases_history, self.bases):
                        basis_history.append(basis.detach().clone())

        if return_history:
            for layer in range(self.L - 1):
                bases_history[layer] = torch.stack(bases_history[layer], dim=0)
            return bases_history

    def inspect_bases(self):
        """Runs the generative model forward for each basis element.

        This allows visual inspection of what each basis function represents
        at the final (bottom) layer.

        Returns
        -------
        bases_viz : List[List[Tensor]], shape [D_L]
            Visualizations of the basis functions at each layer.
        """
        bases_viz = []
        for layer in range(self.L - 1):
            layer_bases_viz = []
            for basis_fn in range(self.bases[layer].shape[0]):
                weights = [torch.zeros((1, dim)) for dim in self.dims]
                weights[layer][0, basis_fn] = 1.
                layer_bases_viz.append(Hierarchical.generate(self.bases, weights)[0])
            bases_viz.append(layer_bases_viz)
        return bases_viz

    def _check_bases_weights(bases, weights):
        """Check bases and weights for shape compatibility.
        """
        if len(weights) != len(bases) + 1:
            raise ValueError(
                f"Must have exactly one more weight than basis "
                f"(`L` layers and `L-1` bases to transform between them), "
                f"got {len(weights)} weights and {len(bases)} bases."
            )
        if not all([
            weights[i].shape[0] == weights[0].shape[0]
            for i in range(1, len(weights))
        ]):
            raise ValueError(
                "Weight tensors must all have the same size in the 0-th dimension."
                "This is the size of the data to generate."
            )
        for (layer, (basis_i, basis_j)) in enumerate(zip(bases[:-1], bases[1:])):
            if basis_i.shape[1] != basis_j.shape[0]:
                raise ValueError(
                    f"Basis between layer {layer} and layer {layer+1} "
                    f"produces weights of dimension {basis_i.shape[1]} "
                    f"for layer {layer+1} but "
                    f"basis between layer {layer+1} and layer {layer+2} "
                    f"expects {basis_j.shape[0]} weights for layer {layer+1}."
                )
        for (layer, (basis, weight)) in enumerate(zip(bases, weights[:-1])):
            if basis.shape[0] != weight.shape[1]:
                raise ValueError(
                    f"Basis between layer {layer} and layer {layer+1} "
                    f"expects {basis.shape[0]} weights for {layer}, "
                    f"but {weight.shape[1]} weights "
                    f"are provided for layer {layer}."
                )
        if bases[-1].shape[1] != weights[-1].shape[1]:
            raise ValueError(
                f"The final basis outputs data with dimension {bases[-1].shape[1]}, "
                f"but final layer weights have dimension {weights[-1].shape[1]}."
            )

    def _check_bases_priors(bases, priors):
        """Check bases and priors for shape compatibility.
        """
        if len(priors) != len(bases) + 1:
            raise ValueError(
                f"Must have exactly one more prior than basis "
                f"(`L` layers and `L-1` bases to transform between them), "
                f"got {len(priors)} priors and {len(bases)} bases."
            )
        for (layer, (basis_i, basis_j)) in enumerate(zip(bases[:-1], bases[1:])):
            if basis_i.shape[1] != basis_j.shape[0]:
                raise ValueError(
                    f"Basis between layer {layer} and layer {layer+1} "
                    f"produces priors of dimension {basis_i.shape[1]} "
                    f"for layer {layer+1} but "
                    f"basis between layer {layer+1} and layer {layer+2} "
                    f"expects {basis_j.shape[0]} priors for layer {layer+1}."
                )
        for (layer, (basis, prior)) in enumerate(zip(bases, priors[:-1])):
            if basis.shape[0] != prior.D:
                raise ValueError(
                    f"Basis between layer {layer} and layer {layer+1} "
                    f"expects {basis.shape[0]} weights for {layer}, "
                    f"but the prior for layer {layer} is over {prior.D} weights."
                )
        if bases[-1].shape[1] != priors[-1].D:
            raise ValueError(
                f"The final basis outputs data with dimension {bases[-1].shape[1]}, "
                f"but final layer prior is over {priors[-1].D} weights."
            )

    def _compute_bottom_weights(
        data: torch.Tensor,
        bases: List[torch.Tensor],
        top_weights: List[torch.Tensor],
    ):
        """Compute the bottom-layer weights for `data`, given weights for all the other layers.

        Parameters
        ----------
        data : Tensor, shape [N, D_L]
            Data to be generated.
        bases : List[Tensor], length L - 1, shape [D_i, D_{i+1}]
            Basis functions to transform between layers.
        weights : List[Tensor], length L - 1, shape [N, D_i]
            Weights at the top `L - 1` layers.
        """
        weights = top_weights + [torch.zeros_like(data)]
        Hierarchical._check_bases_weights(bases, weights)
        bottom_weights = data - Hierarchical.generate(bases, weights)
        return bottom_weights


class SimulSparseCoding(SparseCoding):
    def __init__(self, inference_method, n_basis, n_features, sparsity_penalty,
                 inf_rate=1, learn_rate=1, time_step=1, t_max=1000,
                 device=None):
        super().__init__(inference_method, n_basis, n_features,
                         sparsity_penalty)
        self.inf_rate = inf_rate
        self.learn_rate = learn_rate
        self.time_step = time_step
        self.t_max = t_max
        self.device = device

    def simultaneous_update(self, data, batch_size):
        losses = []
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
        iterloader = iter(dataloader)
        a = torch.rand((batch_size, self.n_basis)).to(self.device)

        for t in range(int(self.t_max/self.time_step)):
            try:
                batch = next(iterloader)
            except StopIteration:
                dataloader = DataLoader(data, batch_size=batch_size,
                                        shuffle=True)
                iterloader = iter(dataloader)
                batch = next(iterloader)

            # update coefficients
            residual = batch - (self.dictionary@a.t()).t()
            a -= (self.time_step/self.inf_rate) * self.inference_method.grad(residual, self.dictionary, a)

            # update dictionary
            self.dictionary -= (self.time_step/self.learn_rate) * self.compute_grad_dict(batch, a)

            # normalize dictionary
            self.normalize_dictionary()

            # compute current loss
            loss = self.compute_loss(batch, a)

            losses.append(loss)
        return np.asarray(losses)
