import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle as pkl


class SparseCoding(torch.nn.Module):
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
    check_for_dictionary_nan : bool, default=False
        Flag to check for nans in the dictionary after gradient
        updates and normalizations. Raises ValueError if nan
        found
    """

    def __init__(self, inference_method, n_basis, n_features,
                 sparsity_penalty=0.2, device=None, check_for_dictionary_nan=False, **kwargs):
        super(SparseCoding, self).__init__()
        self.inference_method = inference_method
        self.n_basis = n_basis
        self.n_features = n_features
        self.check_for_dictionary_nan = check_for_dictionary_nan
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
        if self.check_for_dictionary_nan:
            self.checknan()

    def normalize_dictionary(self):
        """Normalize columns of dictionary matrix to unit norm."""
        self.dictionary = self.dictionary.div_(self.dictionary.norm(p=2, dim=0))
        if self.check_for_dictionary_nan:
            self.checknan()

    def infer(self, data):
        return self.inference_method.infer(data, self.dictionary)

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
        for _ in range(n_epoch):
            loss = 0.0
            for batch in dataloader:
                # infer coefficients
                a = self.infer(batch)
                # update dictionary
                self.update_dictionary(batch, a)
                # normalize dictionary
                self.normalize_dictionary()
                # compute current loss
                loss += self.compute_loss(batch, a)
            losses.append(loss/len(dataloader))
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


class TopographicSparseCoding(SparseCoding):
    """Class for learning a topographic sparse codes

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
    stride : int
        Stride of neighborhoods
    kernel_size : int
        Size of neighborhoods
    dictionary_lr : float, default=1e-2
        Learning rate of dictionary update
    device : torch.device, default=torch.device("cpu")
        Which device to utilize
    check_for_dictionary_nan : bool, default=False
        Flag to check for nans in the dictionary after gradient
        updates and normalizations. Raises ValueError if nan
        found
    n_iterations : int, default=1000
        Number of steps to run forward Euler during inference
    step_size : float, default=0.01
        Forward Eular step size
    """
    def __init__(self, n_basis, n_features, stride, kernel_size, sparsity_penalty=0.2, device=None,
                 check_for_dictionary_nan=False, n_iterations=1000, step_size=0.01, **kwargs):
        # Initialize base class
        super().__init__(
            inference_method=None,  # not used in subclass
            n_basis=n_basis,
            n_features=n_features,
            sparsity_penalty=sparsity_penalty,
            device=device,
            check_for_dictionary_nan=check_for_dictionary_nan,
            **kwargs
        )
        self.stride = stride
        self.kernel_size = kernel_size
        self.topographic_projection = self.build_topographic_projection().to(self.device)
        self.n_iterations = n_iterations
        self.step_size = step_size

    def infer(self, data):
        """Inference method. Currently uses topographic LCA.

        Parameters
        ----------
        data : array-like (batch_size, n_features)
            data to infer sparse code
        """
        return self.topographic_LCA(data)

    def topographic_LCA(self, x):
        """Topographic LCA inference method

        Parameters
        ----------
        x : array-like (batch_size, n_features)
            data to infer sparse code
        """
        G = self.dictionary.t() @ self.dictionary - torch.eye(self.dictionary.shape[1], device=self.device)
        b = (self.dictionary.t() @ x.t()).t()

        u = torch.zeros_like(b).to(x.device)
        for _ in range(self.n_iterations):
            a = self.compute_active_coefficients(u)
            du = b - u - a @ G
            u = u + self.step_size * du
        return self.compute_active_coefficients(u)

    def compute_active_coefficients(self, u):
        """Threshold nonlinearity for topographic LCA

        Parameters
        ----------
        u : array-like (batch_size, n_basis)
            sparse coefficient subthreshold values
        """
        eps = 0.001
        group_norm = torch.sqrt(torch.square(u) @ self.topographic_projection.T + eps)  # [B,G]
        group_norm_reshape = group_norm @ self.topographic_projection  # [B,N]
        mask = (group_norm_reshape > self.sparsity_penalty).float()
        a = mask * (group_norm_reshape - self.sparsity_penalty) * (u / group_norm_reshape)
        return a

    def _build_topographic_projection(self):
        """Builds a matrix W of shape [n_groups, n*n] for topographic projection"""
        n = int(self.n_basis**0.5)
        r = self.kernel_size
        indices = []
        for i in range(0, n - r + 1, self.stride):
            for j in range(0, n - r + 1, self.stride):
                mask = torch.zeros(n, n)
                mask[i:i+r, j:j+r] = 1
                indices.append(mask.view(-1))
        W = torch.stack(indices)
        return W  # shape [n_groups, n*n]
