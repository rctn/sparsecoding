import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle as pkl


class SparseCoding(torch.nn.Module):

    def __init__(
        self,
        inference_method,
        n_basis,
        n_features,
        sparsity_penalty=0.2,
        device=None,
        check_for_dictionary_nan=False,
        **kwargs,
    ):
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
        self.dictionary = torch.add(self.dictionary, self.dictionary_lr * dictionary_grad)
        if self.check_for_dictionary_nan:
            self.checknan()

    def normalize_dictionary(self):
        """Normalize columns of dictionary matrix to unit norm."""
        self.dictionary = self.dictionary.div_(self.dictionary.norm(p=2, dim=0))
        if self.check_for_dictionary_nan:
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
        for _ in range(n_epoch):
            loss = 0.0
            for batch in dataloader:
                # infer coefficients
                a = self.inference_method.infer(batch, self.dictionary)
                # update dictionary
                self.update_dictionary(batch, a)
                # normalize dictionary
                self.normalize_dictionary()
                # compute current loss
                loss += self.compute_loss(batch, a)
            losses.append(loss / len(dataloader))
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

        MSE_loss = torch.square(torch.linalg.vector_norm(data - torch.mm(self.dictionary, a.t()).t(), dim=1))
        sparsity_loss = self.sparsity_penalty * torch.abs(a).sum(dim=1)
        total_loss = torch.sum(MSE_loss + sparsity_loss)
        return total_loss.item() / batch_size

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
