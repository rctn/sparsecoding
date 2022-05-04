import numpy as np
import torch 
from torch.utils.data import DataLoader
import pickle as pkl


class SparseCoding(torch.nn.Module):
    
    def __init__(self, inference_method, n_basis, n_features, sparsity_penalty=0.2, device=None,**kwargs):
        '''
        Class for learning a sparse code via dictionary learning
        
        Parameters
        ----------
        inference_method : sparsecoding.InferenceMethod
            method for inferring coefficients for each image given the dictionary
        n_basis : scalar (1,)
            number of basis functions in dictionary
        n_features : scalar (1,)
            number of features in data
        sparsity_penalty : scalar (1,) default=0.2
            sparsity penalty
        dictionary_lr : scalar (1,) default=1e-2
            learning rate of dictionary update
        device : torch.device defualt=torch.device("cpu")
            which device to utilize
        '''
        
        super(SparseCoding, self).__init__()
        self.inference_method = inference_method
        self.n_basis = n_basis
        self.n_features = n_features
        self.device = torch.device("cpu") if device is None else device
        self.dictionary_lr = torch.tensor(np.float32(kwargs.pop('dictionary_lr',1e-2))).to(self.device)            
        self.sparsity_penalty = torch.tensor(np.float32(sparsity_penalty)).to(self.device)
        self.dictionary = torch.randn((self.n_features, self.n_basis)).to(self.device)
        self.normalize_dictionary()
   

    def update_dictionary(self,data,a):
        '''
        Compute gradient of loss function w.r.t. dictionary elements, and update 

        Parameters
        ----------
        data : torch.tensor (batch_size,n_features)
            input data
        a : torch.tensor (batch_size,n_basis)
            already-inferred coefficients

        Returns
        -------

        '''
        residual = data - torch.mm(self.dictionary,a.t()).t()
        ddictionary = torch.mm(residual.t(),a)
        self.dictionary = torch.add(self.dictionary, self.dictionary_lr*ddictionary)
        self.checknan()
        
        
    def normalize_dictionary(self):
        '''
        Normalize columns of dictionary matrix to unit norm
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        self.dictionary = self.dictionary.div_(self.dictionary.norm(p=2,dim=0))
        self.checknan()
        
        
    def learn_dictionary(self,dataset,n_epoch,batch_size):
        '''
        Learn dictionary for n_epoch epochs

        Parameters
        ----------
        dataset : torch.utils.data.Dataset or scalar (n_samples,n_features)
            input dataset
        n_epoch : scalar (1,)
            number of iterations to learn dictionary
        batch_size : scalar (1,)
            batch size to learn on
        
        Returns
        -------
        scalar (nepoch,)
            losses of each batch
            
        '''
        losses = []
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        iterloader = iter(dataloader)
        for i in range(n_epoch):
            try:
                batch = next(iterloader)
            except StopIteration:
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                iterloader = iter(dataloader)
                batch = next(iterloader)
                
            # infer coefficients
            a = self.inference_method.infer(batch, self.dictionary)
            
            # update dictionary
            self.update_dictionary(batch,a)
            
            # normalize dictionary
            self.normalize_dictionary()
            
            # compute current loss
            loss = self.compute_loss(batch,a)
            
            losses.append(loss)
        return np.asarray(losses)
        
        
    def compute_loss(self,data,a):
        '''
        Compute loss given data and inferred coefficients
        
        Parameters
        ----------
        data : scalar (batch_size,n_features)
        
        a : scalar (batch_size,n_basis)
            inferred coefficients

        Returns
        -------
        float (1,) 
            loss
        '''
        batch_size,_ = data.shape
        
        MSE_loss = torch.square(torch.linalg.vector_norm(data-torch.mm(self.dictionary,a.t()).t(),dim=1)) 
        sparsity_loss = self.sparsity_penalty*torch.abs(a).sum(dim=1)
        total_loss = torch.sum(MSE_loss + sparsity_loss)
        return total_loss.item()/batch_size
        
        
    def get_numpy_dictionary(self):
        '''
        Returns dictionary as numpy array
        
        Parameters
        ----------
        
        Returns
        -------
        scalar (n_features,n_basis)
        '''
        return self.dictionary.cpu().detach().numpy()
    
    
    def checknan(self,data=torch.tensor(0),name='data'):
        '''
        Check for nan values in dictinary, or data
        
        Parameters
        ----------
        data : torch.tensor default=1
            data to check for nans
        name : string
            name to add to error, if one is thrown
        
        Returns
        -------

        '''
        if torch.isnan(data).any():
            raise ValueError('sparsecoding error: nan in %s.'%(name))
        if torch.isnan(self.dictionary).any():
            raise ValueError('sparsecoding error: nan in dictionary.')
            
            
    def load_dictionary(self,filename):
        '''
        Load dictionary from pkl dump
        
        Parameters
        ----------
        filename : string (1,)
            file to load dictionary from
        
        Returns
        -------

        '''
        file = open(filename,'rb')
        nD = pkl.load(file)
        file.close()
        self.dictionary = torch.tensor(nD.astype(np.float32)).to(self.device) 
        
            
    def save_dictionary(self,filename):
        '''
        Save dictionary to pkl dump

        Parameters
        ----------
        filename : string (1,)
            file to save current dictionary to

        Returns
        -------

        '''
        filehandler = open(filename,"wb")
        pkl.dump(self.get_numpy_dictionary(),filehandler)
        filehandler.close()
