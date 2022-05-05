import numpy as np
import torch


class BarsDataset:
    '''Generate bars dictionary and dataset'''
    
    def __init__(self,patch_size,n_samples,coefficient_treshold,device=torch.device('cpu')):
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.coefficient_treshold = coefficient_treshold
        self.device = device
        self.n_basis = 2*patch_size
        
        # generate dictionary
        self.generate_dictionary()
        
        # generate dataset
        self.generate_data()
        
        
    def generate_dictionary(self):
        '''Generate dictionary consisting of all combinations of 
        horizontal/vertical bars
        '''
        bars = np.zeros([self.patch_size,self.patch_size,self.n_basis],dtype=np.float32)
        c = 0
        for i in range(self.patch_size):
            bars[i,:,c] = 1.
            c += 1

        for i in range(self.patch_size):
            bars[:,i,c] = 1.
            c += 1
        self.dictionary = torch.tensor(bars.reshape([self.patch_size**2,self.n_basis])).to(self.device)
        
        
    def generate_data(self):
        '''Generate data consisting of all combinations of 
        horizontal/vertical bars
        '''
        torch.manual_seed(1997)
        # compute coefficients
        self.coefficients = torch.rand([self.n_samples,self.n_basis]).to(self.device)
        self.coefficients[self.coefficients < self.coefficient_treshold] = 0
        # generate dataset
        self.data = (self.dictionary@self.coefficients.t()).t()
        