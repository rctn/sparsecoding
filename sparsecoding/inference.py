import numpy as np
import torch

class InferenceMethod:
    '''Base class for inference method.'''
    
    def __init__(self,solver):
        '''
        Parameters
        ----------
        '''
        self.solver = solver
        
        
    def initialize(self,a):
        '''
        Define initial coefficients.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        raise NotImplementedError
    
    
    def grad(self):
        '''
        Compute the gradient step.
        
        Parameters
        ----------
        
        Returns
        -------
        '''
        raise NotImplementedError
    
    
    def infer(self,dictionary,data):
        '''
        Infer the coefficients given a dataset and dictionary.
        
        Parameters
        ----------
        dictionary : array like (n_features,n_basis)
            
        data : array like (n_samples,n_features)
            
        Returns
        -------
        coefficients : (n_samples,n_basis)
        '''
        raise NotImplementedError
        
    @staticmethod
    def checknan(data=torch.tensor(0), name='data'):
        '''
        Check for nan values in data.
        
        Parameters
        ----------
        data : torch.tensor default=1
            data to check for nans
        name : string
            name to add to error, if one is thrown
        '''
        if torch.isnan(data).any():
            raise ValueError('InferenceMethod error: nan in %s.'%(name))
            

        
        
class LCA(InferenceMethod):
    def __init__(self, n_iter = 100, coeff_lr=1e-3, threshold=0.1, stop_early=False, epsilon=1e-2, solver = None):
        '''
        Method implemented according locally competative algorithm (Rozell 2008) 
        with the ideal soft thresholding function.
        
        Parameters
        ----------
        n_iter : scalar (1,) default=100
            number of iterations to run
        coeff_lr : scalar (1,) default=1e-3
            update rate of coefficient dynamics
        threshold : scalar (1,) default=0.1
            threshold for non-linearity
        stop_early : boolean (1,) default=False
            stops dynamics early based on change in coefficents
        epsilon : scalar (1,) default=1e-2
            only used if stop_early True, specifies criteria to stop dynamics
        solver : default=None
        '''
        super().__init__(solver)
        self.threshold = threshold
        self.coeff_lr = coeff_lr
        self.stop_early = stop_early
        self.epsilon = epsilon
        self.n_iter = n_iter
        
        
    def threshold_nonlinearity(self,u):
        """
        Soft threshhold function according to Rozell 2008
        
        Parameters
        ----------
        u - torch.tensor (batch_size,n_basis)
            membrane potentials
        
        Returns
        -------
        a - torch.tensor (batch_size,n_basis)
            activations
        """
        a = (torch.abs(u) - self.threshold).clamp(min=0.)
        a = torch.sign(u)*a
        return a
    
    
    def grad(self,b,G,u,a):
        '''
        Compute the gradient step on membrane potentials
        
        Parameters
        ----------
        b : scalar (batch_size,n_coefficients)
            driver signal for coefficients 
        G : scalar (n_coefficients,n_coefficients)
            inhibition matrix 
        a : scalar (batch_size,n_coefficients)
            currently active coefficients 
            
        Returns
        -------
        du : scalar (batch_size,n_coefficients)
            grad of membrane potentials
        '''
        du = b-u-(G@a.t()).t()
        return du
    
             
    def infer(self, data, dictionary):
        """
        Infer coefficients for each image in data using dict elements dictionary.
        Method implemented according locally competative algorithm (Rozell 2008)

        Parameters
        ----------
        dictionary : array like (n_features,n_basis)
            
        data : array like (n_samples,n_features)
            
        Returns
        -------
        coefficients : (n_samples,n_basis)
        """
        batch_size, n_features = data.shape
        n_features, n_basis = dictionary.shape
        device = dictionary.device

        # initialize
        u = torch.zeros((batch_size, n_basis)).to(device)

        b = (dictionary.t()@data.t()).t()
        G = dictionary.t()@dictionary-torch.eye(n_basis).to(device)
        for i in range(self.n_iter):
            if self.stop_early:
                old_u = u.clone().detach()
                
            a = self.threshold_nonlinearity(u)
            du = self.grad(b,G,u,a)
            u = u + self.coeff_lr*du
            
            if self.stop_early:
                if  torch.linalg.norm(old_u - u)/torch.linalg.norm(old_u) < self.epsilon:
                    break 
            self.checknan(u,'coefficients')
            
        coefficients = self.threshold_nonlinearity(u)
        return coefficients
  