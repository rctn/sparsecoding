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
        super().__init__(solver)
        self.threshold = threshold
        self.coeff_lr = coeff_lr
        self.stop_early = stop_early
        self.epsilon = epsilon
        self.n_iter = n_iter
        
        
    def threshold_nonlinearity(self,u):
        """
        Soft threshhold function according to Rozell 2008
        
        Parameters:
        u - torch.tensor (batch_size,n_basis)
            membrane potentials
        ---
        Returns: 
        a - torch.tensor (batch_size,n_basis)
            activations
        """
        a = (torch.abs(u) - self.threshold).clamp(min=0.)
        a = torch.sign(u)*a
        return a
        
             
    def infer(self, data, dictionary):
        """
        Infer coefficients for each image in data using dict elements dictionary.
        Method implemented according locally competative algorithm (Rozell 2008)

        Parameters
        ----------
        data : array-like (batch_size, n_features)
            
        dictionary : array-like, (n_features, n_basis)
       
        Returns
        -------
        coefficients : array-like (batch_size, n_basis)
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
            du = b-u-(G@a.t()).t()
            u = u + self.coeff_lr*du
            
            if self.stop_early:
                if (old_u - u).norm(p=2).sum() < self.eps:
                    break 
            self.checknan(u,'coefficients')
            
        coefficients = self.threshold_nonlinearity(u)
        return coefficients
    
    
#     def ista(self,I):
#         """
#         Infer coefficients for each image in I made up of dict elements D
#         Method implemented according to 1996 Olshausen and Field
#         ---
#         Parameters:
#         I - torch.tensor (batch_size,n)
#             input images
#         ---
#         Returns:
#         a - scalar (batch_size,n_basis)
#             sparse coefficients
#         """
#         batch_size = I.size(0)

#         # initialize
#         a = torch.zeros((batch_size,self.n_basis)).to(self.device)
#         residual = I - torch.mm(self.D,a.t()).t()
       
#         for i in range(self.n_itr):
            
#             # update coefficients
#             a = a + self.a_lr*((self.D.t()@residual.t()).t() - self.lmbda*torch.sign(a))
            
#             # check stopping criteria
#             if self.stop_early:
#                 residual_new = I - torch.mm(self.D,a.t()).t()
#                 if (residual_new - residual).norm(p=2).sum() < self.eps:
#                     break
#                 residual = residual_new
#             else:
#                 residual = I - torch.mm(self.D,a.t()).t()    
#             # check for nans
#             self.checknan(a,'coefficients')
#         return a
    
    
#     def lca(self,I):
#         """
#         Infer coefficients for each image in I using dict elements self.D
#         Method implemented according locally competative algorithm (Rozell 2008)
#         ---
#         Parameters:
#         I - torch.tensor (batch_size,n)
#             input images
#         ---
#         Returns:
#         a - torch.tensor (batch_size,n_basis)
#             sparse coefficients
#         """
#         batch_size = I.size(0)

#         # initialize
#         u = torch.zeros((batch_size,self.n_basis)).to(self.device)
#         a = torch.zeros((batch_size,self.n_basis)).to(self.device)

#         b = (self.D.t()@I.t()).t()
#         G = self.D.t()@self.D-torch.eye(self.n_basis).to(self.device)
#         for i in range(self.n_itr):
#             if self.stop_early:
#                 old_u = u.clone().detach()
                
#             a = self.threshold_nonlinearity(u)
#             du = b-u-(G@a.t()).t()
#             u = u + self.a_lr*du
            
#             if self.stop_early:
#                 if (old_u - u).norm(p=2).sum() < self.eps:
#                     break 
#             self.checknan(u,'coefficients')
            
#         return self.threshold_nonlinearity(u)
                
                
#     def Tsoft(self,u):
#         """
#         Soft threshhold function according to Rozell 2008
        
#         Parameters:
#         u - torch.tensor (batch_size,n_basis)
#             membrane potentials
#         ---
#         Returns: 
#         a - torch.tensor (batch_size,n_basis)
#             activations
#         """
#         a = (torch.abs(u) - self.thresh).clamp(min=0.)
#         a = torch.sign(u)*a
#         return a
        