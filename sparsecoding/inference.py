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
        Infer coefficients using provided dictionary
 
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
  
    
class Vanilla(InferenceMethod):
    def __init__(self, n_iter = 100, coeff_lr=1e-3, sparsity_penalty=0.2, stop_early=False, epsilon=1e-2, solver = None):
        '''
        Gradient descent with Euler's method on model in Olhausen & Feild (1997)
        with laplace prior over coefficients (corresponding to l-1 norm penalty).
        
        Parameters
        ----------
        n_iter : scalar (1,) default=100
            number of iterations to run
        coeff_lr : scalar (1,) default=1e-3
            update rate of coefficient dynamics
        sparsity_penalty : scalar (1,) default=0.2
            
        stop_early : boolean (1,) default=False
            stops dynamics early based on change in coefficents
        epsilon : scalar (1,) default=1e-2
            only used if stop_early True, specifies criteria to stop dynamics
        solver : default=None
        '''
        super().__init__(solver)
        self.coeff_lr = coeff_lr
        self.sparsity_penalty = sparsity_penalty
        self.stop_early = stop_early
        self.epsilon = epsilon
        self.n_iter = n_iter
    
    
    def grad(self,residual,dictionary,a):
        '''
        Compute the gradient step on coefficients
        
        Parameters
        ----------
        residual : scalar (batch_size,n_features)
            residual between reconstructed image and original
        dictionary : scalar (n_features,n_coefficients)
        
        a : scalar (batch_size,n_coefficients)
            
        Returns
        -------
        da : scalar (batch_size,n_coefficients)
            grad of membrane potentials
        '''
        da = (dictionary.t()@residual.t()).t() - self.sparsity_penalty*torch.sign(a)
        return da
    
             
    def infer(self, data, dictionary):
        """
        Infer coefficients using provided dictionary

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
        a = torch.rand((batch_size, n_basis)).to(device)
        
        residual = data - (dictionary@a.t()).t()
        for i in range(self.n_iter):
            
            if self.stop_early:
                old_a = a.clone().detach()
                
            da = self.grad(residual,dictionary,a)
            a = a + self.coeff_lr*da
            
            if self.stop_early:
                if  torch.linalg.norm(old_a - a)/torch.linalg.norm(old_a) < self.epsilon:
                    break 
                    
            residual = data - (dictionary@a.t()).t()
            
            self.checknan(a,'coefficients')
        return a
    
    
    
class LSM(InferenceMethod):
    """
    Infer coefficients for each image in data using elements dictionary.
    Method implemented according to "Group Sparse Coding with a Laplacian Scale Mixture Prior" (P. J. Garrigues & B. A. Olshausen, 2010)
    """    

    def __init__(self, n_iter = 100, n_iter_LSM = 6, beta=0.01, alpha=80.0, sigma=0.005, sparse_threshold = 10**-2, solver = None):
        '''
        
        Parameters
        ----------
        n_iter : scalar (1,) default=100
            number of iterations to run for an optimizer

        n_iter_LSM : scalar (1,) default=6
            number of iterations to run the outer loop of  LSM

        beta : scalar (1,) default=0.01
            LSM parameter used to update lambdas          

        alpha : scalar (1,) default=80.0
            LSM parameter used to update lambdas  

        sigma : scalar (1,) default=0.005
            LSM parameter used to compute the loss function

        sparse_threshold : scalar (1,) default=10**-2
            threshold used to discard smallest coefficients in the final solution
            SM parameter used to compute the loss function
            
        solver : default=None
        '''
        super().__init__(solver)
        self.n_iter = n_iter
        self.n_iter_LSM = n_iter_LSM
        self.beta = beta
        self.alpha = alpha
        self.sigma = sigma
        self.sparse_threshold = sparse_threshold

    def lsm_Loss(self, data, dictionary, coefficients, lambdas, sigma):
        """
        Compute LSM loss according to equation (7) in (P. J. Garrigues & B. A. Olshausen, 2010)

        Parameters
        ----------
        data : array-like (batch_size, n_features)
            data to be used in sparse coding
            
        dictionary : array-like, (n_features, n_basis)
            dictionary to be used
            
        coefficients : array-like (batch_size, n_basis)
            the current values of coefficients
        
        lambdas : array-like (batch_size, n_basis)
            the current values of regularization coefficient for all basis
            
        sigma : scalar (1,) default=0.005
            LSM parameter used to compute the loss functions    
            
        Returns
        -------
        loss : array-like (batch_size, 1)
            loss values for each data sample
        """        
        
        # Compute loss 
        loss = (1/(2*(sigma**2)))*torch.pow(torch.norm(data - torch.mm(dictionary,coefficients.t()).t(), p=2, dim=1, keepdim=True),2) + torch.sum(lambdas.mul(torch.abs(coefficients)), 1, keepdim=True)  
        
        return loss        
     
    def infer(self, data, dictionary):
        """
        Infer coefficients for each image in data using dict elements dictionary using Laplacian Scale Mixture (LSM)

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

        # Initialize coefficients for the whole batch
        coefficients=torch.zeros(batch_size, n_basis, requires_grad=True).to(device)
        

        for i in range(0,self.n_iter_LSM):
            
            # Compute the initial values of lambdas
            lambdas = (self.alpha + 1)/(self.beta + torch.abs(coefficients))
            
            # Set coefficients to zero before doing repeating the inference with new lambdas
            coefficients=torch.zeros(batch_size, n_basis, requires_grad=True).to(device)
            
            # Set up optimizer
            optimizer = torch.optim.Adam([coefficients])
            
            # Internal loop to infer the coefficients with the current lambdas
            for t in range(0,self.n_iter):
                
                # compute LSM loss for the current iteration
                loss = self.lsm_Loss(data=data, dictionary=dictionary, coefficients=coefficients, lambdas=lambdas, sigma=self.sigma) 
                
                optimizer.zero_grad()
        
                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward(torch.ones((batch_size,1)),retain_graph=True)
        
                # Calling the step function on an Optimizer makes an update to its parameters
                optimizer.step()
        
        
        # Sparsify the final solution by discarding the small coefficients
        coefficients.data[torch.abs(coefficients.data)<self.sparse_threshold] = 0 
      
        return coefficients.detach()    
    
    
    
    
    
    
    
    
    
    
    
    
      