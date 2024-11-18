import torch
import torch.fft as fft
from typing import Dict, Optional
from functools import lru_cache
from .whiten import whiten, compute_whitening_stats


def check_images(images):
    """Verify that tensor is in the shape [N, C, H, W] and C = 1   
    """

    if len(images.shape) != 4:
        raise ValueError('Images must be in shape [N, C, H, W]')
    if images.shape[1] != 1:
        raise ValueError(f'Images must be grayscale, received {images.shape[1]} channels')


def whiten_images(images: torch.Tensor,
                  algorithm: str,
                  stats: Dict = None,
                  **kwargs) -> torch.Tensor:
    """
    Wrapper for all whitening transformations

    Args:
        images: tensor of shape (N, C, H, W)
        algorithm: what whitening transform we want to use
        stats: dictionary of dataset statistics needed for whitening transformations
    """

    check_images(images)

    if algorithm == 'frequency':
        return frequency_whitening(images, **kwargs)

    elif algorithm in ['zca', 'pca', 'cholesky']:
        N, C, H, W = images.shape
        flattened_images = images.flatten(start_dim=1)
        whitened = whiten(flattened_images, algorithm, stats, **kwargs)
        return whitened.reshape((N, C, H, W))

    else:
        raise ValueError(f"Unknown whitening algorithm: {algorithm}, \
                          must be one of ['frequency', 'pca', 'zca', 'cholesky]")
    

def compute_image_whitening_stats(images: torch.Tensor,
                                  n_components=None) -> Dict:
    check_images(images)
    flattened_images = images.flatten(start_dim=1)
    return compute_whitening_stats(flattened_images, n_components)


def create_frequency_filter(image_size: int, f0_factor: float = 0.4) -> torch.Tensor:
    """
    Create a frequency domain filter for image whitening.
    
    Args:
        image_size: Size of the square image
        f0_factor: Factor for determining the cutoff frequency (default 0.4)
        
    Returns:
        torch.Tensor: Frequency domain filter
    """
    fx = torch.linspace(-image_size/2, image_size/2-1, image_size)
    fy = torch.linspace(-image_size/2, image_size/2-1, image_size)
    fx, fy = torch.meshgrid(fx, fy, indexing='xy')
    
    rho = torch.sqrt(fx**2 + fy**2)
    f_0 = f0_factor * image_size
    filt = rho * torch.exp(-(rho/f_0)**4)
    
    return fft.fftshift(filt)


@lru_cache(maxsize=32)
def get_cached_filter(image_size: int, f0_factor: float = 0.4) -> torch.Tensor:
    """
    Get a cached frequency filter for the given image size.
    
    Args:
        image_size: Size of the square image
        f0_factor: Factor for determining the cutoff frequency
        
    Returns:
        torch.Tensor: Cached frequency domain filter
    """
    return create_frequency_filter(image_size, f0_factor)


def normalize_variance(tensor: torch.Tensor, target_variance: float = 0.1) -> torch.Tensor:
    """
    Normalize the variance of a tensor to a target value.
    
    Args:
        tensor: Input tensor
        target_variance: Desired variance after normalization
        
    Returns:
        torch.Tensor: Normalized tensor
    """
    if torch.var(tensor) < 1e-8:
        return tensor
        
    centered = tensor - tensor.mean()
    current_variance = torch.var(centered)
    
    if current_variance > 0:
        scale_factor = torch.sqrt(torch.tensor(target_variance) / current_variance)
        return centered * scale_factor
    return centered


def whiten_channel(
    channel: torch.Tensor,
    filt: torch.Tensor,
    target_variance: float = 0.1
) -> torch.Tensor:
    """
    Apply frequency domain whitening to a single channel.
    
    Args:
        channel: Single channel image tensor
        filt: Frequency domain filter
        target_variance: Target variance for normalization
        
    Returns:
        torch.Tensor: Whitened channel
    """

    if torch.var(channel) < 1e-8:
        return channel
    
    # Convert to frequency domain and apply filter
    If = fft.fft2(channel)
    If_whitened = If * filt.to(channel.device)
    
    # Convert back to spatial domain and normalize
    whitened = torch.real(fft.ifft2(If_whitened))

    # Normalize variance
    whitened = whitened - whitened.mean()
    variance = torch.var(whitened)
    if variance > 0:
        scale_factor = torch.sqrt(torch.tensor(target_variance) / variance)
        whitened = whitened * scale_factor

    return whitened


def frequency_whitening(
    images: torch.Tensor,
    target_variance: float = 0.1,
    f0_factor: float = 0.4
) -> torch.Tensor:
    """
    Apply frequency domain decorrelation to batched images.
    Method used in original sparsenet in Olshausen and Field in Nature
    and http://www.rctn.org/bruno/sparsenet/
    
    Args:
        images: Input images of shape (N, C, H, W)
        target_variance: Target variance for normalization
        f0_factor: Factor for determining filter cutoff frequency
        
    Returns:
        torch.Tensor: Whitened images
    """
    _, _, H, W = images.shape
    if H != W:
        raise ValueError("Images must be square")
    
    # Get cached filter
    filt = get_cached_filter(H, f0_factor)
    
    # Process each image in the batch
    whitened_batch = []
    for img in images:
        whitened_batch.append(
            whiten_channel(img[0], filt, target_variance)
        )
    
    return torch.stack(whitened_batch).unsqueeze(1)


class WhiteningTransform(object):
    """
    A PyTorch transform for image whitening that can be used in a transform pipeline.
    Supports frequency, PCA, and ZCA whitening methods.
    """
    def __init__(
        self,
        algorithm: str = 'zca',
        stats: Optional[Dict] = None,
        compute_stats: bool = False,
        **kwargs
    ):
        """
        Initialize whitening transform.
        
        Args:
            algorithm: One of ['frequency', 'pca', 'zca']
            stats: Pre-computed statistics for PCA/ZCA whitening
            compute_stats: If True, will compute stats on first batch seen
            **kwargs: Additional arguments passed to whitening function
        """
        self.algorithm = algorithm
        self.stats = stats
        self.compute_stats = compute_stats
        self.kwargs = kwargs
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply whitening transform to images.
        
        Args:
            images: Input images of shape [N, C, H, W] or [C, H, W]
        
        Returns:
            Whitened images of same shape as input
        """
        # Add batch dimension if necessary
        if images.dim() == 3:
            images = images.unsqueeze(0)
            single_image = True
        else:
            single_image = False

        check_images(images)
        # Apply whitening
        whitened = whiten(
            images,
            self.algorithm,
            self.stats,
            **self.kwargs
        )
     
        # Remove batch dimension if input was single image
        if single_image:
            whitened = whitened.squeeze(0)
            
        return whitened

    def __repr__(self):
        return "custom whitening augmentation"
    