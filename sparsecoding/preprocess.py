import numpy as np
import torch


## Sparse Coding Image Processing Code
def patch_images(images, n_divisions=16):
    '''
    Generate a collection of patches for each grayscale image in images.
    
    Parameters
    ----------
    images : torch.Tensor, shape=(W, H, batch)
    
    n_divisions : int
        number of subdivisions of the image
        
    Returns
    -------
    patch_data : torch.Tensor, shape=(n_images, n_patches, patch_size_x, patch_size_y)
    n_images : int
    n_divisions : int
    patch_size_x : int
    patch_size_y : int
    '''
    n_images = images.shape[-1]
    patch_size_x = images.shape[0]//n_divisions
    patch_size_y = images.shape[1]//n_divisions
    
    patch_data = torch.empty(n_images, n_divisions**2, patch_size_x, patch_size_y)
    for i in range(n_images):
        im = images[:,:,i]
        patches = [im[x * patch_size_x : (x + 1) * patch_size_x, y * patch_size_y : (y + 1) * patch_size_y] for x in range(0, n_divisions) for y in range(0,n_divisions)]
        patch_data[i] = torch.stack(patches)
    return patch_data, n_images, n_divisions, patch_size_x, patch_size_y

def unpatch_image(I, n_divisions, patch_size_x):
    final_img = torch.empty(n_divisions*patch_size_x, n_divisions*patch_size_x)
    index = 0
    for row in range(n_divisions):
        for col in range(n_divisions):
            final_img[row*patch_size_x : (row+1)*patch_size_x, 
                      col*patch_size_x : (col+1)*patch_size_x] = I[index]
            index = index + 1
    return final_img

def unpatch_images(patch_images, n_images, n_divisions, patch_size_x, patch_size_y):
    '''
    Convert image patches into full images.
    '''
    patch_images = patch_images.reshape(n_images, n_divisions**2, patch_size_x, patch_size_y)
    final_images = torch.empty(n_images, n_divisions*patch_size_x, n_divisions*patch_size_x)
    for i,patch in enumerate(patch_images):
        final_images[i] = unpatch_image(patch, n_divisions, patch_size_x)
    return final_images
            
def preprocess_images(images, patch_size_x, patch_size_y):
    data = images.detach().clone()
    means = torch.mean(data, dim=(1,2,3), keepdims = True)
    data = data - means
    stds = 10*torch.std(data, dim=(1,2,3), keepdims = True)

    data = data / stds
    data = data.reshape(-1, patch_size_x*patch_size_y)
    return data

# class ImageSet(self, images, whiten, filter, normalization):
#     ''' Class for loading and preprocessing image data.'''

#     def __init__(self, images):
#         self.images =
#         self.n_images =
#         self.image_dims =

#     def generate_patches(self, n_patches, patch_dims, edge_buff):
#         '''
#         Generate randomly extracted patches from full image dataset.

#         Parameters
#         ________
#         n_patches : integer specifying desired number of patches to generate

#         patch_dims : array like or tuple (heigh, width)

#         '''
#         patch_height_range = self.image_dims[0] - 2*edge_buff - patch_dims[0]
#         patch_width_range = self.image_dims[1]  - 2*edge_buff - patch_dims[1]

#         for i in range(n_patches):
#             patch_height_start = int(np.floor(np.random.rand()*patch_height_range + edge_buff))
#             patch_height_end = patch_height_start + self.patch_dims[0]

#             patch_width_start = int(np.floor(np.random.rand()*patch_width_range + edge_buff))
#             patch_width_end = patch_width_start + self.patch_dims[1]

#             image_idx = int(np.floor(np.random.rand()*(self.n_images-1)))
#             patches[:,i] = self.images[patch_height_start:patch_height_end, patch_width_start:patch_width_end, image_idx].reshape(patch_dims[0]*patch_dims[1])

#         self.patches = torch.from_numpy(patches).cuda()

#     def PCA(self, *args, **kwargs):


#     def filter(self, *args, **kwargs):


