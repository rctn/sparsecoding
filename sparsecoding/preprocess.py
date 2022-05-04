import numpy as np
import pytorch as torch

class ImageSet(object):
    ''' Class for loading and preprocessing image data.'''

    def __init__(self, images, whiten, filter, normalization):
        self.images =
        self.n_images =
        self.image_dims =

    def generate_patches(self, n_patches, patch_dims, edge_buff):
        '''
        Generate randomly extracted patches from full image dataset.

        Parameters
        ________
        n_patches : integer specifying desired number of patches to generate

        patch_dims : array like or tuple (heigh, width)

        '''
        patch_height_range = self.image_dims[0] - 2*edge_buff - patch_dims[0]
        patch_width_range = self.image_dims[1]  - 2*edge_buff - patch_dims[1]

        for i in range(n_patches):
            patch_height_start = int(np.floor(np.random.rand()*patch_height_range + edge_buff))
            patch_height_end = patch_height_start + patch_dims[0]

            patch_width_start = int(np.floor(np.random.rand()*patch_width_range + edge_buff))
            patch_width_end = patch_width_start + patch_dims[1]

            image_idx = int(np.floor(np.random.rand()*(self.n_images-1)))
            patches[:,i] = self.images[patch_height_start:patch_height_end, patch_width_start:patch_width_end, image_idx].reshape(patch_dims[0]*patch_dims[1])

        self.patches = torch.from_numpy(patches).cuda()

    def PCA(self, *args, **kwargs):


    def filter(self, *args, **kwargs):
