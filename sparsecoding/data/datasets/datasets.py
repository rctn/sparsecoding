import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


# ================================================================================
#  NATURAL IMAGES
# ================================================================================
class Olshausen(Dataset):

    def __init__(self, img_dir, patch_size, patch_overlap, data_key='IMAGESr', device=None):
        """
        Parameters:
        img_dir - string
            *.mat file to load images from, expected to be of for (pix,pix,n_img)
        patch_size - scalar (1,)
            patch row/column size to extract
        patch_overlap - scalar (1,) default=0
            amount to overlap patches by in pixels. If 0, no overlap
        data_key - string default='IMAGESr'
            key to query mat dict with to get data
        device - torch.device default == cpu
            device to load data on
        ---
        """
        self.device = torch.device("cpu") if device is None else device
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap

        # load image dataset
        images_dict = sio.loadmat(img_dir)
        self.images = np.asarray(images_dict[data_key])  # scalar - (m,m,n_img)

        # make patches
        self.extractpatches()

    def __len__(self):
        '''
        return number of patches
        '''
        return self.patches.shape[0]

    def __getitem__(self, idx):
        '''
        return patch at idx
        '''
        return self.patches[idx, :]

    def extractpatches(self):
        """
        Extracts image patches from images
        ---
        Defines:
        self.patches - scalar torch.tensor (n_patch,patch_dim**2)
            image patches

        Note: if patch_dim doesn't go into pix evenly,
              no error will be thrown left, bottom part of image cut-off
        """
        pix, _, n_img = self.images.shape

        n = (pix-self.patch_size)//(self.patch_size-self.patch_overlap)

        self.patches = []
        for img in range(n_img):
            for i in range(n):
                for j in range(n):
                    rl = i*(self.patch_size - self.patch_overlap)
                    rr = rl+self.patch_size
                    cl = j*(self.patch_size - self.patch_overlap)
                    cr = cl+self.patch_size
                    self.patches.append(self.images[rl:rr, cl:cr, img])
        self.patches = np.asarray(self.patches, dtype=np.float32).reshape(-1, self.patch_size**2)
        self.patches = torch.from_numpy(self.patches).to(self.device)
