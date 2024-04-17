import os
from tqdm import tqdm
# from utils.diffusion_utils import load_latents
from torch.utils.data.dataset import Dataset
import tifffile
from dataset.utils import transform_augment
import numpy as np


class MnistDataset(Dataset):
    r"""
    Nothing special here. Just a simple dataset class for mnist images.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset
    """
    
    def __init__(self, split, im_root, noise_root, im_size, im_channels,
                 use_latents=False, latent_path=None, condition_config=None):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        # Should we use latents or not
        self.latent_maps = None
        self.use_latents = False
        
        # Conditioning for the dataset
        self.condition_types = [] if condition_config is None else condition_config['condition_types']

        self.images, self.labels = self.load_images(im_root)
        self.noise_root = noise_root

    def load_images(self, im_root):
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path:
        :return:
        """
        assert os.path.exists(im_root), "images path {} does not exist".format(im_root)
        ims = []
        labels = []

        for name in tqdm(os.listdir(im_root)):
            im_path = os.path.join(im_root, name)
            lab_path = os.path.join(im_root.replace('interf', 'label'), name)
            ims.append(im_path)
            labels.append(lab_path)
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        lab = tifffile.imread(self.labels[index]).astype(np.float32)
        dint, lab = transform_augment(lab, self.noise_root)
        return {'dint': dint, 'lab': lab}

            
