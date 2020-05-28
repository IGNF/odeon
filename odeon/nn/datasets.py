from torch.utils.data import Dataset
from torchvision import transforms
from skimage.util import img_as_float

from odeon.commons.image import image_to_ndarray

class PatchDataset(Dataset):

    def __init__(self, image_files, mask_files, transformations=[], **kwargs):

        self.image_files = image_files
        self.image_bands = kwargs.get('image_bands', None)
        self.mask_files = mask_files
        self.mask_bands = kwargs.get('mask_bands', None)

        self.width = kwargs.get('width', None)
        self.height = kwargs.get('height', None)
        self.transforms = transforms.Compose(transformations)
        pass

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        # load image file
        image_file = self.image_files[index]
        img = image_to_ndarray(image_file, width=self.width, height=self.height, band_indices=self.image_bands)
        # pixels are normalized to [0, 1]
        img = img_as_float(img)

        # load mask file
        mask_file = self.mask_files[index]
        msk = image_to_ndarray(mask_file, width=self.width, height=self.height)

        sample = {"image": img, "mask": msk}

        # apply transforms
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
