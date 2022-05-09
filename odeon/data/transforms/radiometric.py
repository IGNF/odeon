import random

import numpy as np
from skimage import exposure
from skimage.color import hsv2rgb, rgb2hsv
from skimage.util import random_noise


class Radiometry(object):
    """Apply gamma, hue variation and noise to image and mask
    There is a 50% that each effect is applied"""

    def __call__(self, **sample):

        image, mask = sample["image"], sample["mask"]

        if random.randint(0, 1):
            # gamma correction on the RGB IRC channels
            gamma_factor = random.uniform(0.5, 2.2)
            image[:, :, 0:4] = exposure.adjust_gamma(
                image[:, :, 0:4], gamma=gamma_factor, gain=1
            )

        if random.randint(0, 1):
            # allows up to 10% of color variation
            hue_factor = random.uniform(0, 0.066)

            # the rvb->hsv translation may lead to 'divide by zero' errors
            np.seterr(invalid="ignore", divide="ignore")
            # in HSV space, only the first band is modified (hue)
            hsv_img = rgb2hsv(image[:, :, 0:3])
            hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue_factor) % 1
            image[:, :, 0:3] = hsv2rgb(hsv_img)

        if random.randint(0, 1):
            var = random.uniform(0.001, 0.01)
            image[:, :, 0:4] = random_noise(image[:, :, 0:4], var=var)

        return {"image": image, "mask": mask}
