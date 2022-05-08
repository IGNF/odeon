import albumentations as A
from typing import List
from odeon import LOGGER
from odeon.data.transforms import (
    Compose,
    DeNormalize,
    Rotation90,
    Radiometry,
    ScaleImageToFloat, 
    FloatImageToByte,
    ToDoubleTensor,
    CHW_to_HWC,
    HWC_to_CHW
)


def configure_transforms(data_aug:dict, normalization_weights:dict=None, verbosity:bool=False)-> dict:

    def _parse_data_augmentation(list_tfm: List[str])-> List:
        tfm_dict = {"rotation90": Rotation90(), "radiometry": Radiometry()}
        list_tfm = list_tfm if isinstance(list_tfm, list) else [list_tfm]
        tfm_func = [tfm_dict[tfm] for tfm in list_tfm]
        return tfm_func

    transforms, inv_transforms = {}, {}
    for split_name in ["train", "val", "test"]:
        tfm_func, inv_tfm_func = [], [CHW_to_HWC(img_only=True)]
        if split_name == "train" and data_aug is not None:
            tfm_func = _parse_data_augmentation(data_aug[split_name])

        # Part to define how to normalize the data
        if normalization_weights is not None and split_name in normalization_weights.keys():
            tfm_func.append(A.Normalize(mean=normalization_weights[split_name]["mean"],
                                        std=normalization_weights[split_name]["std"]))

            inv_tfm_func.append(DeNormalize(mean=normalization_weights[split_name]["mean"],
                                            std=normalization_weights[split_name]["std"]))
        else:
            tfm_func.append(ScaleImageToFloat())
            inv_tfm_func.append(FloatImageToByte())

        tfm_func.append(ToDoubleTensor())  # To transform float type arrays to double type tensors
        inv_tfm_func.extend([HWC_to_CHW(img_only=True)])
        transforms[split_name] = Compose(tfm_func)
        inv_transforms[split_name] = Compose(inv_tfm_func)

        if verbosity:
            LOGGER.debug(f"DEBUG: Transforms: {split_name}, {tfm_func}")
            LOGGER.debug(f"DEBUG: Inverse transforms: {split_name}, {inv_tfm_func}")

    return transforms, inv_transforms