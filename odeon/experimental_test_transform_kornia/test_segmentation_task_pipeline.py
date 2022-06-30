""" test data reading, augmentation, normalization """
import os
import matplotlib.pyplot as plt
import torchvision
from kornia import tensor_to_image
from odeon.tasks.segmentation_task.data_module import SegmentationTaskDataModule
from odeon.tasks.segmentation_task.transform import DeNormalize
from odeon.core.default_conf import EnvConf, InputFields
from odeon.core.constants import IMAGENET
from odeon.core.image import tensor_to_image
conf = EnvConf()
input_fields = InputFields()
print(conf)
ROOT = conf.root
data_dir = os.path.join(ROOT, conf.data_path)
db_path = conf.db_name
batch_size = 2
denormalize = DeNormalize(mean=IMAGENET["mean"], std=IMAGENET["std"])
gers_data = SegmentationTaskDataModule(data_dir=data_dir,
                                       db_path=db_path,
                                       fold=1,
                                       batch_size=batch_size,
                                       mode="hard_aug",
                                       debug=True)
gers_data.setup()


def test():

    fig = plt.figure(figsize=(20, 20))
    print(gers_data.db.columns)
    # exit(0)
    train_loader = gers_data.train_dataloader()
    samples = next(iter(train_loader))
    print(samples["img"].shape)

    for index, img, mask, img_aug, mask_aug in zip(samples["index"],
                                                   samples["ori_img"],
                                                   samples["ori_mask"],
                                                   samples["img"],
                                                   samples["mask"]):

        img_name = gers_data.db.loc[index.numpy(), input_fields.img_2]
        print(img_aug.max())
        img_aug = tensor_to_image(denormalize(img_aug))
        print(img_aug.max())
        print(mask_aug.shape)
        mask_aug = tensor_to_image(mask_aug)
        print(mask_aug.shape)
        img = tensor_to_image(img)
        mask = tensor_to_image(mask)
        fig.suptitle(f'example image {index}', fontsize=16)
        ax = []
        ax.append(fig.add_subplot(2, 2, 1))
        ax[-1].clear()
        ax[-1].set_title("image before aug")
        plt.imshow(img)
        # axarr[0, 0].set_title("img 1")
        ax.append(fig.add_subplot(2, 2, 2))
        ax[-1].clear()
        ax[-1].set_title("image after aug")
        plt.imshow(img_aug)
        ax.append(fig.add_subplot(2, 2, 3))
        ax[-1].clear()
        ax[-1].set_title("mask before mask")
        plt.imshow(mask)
        # axarr[0, 0].set_title("img 1")
        ax.append(fig.add_subplot(2, 2, 4))
        ax[-1].clear()
        ax[-1].set_title("mask after mask")
        plt.imshow(mask_aug)
        # axarr[0, 1].set_title("img 2")
        # axarr[1, 1].set_title("sub img 2")
        plt.pause(3)


if __name__ == "__main__":

    test()