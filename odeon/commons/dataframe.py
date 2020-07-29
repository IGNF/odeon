import numpy as np
import os


def split_dataset_from_df(df, percentage):

    msk = np.random.rand(len(df)) < percentage
    return df[msk], df[~msk]


def set_path_to_center(df, img_path, msk_path):

    for idx, row in df.iterrows():

        x = "{:.4f}".format(row.x).replace(".", "-")
        y = "{:.4f}".format(row.y).replace(".", "-")

        df.at[idx, "msk_file"] = os.path.join(msk_path, "{}_{}_{}.tif".format(x, y, idx))
        df.at[idx, "img_file"] = os.path.join(img_path, "{}_{}_{}.tif".format(x, y, idx))
    return df
