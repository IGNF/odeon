import os

import geopandas as gpd
from sklearn.model_selection import StratifiedKFold, train_test_split

from odeon.core.io_utils import create_path_if_not_exists


class StratifiedKFold3(StratifiedKFold):
    # see https://stackoverflow.com/questions/45221940/creating-train-test-val-split-with-stratifiedkfold
    def split(self, X, y, groups=None):
        s = super().split(X, y, groups)
        for train_indxs, test_indxs in s:
            y_train = y[train_indxs]
            train_indxs, cv_indxs = train_test_split(train_indxs, stratify=y_train, test_size=(1 / (self.n_splits - 1)))
            yield train_indxs, cv_indxs, test_indxs


def run(db_file: str, n_split=10):
    output_dir = os.path.dirname(db_file)
    print(output_dir)
    gdf = gpd.read_file(db_file)
    crs = gdf.crs
    # gdf = gdf.sample(len(gdf))  # shuffle
    # print(gdf.columns)
    # gdf['area'] = gdf.apply(lambda x: x.geometry.area, axis=1)
    index = gdf.sort_values(['area']).index
    print(index)
    # print(index)
    # print(gdf['area'])
    # count = len(gdf)
    # n_row_by_split = count // n_split
    for idx in index:
        # Count
        gdf.loc[idx, 'fold'] = int(idx % n_split)
    print(gdf.groupby(['fold']).agg({'area': ['mean', 'count', 'std']}))
    print(f"std: {gdf.groupby(['fold']).agg({'area': ['mean', 'count', 'std']}).std()}")
    # grouped_gdf = gdf.groupby(['fold'])
    # splits = [i for i in range(10)]
    val_split = [i for i in range(10)]
    test_split = [i + 1 if i != 9 else 0 for i in range(10)]
    for val_idx, test_idx in zip(val_split, test_split):
        output_split_dir = os.path.join(output_dir, f'fold-{str(int(val_idx))}')
        create_path_if_not_exists(output_split_dir)
        # train_idx = set(splits) - {val_idx, test_idx}
        # print(f'train idx: {train_idx}, val idx: {val_idx}, test idx: {test_idx}')
        assert val_idx != test_idx
        train_gdf = gdf[~gdf['fold'].isin([float(val_idx), float(test_idx)])]
        val_gdf = gdf[gdf['fold'].isin([float(val_idx)])]
        test_gdf = gdf[gdf['fold'].isin([float(test_idx)])]
        print(len(train_gdf))
        print(len(val_gdf))
        print(len(test_gdf))
        train_gdf = gpd.GeoDataFrame(train_gdf, crs=crs)
        train_gdf.to_file(os.path.join(output_split_dir, f'train_split_{val_idx}.geojson'), driver='GeoJSON')
        val_gdf = gpd.GeoDataFrame(val_gdf, crs=crs)
        val_gdf.to_file(os.path.join(output_split_dir, f'val_split_{val_idx}.geojson'), driver='GeoJSON')
        test_gdf = gpd.GeoDataFrame(test_gdf, crs=crs)
        test_gdf.to_file(os.path.join(output_split_dir, f'test_split_{val_idx}.geojson'), driver='GeoJSON')
    gdf.to_file(db_file)


if __name__ == "__main__":
    root = "/media/HP-2007S005-data"
    db_file = os.path.join(root, "gers/change_dataset/patches/patch_dataset.geojson")
    run(db_file=db_file)
