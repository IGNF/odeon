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


def run(db_file: str, patch_db_file: str, n_split=10, ):
    output_dir = os.path.dirname(db_file)
    print(output_dir)
    gdf = gpd.read_file(db_file)
    patch_gdf = gpd.read_file(patch_db_file)
    output_patch_dir = os.path.dirname(patch_db_file)
    crs = gdf.crs
    # gdf = gdf.sample(len(gdf))  # shuffle
    # print(gdf.columns)
    # gdf['area'] = gdf.apply(lambda x: x.geometry.area, axis=1)
    gdf = gdf.sort_values(['area'])
    gdf.reset_index(inplace=True, drop=True)
    # print(index)
    # print(index)
    # print(gdf['area'])
    # count = len(gdf)
    # n_row_by_split = count // n_split
    for idx, row in gdf.iterrows():
        gdf.loc[idx, 'fold'] = idx % n_split
        print(f'index {idx}, zone {row["id_zone"]}, area {row["area"]}, fold {gdf.loc[idx, "fold"]}, n split {n_split}')

    print(gdf.groupby(['fold']).agg({'area': ['mean', 'count', 'std']}))
    print(f"std: {gdf.groupby(['fold']).agg({'area': ['mean', 'count', 'std']}).std()}")
    print(f'zones sorted by area \n {gdf[["id_zone", "area", "fold"]].sort_values(["area"])}')

    # grouped_gdf = gdf.groupby(['fold'])
    # splits = [i for i in range(10)]
    for i in range(n_split):
        val_fold = i + 1 if i != 9 else 0
        test_fold = i
        output_split_dir = os.path.join(output_dir, f'split-{str(int(i))}')
        output_patch_split_dir = os.path.join(output_patch_dir, f'split-{str(int(i))}')
        create_path_if_not_exists(output_split_dir)
        create_path_if_not_exists(output_patch_split_dir)
        # train_idx = set(splits) - {val_idx, test_idx}
        # print(f'train idx: {train_idx}, val idx: {val_idx}, test idx: {test_idx}')
        assert val_fold != test_fold
        train_gdf = gdf[~gdf['fold'].isin([float(val_fold), float(test_fold)])]
        val_gdf = gdf[gdf['fold'].isin([float(val_fold)])]
        test_gdf = gdf[gdf['fold'].isin([float(test_fold)])]
        train_patch_gdf = patch_gdf[patch_gdf['id_zone'].isin(list(train_gdf['id_zone'].unique()))]
        val_patch_gdf = patch_gdf[patch_gdf['id_zone'].isin(list(val_gdf['id_zone'].unique()))]
        test_patch_gdf = patch_gdf[patch_gdf['id_zone'].isin(list(test_gdf['id_zone'].unique()))]
        print(len(train_gdf))
        print(len(val_gdf))
        print(len(test_gdf))
        print(len(train_patch_gdf))
        print(len(val_patch_gdf))
        print(len(test_patch_gdf))
        train_gdf = gpd.GeoDataFrame(train_gdf, crs=crs)
        train_gdf.to_file(os.path.join(output_split_dir, f'train_split_{i}.geojson'), driver='GeoJSON')
        train_patch_gdf = gpd.GeoDataFrame(train_patch_gdf, crs=crs)
        train_patch_gdf.to_file(os.path.join(output_patch_split_dir, f'train_split_{i}.geojson'),
                                driver='GeoJSON')
        val_gdf = gpd.GeoDataFrame(val_gdf, crs=crs)
        val_gdf.to_file(os.path.join(output_split_dir, f'val_split_{i}.geojson'), driver='GeoJSON')
        val_patch_gdf = gpd.GeoDataFrame(val_patch_gdf, crs=crs)
        val_patch_gdf.to_file(os.path.join(output_patch_split_dir, f'val_split_{i}.geojson'),
                              driver='GeoJSON')
        test_gdf = gpd.GeoDataFrame(test_gdf, crs=crs)
        test_gdf.to_file(os.path.join(output_split_dir, f'test_split_{i}.geojson'), driver='GeoJSON')
        test_patch_gdf = gpd.GeoDataFrame(test_patch_gdf, crs=crs)
        test_patch_gdf.to_file(os.path.join(output_patch_split_dir, f'test_split_{i}.geojson'),
                               driver='GeoJSON')

    gdf.to_file(db_file)


if __name__ == "__main__":
    root = "/media/HP-2007S005-data"
    db_file = os.path.join(root, "gers/change_dataset/zones/dataset_v1.shp")
    patch_db_file = os.path.join(root, "gers/change_dataset/patches/patch_dataset.geojson")
    run(db_file=db_file, patch_db_file=patch_db_file)
