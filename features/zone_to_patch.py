import functools
import operator
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
from tqdm import tqdm

from odeon.core.io_utils import create_path_if_not_exists
from odeon.core.raster import read
from odeon.core.tile import tile
from odeon.core.types import OPT_URI, URI


@dataclass
class Stats:
    midpoints: Optional[float] = None
    mean: Optional[float] = None
    var: Optional[float] = None
    std: Optional[float] = None
    skew: Optional[float] = None
    kurtosis: Optional[float] = None


class RadiometricStats(Stats):
    ...


class HistogramAccumulator:
    def __init__(self,
                 range: List,
                 max_value: int,
                 histogram: Optional[np.ndarray] = None,
                 bins: Optional[np.ndarray] = None,
                 n_sample: Optional[int] = None
                 ):
        # super.__init__()
        self._range = range
        self._max_value = max_value
        self._histogram: Optional[np.ndarray] = histogram if histogram else None
        if histogram:
            assert bins is not None
        self._bins: Optional[np.ndarray] = bins if bins else None
        self._n_sample: int = n_sample if n_sample else 0
        self._stats: Optional[Stats] = None

    def update(self,
               data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        self._histogram, self._bins = HistogramAccumulator.accumulate_histograms(data=data,
                                                                                 n=self._max_value,
                                                                                 range=self._range,
                                                                                 histogram=self._histogram,
                                                                                 bins=self._bins,
                                                                                 n_sample=self._n_sample
                                                                                 )
        self._n_sample += 1
        return self._histogram, self._bins, self._n_sample

    def compute(self):
        assert self._histogram is not None
        if self._stats:
            return self._stats
        else:
            return HistogramAccumulator.computer_stats_raster_from_histogram(self._histogram,
                                                                             bins=self._bins)

    @staticmethod
    def computer_stats_raster_from_histogram(histogram: np.ndarray, bins: np.ndarray) -> RadiometricStats:
        midpoints = 0.5 * (bins[1:] + bins[:-1])
        mean = np.average(midpoints, weights=histogram)
        var = np.average((midpoints - mean) ** 2, weights=histogram)
        std = np.sqrt(var)
        return RadiometricStats(midpoints=midpoints,
                                mean=mean,
                                var=var,
                                std=std)

    @staticmethod
    def accumulate_histograms(data: np.ndarray,
                              n: int,
                              range: List,
                              histogram: np.ndarray = None,
                              bins: np.ndarray = None,
                              n_sample: int = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
        _histo, _bins = np.histogram(data.ravel(), bins=n, range=range)
        if histogram:
            assert bins
            assert n_sample
            """
            _weights = np.array([n_sample, 1])
            _histo_array = np.array([histogram, _histo])
            _sum = np.dot(_histo_array, _weights)
            """
            _sum = histogram * n_sample + _histo
            return _sum, _bins
        else:
            return _histo, _bins


def get_window(bounds, width: int, height: int):
    return from_bounds(west=bounds[0],
                       south=bounds[1],
                       east=bounds[2],
                       north=bounds[3],
                       width=width,
                       height=height)


def extract_patch_from_zone(src: rio.DatasetReader, bounds: List, width: int, height: int) -> Tuple[np.ndarray, Dict]:
    window = window_from_bounds(left=bounds[0], bottom=bounds[1], right=bounds[2], top=bounds[3],
                                transform=src.transform)
    patch = read(src=src, window=window, width=width, height=height)
    return patch, src.profile


def write_patch(patch: np.ndarray, profile, output_path):
    with rio.open(output_path, 'w+', **profile) as dst:
        dst.write(patch)


def compute_stats_label(data: np.ndarray,
                        label_value: Union[int, float],
                        compute_has_value=False) -> Union[float, Tuple[float, bool]]:
    _msk = data[data == label_value]
    _shape = data.shape
    _n_pixel = functools.reduce(operator.mul, _shape, 1)
    stat_data = _msk.sum() / _n_pixel
    if compute_has_value:
        return float(stat_data), stat_data > 0.0
    else:
        return float(stat_data)


def split_zone_to_patch(raster_fields: Dict,
                        mask_fields: Dict,
                        output_dir: URI,
                        input_file: URI,
                        output_prefix: OPT_URI = None,
                        root: OPT_URI = None,
                        patch_size: int = 512,
                        resolution: float = 0.2,
                        output_driver: str = 'GTiff'):
    """
    Build change dataset as Patch from Zone
    step 1/ tile dataframe
    step 2/ create output directories for each modality
    step 3/ open raster connections
    step 4/ create patches
        substep 1/ read rasters and write
        substep 2/ read mask,
                   compute stats,
                   add stats per_change / has_change write
    step 5/ save patches geodataframe

    Parameters
    ----------
    raster_fields
    mask_fields
    input_file
    output_dir
    root
    output_prefix
    patch_size
    resolution

    Returns
    -------

    """

    # step 1/ tile dataframe
    patch_size_u: float = float(patch_size * resolution)
    gdf: gpd.GeoDataFrame = gpd.read_file(os.path.join(root, input_file)) if root else gpd.read_file(input_file)
    crs = gdf.crs
    d = []

    for idx, i in gdf.iterrows():
        zone_row = i.to_dict()
        del zone_row['geometry']
        for row in tile(bounds=i.geometry.bounds,
                        overlap=0,
                        tile_size=patch_size_u,
                        strict_inclusion=False):
            row.update(zone_row)
            assert 'geometry' in row.keys()
            d.append(row)

    patch_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(data=d,
                                                   crs=crs)
    # patch_gdf = patch_gdf.sample(n=10)

    print(len(gdf))
    print(len(patch_gdf))
    print(patch_gdf.iloc[0].geometry.bounds[2] - patch_gdf.iloc[0].geometry.bounds[0])
    print(gdf.iloc[0].geometry.bounds[2] - gdf.iloc[0].geometry.bounds[0])
    # patch_gdf.to_file('/media/HP-2007S005-data/gers/change_dataset/test.shp')

    # step 2/ create output directories for each modality
    output_dir = os.path.join(output_prefix, output_dir) if output_prefix else output_dir
    # [v.update({'output': os.path.join(output_dir, v['output'])}) for v in raster_fields.values()]
    # [v.update({'output': os.path.join(output_dir, v['output'])}) for v in mask_fields.values()]
    print(type(raster_fields))
    print(raster_fields)
    [create_path_if_not_exists(v['output']) for v in raster_fields.values()]
    [create_path_if_not_exists(v['output']) for v in mask_fields.values()]
    # assert os.path.isdir('/media/HP-2007S005-data/gers/change_dataset/patches/T0')

    print(raster_fields.values())
    # step 3/ open raster connections
    for v in raster_fields.values():
        v['src'] = rio.open(os.path.join(root, v['input']))
    for v in mask_fields.values():
        v['src'] = rio.open(os.path.join(root, v['input']))
    print(raster_fields)
    # step 4/ create patches
    for idx, row in tqdm(patch_gdf.iterrows(), total=len(patch_gdf)):
        bounds = row.geometry.bounds
        xmin = '-'.join(str(bounds[0]).split('.'))
        ymax = '-'.join(str(bounds[3]).split('.'))

        out_transform = get_window(bounds=bounds, width=patch_size, height=patch_size)
        for k, v in raster_fields.items():
            basename = f'{xmin}_{ymax}_{os.path.basename(row[k])}'
            relative_path = os.path.join(v['output'], basename)
            output_path = os.path.join(output_dir, relative_path)
            patch, profile = extract_patch_from_zone(src=v['src'], bounds=bounds, width=patch_size, height=patch_size)
            profile.update({'transform': out_transform, 'width': patch_size, 'height': patch_size,
                            'driver': output_driver})
            write_patch(patch=patch, profile=profile, output_path=output_path)
            patch_gdf.loc[idx, v['output']] = relative_path

        for k, v in mask_fields.items():
            basename = f'{xmin}_{ymax}_{os.path.basename(row[k])}'
            relative_path = os.path.join(v['output'], basename)
            output_path = os.path.join(output_dir, relative_path)
            patch, profile = extract_patch_from_zone(src=v['src'], bounds=bounds, width=patch_size,
                                                     height=patch_size)
            labels = v['labels']
            for label in labels:
                stats = compute_stats_label(data=patch,
                                            label_value=label['value'],
                                            compute_has_value=label['has_value']
                                            )
                # print(stats)
                patch_gdf.loc[idx, f"{str(label['name'])}-stat"] = stats[0]
                patch_gdf.loc[idx, f"{str(label['name'])}-has_value"] = int(stats[1])

            profile.update({'transform': out_transform, 'width': patch_size, 'height': patch_size,
                            'driver': output_driver})
            write_patch(patch=patch, profile=profile, output_path=output_path)
            patch_gdf.loc[idx, v['output']] = relative_path
    # step 5/ save patches dataframe
    patch_gdf.to_file(os.path.join(output_dir, 'patch_dataset.geojson'), driver='GeoJSON')


if __name__ == '__main__':
    prefix = '/var/data/dl'
    raster_fields = {'T0_path': {'output': 'T0', 'input': 'T0.vrt'},
                     't1_path': {'output': 'T1', 'input': 'T1.vrt'}}
    mask_fields = {'change_pat': {'output': 'change',
                                  'input': 'change.vrt', 'labels': [{'name': 'change',
                                                                     'value': 1,
                                                                     'computer_stats': True,
                                                                     'has_value': True
                                                                     }
                                                                    ]
                                  }}

    split_zone_to_patch(raster_fields,
                        mask_fields,
                        output_dir='patches',
                        output_prefix=os.path.join(prefix, 'gers/change_dataset/'),
                        root=os.path.join(prefix, 'gers/change_dataset/'),
                        input_file='dataset_v1.shp',
                        patch_size=512,
                        resolution=0.2)
