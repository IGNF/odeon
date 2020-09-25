"module of Jobs classes, typically detection jobs"
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds, transform
from odeon.commons.rasterio import affine_to_tuple
from odeon import LOGGER
from odeon.commons.shape import create_box_from_bounds


class BaseDetectionJob:

    pass


class PatchJobDetection(BaseDetectionJob):
    """
    Job class used for patch based detection
    It simply encapsulates a pandas.DataFrame
    """

    def __init__(self, df: pd.DataFrame, path, recover=False):

        self._df = df
        self._job_done = None
        self._path = path
        self._job_file = os.path.join(self._path, "job_detection.csv")
        self._recover = recover

        if self._recover:

            if os.path.isfile(self._job_file):

                df = pd.read_csv(self._job_file)
                self._df = df
                self.keep_only_todo_list()

    def __len__(self):

        return len(self._df)

    def __str__(self):

        return f" PatchJobDetection with dataframe {self._df}"

    def get_row_at(self, index):

        return self._df.iloc[index]

    def get_cell_at(self, index, column):

        return self._df.at[index, column]

    def set_cell_at(self, index, column, value):

        self._df.at[index, column] = value

    def get_job_done(self):

        return self._df.loc[self._df.job_done]

    def get_todo_list(self):

        return self._df.loc[~self._df.job_done]

    def keep_only_todo_list(self):

        self._job_done = self.get_job_done()
        self._df = self.get_todo_list()

    def save_job(self):

        out = self._df

        if self._job_done is not None:

            pd.concat([out, self._job_done])

        out.to_csv(os.path.join(self._path, self._job_file))


class ZoneDetectionJob(PatchJobDetection):

    def __init__(self, df: pd.DataFrame, path, recover=False, file_name="job_detection.shp"):

        self._df = df
        self._job_done = None
        self._path = path
        self._job_file = os.path.join(self._path, file_name)
        self._recover = recover

        if self._recover:

            if os.path.isfile(self._job_file):

                df = pd.read_csv(self._job_file)
                self._df = df
                self.keep_only_todo_list()

    def save_job(self):

        out = self._df

        if self._job_done is not None:

            pd.concat([out, self._job_done])

        out.to_file(os.path.join(self._path, self._job_file))

    def get_bounds_at(self, idx):
        LOGGER.debug(f"index {idx}")
        LOGGER.debug(f"indices:\n {self._df.index.values.tolist()}")
        return self.get_cell_at(idx, "geometry").bounds

    @staticmethod
    def build_job(gdf, output_size, resolution, meta, overlap=0, out_dalle_size=None, write_job=True):

        output_size_u = output_size * resolution
        overlap_u = overlap * resolution
        step = output_size_u - (2 * overlap_u)
        tmp_list = []

        if out_dalle_size is None:

            for idx, row in gdf.iterrows():

                bounds = row["geometry"].bounds

                min_x, min_y = int(bounds[0]), int(bounds[1])
                max_x, max_y = int(bounds[2]), int(bounds[3])
                # LOGGER.debug(f"minx: {min_x}, miny: {min_y}, maxx: {max_x}, maxy: {max_y}")

                for i in np.arange(min_x - overlap_u, max_x + overlap_u, step):

                    for j in np.arange(min_y - overlap_u, max_y + overlap_u, step):

                        "handling case where the extent is not a multiple of step"
                        if i + output_size_u > max_x + overlap_u:

                            i = max_x + overlap_u - output_size_u

                        if j + output_size_u > max_y + overlap_u:

                            j = max_y + overlap_u - output_size_u

                        # LOGGER.debug(f"after: i {i}, j {j}")
                        # computes receptive fields and receptive field Affine transform
                        left = i + overlap_u
                        bottom = i + output_size_u - overlap_u
                        right = j + overlap_u
                        top = j + output_size_u - overlap_u

                        height = output_size - overlap
                        width = height
                        # LOGGER.debug(width)
                        """
                        window = from_bounds(left, bottom, right, top, transform=meta["transform"])
                        window = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
                        LOGGER.debug(str(window))
                        affine = transform(window, meta["transform"])
                        LOGGER.debug(str(window))
                        LOGGER.debug(str(affine))
                        """
                        col, row = int((j - min_y) // resolution), int((i - min_x) // resolution)
                        row_d = {
                                 "id": f"{idx}-{row}-{col}",
                                 "zone": idx,
                                 "job_done": False,
                                 "left": left,
                                 "bottom": bottom,
                                 "right": right,
                                 "top": top,
                                 "geometry": create_box_from_bounds(i,  i + output_size_u, j, j + output_size_u)
                                }
                        tmp_list.append(row_d)
            gdf_output = gpd.GeoDataFrame(tmp_list, crs=gdf.crs, geometry="geometry")
            gdf_output

            if write_job:

                return gdf_output, gdf

            else:

                return gdf_output
        else:

            pass


class WriteJob(ZoneDetectionJob):

    def __init__(self, df: pd.DataFrame, path, recover=False, file_name="job_write.shp"):

        super(WriteJob, self).__init__(df, path, recover=recover, file_name=file_name)
