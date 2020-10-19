"module of Jobs classes, typically detection jobs"
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from odeon import LOGGER
from odeon.commons.shape import create_box_from_bounds


class BaseDetectionJob:

    pass


class PatchJobDetection(BaseDetectionJob):
    """
    Job class used for patch based detection
    It simply encapsulates a pandas.DataFrame
    """

    def __init__(self, df: pd.DataFrame, path, recover=False, file_name="detection_job.csv"):

        self._df = df
        self._job_done = None
        self._path = path
        self._job_file = os.path.join(self._path, file_name)
        self._recover = recover

        if self._recover and os.path.isfile(self._job_file):

            self._df = self.read_file(self._job_file)

        self.keep_only_todo_list()

    def __len__(self):

        return len(self._df)

    def __str__(self):

        return f" PatchJobDetection with dataframe {self._df}"

    @classmethod
    def read_file(cls, file_name):

        return pd.read_csv(file_name)

    def get_row_at(self, index):

        return self._df.iloc[index]

    def get_cell_at(self, index, column):

        return self._df.at[index, column]

    def set_cell_at(self, index, column, value):

        self._df.at[index, column] = value

    def get_job_done(self):

        return self._df[self._df["job_done"] == 1]

    def get_todo_list(self):

        return self._df[self._df["job_done"] == 0]

    def keep_only_todo_list(self):

        self._job_done = self.get_job_done()
        self._df = self.get_todo_list()
        self._df.reset_index(drop=True)
        self._job_done.reset_index(drop=True)

    def save_job(self):

        out = self._df

        if self._job_done is not None:

            out = pd.concat([out, self._job_done])

        out.to_csv(os.path.join(self._path, self._job_file))


class ZoneDetectionJob(PatchJobDetection):

    def __init__(self,
                 df: pd.DataFrame,
                 path,
                 recover=False,
                 file_name="detection_job.shp"):

        super(ZoneDetectionJob, self).__init__(df, path, recover=recover, file_name=file_name)

    def __str__(self):

        return f" Zone Job Detection with dataframe {self._df}"

    @classmethod
    def read_file(cls, file_name):

        return gpd.read_file(file_name)

    def get_job_done(self):
        """
        from random import choices
        population = [0, 1]
        weights = [0.01, 0.99]
        self._df["job_done"] = self._df["job_done"].apply(lambda x: choices(population, weights)[0])
        """
        df_grouped = self._df.groupby(["output_id"]).agg({"job_done": ["sum", "count"]})
        LOGGER.debug(df_grouped)
        LOGGER.debug(df_grouped.index.values.tolist())
        LOGGER.debug(df_grouped.columns)
        job_done_id = df_grouped[df_grouped["job_done", "sum"] == df_grouped["job_done", "count"]]
        LOGGER.debug(job_done_id.index.values.tolist())
        LOGGER.debug(len(job_done_id))

        return self._df[~self._df["output_id"].isin(job_done_id)]

    def get_todo_list(self):

        if self._job_done is None:

            return self._df

        else:

            return self._df[~self._df["output_id"].isin([self._job_done["output_id"]])]

    def save_job(self):

        out = self._df

        if self._job_done is not None:

            out = pd.concat([out, self._job_done])

        LOGGER.debug(len(out))
        LOGGER.debug(out)
        LOGGER.debug(out.columns)
        LOGGER.debug(out.dtypes)
        LOGGER.debug(out["id"])
        out.to_file(self._job_file)

    def get_bounds_at(self, idx):
        LOGGER.debug(f"index {idx}")
        LOGGER.debug(f"indices:\n {self._df.index.values.tolist()}")
        return self.get_cell_at(idx, "geometry").bounds

    def job_finished_for_output_id(self, output_id):

        dalle_df = self._df[self._df["output_id"] == output_id]
        if len(dalle_df[dalle_df["job_done"] == 1]) == len(dalle_df):

            return True
        else:
            return False

    def mark_dalle_job_as_done(self, output_id):

        self._df = self._df[~self._df["output_id"].isin([output_id])]
        self._job_done = pd.concat([self._job_done, self._df[self._df["output_id"].isin([output_id])]])

    @staticmethod
    def build_job(gdf, output_size, resolution, meta, overlap=0, out_dalle_size=None):

        output_size_u = output_size * resolution
        overlap_u = overlap * resolution
        step = output_size_u - (2 * overlap_u)
        tmp_list = []
        write_gdf = None

        for idx, df_row in gdf.iterrows():

            bounds = df_row["geometry"].bounds

            min_x, min_y = int(bounds[0]), int(bounds[1])
            max_x, max_y = int(bounds[2]), int(bounds[3])
            name = df_row["id"] if "id" in gdf.columns else idx

            if out_dalle_size is not None:

                for i in np.arange(min_x, max_x, out_dalle_size):

                    for j in np.arange(min_y, max_y, out_dalle_size):

                        "handling case where the extent is not a multiple of step"
                        if i + out_dalle_size > max_x:

                            i = max_x - out_dalle_size

                        if j + output_size_u > max_y:

                            j = max_y - output_size_u

                        # LOGGER.debug(f"after: i {i}, j {j}")
                        # computes ROI and ROI Affine transform
                        left = i
                        right = i + out_dalle_size
                        bottom = j
                        top = j + out_dalle_size

                        # width = height = output_size - overlap
                        # LOGGER.debug(width)
                        """
                        window = from_bounds(left, bottom, right, top, transform=meta["transform"])
                        window = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
                        LOGGER.debug(str(window))
                        affine = transform(window, meta["transform"])
                        LOGGER.debug(str(window))
                        LOGGER.debug(str(affine))
                        """
                        col, row = int((j - min_y) // resolution) + 1, int((i - min_x) // resolution) + 1

                        row_d = {
                                    "id": f"{name}-{row}-{col}",
                                    "name": name,
                                    "job_done": False,
                                    "left": left,
                                    "bottom": bottom,
                                    "right": right,
                                    "top": top,
                                    "affine": "",
                                    "patch_count": 0,
                                    "nb_patch_done": 0,
                                    "geometry": create_box_from_bounds(i,  i + out_dalle_size, j, j + out_dalle_size)
                                }
                        tmp_list.append(row_d)

                write_gdf = gpd.GeoDataFrame(tmp_list, crs=gdf.crs, geometry="geometry")

            else:

                """
                col, row = int((min_y) // resolution) + 1, int((min_x) // resolution) + 1
                row_d = {
                                    "id": f"{name}-{row}-{col}",
                                    "name": name,
                                    "job_done": 0,
                                    "left": min_x,
                                    "bottom": min_y,
                                    "right": max_x,
                                    "top": max_y,
                                    "affine": "",
                                    "patch_count": 0,
                                    "nb_patch_done": 0,
                                    "geometry": df_row["geometry"]
                        }

                tmp_list.append(row_d)
                """
                write_gdf = gdf

            # write_gdf = gpd.GeoDataFrame(tmp_list, crs=gdf.crs, geometry="geometry")

        tmp_list = []
        for idx, df_row in write_gdf.iterrows():

            bounds = df_row["geometry"].bounds
            LOGGER.debug(bounds)
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
                    # computes ROI and ROI Affine transform
                    left = i + overlap_u
                    right = i + output_size_u - overlap_u
                    bottom = j + overlap_u
                    top = j + output_size_u - overlap_u

                    # width = height = output_size - overlap
                    # LOGGER.debug(width)
                    """
                    window = from_bounds(left, bottom, right, top, transform=meta["transform"])
                    window = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
                    LOGGER.debug(str(window))
                    affine = transform(window, meta["transform"])
                    LOGGER.debug(str(window))
                    LOGGER.debug(str(affine))
                    """
                    col, row = int((j - min_y) // resolution) + 1, int((i - min_x) // resolution) + 1
                    if out_dalle_size is not None:
                        row_d = {
                                    "id": str(f"{idx + 1}-{row}-{col}"),
                                    "output_id": df_row["id"],
                                    "dalle_done": 0,
                                    "job_done": 0,
                                    "left": left,
                                    "bottom": bottom,
                                    "right": right,
                                    "top": top,
                                    "left_o": df_row["left"],
                                    "bottom_o": df_row["bottom"],
                                    "right_o": df_row["right"],
                                    "top_o": df_row["top"],
                                    "geometry": create_box_from_bounds(i,  i + output_size_u, j, j + output_size_u)
                                }
                    else:
                        row_d = {
                                    "id": str(f"{idx + 1}-{row}-{col}"),
                                    "output_id": str(f"{idx + 1}-{row}-{col}"),
                                    "dalle_done": 0,
                                    "job_done": 0,
                                    "left": left,
                                    "bottom": bottom,
                                    "right": right,
                                    "top": top,
                                    "left_o": left,
                                    "bottom_o": bottom,
                                    "right_o": right,
                                    "top_o": top,
                                    "geometry": create_box_from_bounds(i,  i + output_size_u, j, j + output_size_u)
                                }
                    tmp_list.append(row_d)
                    if out_dalle_size is not None:

                        write_gdf.at[idx, "patch_count"] += 1

        gdf_output = gpd.GeoDataFrame(tmp_list, crs=gdf.crs, geometry="geometry")

        return gdf_output, write_gdf


class ZoneDetectionJobNoDalle(PatchJobDetection):

    def __init__(self, df: pd.DataFrame, path, recover=False, file_name="detection_job.shp"):

        super(ZoneDetectionJobNoDalle, self).__init__(df, path, recover, file_name=file_name)
        # LOGGER.info(self._df.index)
        # exit(0)

    @classmethod
    def read_file(cls, file_name):

        return gpd.read_file(file_name)

    def get_bounds_at(self, idx):
        LOGGER.debug(f"index {idx}")
        LOGGER.debug(f"indices:\n {self._df.index.values.tolist()}")
        return self.get_cell_at(idx, "geometry").bounds

    def save_job(self):

        out = self._df

        if self._job_done is not None:

            out = pd.concat([out, self._job_done])

        LOGGER.debug(len(out))
        LOGGER.debug(out)
        LOGGER.debug(out.columns)
        LOGGER.debug(out.dtypes)
        LOGGER.debug(out["id"])
        out.to_file(self._job_file)
