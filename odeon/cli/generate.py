"""
# Generation grid module
main entry point to generation tool


"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import rasterio
import fiona
from rasterio.windows import transform
from rasterio.features import rasterize, geometry_window
from odeon.commons.image import CollectionDatasetReader
from odeon.commons.rasterio import get_max_type
from odeon import LOGGER
from odeon.commons.dataframe import set_path_to_center, split_dataset_from_df
from odeon.commons.folder_manager import build_directories
from odeon.commons.logger.logger import get_new_logger, get_simple_handler
from odeon.commons.guard import geo_projection_raster_guard, geo_projection_vector_guard
from odeon.commons.guard import vector_driver_guard, files_exist, dirs_exist, raster_bands_exist
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.commons.core import BaseTool

" A logger for big message "
STD_OUT_LOGGER = get_new_logger("stdout_generation")
ch = get_simple_handler()
STD_OUT_LOGGER.addHandler(ch)


class Generator(BaseTool):

    def __init__(self,
                 image_layers,
                 vector_classes,
                 output_path=None,
                 poi_pattern=None,
                 train_test_split=-1,
                 train_val_split=-1,
                 compute_only_masks=False,
                 dem=False,
                 append=False,
                 image_size_pixel=512,
                 resolution=None
                 ):
        """Main generation function called when you use the cli tool generation

            Parameters
            ----------
            image_layers : dict
             a dict in the form {band_name: {path: file_path,band: []}
            vector_classes : dict
             a dict in the form {class_name: file_path}
            output_path : str
             the output path of generaiton
            poi_pattern : regex
             a regex to list the files of poi in input
            train_test_split : Union[float, None]
             the percentage of train in train test split
            train_val_split : Union[float, None]
             the percentage of train in train val split
            compute_only_masks : bool
             rather generate only the mask or not
            dem : bool
             create the DSM - DTM band or not if DSM and DTM are here
            append : bool
             append generated data to a previous a generation output
            image_size_pixel : int
             size of patch in pixel
            resolution : Union(float, list of float)
             resolution in x and y of the output patches

            Returns
            -------

            """
        self.shape_files = vector_classes
        self.raster_out = os.path.join(output_path, "full_mask.tiff")
        self.dict_of_raster = image_layers
        self.vector_classes = vector_classes
        self.img_size = image_size_pixel
        self.files = [glob.glob(poi_pattern)] if isinstance(poi_pattern, str) else [glob.glob(i) for i in poi_pattern]
        self.train_val_split = train_val_split
        self.train_test_split = train_test_split
        self.compute_only_masks = compute_only_masks
        self.dem = dem
        self.append = append
        self.resolution = resolution if isinstance(resolution, list) else [resolution, resolution]
        self.nb_of_image_band = 0
        self.output_path = output_path
        self.meta_img = None
        self.meta_msk = None
        self.splits = None
        self.paths = None
        self.df = None
        self.extent = None
        self.num_seq = None
        self.check()

    def __call__(self):
        """
        Callable object, we run Generation by calling the Object
        example: generator = Generator(**options)
                 generator() # run the generation
        Returns
        -------

        """
        try:

            LOGGER.info("configuration")
            self.get_bounds()
            self.set_sequence()
            self.split()
            self.set_number_of_image_band()
            self.set_meta_img_msk()

            for i in range(self.num_seq):
                self.clean()
                self.pre_rasterize_mask(i)
                self.generate(i)

        except Exception as error:

            raise OdeonError(ErrorCodes.ERR_GENERATION_ERROR,
                             "something went wrong during generation",
                             stack_trace=error)
        finally:

            self.clean()

    def check(self):
        """
        check confiuration inputs to be suitale for the generation process

        Returns
        -------

        """
        try:
            for elt in self.files:

                if len(elt) == 0:

                    raise OdeonError(ErrorCodes.ERR_FILE_NOT_EXIST,
                                     "couldn't find file with poi_pattern")

            files_exist(elt)
            dirs_exist([self.output_path])

        except OdeonError as oe:

            raise OdeonError(ErrorCodes.ERR_GENERATION_ERROR,
                             "something went wrong during generation", stack_trace=oe)

        # we compute the number of bands for the patch image output
        # and the scales factors for each raster with the targeted resolution
        for raster_name, raster in self.dict_of_raster.items():

            try:

                files_exist([raster["path"]])
                geo_projection_raster_guard(raster["path"])
                # raster_driver_guard(raster["path"])
                raster_bands_exist(raster["path"], raster["bands"])

            except OdeonError as oe:

                raise OdeonError(ErrorCodes.ERR_GENERATION_ERROR,
                                 "something went wrong during generation configuration",
                                 stack_trace=oe)

        for name, vector in self.shape_files.items():

            try:

                files_exist([vector])
                geo_projection_vector_guard(vector)
                vector_driver_guard(vector)

            except OdeonError as oe:

                raise OdeonError(ErrorCodes.ERR_GENERATION_ERROR,
                                 "something went wrong during generation configuration",
                                 stack_trace=oe)

    def set_sequence(self):
        """

        Returns
        -------

        """
        len_seq = None
        for name, raster in self.dict_of_raster.items():

            LOGGER.debug(f"len of seq {len_seq}")

            if isinstance(raster["path"], str):

                len_seq = 1 if len_seq is None else len_seq

                if len_seq != 1:

                    raise OdeonError(ErrorCodes.ERR_GENERATION_ERROR,
                                     "we expect the same type in each rater layer,"
                                     " but one is array and the other is string")

                self.dict_of_raster[name]["path"] = [raster["path"]]

            else:

                len_seq = len(self.dict_of_raster[name]["path"]) if len_seq is None else len_seq

                if len_seq != len(self.dict_of_raster[name]["path"]):
                    raise OdeonError(ErrorCodes.ERR_GENERATION_ERROR,
                                     "we expect the same length of array in each image layers")

        for name, vector in self.vector_classes.items():

            LOGGER.debug(f"len of seq {len_seq}")

            if isinstance(vector, str):

                if len_seq != 1:

                    raise OdeonError(ErrorCodes.ERR_GENERATION_ERROR,
                                     "something went wrong during generation configuration:"
                                     "\nyou declared an array in image_layers but a string is found in vector_layers")

                else:

                    self.vector_classes[name] = [vector]

            else:

                if len_seq != len(vector):

                    raise OdeonError(ErrorCodes.ERR_GENERATION_ERROR,
                                     "something went wrong during generation configuration:"
                                     "\nthe length of image_layers array is different of "
                                     "of poi_pattern array")

        LOGGER.debug(f"len of seq {len_seq}")

        if len_seq != len(self.files):
            LOGGER.debug(f"length of file  {self.files}: {len(self.files)}")
            raise OdeonError(ErrorCodes.ERR_GENERATION_ERROR,
                             "something went wrong during generation configuration:"
                             "\nthe length of image_layers array is different of "
                             "of poi_pattern array")
        LOGGER.debug(f" image_layers: {self.dict_of_raster}, vector layers: {self.vector_classes}, "
                     f" poi_pattern: {self.files}")
        self.num_seq = len_seq

    def get_bounds(self):

        self.df: pd.DataFrame = pd.DataFrame(columns=["num_seq", "x", "y"])

        """ iterates over csv files """
        if isinstance(self.files, str):

            for file in self.files:

                df_in = pd.read_csv(file, delimiter=";",
                                    header=None,
                                    names=["x", "y"]).sample(frac=1)
                df_in["num_seq"] = 1
                self.df = self.df.append(df_in,
                                         ignore_index=True)

        else:

            for idx, l in enumerate(self.files):

                for file in l:

                    df_in = pd.read_csv(file, delimiter=";",
                                        header=None,
                                        names=["x", "y"]).sample(frac=1)

                    df_in["num_seq"] = idx
                    self.df = self.df.append(df_in,
                                             ignore_index=True)

        left, bottom, top, right = self.df["x"].min(), self.df["y"].min(), self.df["y"].max(), self.df["x"].max()
        self.extent = {"left": left, "bottom": bottom, "top": top, "right": right}
        LOGGER.debug(self.extent)
        LOGGER.debug(self.df)

    def set_number_of_image_band(self):
        """
        Compute and set the number of band for each image Patch
        Returns
        -------

        """

        for raster_name, raster in self.dict_of_raster.items():

            self.nb_of_image_band += len(raster["bands"])

        if ("DSM" and "DTM") in self.dict_of_raster and self.dem:

            dsm_length = len(self.dict_of_raster["DSM"]["bands"])
            dtm_length = len(self.dict_of_raster["DTM"]["bands"])
            self.nb_of_image_band -= (dsm_length + dtm_length - 1)

        LOGGER.debug(f"number of raster band in img patch: {self.nb_of_image_band}")

    def set_meta_img_msk(self):
        """
        Set metadata of image patch and mask patch in rasterio format

        Returns
        -------

        """
        with rasterio.open(next(iter(self.dict_of_raster.values()))["path"][0]) as dst:
            self.meta_msk = {"crs": dst.meta["crs"],
                             "transform": dst.meta["transform"],
                             "driver": "GTiff",
                             "count": len(self.vector_classes) + 1,
                             "width": self.img_size,
                             "height": self.img_size,
                             "dtype": rasterio.uint8}

            self.meta_img = self.meta_msk.copy()
            self.meta_img["count"] = self.nb_of_image_band
            self.meta_img["dtype"] = get_max_type(self.dict_of_raster)
            self.meta_img["resolution"] = self.resolution

        LOGGER.debug(f"meta mask: {self.meta_msk}, meta img: {self.meta_img}")

    def split(self):
        """
        Merge the input CSV in a panda DataFrame
        and split the dataset in train, val, test format
        depending on input options.

        Returns
        -------

        """

        self.splits = {}
        self.paths = {}
        self.df["img_file"] = ""
        self.df["msk_file"] = ""

        if self.train_val_split == -1 and self.train_test_split == -1:

            # case only train
            train = self.df
            train_path = os.path.join(self.output_path, "train")
            train_img_path = os.path.join(train_path, "img")
            train_msk_path = os.path.join(train_path, "msk")
            self.paths["train_img_path"] = train_img_path
            self.paths["train_msk_path"] = train_msk_path
            train = set_path_to_center(train,
                                       train_img_path,
                                       train_msk_path)
            self.splits["train"] = train

        elif self.train_val_split == 0 and self.train_test_split == -1:

            # case only val
            val = self.df
            val_path = os.path.join(self.output_path, "val")
            val_img_path = os.path.join(val_path, "img")
            val_msk_path = os.path.join(val_path, "msk")
            self.paths["val_img_path"] = val_img_path
            self.paths["val_msk_path"] = val_msk_path
            val = set_path_to_center(val,
                                     val_img_path,
                                     val_msk_path)
            self.splits["val"] = val

        elif self.train_val_split == -1 and self.train_test_split == 0:

            # case only test
            test = self.df
            test_path = os.path.join(self.output_path, "test")
            test_img_path = os.path.join(test_path, "img")
            test_msk_path = os.path.join(test_path, "msk")
            self.paths["test_img_path"] = test_img_path
            self.paths["test_msk_path"] = test_msk_path
            test = set_path_to_center(test,
                                      test_img_path,
                                      test_msk_path)
            self.splits["test"] = test

        elif self.train_val_split > 0 and self.train_test_split == -1:

            LOGGER.debug("train/val only case")
            # case only train/val split
            train = self.df
            train, val = split_dataset_from_df(train,
                                               self.train_val_split)
            train_path = os.path.join(self.output_path, "train")
            train_img_path = os.path.join(train_path, "img")
            train_msk_path = os.path.join(train_path, "msk")
            self.paths["train_img_path"] = train_img_path
            self.paths["train_msk_path"] = train_msk_path
            train = set_path_to_center(train,
                                       train_img_path,
                                       train_msk_path)
            self.splits["train"] = train

            val_path = os.path.join(self.output_path, "val")
            val_img_path = os.path.join(val_path, "img")
            val_msk_path = os.path.join(val_path, "msk")
            self.paths["val_img_path"] = val_img_path
            self.paths["val_msk_path"] = val_msk_path

            val = set_path_to_center(val,
                                     val_img_path,
                                     val_msk_path)
            self.splits["val"] = val

        elif self.train_val_split == -1 and self.train_test_split > 0:

            # case only train/test split
            train = self.df
            train, test = split_dataset_from_df(train, self.train_test_split)
            train_path = os.path.join(self.output_path, "train")
            train_img_path = os.path.join(train_path, "img")
            train_msk_path = os.path.join(train_path, "msk")
            self.paths["train_img_path"] = train_img_path
            self.paths["train_msk_path"] = train_msk_path
            train = set_path_to_center(train,
                                       train_img_path,
                                       train_msk_path)
            self.splits["train"] = train

            test_path = os.path.join(self.output_path, "test")
            test_img_path = os.path.join(test_path, "img")
            test_msk_path = os.path.join(test_path, "msk")
            self.paths["test_img_path"] = test_img_path
            self.paths["test_msk_path"] = test_msk_path

            test = set_path_to_center(test,
                                      test_img_path,
                                      test_msk_path)
            self.splits["test"] = test

        elif self.train_val_split == 0 and self.train_test_split > 0:

            # case only val/test split
            val = self.df
            val, test = split_dataset_from_df(val, self.train_test_split)

            val_path = os.path.join(self.output_path, "val")
            val_img_path = os.path.join(val_path, "img")
            val_msk_path = os.path.join(val_path, "msk")
            self.paths["val_img_path"] = val_img_path
            self.paths["val_msk_path"] = val_msk_path
            val = set_path_to_center(val,
                                     val_img_path,
                                     val_msk_path)
            self.splits["val"] = val

            test_path = os.path.join(self.output_path, "test")
            test_img_path = os.path.join(test_path, "img")
            test_msk_path = os.path.join(test_path, "msk")
            self.paths["test_img_path"] = test_img_path
            self.paths["test_msk_path"] = test_msk_path

            test = set_path_to_center(test,
                                      test_img_path,
                                      test_msk_path)
            self.splits["test"] = test

        else:

            # general case train/val/test split
            train = self.df
            train, test = split_dataset_from_df(train, self.train_test_split)
            train, val = split_dataset_from_df(train,
                                               self.train_val_split)

            train_path = os.path.join(self.output_path, "train")
            train_img_path = os.path.join(train_path, "img")
            train_msk_path = os.path.join(train_path, "msk")
            self.paths["train_img_path"] = train_img_path
            self.paths["train_msk_path"] = train_msk_path
            train = set_path_to_center(train,
                                       train_img_path,
                                       train_msk_path)
            self.splits["train"] = train

            val_path = os.path.join(self.output_path, "val")
            val_img_path = os.path.join(val_path, "img")
            val_msk_path = os.path.join(val_path, "msk")
            self.paths["val_img_path"] = val_img_path
            self.paths["val_msk_path"] = val_msk_path

            val = set_path_to_center(val,
                                     val_img_path,
                                     val_msk_path)
            self.splits["val"] = val

            test_path = os.path.join(self.output_path, "test")
            test_img_path = os.path.join(test_path, "img")
            test_msk_path = os.path.join(test_path, "msk")
            self.paths["test_img_path"] = test_img_path
            self.paths["test_msk_path"] = test_msk_path

            test = set_path_to_center(test,
                                      test_img_path,
                                      test_msk_path)
            self.splits["test"] = test

        build_directories(self.paths, self.append)

        LOGGER.debug(self.splits)

    def pre_rasterize_mask(self, pointer):
        """

        Returns
        -------

        """

        stdout = f'''
                ##############################################
                #                                            #
                #   Preprocess: Shapefile rasterization      #
                #    part {pointer + 1} of {self.num_seq}    #
                ##############################################
                '''

        with rasterio.open(next(iter(self.dict_of_raster.values()))["path"][pointer]) as dataset:

            meta = dataset.meta
            meta["dtype"] = rasterio.uint8
            meta["count"] = len(self.shape_files) + 1
            meta["driver"] = "GTiff"

        STD_OUT_LOGGER.info(stdout)

        with rasterio.open(self.raster_out, 'w+', **meta,
                           NBITS=1,
                           BIGTIFF="IF_NEEDED",
                           tiled=True,
                           blockxsize=self.img_size,
                           blockysize=self.img_size,
                           sparse_ok="TRUE") as new_dataset:

            for idx, (name, shape_file) in enumerate(self.shape_files.items(), start=1):
                LOGGER.info(f"shape file: {shape_file[pointer]}")

                with fiona.open(shape_file[pointer]) as polygons:

                    for i, polygon in tqdm(enumerate(polygons), total=len(polygons)):

                        try:

                            geometry = polygon['geometry']

                            polygon_window = geometry_window(
                                new_dataset,
                                [geometry],
                                pixel_precision=6).round_shape(op='ceil',
                                                               pixel_precision=4)

                            polygon_shape = (polygon_window.height, polygon_window.width)
                            tuples = [geometry]
                            polygon_transform = transform(polygon_window, new_dataset.transform)

                            polygon_band = rasterize(tuples,
                                                     out_shape=polygon_shape,
                                                     default_value=1,
                                                     transform=polygon_transform,
                                                     dtype=rasterio.uint8,
                                                     fill=0)

                            old_band = new_dataset.read(idx, window=polygon_window)
                            new_band = old_band.astype(np.bool) + polygon_band.astype(np.bool)
                            new_band = new_band.astype(np.uint8)
                            new_dataset.write_band(idx,
                                                   new_band,
                                                   window=polygon_window)

                            # building the no label band
                            """
                            bands = new_dataset.read([1, len(self.shape_files)], window=polygon_window).astype(np.uint8)
                            bands = bands.astype(np.bool).astype(np.uint8)
                            other_band = np.sum(bands, axis=0, dtype=np.uint8)
                            other_band = (other_band == 0).astype(np.uint8)
                            new_dataset.write_band(len(self.shape_files) + 1, other_band, window=polygon_window)
                            """

                        except rasterio.errors.WindowError as error:

                            LOGGER.warning(f"shape file: {shape_file}, polygon id: {i}")
                            """this excepion occurs if a polygon it out of the raster bounds"""
                            LOGGER.warning(f"window error during the rasterization"
                                           f"of polygon {i} of classe {name} \n "
                                           f"error: {error}")

                        except MemoryError as error:

                            LOGGER.warning(f"shape file: {shape_file}, polygon shape: , polygon id: {i}")
                            """this excepion occurs if a polygon it out of the raster bounds"""
                            LOGGER.warning(f"window error during the rasterization"
                                           f"of polygon {i} of classe {name} \n "
                                           f"error: {error}")
                            """
                            raise OdeonError(ErrorCodes.ERR_GENERATION_ERROR,
                                             f"At least one of the polygon of shape file {shape_file} is too big",
                                             error=error)
                            """

                        except ValueError as error:

                            LOGGER.warning(f"shape file: {shape_file}, polygon id: {i}")
                            """this excepion occurs if a polygon it out of the raster bounds"""
                            LOGGER.warning(f"window error during the rasterization"
                                           f"of polygon {i} of classe {name} \n "
                                           f"error: {error}")

    def generate(self, pointer):
        """

        Returns
        -------

        """

        def generate_data(df,
                          meta_msk,
                          meta_img,
                          raster_out,
                          dict_of_raster,
                          dem,
                          compute_only_masks):
            """
            generate and writes a Patch (a tuple image, mask) for each
            row of the dataframe representing the Dataset.

            Parameters
            ----------
            df : pandas.DataFrame
             the list of center point for each patch
            meta_msk : dict
             rasterio metadata for mask generation
            meta_img : dict
             rasterio metadata for img generation
            raster_out : str
             path to write
            dict_of_raster : dict
             a dictionary of raster name, raster file
            dem : bool
             rather calculate or not DSM - DTM and create a new band with it
            compute_only_masks : bool
             computer only mask or not

            Returns
            -------
            None

            """
            for _, center in tqdm(df.iterrows(), total=len(df)):

                try:
                    # Verification if the future output file already exists. If True, the former one will be deleted.
                    if os.path.isfile(center["img_file"]):
                        os.remove(center["img_file"])

                    CollectionDatasetReader.stack_window_raster(center,
                                                                dict_of_raster,
                                                                meta_img,
                                                                dem,
                                                                compute_only_masks,
                                                                raster_out,
                                                                meta_msk)
                except Exception as error:

                    raise OdeonError(ErrorCodes.ERR_GENERATION_ERROR,
                                     "something went wrong during generation",
                                     stack_trace=error)

        stdout = f'''
                ##############################################
                #                                            #
                #   Generating images and masks              #
                #    part {pointer + 1} of {self.num_seq}    #
                ##############################################
                '''

        STD_OUT_LOGGER.info(stdout)

        # Modifications on dict_of_raster from generation to match with the dict_of_raster in detection
        # in order to use the same function get_stacked_window_collection for DEM computation.

        for i, source_type in enumerate(self.dict_of_raster.keys()):
            self.dict_of_raster[source_type]['connection'] = \
                rasterio.open(self.dict_of_raster[source_type]['path'][pointer])
            if i == 0:
                self.meta_msk['transform'] = self.dict_of_raster[source_type]['connection'].transform
                self.meta_img['transform'] = self.meta_msk['transform']

        for split_name, split in self.splits.items():
            LOGGER.info(f"generating {split_name} data")
            s = split[split["num_seq"] == pointer]
            LOGGER.debug(s)
            generate_data(s,
                          self.meta_msk,
                          self.meta_img,
                          self.raster_out,
                          self.dict_of_raster,
                          self.dem,
                          self.compute_only_masks)

            output_split = os.path.join(self.output_path, f"{split_name}.csv")

            if (self.append or pointer > 0) is True and os.path.isfile(output_split):
                df = pd.read_csv(output_split, header=None, names=["img_file", "msk_file"])
                df = df.append(split, ignore_index=True)
                df[["img_file", "msk_file"]].to_csv(output_split,
                                                    index=False,
                                                    header=False)

            else:
                split[["img_file", "msk_file"]].to_csv(output_split,
                                                       index=False,
                                                       header=False)
        # Close all opened rasters
        for source_type in self.dict_of_raster.keys():
            self.dict_of_raster[source_type]['connection'].close()

    def clean(self):
        """Clean temporary pre-rasterized mask

        Returns
        -------

        """
        if os.path.isfile(self.raster_out):

            os.remove(self.raster_out)
