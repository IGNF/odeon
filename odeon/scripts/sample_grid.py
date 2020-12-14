"""Grid sampling

This module performs a grid sampling from a file containing (multi)polygons. This is the first step in the ODEON
suite.

From a single shapefile containing several (multi)polygons, this code generates as many csv files as features.

Each csv file will contain the (x,y) coordinates in the same coordinate reference system as the input shapefile.
The distance between the points are computed in order to extract images with the dimension expressed in the json file
without overlap.


Example
-------
    Call this module from the root of the project:

    $ python -m src.grid_sampling -c src/json/grid_sampling.json -v

    This will read the configuration from a json file and create as many csv files as there are polygons in the
    shapefile.


Notes
-----
    skipped in json (see json_interpreter)

"""

import os
import fiona
import math
from shapely.geometry import shape, box, mapping, Point
from tqdm import tqdm
import odeon.commons.folder_manager as fm
from odeon.commons.logger.logger import get_file_handler
from odeon import LOGGER
from odeon.commons.core import BaseTool


class SampleGrid(BaseTool):
    """
    Callable class as entry point of the sample_grid tool
    """

    def __init__(self,
                 verbose,
                 input_file,
                 output_pattern,
                 image_size_pixel,
                 resolution,
                 strict_inclusion=False,
                 shift=False,
                 tight_mode=False):
        """

        Parameters
        ----------
        verbose : bool
            verbose level
        input_file : str
            shapefile containing the zones of interest for one "departement"
        output_pattern : str
            filename pattern used for each zone
        image_size_pixel : int
            number of pixel of an images
        resolution : Union[float, list of float]
            resolution for x and y dimension if different
        strict_inclusion : boolean
            True if the tile must be strictly included in the polygonal shape
        shift : boolean
            True to shift samples by half the size of a tile
        tight_mode : boolean
         sample points stick to the upper left corner
        """

        LOGGER.addHandler(get_file_handler(LOGGER, os.path.split(output_pattern)[0]))

        LOGGER.debug("Configuration :")
        LOGGER.debug(f"\tinput shapefile: {input_file}")
        LOGGER.debug(f"\toutput pattern: {output_pattern}")
        LOGGER.debug(f"\timage size (pixel): {image_size_pixel}")
        LOGGER.debug(f"\tpixel size (CRS unit): {resolution}")
        LOGGER.debug(f"\tstrict_inclusion: {strict_inclusion}")
        LOGGER.debug(f"\tshift (1 to shift centers): {shift}")

        if not os.path.isfile(input_file):
            LOGGER.debug(f"ERROR: file nor found: {input_file}")
        fm.create_folder(os.path.dirname(output_pattern))
        self.verbose = verbose
        self.input_file = input_file
        self.output_pattern = output_pattern
        self.image_size_pixel = image_size_pixel
        self.resolution = resolution if isinstance(resolution, list) else [resolution, resolution]
        self.strict_inclusion = strict_inclusion
        self.shift = shift
        self.tight_mode = tight_mode

    def __call__(self):
        """

        Returns
        -------

        """
        # read shape file and count shapes, order them by x
        LOGGER.info("retrieve geometry")
        geometry_list, crs = self.get_geometries_from_shp(self.input_file)
        # build output filename and create if necessary the output folder
        LOGGER.info("generate filename")
        filename_list = self.generate_filename(self.output_pattern, len(geometry_list))
        # get side of the expected images
        side = [self.image_size_pixel * x for x in self.resolution]
        # generate csv
        LOGGER.info("generate csv")
        self.generate_csv(geometry_list,
                          filename_list,
                          side,
                          crs,
                          self.strict_inclusion,
                          self.shift,
                          self.tight_mode,
                          self.verbose)
        return

    @staticmethod
    def setup_output(output_pattern):
        """
        Create the folder for the files created during the sampling
        :param output_pattern: the pattern to follow for the files
        """
        path = os.path.split(output_pattern)
        fm.create_folder(path)

    @staticmethod
    def get_geometries_from_shp(shapefile):
        """
        Returns a list of geometries ordered by x, then y
        Parameters
        ----------
        shapefile : str
            shapefile containing polygons (usually within a "departement")

        Returns
        -------
        list_of_geometry : list
            list of geometries (Shapely Polygons),
        crs : dict
            coordinates system used in the shapefile

        """
        geometry_list = []
        with fiona.open(shapefile, 'r') as layer:

            crs = layer.crs
            for feature in tqdm(layer):
                geometry = shape(feature['geometry'])
                geometry_list.append(geometry)

        return geometry_list, crs

    @staticmethod
    def generate_filename(output_pattern, no_of_samples):
        """
        Returns a list of filenames with complete path
        Parameters
        ----------
        output_pattern : str
            the pattern to follow when generating filenames
        no_of_samples : int
            number of filename to generate

        Returns
        -------
        filename_list : list
            a list of filenames
        """
        filename_list = []
        path, pattern = os.path.split(output_pattern)
        if "*" not in pattern:
            pattern = "*_" + pattern

        for i in range(no_of_samples):
            filename = os.path.join(path, pattern.replace("*", "zone" + str(i + 1)))
            filename_list.append(filename)

        return filename_list

    @staticmethod
    def generate_csv(geometry_list, filename_list, side, crs, strict_inclusion, shift, tight_mode, verbose):
        """
        Generate a list of coordinates (x,y) for each geometry and save this list
        Parameters
        ----------
        geometry_list : list of geometry
            list of geometry (square shaped areas)
        filename_list : list
            list of filename that will
        side : list of float
            size of image in CRS unit
        crs : dict
            coordinate system
        strict_inclusion : boolean
            True if the tile must be strictly included in the polygonal shape
        shift : boolean
            True to shift samples by half the size of a tile
        tight_mode : boolean
            sample points stick to the upper left corner
        verbose : bool
            verbose level
        """
        geometry_list.sort(key=lambda g: (g.bounds[0], g.bounds[1]))

        """ bbox: (x_1, y_1 (SW Point) x_2, y_2 (NE Point)) """
        bounding_box_list = [geometry.bounds for geometry in geometry_list]
        # TODO ? bounding_box_list.sort(key=lambda x: (x[0], x[1]))  # sort by x1 then y_2
        data = zip(filename_list, bounding_box_list, geometry_list)

        for filename, bounding_box, geometry in tqdm(data):
            # find limits
            x_1, y_1, x_2, y_2 = bounding_box
            if shift:
                x_1, y_1, x_2, y_2 = x_1 + side[0] / 2, y_1 + side[1] / 2, x_2 - side[0] / 2, y_2 - side[1] / 2

            LOGGER.debug(f"bounding_box: {bounding_box}")

            x_num, y_num = (x_2 - x_1) / side[0], (y_2 - y_1) / side[1]  # number of samples
            LOGGER.debug(f"x_num, y_num: {x_num}, {y_num}")

            coordinates = []
            tile_joint = [0, 0]
            if tight_mode is not True:
                tile_joint = [
                    ((x_2 - x_1) % side[0]) / x_num,
                    ((y_2 - y_1) % side[1]) / y_num
                ]

            LOGGER.debug(f"tile_joint: {tile_joint}")
            for i in range(math.ceil(x_num)):
                for j in range(math.ceil(y_num)):
                    coordinates.append((x_1 + (2*i+1)*side[0] / 2 + (i+1)*tile_joint[0],
                                        y_1 + (2*j+1)*side[1] / 2 + (j+1)*tile_joint[1]))

            if strict_inclusion:
                coordinates = [c for c in coordinates if SampleGrid.included(c[0], c[1], side, geometry)]

            SampleGrid.save_output(coordinates, filename, side, crs, verbose)

            LOGGER.debug(f"[{filename}]: {len(coordinates)} points")

    @staticmethod
    def included(x, y, side, geometry):

        tile = box(x - side[0] / 2, y - side[1] / 2, x + side[0] / 2, y + side[1] / 2)
        return tile.intersection(geometry).area / (side[0] * side[1]) > 0.999

    @staticmethod
    def save_output(coordinates, filename, side, crs, verbose):
        """
        Save coordinates and, if verbose is on, save shapefiles (area and center) for visual
        inspection
        Parameters
        ----------
        coordinates : list
            list of coordinates
        filename : str
            output file containing the coordinate in csv format
        side : list of float
            size of an image in CRS unit
        crs : dict
            coordinate system
        verbose : bool
            verbose level
        """

        # Save into a csv file
        csv_file = open(filename, 'w', encoding='utf-8', errors='ignore')
        for (x, y) in coordinates:
            csv_file.write(f"{round(x, 8)}; {round(y, 8)}\n")
        if verbose:
            # Save the patch shape into a shp file
            shp_filename = filename[:-4] + "_area.shp"
            shp_schema = {'geometry': 'Polygon', 'properties': {'id_sample': 'int'}}
            shp_file = fiona.open(shp_filename, 'w', crs=crs, driver='ESRI Shapefile', schema=shp_schema)
            for i, (x, y) in enumerate(coordinates):
                dx = side[0] / 2
                dy = side[1] / 2
                shp_file.write({
                    'properties': {'id_sample': i},
                    'geometry': mapping(box(x - dx, y - dy, x + dx, y + dy))
                })
            # Save the patch center into a shp file
            shp_filename = filename[:-4] + "_center.shp"
            shp_schema = {'geometry': 'Point', 'properties': {'id_sample': 'int'}}
            shp_file = fiona.open(shp_filename, 'w', crs=crs, driver='ESRI Shapefile', schema=shp_schema)

            for i, (x, y) in enumerate(coordinates):

                shp_file.write({
                    'properties': {'id_sample': i},
                    'geometry': mapping(Point(float(x), float(y)))
                })
