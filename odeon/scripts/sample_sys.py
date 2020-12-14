"""Systematic sampling
this modules is the main entry point of sample_sys and performs a
systematic sampling on a list of polygon geo-referenced
"""
from odeon import LOGGER
from odeon.commons.logger.logger import get_new_logger, get_simple_handler
from odeon.commons.core import BaseTool
from odeon.commons.sampling import get_roi_limits, get_roi_limits_with_filter
from odeon.commons.sampling import CountFunctor
from odeon.commons.sampling import apply_tile_functor, sum_area
from odeon.commons.sampling import SampleFunctor
from odeon.commons.exception import OdeonError


class SampleSys(BaseTool):
    """
    Computes a systematic sampling

    (see https://en.wikipedia.org/wiki/Systematic_sampling)
    on an input shape file defining the extent based on a list of geometry contained in another shape file.

    The pseudo code is

    - initialisation with the global configuration parameters
    - retrieve the ROI limits  based on configuration and the associated geometries
    - compute statistics on the ROIs

      - retrieve the associated Functor class and initialize the object to get
        pixel based or patch based statistics
      - For each geometry:

        - compute the associated bounding box
        - for each block of each ROI included in the bounding box of the geometry:

          - compute by block the number of patch or pixel and accumulate the result

        - Retrieving of the computed global statistics

    - sampling on the global extent

      - computation of the sampling rate (total number of pixel / number of sample)
      - configuration of the sampling functor (pixel based or patch based)
      - for each tile containing in the global extent:

        - sampling of pixels contained in the intersection of mask and extent.
        - writing / updating of the output files with the sampled pixels.

    """

    def __init__(self,
                 mask_path,
                 output_path,
                 output_type,
                 sample_type,
                 number_of_sample,
                 invert,
                 buffer_size,
                 extent_path,
                 filter_field,
                 filter_value,
                 resolution,
                 tile_size_mo,
                 patch_size,
                 patch_min_density
                 ):
        """

        Parameters
        ----------
        mask_path : str
        output_path
        output_type
        sample_type
        number_of_sample
        invert
        buffer_size
        extent_path
        filter_field
        filter_value
        resolution
        tile_size_mo
        patch_size
        patch_min_density
        """

        self.mask_path = mask_path
        self.output_path = output_path
        self.output_type = output_type
        self.sample_type = sample_type
        self.number_of_sample = number_of_sample
        self.invert = invert
        self.buffer_size = buffer_size
        self.extent_path = extent_path
        self.filter_field = filter_field if filter_field != "" else None
        self.filter_value = filter_value if filter_value != "" else None
        self.resolution = resolution if isinstance(resolution, list) else [resolution, resolution]
        self.tile_size_mo = tile_size_mo
        self.patch_size = patch_size
        self.patch_min_density = patch_min_density

        ####################################

        self.limit_geoms = []
        " A logger for big message "
        self.STD_OUT_LOGGER = get_new_logger("stdout_generation")
        ch = get_simple_handler()
        self.STD_OUT_LOGGER.addHandler(ch)

        if self.filter_field is not None and self.filter_value is not None:

            try:

                geoms, list_bbox, limit_crs = get_roi_limits_with_filter(self.extent_path,
                                                                         self.filter_value,
                                                                         self.filter_field)
                self.limit_geoms.append(geoms)

            except OdeonError as error:

                raise OdeonError(error.error_code, "something went wrong during sampling", stack_trace=error)

        else:
            geoms, list_bbox, limit_crs = get_roi_limits(self.extent_path)
            self.limit_geoms.extend(geoms)

        # si buffer on applique un buffer négatif à chaque géométrie de la liste
        if self.buffer_size != 0:

            self.limit_geoms = [geom.buffer(-self.buffer_size) for geom in self.limit_geoms]

    def __call__(self):
        """Operate the two phases systematic sampling

        Returns
        -------
        None
        """
        stdout = """
        #############################################################
        # first step: statistic computation on sampling area/limits,#
        #  i.e  count pixel inside vector mask AND sampling limits  #
        #############################################################
        """
        self.STD_OUT_LOGGER.info(stdout)
        LOGGER.info("count pixel")
        patch_size = self.patch_size if self.sample_type == "patch" else None

        if self.sample_type == "pixel":

            count_functor = CountFunctor(self.mask_path,
                                         self.resolution,
                                         self.invert)

        else:

            count_functor = CountFunctor(self.mask_path,
                                         self.resolution,
                                         self.invert,
                                         self.patch_size,
                                         self.patch_min_density,
                                         )
        apply_tile_functor(count_functor,
                           self.limit_geoms,
                           self.tile_size_mo,
                           self.resolution,
                           patch_size=patch_size,
                           with_tqdm=True)

        count_functor.close()
        count_msg = f"count {self.sample_type} = {count_functor.count}"
        LOGGER.info(count_msg)
        # sum polygon area
        LOGGER.info("sum polygon area")
        area = sum_area(self.mask_path)
        LOGGER.info("area = {0} m2".format(area))
        stdout = """
        #####################################
        #                                   #
        # second step: sampling             #
        #                                   #
        #####################################
        """
        self.STD_OUT_LOGGER.info(stdout)
        LOGGER.info("sampling")
        # configure export file. Create out shapefile if needed.
        # configure sampling strategy

        if self.sample_type == "pixel":

            sample_functor = SampleFunctor(
                self.mask_path,
                self.resolution,
                self.invert,
                self.number_of_sample,
                count_functor.count,
                self.patch_size,
                self.output_path,
                self.output_type,
            )

        else:

            sample_functor = SampleFunctor(
                self.mask_path,
                self.resolution,
                self.invert,
                self.number_of_sample,
                count_functor.count,
                self.patch_size,
                self.output_path,
                self.output_type,
                self.patch_min_density)

        apply_tile_functor(sample_functor,
                           self.limit_geoms,
                           self.tile_size_mo,
                           self.resolution,
                           patch_size=patch_size,
                           with_tqdm=True)

        LOGGER.info(f"total of patch = {sample_functor.tot_patch}")
        LOGGER.info(f"total of sample = {sample_functor.tot_sample}")
        sample_functor.close()
