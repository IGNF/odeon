from odeon import LOGGER
import math
from tqdm import tqdm
import numpy as np
import fiona
from shapely.geometry import shape, box, Point, mapping
import rasterio
from rasterio import features
from odeon.commons.exception import OdeonError, ErrorCodes


class BaseFunctor:
    """
    Base class for the two phase systematic sampling
    """

    def __init__(self, mask_shp, resolution, invert):
        """

        Parameters
        ----------
        mask_shp : str
         shape file of mask
        resolution : list of float
         pixel resolution value
        invert : bool
         rather finding the non overlapping geometries patch or not
        """

        self.pixel_size = resolution
        self.feat_layer = fiona.open(mask_shp, 'r')
        self.count = 0
        self.invert = invert

    def __call__(self, min_x, min_y, max_x, max_y, mask=None):
        """

        Parameters
        ----------
        min_x : float
        min_y : float
        max_x : float
        max_y : float
        mask : NDArray

        Returns
        -------
        image : NDArray
        img_mask : NDArray
        tile_size : int
        tile_affine : rasterio.Affine
        """

        bbox_tile = box(min_x, min_y, max_x, max_y)  # minx, miny, maxx, maxy,

        feat_bbox_select = self.feat_layer.items(bbox=bbox_tile.bounds)
        feat_gen = [(feature[1]["geometry"], 1) for feature in feat_bbox_select]

        if len(feat_gen) == 0:

            raise OdeonError(error_code=ErrorCodes.ERR_EMPTY_ITERABLE_OBJECT,
                             message="empty list, there is no geometry in the object feat_bbox_select")

        tile_shape = (
            int(round((max_x - min_x) / self.pixel_size[0])),
            int(round((max_y - min_y) / self.pixel_size[1])),
        )
        tile_affine = rasterio.transform.from_origin(
            min_x,
            max_y,
            self.pixel_size[0],
            self.pixel_size[1])

        image = features.rasterize(feat_gen,
                                   out_shape=tile_shape,
                                   transform=tile_affine)

        if self.invert:

            image ^= 1  # in place element wise XOR operation. image = image ^ 1.

        if mask is not None:

            if not bbox_tile.intersects(mask):

                raise OdeonError(error_code=ErrorCodes.ERR_INTERSECTION_ERROR,
                                 message=f"the bounding box {bbox_tile} does not intersects the mask {mask}")

            feat_mask = [mask]
            img_mask = rasterio.features.geometry_mask(
                feat_mask,
                out_shape=tile_shape,
                transform=tile_affine,
                all_touched=True,
                invert=True)

        else:

            img_mask = np.ones(shape=tile_shape, dtype=np.bool)

        return image, img_mask, tile_shape, tile_affine

    def process_patch(self, image, img_mask):
        """
        The function called when we functor at the patch level
        Parameters
        ----------
        image : NDArray
        img_mask : NDArray

        Returns
        -------

        """

        pass

    def process_pixel(self, image, img_mask):
        """
        The function called when we functor at the pixel level
        Parameters
        ----------
        image : NDArray
        img_mask : NDArray

        Returns
        -------

        """

        pass

    def close(self):
        """
        close opened fiona drivers
        Returns
        -------

        """

        self.feat_layer.close()


class SampleFunctor(BaseFunctor):
    """
    The functor to compute and write on csv/shape file the valid samples based
    on the number of valid samples computed in the counting phase.
    The functor is applied on tile with parametric resolution,
    and the intersection computation are made at pixel level or  patch level.
    At the patch level, a patch is considered as valid when the density of the vector mask
    on the patch reach a threshold (density_threshold).
    The computation of intersection are made after a rasterization of patch and mask.
    The size of tile should be a multiple of the patch size.
    """

    def __init__(self,
                 mask_shp,
                 resolution,
                 invert,
                 num_sample,
                 patch_count,
                 patch_size,
                 out_sample,
                 out_shp_type=None,
                 density_threshold=None):
        """

        Parameters
        ----------
        mask_shp : str
         shape file of mask
        resolution : float
         pixel resolution value
        invert : bool
         rather finding the non overlapping geometries patch or not
        num_sample : int
         desired number of sample
        patch_count : int
         number of patch counted in the counting phase
        patch_size : int
         size of sampled patches
        out_sample : str
         output csv file
        out_shp_type :
         output shape file
        density_threshold : float
         minimum dentity of good pixel in a patch to be considered as a candidate for sampling
        """
        super(SampleFunctor, self).__init__(mask_shp, resolution, invert)
        self.out_crs = self.feat_layer.crs
        self.tot_patch = 0
        self.tot_sample = 0
        self.f_coord = open(out_sample, 'w')

        if patch_size is not None:

            patch_size_p = [
                patch_size * resolution[0],
                patch_size * resolution[1]
            ]

        else:

            patch_size_p = None

        self.feat_sample, self.write_sample = init_out_shp(
            self.f_coord, self.out_crs, patch_size_p, out_shp_type)
        self.patch_size = patch_size
        self.density_threshold = density_threshold
        self.patch_stride = max(int(patch_count / num_sample), 1)

    def __call__(self, min_x, min_y, max_x, max_y, mask=None):
        """

        Parameters
        ----------
        min_x : float
        min_y : float
        max_x : float
        max_y : float
        mask : NDArray

        Returns
        -------

        """

        try:
            if mask is not None:

                image, img_mask, tile_size, tile_affine = super(SampleFunctor, self).__call__(min_x,
                                                                                              min_y,
                                                                                              max_x,
                                                                                              max_y,
                                                                                              mask=mask)
            else:

                image, img_mask, tile_size, tile_affine = super(SampleFunctor, self).__call__(min_x,
                                                                                              min_y,
                                                                                              max_x,
                                                                                              max_y)

            if self.density_threshold:

                self.process_patch(image, img_mask, tile_affine, tile_size)

            else:

                self.process_pixel(image, img_mask, tile_affine)

        except OdeonError as error:

            LOGGER.warning(f"{error}")

    def process_patch(self, image, img_mask, tile_affine, tile_size):
        """
        The function called when we sample at the patch level

        Parameters
        ----------
        image : NDArray
         input image as numpy array
        img_mask : NDArray
         input mask as numpy array
        tile_affine : rasterio.Affine
         affine object associated with the input image
        tile_size : list of float
         size of the tile where we are sampling

        Returns
        -------

        """

        super(SampleFunctor, self).process_patch(image, img_mask)
        nb_patch_x = int(tile_size[0] / self.patch_size)
        nb_patch_y = int(tile_size[1] / self.patch_size)

        if (nb_patch_x * self.patch_size != tile_size[0]) or (nb_patch_y * self.patch_size != tile_size[1]):

            LOGGER.warning(f"tile size {tile_size} is not a multiple of patch size {self.patch_size}")

        image_masked = np.logical_and(image, img_mask).astype(rasterio.uint8)
        patch_array = np.zeros((nb_patch_x, nb_patch_y))

        for patch_x in range(0, nb_patch_x):

            x_min_p = patch_x * self.patch_size
            x_max_p = patch_x * self.patch_size + self.patch_size

            for patch_y in range(0, nb_patch_y):

                y_min_p = patch_y * self.patch_size
                y_max_p = patch_y * self.patch_size + self.patch_size
                patch_mean = image_masked[x_min_p:x_max_p, y_min_p:y_max_p].mean()

                if patch_mean >= self.density_threshold:

                    patch_array[patch_x, patch_y] = 1

        patch_count = patch_array.sum()

        if (self.tot_patch + patch_count) < (self.tot_sample * self.patch_stride + self.patch_stride):

            self.tot_patch += patch_count

            return

        # si oui
        pix_i, pix_j = np.where(patch_array)
        sample_idx = int((self.tot_sample * self.patch_stride + self.patch_stride) - self.tot_patch)
        # LOGGER.info("pix_count = {0}".format(pix_count))
        try:

            while sample_idx < patch_count:

                # LOGGER.info("sample_idx = {0}".format(sample_idx))
                sample_i = pix_i[sample_idx]
                sample_j = pix_j[sample_idx]

                coord_x_min, coord_y_min = tile_affine * (sample_j * self.patch_size, sample_i * self.patch_size)
                dist_to_center = [
                    self.patch_size * self.pixel_size[0] / 2,
                    self.patch_size * self.pixel_size[1] / 2
                ]
                self.write_sample(coord_x_min + dist_to_center[0], coord_y_min + dist_to_center[1], self.tot_sample)
                sample_idx += self.patch_stride
                self.tot_sample += 1

        except IndexError as error:

            LOGGER.warning(f"out of bound index {sample_idx} \n {error}")

        self.tot_patch += patch_count

    def process_pixel(self, image, img_mask, tile_affine):
        """
        The function called when we sample at the pixel level

        Parameters
        ----------
        image : NDArray
         input image as numpy array
        img_mask : NDArray
         input mask as numpy array
        tile_affine : rasterio.Affine
         affine object associated with the input image

        Returns
        -------

        """

        pix_count = image[img_mask].sum()

        # we check if we need new pixels in the current tile
        if (self.tot_patch + pix_count) < (self.tot_sample * self.patch_stride + self.patch_stride):

            self.tot_patch += pix_count
            return

        # if we need new pixel
        pix_i, pix_j = np.where(np.logical_and(image, img_mask).astype(np.uint8))
        sample_idx = int((self.tot_sample * self.patch_stride + self.patch_stride) - self.tot_patch)

        try:

            while sample_idx < pix_count:

                sample_i = pix_i[sample_idx]
                sample_j = pix_j[sample_idx]

                coord_x, coord_y = tile_affine * (sample_j, sample_i)

                self.write_sample(coord_x, coord_y, self.tot_sample)
                sample_idx += self.patch_stride
                self.tot_sample += 1

        except IndexError as error:

            LOGGER.warning(f"out of bound index {sample_idx} \n {error}")

        self.tot_patch += pix_count

    def close(self):
        """
        close opened fiona drivers
        Returns
        -------

        """

        super(SampleFunctor, self).close()
        self.f_coord.close()

        if self.feat_sample is not None:

            self.feat_sample.close()


class CountFunctor(BaseFunctor):
    """
    Foncteur de calcul du nombre de patch d'apprentissage valide présents dans un masque vecteur

    le foncteur s'applique par dalle de traitement et on spécifie la résolution à laquelle appliquer le comptage.
    On passe par une rasterisation du vecteur par dalle de traitement.

    Les patchs candidats sont définis selon une grille régulière. Un patch est considéré comme valide quand la densité
    du masque vecteur dans le patch dépasse un seuil donné par l'utilisateur.

    La taille des dalles de traitements dans être un nombre entier de la taille de patch
    """

    def __init__(self, mask_shp, resolution, invert, patch_size=None, density_threshold=None):
        """constructeur du foncteur de comptage de patch d'entrainement

        Parameters
        ----------
        mask_shp : str
         couche du masque vecteur ne format shapefile
        resolution : list of float
         resolution in x and y of the output patches
        invert : bool
         False/True utilise le complémentaire du masque vecteur pour l'échantillonage
        patch_size : int
         taille d'un patch
        density_threshold : float
         seuil de densité par la selection des patchs
        """

        super(CountFunctor, self).__init__(mask_shp, resolution, invert)
        self.patch_size = patch_size
        self.density_threshold = density_threshold

    def __call__(self, min_x, min_y, max_x, max_y, mask=None):
        """operateur sur une dalle définie par les coordonnées [min_x,max_x][min_y, max_y]

                Parameters
                ----------
                min_x : float
                min_y : float
                max_x : float
                max_y : float
                mask : NDArray

                Returns
                -------

        """
        try:

            if mask is not None:

                image, img_mask, tile_size, tile_affine = super(CountFunctor, self).__call__(min_x,
                                                                                             min_y,
                                                                                             max_x,
                                                                                             max_y,
                                                                                             mask=mask)

            else:

                image, img_mask, tile_size, tile_affine = super(CountFunctor, self).__call__(min_x,
                                                                                             min_y,
                                                                                             max_x,
                                                                                             max_y)

            if self.density_threshold is not None:

                self.process_patch(image, img_mask, tile_size)

            else:

                self.process_pixel(image, img_mask)

        except OdeonError as error:

            LOGGER.warning(f"{error}")

    def process_patch(self, image, img_mask, tile_size):
        """

        The function called when we count at the patch level

        Parameters
        ----------
        image : NDArray
         input image as numpy array
        img_mask : NDArray
         input mask as numpy array
        tile_size : int
         size of the tile where we are counting

        Returns
        ----------
        None

        """
        LOGGER.info(tile_size)
        LOGGER.info(self.patch_size)
        nb_patch_x = int(tile_size[0] / self.patch_size)
        nb_patch_y = int(tile_size[1] / self.patch_size)

        if (nb_patch_x * self.patch_size != tile_size[0]) or (nb_patch_y * self.patch_size != tile_size[1]):

            LOGGER.warning(f"tile size {tile_size} is not \
                            a multiple of patch size {self.patch_size} on one of its axis")

        image_masked = np.logical_and(image, img_mask).astype(rasterio.uint8)

        for patch_x in range(0, nb_patch_x):

            x_min_p = patch_x * self.patch_size
            x_max_p = patch_x * self.patch_size + self.patch_size

            for patch_y in range(0, nb_patch_y):

                y_min_p = patch_y * self.patch_size
                y_max_p = patch_y * self.patch_size + self.patch_size
                # LOGGER.debug(f"x_min_p, x_max_p, y_min_p, y_max_p: {x_min_p, x_max_p, y_min_p, y_max_p}")
                patch_mean = image_masked[x_min_p:x_max_p, y_min_p:y_max_p].mean()

                if patch_mean >= self.density_threshold:

                    self.count += 1

    def process_pixel(self, image, img_mask):
        """
        The function called when we count at the pixel level

        Parameters
        ----------
        image : NDArray
         input image as numpy array
        img_mask : NDArray
         input mask as numpy array

        Returns
        ----------
        None

        """

        self.count += image[img_mask].sum()


def init_out_shp(out_sample, out_crs, patch_size_p, out_shp_type=None):
    """
    initialize a shapefile (in ESRI shapefile format) based on the type
    of input data and the options of the the associated write_sample function.
    Parameters
    ----------
    out_sample : file
     file object of the output shapefile
    out_crs
    patch_size_p
    out_shp_type

    Returns
    -------
    Union[fiona object file, function]
     the output shape file and the associated write sample function
    """

    out_sample_shp = out_sample.name[:-4] + ".shp"
    feat_sample = None

    if out_shp_type == "pixel":

        sample_schema = {'geometry': 'Point', 'properties': {'id_sample': 'int'}}
        feat_sample = fiona.open(out_sample_shp, 'w', crs=out_crs, driver='ESRI Shapefile', schema=sample_schema)
        LOGGER.info("export shapefile des échantillons au format point/pixel")

        def write(x, y, id_sample):

            write_sample(out_sample, x, y, id_sample=id_sample, out_shp=feat_sample)

    elif out_shp_type == "patch":

        sample_schema = {'geometry': 'Polygon', 'properties': {'id_sample': 'int'}}
        feat_sample = fiona.open(out_sample_shp, 'w', crs=out_crs, driver='ESRI Shapefile', schema=sample_schema)
        LOGGER.info("export shapefile des échantillons au format polygones/patch")

        def write(x, y, id_sample):

            write_sample(out_sample, x, y, id_sample=id_sample, out_shp=feat_sample, patch_size_p=patch_size_p)

    else:

        LOGGER.info("pas d'export shapefile des échantillons")

        def write(x, y, id_sample=None):

            write_sample(out_sample, x, y)

    return feat_sample, write


def get_roi_limits_with_filter(file, value, field_name='INSEE_DEP'):
    """
    return the geometry, the bbox and the crs of a filtered polygon
    the filtering is based on a field property of the shape file schema
    We assume this property has unique value

    Parameters
    ----------
    file : str
     a shape file compatible with fiona drivers
    value : str
     the target value in the filtered property
    field_name : str
     the property name to look up in the shape file schema

    Returns
    -------
    geometry in shapely format,bounding box, and crs

    Raises
    -------
    OdeonError
    """
    # we start by retrieving the bounding box and the geometry of the filtered polygon
    geom = None
    bbox = None
    out_crs = None

    with fiona.open(file, 'r') as layer_dep:

        out_crs = layer_dep.crs
        schema = layer_dep.schema
        properties = schema["properties"]

        if field_name in properties.keys():

            for feat in layer_dep:

                test_value = feat['properties'][field_name]

                if str(test_value) == str(value):

                    geom = shape(feat['geometry'])
                    bbox = geom.bounds
                    break

            return geom, [bbox], out_crs

        else:

            raise OdeonError(ErrorCodes.ERR_FIELD_NOT_FOUND,
                             f"the field {field_name} has not been found in your shape file {file}")


def get_roi_limits(roi_shp):
    """
    return a list of geometry and bouding box based on a shape file containing one
    or more shape.
    Parameters
    ----------
    roi_shp : str
     shapefile containing the input shapes (shapes must be of type Polygon)

    Returns
    -------
    Union[list, list, str]
     geom_list, bbox_list, out_crs: shapely bjects cotaining respectively: the list of geometries,
     the list of bounding boxes and the crs associated to the shape file.
    """

    geom_list = []
    bbox_list = []
    with fiona.open(roi_shp, 'r') as layer_roi:

        out_crs = layer_roi.crs

        for feat in layer_roi:

            geom = shape(feat['geometry'])
            bbox = geom.bounds
            geom_list.append(geom)
            bbox_list.append(bbox)

    return geom_list, bbox_list, out_crs


def sum_area(in_shp_path):
    """
    compute the total surface of the shapes contained in a shape file

    Parameters
    ----------
    in_shp_path : str
     fichier shapefile en entrée. De type surfacique.

    Returns
    -------
    float
     total surface in crs UNIT (meter, degree)
    """

    area = 0

    with fiona.open(in_shp_path, 'r') as in_feat:

        for feat in in_feat:

            feat_area = shape(feat["geometry"]).area
            area += feat_area

    return area


def write_sample(out_sample,
                 coord_x,
                 coord_y,
                 id_sample=None,
                 out_shp=None,
                 patch_size_p=None):
    """
    add a sample to a csv file and optionnally to a shape file
    in a point or polygon (depending on the input params) shapely format
    Parameters
    ----------
    out_sample : file
     the file object to write in
    coord_x : float
     x coordinate of sample
    coord_y : float
     y coordinate of sample
    id_sample : int
     shape id in the case you add a shape to a shape file for the sampled point
    out_shp : file
     the shape file to write in if necessary
    patch_size_m : list of float
     size of the sample patch used to compute the bounds of the sampled patch

    Returns
    -------

    """
    out_sample.write(f"{coord_x}; {coord_y}\n")

    if id_sample is not None and out_shp is not None and patch_size_p is not None:

        dist_to_center = [
            patch_size_p[0] / 2,
            patch_size_p[1] / 2
        ]
        coord_x_min = coord_x - dist_to_center[0]
        coord_y_min = coord_y - dist_to_center[1]
        coord_x_max = coord_x + dist_to_center[0]
        coord_y_max = coord_y + dist_to_center[1]
        patch = box(coord_x_min, coord_y_min, coord_x_max, coord_y_max)
        out_shp.write(
            {'properties': {'id_sample': id_sample},
             'geometry': mapping(patch)})

    elif id_sample is not None and out_shp is not None and patch_size_p is None:

        point = Point(float(coord_x), float(coord_y))
        out_shp.write(
            {'properties': {'id_sample': id_sample},
             'geometry': mapping(point)})


def get_processing_tiles_limits(geom_list, mem_tile_size_mo, resolution, patch_size=None):
    """compute the processing limits by tiles coupled with a list of ROI

    Parameters
    ----------
    geom_list : list
    mem_tile_size_mo : int
     max size memory allowed during computation
    resolution : list of float
     pixel size in crs UNIT
    patch_size : int
     size of the patch of interest

    Returns
    -------
    tiles_limits : list
     list of bounding box + geometry
    length_tile_size_m : float
     size in crs UNIT of the extent represented by the list of tiles
    num_tiles : int
     number of max tiles to fit the max size memory allowed
    length_tile_size: int
     size in pixel of the extent represented by the list of tiles
    """

    # the process take one byte by pixel, so we compute the max number of pixel we can process
    # in one batch without exceding the memory limit in MO
    pix_tile_size = mem_tile_size_mo * 1024 * 1024
    # we compute the max size of a tile in pixel. If we compute by patch, we make sure that
    # the max number of pixel is a multiple of patch size.
    length_tile_size = int(math.sqrt(pix_tile_size))

    length_tile_size_p = [
        length_tile_size * resolution[0],
        length_tile_size * resolution[1]
    ]
    LOGGER.debug(patch_size)

    if patch_size is not None:

        patch_size_p = [
            patch_size * resolution[0],
            patch_size * resolution[1]
        ]
        length_tile_size_p = [
            int(length_tile_size_p[0] // patch_size_p[0]) * patch_size_p[0],
            int(length_tile_size_p[1] // patch_size_p[1]) * patch_size_p[1]
        ]

    LOGGER.info("memory length_tile_size = {0}".format(length_tile_size))
    LOGGER.info("memory length_tile_size_p = {0}".format(length_tile_size_p))

    tiles_limits = []
    num_tiles = 0

    for geom in geom_list:

        bbox = geom.bounds
        min_x = int(bbox[0] / length_tile_size_p[0]) * length_tile_size_p[0]
        nb_tile_x = int((bbox[2] - min_x) / length_tile_size_p[0]) + 1
        min_y = int(bbox[1] / length_tile_size_p[1]) * length_tile_size_p[1]
        nb_tile_y = int((bbox[3] - min_y) / length_tile_size_p[1]) + 1
        tiles_limits.append((min_x, min_y, nb_tile_x, nb_tile_y, geom))
        num_tiles += nb_tile_x * nb_tile_y

    return tiles_limits, length_tile_size_p, num_tiles, length_tile_size


def apply_tile_functor(functor, geom_list, mem_tile_size_mo, resolution, patch_size=None, with_tqdm=False):
    """apply a computation treatment (a functor object) to a tile on a list of geometry/ROI

    Each geometry/ROI is divided in tile (the tile size is based on the mem_tile_size_mo param)
    au traitements. the tile are browsed from left to right, bottom to top.

    Parameters
    ----------
    functor : BaseFunctor
     a functor object to apply to each tile of the extent
    geom_list: list
     a list of geometry/ROI forming the global extent of the sampling process
    mem_tile_size_mo: int
     the tile size to browse the global extent
    resolution : float
     size of pixel in crs UNIT (meter, degree, etc.)
    patch_size : Union[None, int]
     size in pixel of each patch
    with_tqdm : bool
     use tqdm bar or not

    Returns
    -------
    None

    """
    _, length_tile_size_p, num_tiles, length_tile_size = get_processing_tiles_limits(
        geom_list, mem_tile_size_mo, resolution, patch_size)

    LOGGER.info("memory length_tile_size = {0}".format(length_tile_size))
    LOGGER.info("memory length_tile_size_m = {0}".format(length_tile_size_p))

    def apply_functor(geom_list, length_tile_size_p, pbar=None):
        """inner helper function

        Parameters
        ----------
        geom_list : list
         a list of geometry/ROI forming the global extent of the sampling process
        length_tile_size_m : list of float
         size in crs UNIT of the extent represented by the list of tiles
        pbar : tqdm
         a tqdm object to enhance visualization

        Returns
        -------

        """
        for geom in geom_list:

            bbox = geom.bounds
            min_x = int(bbox[0] / length_tile_size_p[0]) * length_tile_size_p[0]
            x_nb_tile = int((bbox[2] - min_x) / length_tile_size_p[0]) + 1
            min_y = int(bbox[1] / length_tile_size_p[1]) * length_tile_size_p[1]
            y_nb_tile = int((bbox[3] - min_y) / length_tile_size_p[1]) + 1

            for x in range(0, x_nb_tile):

                for y in range(0, y_nb_tile):

                    if pbar is not None:

                        pbar.update(1)

                    # get current tile bbox
                    x_img = min_x + x * length_tile_size_p[0]
                    y_img = min_y + y * length_tile_size_p[1]
                    x_img_max = x_img + length_tile_size_p[0]
                    y_img_max = y_img + length_tile_size_p[1]
                    functor(x_img, y_img, x_img_max, y_img_max, mask=geom)

    if with_tqdm:

        with tqdm(total=num_tiles) as pbar:

            apply_functor(geom_list, length_tile_size_p, pbar)

    else:

        apply_functor(geom_list, length_tile_size_p)
