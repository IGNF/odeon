**************************
Systematic Sampling how-to
**************************

The command line to generate sampled points is :

``odeon sample_sys -c /path/to/my/json/config/file.json``
with an optional ``-v`` to get a verbose process (debug mode). 

An example is given bellow.


The draw of the sampling process is periodic, and has no random function
of any kind (uniform, gaussian, etc.). Moreover, the draw is done on the
pixels in the global extent (the area constitued by all the region of interest
of the sampling process). In a nutshell, the union of the extent and the mask.

JSON examples
=============

**minimalist json** (the minimum configuration required to start the sampling)

.. code-block:: json

   {
      "io": {
            "extent_path": "/media/hd/zones.shp",
            "mask_path" : "/home/mask_bati_33_2018_all.shp",
            "output_path": "/media/hd/out/test_systematic_sampling/bati.csv"}
   }


**full json example**

.. code-block:: json
   
   {
    "io": {
        "extent_path": "/media/hd/zones.shp",
        "mask_path" : "/home/mask_bati_33_2018_all.shp",
        "output_path": "/media/hd/out/test_systematic_sampling/bati.csv"
        },
    "sampling": { 
        "output_type": "patch",
        "sample_type": "pixel",
        "number_of_sample": 1000,
        "invert": false,
        "buffer_size" : 50,
        "filter_field": "my-filter-field(ex: code insee)",
        "filter_value": "my-filter_value"
        },
    "patch": {
        "resolution":  0.2,
        "tile_size_mo": 38,
        "patch_size" : 256,
        "patch_min_density": 0.3
        }
   }


Configuration
=============

The configuration is made of 3 sections as you can see in the json examples: 

* `io <io section_>`_
* `sampling <sampling section_>`_
* `patch <patch section_>`_

io section
----------

where you configure the main input and output files.

* ``extent_path (required, string URI of an existing shape file)`` : 
  path to the zone of interest (extent) of the sampling in a fiona compatible format
  (see `fiona manual <https://fiona.readthedocs.io/en/latest/manual.html>`_)
  It can contain a collection of disjointed geometry.
  
* ``mask_path (required, string representing a valid URI of an existing shape file)`` :
  the input path of your mask file in a fiona compatible format
  (see `fiona manual <https://fiona.readthedocs.io/en/latest/manual.html>`_)

* ``output_path (required, string representinf a valid URI with a .csv exention)`` :
  the output path of your csv


sampling section
----------------

optional sampling configurations 

* ``output_type (optional, default value "patch", string, accepted values:
  "pixel", "patch" or "no")`` : 
  you can add an optional output shapefile wit this option, to visualize your sampling.

  - "no": just a csv file
  - "pixel": a csv file and a shapefile generated (same name as the csv
    file and a .shp extention) with the sampled points.
  - "patch": a csv file and a shapefile generated (same name as the csv
    file and a .shp extention) with the geometries of type Polygon
    representing the patch limits of each sample.
    The patches/polygons are centered on the sample points coordinates.
    This parameter is dependent of the patch_size value option (see bellow).
    
    - Patch output:
      
      .. figure:: assets/output_patch.png
         :align: center
         :figclass: align-center
     
    - Pixel output:  

      .. figure:: assets/output_pixel.png
         :align: center
         :figclass: align-center

* ``sample_type (optional, default value: "pixel", string, accepted values:
  pixel or "patch")``: sampled population to use, "pixel" ou "patch".
  
  - "pixel" : the sampled population are the pixels in the union of the
    global extent and the mask.
  
  - "patch" : the sampled population are the patches of size patch size
    scaled by the resolution option constituting a grid of the global extent.
    They are filtered based on the patch_density parameter (see bellow).

* ``number_of_sample (optional, default value 1000, positive integer)``: 
  number of sampled point to generate.
    
  .. note:: 
     
     if the number of sampled point if superior to the number of
     counted point, the number of sampled point will be the number of
     counted point.

* ``invert (optional, default value false, boolean)`` : if true, the draw
  will be done in the areas of the extent not intersecting the mask polygons.
  It can be useful in a monoclass case, when you want to select an ensemble
  of complementary patches without the learning class to improve the
  discrimination of your model.
  
  .. figure:: assets/sample_sys_invert_patch.png
     :align: center
     :figclass: align-center
  
* ``buffer_size (optional, default value 50, positive float)``: 
  Size of the padding (space inside the extent and starting at the borders
  of the extent, interior margin of the global extent) value to apply to
  the extent to avoid sampled points close to the borders. Expressed in CRS unit.  
  
  - no padding (buffer_size = 0):
    
    .. figure:: assets/no_padding.png
       :align: center
       :figclass: align-center   
  
  - padding of 200 (buffer_size = 200):
    
    .. figure:: assets/padding_200.png
       :align: center
       :figclass: align-center   

* `filter_field (optional, default value "", string)`: string representing a field
  option in your shape field to select a specific shape as
  your input extent. By example, you could have a shape file with a collection of shape
  represented with an administrative geographical code like a zip code.

* `filter_value (optional, default value "", string)`: the filter value to
  use in your filter field. In this case by example, your shape file has
  2 fields, and you could filter and get a specific polygon by id or by
  zip code.   
  
  .. figure:: assets/filter_field.png
     :align: center
     :figclass: align-center  

.. note:: 
   
   you need to fill the filter_field AND the filter_value options to filter your shape file.

patch section
-------------

optional patch configuration section

* ``resolution (optional, default value 0.2, positive float of list of float)``:
  resolution, size of pixel in CRS unit.


* ``tile_size_mo (optional, default vallue 38, positive integer)``:
  Maximum memory size to use for the sampling process. To be be clearer,
  the number of patch to use in memory during the sampling process
  (the tile size to use to move into the extent). 

* ``patch_size (optional, default value 256, positive integer)`` :
  patch size in pixel of the output sample. Works also when the output_type
  is set to "patch" (see above)

* ``patch_min_density (optional, default value 0.3, positive float between 0 and 1)``:
  Works when the sample_type option is set to "patch" (see above). Minimal
  density of the intersection of a patch and the vector mask to make a patch
  candidate for sampling.


Algorithmic process
===================

The steps used during the computation are: ::

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


Tips
====

* If you want to use invert to get counter examples, a good option is to
  use "patch" as sample_type option and a patch_min_density equal to 1.

* Use "patch" in the sample_type option in a low patch_min_density if you
  want to sample small and isolated element. On the contrary, use a high
  patch_min_density if you want to privilege big elements and/or small
  elements in high density areas. You need to adjust the concept of high
  and low patch_min_density, depending on your patch size and the size of
  your objects.
 
* if you have no specific criterias, you can just use "pixel" in sample_type.

* In the buffer_size option, a large value can slow down the computation
  actually, you should prefer a moderate value.
