# Generate how-to

Generation_grid performs a dataset generation (with possible split) over a geographical with one or more
learning zones on this geographical zone.

This script takes as input:
 * one or more csv files of sampled points (sampled with the sampler_grid script)
 * GeoTiff files, with bands (rgb tif, MNT, MNS, etc.)
 * vector file, in a fiona ( see [fiona manual](https://fiona.readthedocs.io/en/latest/manual.html) )compatible format, one by classe


The script output 3 directories (train, val, and test) and 2 subdirectories in each: img and mask (img for the
tile image and mask for the tile mask). The size of the tiles are based on the configuration option image_size.
The total number of tuple(img, msk) of tiles is equal to the total number of sampled point.

The script is run a the top level of the project (odeon-landcover) like this: 
```bash
odeon generate -c path_to_my_json_config file/my_json_config_file.json
```
You have a -v option for debug.

The json configuration file in input of CLI command contains 4 sections: 
* _image: configuration of the output tile
    * image_size_pixel size of height and width
    * resolution: scale of image file 
* image_layers: a dict of geotiff layer. They are used to
build the tile images. A Geotiff layer is declared as follow: "name_of_geotiff": {"path": "path_to_geotif", "bands": the bands to extracts from 
geotiff. 

    Example of declaration: "RGB": {
      "path": "/media/hd/tests_data/33/RVB/33_2015_zone_1_1.tif",
      "bands": [1, 2, 3]
    }
    
* vector_classes: a dictionary of shape file, one by class. They're will be used to generate the tile masks.
* generator: this section includes configuration options for spliting the sampled points in train/test/val
and directory paths:
    * output_path: where to generate the dataset. 3 directories will
    be generated inside: train, val, test.
    * poi_pattern: the path where you have your sampled center points with pattern option( *csv)
    * train_test_split: the percentage of the training part over the testing part when we do the train/test split
    * train_val_split": the percentage of the training part over the validation part when we do the train/valt split
    * compute_only_masks": when we activate it, we compute only the mask
    * mnt_mns: rather we create a band wich is the substration of DSM and DTM
    * append: Append patches in the existing pathces in train/val/test if True, or purge everything
    and create a fresh new Dataset.

The json schema of the generation configuration file can be found at odeon/scripts/json_defaults/generation_schema.json
here in a example of json configuration
```json

{
  "image": {
    "image_size_pixel": 128,
    "resolution": [0.2, 0.2]
  },
  "image_layers": {
    "RGB": {
      "path": "/media/hd/tests_data/33/RVB/33_2015_zone_1_1.tif",
      "bands": [1, 2, 3]
    },
    "CIR": {
      "path": "/media/hd/tests_data/33/IRC/33_2015_zone_1_1.tif",
      "bands": [1, 2, 3]
    }
  },
  "vector_classes": {
    "building": "/media/hd/tests_data/33/VECTOR/33_2015_zone_1_1_bati.shp",
    "bitumen" : "/media/hd/tests_data/33/VECTOR/33_2015_zone_1_1_bitume.shp",
    "water" : "/media/hd/tests_data/33/VECTOR/33_2015_zone_1_1_eau.shp",
    "tree": "/media/hd/tests_data/33/VECTOR/33_2015_zone_1_1_ligneux.shp",
    "minerals": "/media/hd/tests_data/33/VECTOR/33_2015_zone_1_1_mineraux.shp",
    "pool": "/media/hd/tests_data/33/VECTOR/33_2015_zone_1_1_piscine.shp"
  },
  "generator": {
    "output_path": "/media/hd/out/test_dev/generation",
    "poi_pattern": "/media/hd/out/test_dev/grid_sampler/zone1_2015_zone_33_1_1.csv",
    "train_test_split": 0.8,
    "train_val_split": 0.8,
    "compute_only_masks": false,
    "mns_mnt": true,
    "append": false
  }
}
```