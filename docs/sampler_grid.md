# Sampler grid how-to

`Sampler_grid` performs a regular sampling over one or several zones.

This is one possibility for the first step (out of two) to build a dataset. This is suited if you have areas with ground truth for all your classes.   

The goal is to generate a list of (x,y) coordinates :
* from each shape in a shapefile
* equally distributed
* allowing an (image,mask) to be completely inside the shape  

The output contains as many files as there are shapes in the given shapefile. Each file is a list of coordinates from one shape.
The shape are ordered in x, then y.


To launch the code, simply type `python sampler_grid.py -c <config.json>`. You can add `-v` in order to have a verbose output.

Example :
```bash
(dlxx) nil@arthur:~/dev/dl/odeon-landcover/odeon$ python sampler_grid.py -c ../tests/test_sampler_grid.json -v
Configuration :
	input shapefile: ../tests/data/sampler_test.shp
	output pattern: ../../odeon_test/*_regularsampling.csv
	image size (pixel): 256
	pixel size (meter per pixel): 0.2
	strict_inclusion: True
	shift (1 to shift centers): 0
[../../odeon_test/zone1_regularsampling.csv]: 247 points
[../../odeon_test/zone2_regularsampling.csv]: 198 points
Code block 'Sampling' took: 00h:00m:00.172s
```

## Json file content

The full json file is :

```json
{
  "image": {
    "image_size_pixel": 256,
    "pixel_size_meter_per_pixel": 0.2
  },
  "sampler": {
    "input_file": "../tests/data/sampler_test.shp",
    "output_pattern": "../../odeon_test/*_regularsampling.csv",
    "strict_inclusion": 1,
    "shift": 0
  }
}
```

### Section image

"image" describes the size and resolution of the future samples in the dataset:
* `image_size_pixel` : the number of pixels per side (usually a power of 2)
*  `pixel_size_meter_per_pixel` : the resolution, 0.2 means 0.2m per pixel

This information has to be consistent with the following steps : generation of teh dataset and training.

### Section sampler

"sampler" contains information regarding the way the sammpling will be performed:
* "input_file": a shapefile containing a list of shapes (avoid concave polygons)
* "output_pattern": "../../odeon_test/*_regularsampling.csv",
* "strict_inclusion": 1,
* "shift": 0
