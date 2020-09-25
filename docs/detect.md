# Detect How To
`detect` performs two distinct types of detection:
* batch detection on a dataset: based on a directory path, a pattern URI list, or a csv file of file.
* batch detection on a zone: based on a geographical extent.



**La commande detect.py effectue une prédiction (inférence) d'un réseaux de neurones sur patchs ou sur une zone d'intérêt**


La commande de base pour faire de la détection est :

* `python detect.py /chemin/du/<fichier_configuration>.json`  

un exemple de fichier de configuration est 


## Configuration

### 1. Common Sections
 
#### image (optional section)

`img_size_pixel (optional, integer, default value: 256, minimal value: 1)`:
the size of in pixel of your input images in case of a dataset detection. The size of your
patch window in case of a detection by zone.

`resolution (optional, float, default value: 0.2, minimal value: 1)`:
the resolution in x and y axis of your input patches / images in the model. A resampling will be applied
on the fly if it's necessary.

#### model (required section)
`model_name (required, string)`: the name of the chosen model for your detection.

`file_name (required, string)`: absolute path of your saved model file. We actually only accept state dictionary saved
model, not the entire model method (to see the distinction in pytorch, go to 
[saving and loading model recipes for pytorch](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html)

`n_classes (required, integer, minimum 1)`: number of class of your trained model.

#### output_param (required)

`output_path (required, string)`: output path for your model outputs.

`output_type (required, string, one value among 'bit', 'int8', 'float32')`: type of your output 
predictions, between bit, 8 bits integer and float 32 bits.

`sparse_mode (optional, boolean, default value false)`: rather output sparse geotif files, to minimize space
 occupation on disk. This option is only used in combination of output_type set as 'bit' and threshold.

`threshold (optional, float, default value 0.5)`: threshold used to output 0/1 value 
when output_type is set to "bit"

#### detect_param (required)
`batch_size (optional, integer, default value 1, minimum value 1)`:
 
 size of input batch in your model

`use_gpu (optional, boolean, default value true)`: 

rather use a gpu for your inference.

`booster (optional, boolean, default value false)`: 

work in progress option, will boost your cpu/gpu params to the max depending on your 
configuration.

`interruption_recovery (optional, boolean, default value false)`: 

recovery option. The detect process logs every operation done and to be done and
save them in a file.
If set to true, the process will load an existing job and start when the job has been
interupted.

`mutual_exclusion (optional, default value false)`:

In a multiclass detection contest, rather use a softmax activation function
or a sigmoïd. In general, you may use a softmax for a multiclass detection with monolabel
and no background pixels. In the other cases like multilabel or monolabel with background,
you may prefer sigmoïd.

### 2. Task Sections

 One of the two sections below is required by odeon detect.

#### Dataset Section (required if Zone section is not filled)

`path (required, string)`: The path of your dataset, or a csv 
with a list of your patch files on the first column.

`image_bands (optional, array)`: a list of integer representing the band to extract from your raster(s)

#### Zone Section (required if dataset section is not filled)

`sources (required, dictionary of band_name:path)`:
A dictionary of raster layer. They will be used to
build the tile images. A raster layer is declared as follow: 
"name_of_raster_entity": {"path": "/path/to/raster",
"bands": a list of integer representing the band to extract from your raster(s)}

`extent (required, string or array)`: A shape file with one or more region of interest
representing the zone(s) where the detection will be done.

`out_dalle_size (optional, integer)`: you can use it if you want an output in a list of file
with a size in the unit format of your crs.

`export_input (optional, default value false)`: rather export input with the same cutting than the output dalle
when output_dalle_size is set to true.
