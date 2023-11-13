Feature: Detect is a cli tool to perform an inference with a neural network model, on patch or region of interest.
  As user
  I want to be able to perform a detection on a dataset (train, val or test),
  if the dataset is formatted as the output of the generate cli tool format its dataset outputs.
  I want to be able to perform a detection on a region of interest, defined by a shape file or bounds,
  with a margin or not, otput by dal or not, with a tile factor option for memory optimization.
  I want to be able to get back to the last finished step of a detection process, if for any reason,
  the process has been prematurely interrupted.
  I want to be able to choose the output path, output type,
  to output in 0/1 bit with a threshold and / or as a sparse matrix,
  to export input data by tile in the case of detection by zone with an output by tile.
  I want to be able to pick my model by type.
  I want to be able to make a detection with gpu or not, with parallelism or not.
  I want to be able to pick the size of the image patch, the resolution, and a margin to avoir bad inference on border.

  Scenario: Detection on a patch dataset
    Given a json configuration file with the dataset section filled.
    When I call odeon detect tool.
    Then I will have an output folder filled of output mask, one by patch.

