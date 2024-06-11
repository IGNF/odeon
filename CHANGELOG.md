# Changes

## 0.5.1 (2024-06-11)
 ***V.0.5.1 issue 25 Add meta data input fields***
- Add meta data input fields in preprocessing class

## 0.5.0 (2024-03-20)
 ***V.0.5.0 first stable version of core API with plugins***
- Full oriented plugin architecture
- Plugin system is stable
## 0.3.0 (2023-11-13)
 ***V.0.3.0 merge the new version of Odeon into the main code***
- Old code has been totally removed excepted changelogs
- Changelogs has been converted into markdown
- New version is based on a modular architecture based on Generic Registry Pattern
- Segmentation Lightning module and Binary Change detection module are integrated
- Data Module is multi modal and multi task by nature
- OdeonFit is the App for training deep learning models
- The first augmentation library is Albumentation
- Plugin system is still in development

## 0.2.20 (2023-03-19)
- missing one odeon-landcover occurrence to update

## 0.2.19 (2023-03-19)
- Update repository name from odeon-landcover to odeon, corrections on readme and doc
- Change changelogs order from asc chronological order to desc.
- update url of repository in setup.py

## 0.2.18 (2022-06-24)
- Update documentation and doc configuration after transfer from gitlab to github

## 0.2.17 (2022-06-23)
- Fix issue #19: batch mode of the generate dataset tool makes multiple copies of sample in csv
- Fix issue #46: bug in generate when using shapefile having geometries with Z attribute
- Fix issue #47: Detect by zone has a shift of 1 pixel on result on the same case
- Fix Issue #49: Detect zone wrong extend and resolution output when asking a resolution different from source data
- remove test on gdal tools in setup.py (see #48)

## 0.2.16 (2022-06-20)
- Implement Feature issue #8: implement "continue training" feature
- Fix issue #15: grid sampling: tile can be outside the zone
- Fix issue #38: spatial shift with sampling_sys

## 0.2.15 (2022-02-21)
- Fix issue 45: Metrics - Problem reading file name during dataset creation

## 0.2.14 (2021-01-11)
- Fix issue 44: Deployment script changes in relation to pylint developments.
- Fix issues 43: Metrics: use the tool with predictions and masks of different shapes.
- Refactoring of metrics code (to correct few problems on testing parameters and to gather plot functions).

## 0.2.13 (2021-11-18)
- Fix issue 40: Metrics: problem with labels in the case of class selection.
- Fix issue 41: Forgot to change version and do modifications on CHANGELOG.rst file.

## 0.2.12 (2021-10-22)
- fix issue 39: Reports become unreadable when the number of classes is too high and we would like to add the
  possibility to output in a CSV file the threshold values corresponding to each point of the ROC/PR curves.

## 0.2.11 (2021-10-21)
- fix issue 31: metric script don't work when prediction and mask files do not have the same number of bands.
- add the feature to select the classes on which we want to use the metrics
- correction on cm figure size when the number of classes is large.
- correction on the export of metrics in a JSON file.

## 0.2.9 (2021-10-05)
- fix issue 34: regression introduced in generate dataset on batch mode.

## 0.2.8 (2021-10-04)
- fix regression introduce in v0.2.4 in detection zone.

## 0.2.7 (2021-09-31)
- improve miou computation speed during training
  use pytorch in place of numpy to do MiOu computation.

## 0.2.6 (2021-09-30)
- fix issue with new mobilenetv2 module on new torchvision version without regression

## 0.2.5 (2021-09-27)
- fix issue with BCE loss.
  Don't call torch loss directly and convert target tensor to long only for CrossEntropyWithLogitLoss.

## 0.2.4 (2021-09-27)
- Refactoring of DEM computation to align detection and generation. Fix issue #21 (closed)
- Correction of bug when bands are after DSM or DTM when dem option is set to true. Fix issue #20 (closed)
- Fixing the option compute_only_mask to compute only masks and not images. Fix issue #18 (closed)
- Correct problems in stats related to the modifications made to remove gdal dependencies. Fix issue #24 (closed)

## 0.2.3 (2021-09-24)
- Correct the problem in generation when a source has NoDataValue. Should fix issue 23.
- fix minimum version number for packages pandas and geopandas. Fix issue #30 (closed)
- fix problem of homogeneity in band indices between tools in code and in documentation. Fix issue #29 (closed)

## 0.2.2 (2021-09-14)
- removed the image_to_ndarray (using gdal) replacing every call to raster_to_ndarray (using rasterio). Should fix issue 22 (closed)

## 0.2.1 (2021-09-14)
- fix zone detection problem in tile generation

## 0.2 (2021-09-13)
- add tool stats
- add tool metrics
- add of the documentation relative to these tools
- add module report

## 0.1.1 (2021-01-06)
- fix issue 16
- fix resolution change in detection

## 0.1 (2020-12-14)
- add grid sampling
- add generation of dataset
- add systematic sampling
- add zone detection based on extent and raster dalle
- add patch detection
- add documentation
- add CI/CD with static code analysis, build doc, publish doc job
