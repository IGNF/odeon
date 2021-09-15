Changes
=======

0.1 (2020-12-14)
----------------
- add grid sampling
- add generation of dataset
- add systematic sampling
- add zone detection based on extent and raster dalle
- add patch detection
- add documentation
- add CI/CD with static code analysic, build doc, publish doc job

0.1.1 (2021-01-06)
------------------
- fix issue 16
- fix resolution change in detection

0.2 (2021-09-13)
------------------
- add tool stats
- add tool metrics
- add of the documentation relative to these tools
- add module report

0.2.1 (2021-09-14)
------------------
- fix zone detection problem in tile generation

0.2.2 (2021-09-14)
------------------
- removed the image_to_ndarray (using gdal) replacing every call to raster_to_ndarray (using rasterio). Should fix issue 22 (closed)
