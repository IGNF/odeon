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

0.2.3 (2021-09-24)
------------------
- Correct the problem in generation when a source have NoDataValue. Should fix issue 23.
- fix minimum version number for packages pandas and geopandas. Fix issue #30 (closed)
- fix problem of homogeneity in band indices between tools in code and in documentation. Fix issue #29 (closed)
