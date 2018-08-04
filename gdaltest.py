import time
import logging
from osgeo import gdal
from skimage import filters


class gdaltest:
    logger = logging.getLogger(__name__)

    def __init__(self, file_path, tar_file_path):
        self.file_path = file_path
        self.tar_file_path = tar_file_path
        gdal.AllRegister()

    def testwarp(self) -> bool:
        print("processing %s", self.file_path)
        tifpath = self.file_path
        Raster = gdal.Open(tifpath)
        Projection = Raster.GetProjectionRef()
        a = 1

