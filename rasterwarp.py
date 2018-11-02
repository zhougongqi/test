import glob
import math
import logging
from osgeo import gdal
import osr
from gdalconst import *
from skimage import filters
import numpy as np
import cv2
import matplotlib.pyplot as pyplot


class rasterwarp:
    logger = logging.getLogger(__name__)

    def __init__(self, file_path, tar_file_path):
        self.file_path = file_path
        self.tar_file_path = tar_file_path
        gdal.AllRegister()

    def testwarp(self) -> bool:
        print("reading img... %s" % self.file_path)
        tifpath = self.file_path
        outpath = 
        img = gdal.Open(tifpath)
        Projection = img.GetProjectionRef()
        img_srs = osr.SpatialReference()
        img_srs.ImportFromWkt(Projection)
        img_w = img.RasterXSize
        img_h = img.RasterYSize
        img_count = img.RasterCount
        img_trans = img.GetGeoTransform()

        oriLXimg = img_trans[0]
        oriTYimg = img_trans[3]
        PWimg = img_trans[1]  # pixel width
        PHimg = img_trans[5]
        originRXimg = oriLXimg + PWimg * img_w
        originBYimg = oriTYimg + PHimg * img_h

        print("reading img... %s", self.tar_file_path)
        imgt = gdal.Open(self.tar_file_path)
        Projectiont = imgt.GetProjectionRef()
        imgt_srs = osr.SpatialReference()
        imgt_srs.ImportFromWkt(Projectiont)

        # create output
        driver = gdal.GetDriverByName("GTiff")
        driver.Register()
        outdata = driver.Create("")

        # reproj = gdal.AutoCreateWarpedVRT()
        a = 1
        return True


if __name__ == "__main__":
    filepath = "/home/tq/zgq/zdata/flood/malaysia2018/s1/S1A_IW_GRDH_1SDV_20180105T220552_20180105T220619_020025_0221DE_0CC6_NR_Cal_Deb_ML_Spk_SRGR_TC.data/elevation.img"  # "/home/tq/zgq/zdata/flood/2014/maskout/dem/miri.tif"
    tarfilepath = "/home/tq/zgq/zdata/flood/2014/maskout/dem/elevation_Cwater.tif"
    rw = rasterwarp(filepath, tarfilepath)
    rw.testwarp()
