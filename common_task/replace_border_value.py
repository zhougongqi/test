import os
import gc
import cv2
import sys
import math
import time
import glob
import logging
import numpy as np
from osgeo import gdal, osr, gdalnumeric
import subprocess

home_dir = os.path.expanduser("~")


def replace_border(clspath, oripath):
    """
    """
    # read cls img
    ds = gdal.Open(clspath)
    nbands = ds.RasterCount
    geo_trans = ds.GetGeoTransform()
    w = ds.RasterXSize
    h = ds.RasterYSize
    img_shape = [h, w]
    print(img_shape)
    proj = ds.GetProjection()
    clsarr = ds.GetRasterBand(1).ReadAsArray()

    # read cls img
    ds = gdal.Open(oripath)
    nbands = ds.RasterCount
    geo_trans = ds.GetGeoTransform()
    w = ds.RasterXSize
    h = ds.RasterYSize
    img_shape = [h, w]
    print(img_shape)
    proj = ds.GetProjection()
    oriarr = ds.GetRasterBand(1).ReadAsArray()

    mask = oriarr.copy()
    mask[mask > 0] == 1
    mask[mask < 0] == 1

    idx = np.where(mask == 0)
    clsarr[idx] = 255

    # write ndvi file
    # build output path
    outpath = clspath.replace(".tif", ".nodata255.tif")
    # write output into tiff file
    out_ds = gdal.GetDriverByName("GTiff").Create(outpath, w, h, 1, gdal.GDT_Byte)
    out_ds.GetRasterBand(1).WriteArray(clsarr)
    out_ds.SetProjection(proj)
    out_ds.SetGeoTransform(geo_trans)
    out_ds.FlushCache()
    out_ds = None

    print("fin")


if __name__ == "__main__":
    """
    replace l8 border value from 0 to 255
    """

    l8cls_path = "/home/tq/data_pool/Y_ALL/crop_models/demo2/fujin_final_ensemble/MLPClassifier_result_LC08_L1TP_114027_20180831_20180912_01_T1_P20181203181831_rice.tif"
    l8ori_path = "/home/tq/data_pool/Y_ALL/crop_models/demo2/fujin_final_ensemble/LC08_L1TP_114027_20180831_20180912_01_T1_stacked.tif"

    replace_border(l8cls_path, l8ori_path)

    print("fin")
