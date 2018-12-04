import os, re
import sys
import math
import glob
import pprint
import subprocess
import numpy as np
from osgeo import gdal, osr, gdalnumeric
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label
import cv2


home_dir = os.path.expanduser("~")

if __name__ == "__main__":
    """
    remove small objects and fill small holes for Rocks
    """
    clspath = "/home/tq/data_pool/Y_ALL/crop_models/demo2/fujin_final_ensemble/final_merge_clipped.tif"

    # read img
    ds = gdal.Open(clspath)
    nbands = ds.RasterCount
    geo_trans = ds.GetGeoTransform()
    w = ds.RasterXSize
    h = ds.RasterYSize
    img_shape = [h, w]
    proj = ds.GetProjection()
    data = ds.GetRasterBand(1).ReadAsArray()

    # binarize each class
    c1 = np.zeros_like(data).astype(np.bool)
    idx = np.where(data == 1)
    c1[idx] = 1

    c2 = np.zeros_like(data).astype(np.bool)
    idx = np.where(data == 2)
    c2[idx] = 1

    c3 = np.zeros_like(data).astype(np.bool)
    idx = np.where(data == 3)
    c3[idx] = 1

    # remove and fill
    c1 = remove_small_objects(c1, 6, connectivity=2)
    c1 = remove_small_objects(c1, 2, connectivity=1)
    c1 = remove_small_holes(c1, 6, connectivity=2)

    c2 = remove_small_objects(c2, 6, connectivity=2)
    c2 = remove_small_objects(c2, 2, connectivity=1)
    c2 = remove_small_holes(c2, 6, connectivity=2)

    c3 = remove_small_objects(c3, 6, connectivity=2)
    c3 = remove_small_objects(c3, 2, connectivity=1)
    # c3 = remove_small_holes(c3, 20, connectivity=1)

    # merge all
    ca = np.zeros_like(data).astype(np.int8)
    idx = np.where(c1 == 1)
    ca[idx] = 1

    idx = np.where(c2 == 1)
    ca[idx] = 2

    idx = np.where(c3 == 1)
    ca[idx] = 3

    # write ndvi file
    # build output path
    outpath = clspath.replace(".tif", ".no_freckles_4.tif")
    # write output into tiff file
    out_ds = gdal.GetDriverByName("GTiff").Create(outpath, w, h, 1, gdal.GDT_Byte)
    out_ds.GetRasterBand(1).WriteArray(ca)
    out_ds.SetProjection(proj)
    out_ds.SetGeoTransform(geo_trans)
    out_ds.FlushCache()
    out_ds = None

    print("fin")
