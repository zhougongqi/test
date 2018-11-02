from osgeo import gdal
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

from general import *


def f_1(x, A, B):
    return A * x + B


if __name__ == "__main__":
    """
    status of tqdi
    """

    path = "/home/tq/data_pool/china_crop/20181101/tqdi_corn_newclass.tif"
    # open vi
    print("opening vi")
    ds = gdal.Open(path)
    vi_width = ds.RasterXSize
    vi_height = ds.RasterYSize
    vi_geotrans = ds.GetGeoTransform()
    vi_proj = ds.GetProjection()
    data = ds.ReadAsArray(0, 0, int(vi_width), int(vi_height))
    del ds

    data = data.astype(np.float)

    idx = np.where(data > 0)
    idx2 = np.where(data > 0.35)
    ttarea = len(idx[0]) - len(idx2[0])
    area0 = len(np.where(data == 0))

    # data = stretch_data(data, 0, 0.35)
    # for tvdi (0,1)

    idx = np.where(data <= 0)
    idx2 = np.where(data < 0.03)
    subarea = len(idx2[0]) - len(idx[0])
    print("0-0.00126", subarea, subarea / ttarea)

    idx = np.where(data <= 0.03)
    idx2 = np.where(data < 0.13)
    subarea = len(idx2[0]) - len(idx[0])
    print("0.00126-0.05", subarea, subarea / ttarea)

    idx = np.where(data <= 0.13)
    idx2 = np.where(data < 0.35)
    subarea = len(idx2[0]) - len(idx[0])
    print("0.05-0.35", subarea, subarea / ttarea)

    print("fin")
