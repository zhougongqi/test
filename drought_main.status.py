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

    path = "/home/tq/data_pool/china_crop/20181030/rice_tvdi.tif"
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
    ttarea = len(idx[0])
    area0 = len(np.where(data == 0))

    # # for dndvi (-0.35-0.35)=========================================
    # idx = np.where(data > 0.001)
    # ttarea = len(idx[0])
    # idx = np.where(data < -0.001)
    # ttarea = ttarea + len(idx[0])

    # idx = np.where(data < -0.001)
    # idx2 = np.where(data <= 0.001)
    # area0 = len(idx2[0]) - len(idx[0])

    # idx = np.where(data < -0.35)
    # subarea = len(idx[0])
    # print("[", -1, "~", -0.35, "]", subarea, subarea / ttarea)
    # for fl in range(7):
    #     flf = fl * 1.0
    #     f = (flf - 3.5) / 10
    #     ff = (flf + 1 - 3.5) / 10
    #     idx = np.where(data < f)
    #     idx2 = np.where(data <= ff)
    #     subarea = len(idx2[0]) - len(idx[0])
    #     if fl == 3:
    #         subarea = subarea - area0
    #     print("[", f, "~", ff, "]", subarea, subarea / ttarea)
    # idx = np.where(data < 0.35)
    # idx2 = np.where(data <= 1)
    # subarea = len(idx2[0]) - len(idx[0])
    # print("[", 0.35, "~", 1, "]", subarea, subarea / ttarea)
    ######################################################################

    # for tvdi (0,1)
    for fl in range(10):
        f = fl / 10
        ff = (fl + 1) / 10
        idx = np.where(data < f)
        idx2 = np.where(data <= ff)
        subarea = len(idx2[0]) - len(idx[0])
        print("[", f, ":", ff, "]", subarea, subarea / ttarea)

    print("fin")
