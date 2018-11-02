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
    combines two drought index into one (TVDI, dNDVI) -> TQDI
    use new classification results and  shape
    """
    outdir = "/home/tq/data_pool/china_crop/20181101/"
    cropname = "corn"

    path_tvdi = "/home/tq/data_pool/zgq/crop_class/tvdi_ndvi_class.tif"

    # open tvdi
    print("opening tvdi")
    ds = gdal.Open(path_tvdi)
    vi_width = ds.RasterXSize
    vi_height = ds.RasterYSize
    vi_geotrans = ds.GetGeoTransform()
    vi_proj = ds.GetProjection()
    data_tvdi = ds.GetRasterBand(1).ReadAsArray() * 1.0
    data_dndvi = ds.GetRasterBand(2).ReadAsArray() * 1.0
    data_mask = ds.GetRasterBand(3).ReadAsArray()

    data_mask[data_mask == 1] = 0  # rice
    data_mask[data_mask == 2] = 1  # corn
    data_mask[data_mask == 3] = 0  # soybean
    data_mask[data_mask == 6] = 0.0

    # ===============stretch index to 0,1
    # ----------dndvi-------
    minv = 0
    maxv = -0.35

    new_dndvi = (data_dndvi - minv) / (maxv - minv)
    new_dndvi[new_dndvi > 1] = 1
    new_dndvi[new_dndvi < 0] = 0

    tqdi = new_dndvi * data_tvdi * data_mask

    # data_mask[data_mask == 255] = 0
    # tqdi = tqdi * data_mask

    # save file
    print("save")
    outpath = outdir + "tqdi_" + cropname + "_newclass.tif"
    out_ds = gdal.GetDriverByName("GTiff").Create(
        outpath, vi_width, vi_height, 1, gdal.GDT_Float32
    )
    out_ds.SetProjection(vi_proj)
    out_ds.SetGeoTransform(vi_geotrans)
    out_ds.GetRasterBand(1).WriteArray(tqdi)
    out_ds.FlushCache()
    out_ds = None
    print("save fin")

    print("fin")
