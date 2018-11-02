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
    """
    outdir = "/home/tq/data_pool/china_crop/20180831/"
    cropname = "soybean"

    path_tvdi = "/home/tq/data_pool/china_crop/20181030/" + cropname + "_tvdi.tif"
    path_dndvi = "/home/tq/data_pool/china_crop/20181030/" + cropname + "_dndvi.tif"
    path_cropmask = (
        "/home/tq/data_pool/china_crop/s1_rice_test/clip_tif/"
        + cropname
        + "_clip.tif"
        # /home/tq/data_pool/china_crop/s1_rice_test/clip_tif/rice_clip.tif
        # / home / tq / data_pool / china_crop / 20181030 / corn_dndvi.tif
    )
    # open tvdi
    print("opening tvdi")
    ds = gdal.Open(path_tvdi)
    vi_width = ds.RasterXSize
    vi_height = ds.RasterYSize
    vi_geotrans = ds.GetGeoTransform()
    vi_proj = ds.GetProjection()
    data = ds.ReadAsArray(0, 0, int(vi_width), int(vi_height))
    del ds
    data_tvdi = data.astype(np.float)

    # open dndvi
    print("opening tvdi")
    ds = gdal.Open(path_dndvi)
    vi_width = ds.RasterXSize
    vi_height = ds.RasterYSize
    vi_geotrans = ds.GetGeoTransform()
    vi_proj = ds.GetProjection()
    data = ds.ReadAsArray(0, 0, int(vi_width), int(vi_height))
    del ds
    data_dndvi = data[0:2436, :].astype(np.float)

    # open cropmask
    print("opening cropmask")
    ds = gdal.Open(path_cropmask)
    vi_width = ds.RasterXSize
    vi_height = ds.RasterYSize
    vi_geotrans = ds.GetGeoTransform()
    vi_proj = ds.GetProjection()
    data = ds.ReadAsArray(0, 0, int(vi_width), int(vi_height))
    del ds
    data_mask = data

    # stat
    idx = np.where(data > 0)
    ttarea = len(idx[0])
    area0 = len(np.where(data == 0))

    # ===============stretch index to 0,1
    # ----------dndvi-------
    minv = 0
    maxv = -0.35

    new_dndvi = (data_dndvi - minv) / (maxv - minv)
    new_dndvi[new_dndvi > 1] = 1
    new_dndvi[new_dndvi < 0] = 0

    tqdi = new_dndvi * data_tvdi

    # data_mask[data_mask == 255] = 0
    # tqdi = tqdi * data_mask

    # save file
    print("save")
    outpath = outdir + "tqdi" + cropname + ".tif"
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
