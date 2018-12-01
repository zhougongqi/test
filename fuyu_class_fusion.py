import logging
import math
import os
import numpy as np

from osgeo import gdal, gdalnumeric
from osgeo import ogr
from osgeo import osr

img_path = "/home/tq/data_pool/china_crop/Fuyu/fusion/fusion-3bands-l8s2-s2-cmask.tif"

# read the first one to get some paras
ds = gdal.Open(img_path)
geo_trans = ds.GetGeoTransform()
w = ds.RasterXSize
h = ds.RasterYSize
img_shape = [h, w]
proj = ds.GetProjection()
cls_l8s2 = ds.GetRasterBand(1).ReadAsArray()
cls_s2 = ds.GetRasterBand(2).ReadAsArray()
cmask = ds.GetRasterBand(3).ReadAsArray().astype(np.int8)
cmask[cmask == 2] = 0
print(np.max(cmask), np.min(cmask))

cls_l8s2[cls_l8s2 == 3] = 33
cls_l8s2[cls_l8s2 == 2] = 3
cls_l8s2[cls_l8s2 == 33] = 2

cliped = gdalnumeric.choose(cmask, (cls_s2, cls_l8s2))

print("rice: ", len(cliped[cliped == 1]))
print("soybean: ", len(cliped[cliped == 2]))
print("corn: ", len(cliped[cliped == 3]))
# build output path
outpath = img_path.replace(".tif", "_masked.tif")
# write output into tiff file
out_ds = gdal.GetDriverByName("GTiff").Create(outpath, w, h, 1, gdal.GDT_Byte)
out_ds.SetProjection(ds.GetProjection())
out_ds.SetGeoTransform(geo_trans)
out_ds.GetRasterBand(1).WriteArray(cliped)
out_ds.FlushCache()

print("done")
