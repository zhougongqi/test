import os, re
import sys
import math
import glob
import pprint
import subprocess
import numpy as np
from osgeo import gdal, osr, gdalnumeric

from calc_mask_by_shape import *

home_dir = os.path.expanduser("~")


def replace_invalid_value(array: np.ndarray, new_value: int) -> np.ndarray:
    """
    Function:
        replace the  NaN, Inf, -Inf values in given array $array
    return:
        a new array without NaN.
    """
    where_are_nan = np.isnan(array)
    array[where_are_nan] = new_value

    where_are_inf = np.isinf(array)
    array[where_are_inf] = new_value

    where_are_isneginf = np.isneginf(array)
    array[where_are_isneginf] = new_value
    return array


def calc_ndvi(l8list: str, outdir: str):
    """
    input:
        l8dir:  a list contains l8 imgs to be calculated
        outdir: dir of output ndvi imgs
    """
    nfiles = len(l8list)

    for n in range(nfiles):
        fname = os.path.basename(l8list[n])
        print(fname)

        # read img
        ds = gdal.Open(l8list[n])
        nbands = ds.RasterCount
        geo_trans = ds.GetGeoTransform()
        w = ds.RasterXSize
        h = ds.RasterYSize
        img_shape = [h, w]
        proj = ds.GetProjection()
        red = ds.GetRasterBand(4).ReadAsArray()
        nir = ds.GetRasterBand(5).ReadAsArray()

        # calc ndvi
        ndvi = (nir * 1.0 - red) / (nir * 1.0 + red)
        ndvi[ndvi > 1] = 1
        ndvi[ndvi < 0] = 0
        ndvi = replace_invalid_value(ndvi, 0.0)

        # write ndvi file
        # build output path
        outpath = outdir + fname.replace(".tif", "ndvi.tif")
        # write output into tiff file
        out_ds = gdal.GetDriverByName("GTiff").Create(
            outpath, w, h, 1, gdal.GDT_Float32
        )
        out_ds.GetRasterBand(1).WriteArray(ndvi)
        out_ds.SetProjection(proj)
        out_ds.SetGeoTransform(geo_trans)
        out_ds.FlushCache()
        out_ds = None

    return True


def calc_ndvi_and_ww(ndvi_list: str, outdir: str):
    """
    input:
        l8dir:  a list contains l8 imgs to be calculated
        outdir: dir of output ndvi imgs
    """
    nfiles = len(ndvi_list)

    # stack all ndvi files
    # run gdal_merge.py to stack them
    tmp_file = outdir + "tmp.tif"
    cmd_str = (
        "gdal_merge.py -separate -of GTiff -o "  # -n 0 -a_nodata 0
        + tmp_file
        + " "
        + ndvi_list[0]
        + " "
        + ndvi_list[1]
        + " "
        + ndvi_list[2]
    )
    print("cmd string is :", cmd_str)
    process_status = subprocess.run(cmd_str, shell=True)

    # read img
    ds = gdal.Open(tmp_file)
    nbands = ds.RasterCount
    geo_trans = ds.GetGeoTransform()
    w = ds.RasterXSize
    h = ds.RasterYSize
    img_shape = [h, w]
    proj = ds.GetProjection()
    n1 = ds.GetRasterBand(1).ReadAsArray()
    n2 = ds.GetRasterBand(2).ReadAsArray()
    n3 = ds.GetRasterBand(3).ReadAsArray()

    ww = n1.copy().astype(np.int8)
    ww[:, :] = 0

    n0 = n3 - n1
    idx = np.where(n0 > 0.1)
    ww[idx] = 1

    # idx = np.where(n0 > 0)
    # ww[idx] += 1

    ww[ww < 1] = 0  # (nfiles - 1) keep the largest pixels

    idx = np.where(n3 <= 0.35)
    ww[idx] = 0

    # write ndvi file
    # build output path
    outpath = outdir + "ww.tif"
    # write output into tiff file
    out_ds = gdal.GetDriverByName("GTiff").Create(outpath, w, h, 1, gdal.GDT_Byte)
    out_ds.GetRasterBand(1).WriteArray(ww)
    out_ds.SetProjection(proj)
    out_ds.SetGeoTransform(geo_trans)
    out_ds.FlushCache()
    out_ds = None

    return True


def cut_and_count_pixels(path: str):
    # read img
    ds = gdal.Open(path)
    nbands = ds.RasterCount
    geo_trans = ds.GetGeoTransform()
    w = ds.RasterXSize
    h = ds.RasterYSize
    img_shape_wh = [w, h]
    proj = ds.GetProjection()
    ww = ds.GetRasterBand(1).ReadAsArray()
    ww[ww > 0] = 1

    # clip by shape
    shppath = "/home/tq/data_pool/china_crop/NCP_winter_wheat_test/shape/hebi-shi.shp"
    mask, tmp1, tmp2 = calc_mask_by_shape(shppath, geo_trans, img_shape_wh)
    print(mask.shape)

    ww = ww * mask

    num = np.sum(ww)
    area = num * 30.0 * 30 / 666 / 10000

    print(area, "wan mu")

    return True


if __name__ == "__main__":
    # winter wheat extraction
    # using Oct, Nov, Dec imgs, 2-3 time spot is needed

    l8_path = home_dir + "/data_pool/china_crop/NCP_winter_wheat_test/L8-124-35/"
    # "/data_pool/china_crop/NCP_winter_wheat_test/L8-122-37/"
    l8_list = [
        "/home/tq/data_pool/china_crop/NCP_winter_wheat_test/L8-124-35/LC08_L1TP_124035_20171106_20171121_01_T1_stacked_decloud.tif",
        "/home/tq/data_pool/china_crop/NCP_winter_wheat_test/L8-124-35/LC08_L1TP_124035_20171122_20171206_01_T1_stacked_decloud.tif",
        "/home/tq/data_pool/china_crop/NCP_winter_wheat_test/L8-124-35/LC08_L1TP_124035_20171208_20171223_01_T1_stacked_decloud.tif",
    ]
    # [
    #     "/home/tq/data_pool/china_crop/NCP_winter_wheat_test/L8-122-37/LC08_L1TP_122037_20171108_20171121_01_T1_stacked_decloud.tif",
    #     "/home/tq/data_pool/china_crop/NCP_winter_wheat_test/L8-122-37/LC08_L1TP_122037_20171124_20171206_01_T1_stacked_decloud.tif",
    #     "/home/tq/data_pool/china_crop/NCP_winter_wheat_test/L8-122-37/LC08_L1TP_122037_20171210_20171223_01_T1_stacked_decloud.tif",
    # ]

    out_path = home_dir + "/data_pool/china_crop/NCP_winter_wheat_test/L8-124-35-out/"

    stat = calc_ndvi(l8_list, out_path)

    ndvi_list = glob.glob(out_path + "*ndvi.tif")
    ndvi_list.sort()
    stat = calc_ndvi_and_ww(ndvi_list, out_path)

    # # count pixels
    stat = cut_and_count_pixels(out_path + "ww.tif")

    print("fin")
