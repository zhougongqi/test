import os, re
import sys
import math
import glob
import pprint
import subprocess
import numpy as np
from osgeo import gdal, osr, gdalnumeric
from skimage.morphology import remove_small_objects
from skimage.measure import label
import cv2

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
    outlist = []

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
        outlist.append(outpath)

    # calc red-ndvi of t3
    # read img
    ds = gdal.Open(l8list[-1])
    nbands = ds.RasterCount
    geo_trans = ds.GetGeoTransform()
    w = ds.RasterXSize
    h = ds.RasterYSize
    img_shape = [h, w]
    proj = ds.GetProjection()
    green = ds.GetRasterBand(3).ReadAsArray()
    red = ds.GetRasterBand(4).ReadAsArray()

    r_g = red * 1.0 - green

    # build output path
    outpath = outdir + "red-green.tif"
    # write output into tiff file
    out_ds = gdal.GetDriverByName("GTiff").Create(outpath, w, h, 1, gdal.GDT_Float32)
    out_ds.GetRasterBand(1).WriteArray(r_g)
    out_ds.SetProjection(proj)
    out_ds.SetGeoTransform(geo_trans)
    out_ds.FlushCache()
    out_ds = None
    outlist.append(outpath)

    return outlist


def calc_ndvi_and_ww(ndvi_list: str, outdir: str):
    """
    input:
        ndvi_list:  a list contains l8 imgs to be calculated
        outdir: dir of output ndvi imgs

        ndvi_list is like:
            ndvi of t1
            ndvi of t2
            ndvi of t3
            red-green of t3
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
        + " "
        + ndvi_list[3]
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
    r_g = ds.GetRasterBand(4).ReadAsArray()

    ww = n1.copy().astype(np.int8)
    ww[:, :] = 0

    n0 = n3 - n1
    idx = np.where(n0 > 0)
    ww[idx] = 1

    n0 = n2 - n1
    idx = np.where(n0 > 0)
    ww[idx] += 1

    idx = np.where(n3 <= 0.22)  # delete small ndvi pixels at t3
    ww[idx] = 0
    idx = np.where(r_g < 0)  # delete red > green pixels
    ww[idx] = 0

    ww[ww < 2] = 0
    ww[ww == 2] = 1

    wwb = ww.astype(np.bool)
    wwb = remove_small_objects(wwb, 8)
    ww = wwb.astype(np.int8)

    # write ndvi file
    # build output path
    outpath = outdir + "ww-rso_sk.tif"
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


def stat_area_zonal_county15(raster_path: str, shp_path: str):
    """
    use 2015 county bound shp
    """
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
    gdal.SetConfigOption("SHAPE_ENCODING", "")  # chinese char
    ogr.RegisterAll()

    # open vector shape_path
    try:
        driver = ogr.GetDriverByName("ESRI Shapefile")
        shapef = driver.Open(shp_path)
        # shapef = ogr.Open(shp_path)
        lyr = shapef.GetLayer(0)
        poly = lyr.GetNextFeature()
        lyrdn = lyr.GetLayerDefn()
        maxpoly = lyr.GetFeatureCount()
    except Exception as e:
        raise Exception("open shape failed!")

    # open raster

    ds = gdal.Open(raster_path)
    geo_trans = ds.GetGeoTransform()
    x_size = ds.RasterXSize
    y_size = ds.RasterYSize
    img_shape = [x_size, y_size]
    data = ds.GetRasterBand(1).ReadAsArray()

    lyr.ResetReading()
    npoly = 0
    tmplist = []
    for poly in lyr:
        # loop poly in lyr, draw ROIs in Image
        polyname = poly.GetField("label")
        npoly = npoly + 1

        mask, num_label, list_label = calc_mask_by_shape(
            shp_path,
            geo_trans,
            img_shape,
            specified_field="label",
            condition=[polyname],
            mask_value=1,
            field_strict=True,
        )
        if mask is None:
            raise Exception("mask is wrong")

        pixels = get_valid_data(data, mask)
        idx = np.where(pixels == 1)
        n_vali_pixels = len(idx[0])
        tmplist.append([polyname, n_vali_pixels])

    for l in tmplist:
        print("{},\t{}".format(l[0], l[1]))
    return True


def get_valid_data(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    nodata_value: list = [0],
    validRange: list = [0, 10000],
):
    """

    """
    mask_idx = np.where(mask > 0)
    train_data = data[mask_idx]
    mask_t = mask[mask_idx]
    if train_data.shape == mask_t.shape:
        pass  # print(train_data.shape)
    else:
        raise Exception("get_valid_data_new(): shape not match! skip")

    return train_data.flatten()


def remove_small_blocks(ww):
    # remove small objects
    ww[1, 1] = 1
    # ww1 = remove_small_objects(ww, 800)

    nb_com, output, stats, cen = cv2.connectedComponentsWithStats(ww, connectivity=4)
    sizes = stats[1:, -1]
    nb_com = nb_com - 1
    min_size = 8
    ww1 = np.zeros((output.shape))
    for i in range(0, nb_com):
        if i % 100 == 0:
            print(i, "--", nb_com)
        if sizes[i] >= min_size:
            ww1[output == i + 1] = 1
    total_area = np.sum(ww)
    print(total_area)
    total_area = np.sum(ww1)
    print(total_area)
    return ww1


if __name__ == "__main__":
    # winter wheat extraction
    # using Oct, Nov, Dec imgs, 2-3 time spot is needed
    # add red-green as a criterion

    l8_path = home_dir + "/data_pool/china_crop/NCP_winter_wheat_test/L8-124-35/"
    # "/data_pool/china_crop/NCP_winter_wheat_test/L8-122-37/"
    l8_list = [
        "/home/tq/data_pool/china_crop/NCP_winter_wheat_test/L8/LC08_L1TP_123034_20171030_20171109_01_T1_stacked_decloud.tif",
        "/home/tq/data_pool/china_crop/NCP_winter_wheat_test/L8/LC08_L1TP_123034_20171115_20171122_01_T1_stacked_decloud.tif",
        # "/home/tq/data_pool/china_crop/NCP_winter_wheat_test/L8/LC08_L1TP_123034_20171201_20171207_01_T1_stacked_decloud.tif",
        "/home/tq/data_pool/china_crop/NCP_winter_wheat_test/L8/LC08_L1TP_123034_20171217_20171224_01_T1_stacked_decloud.tif",
    ]
    # [
    #     "/home/tq/data_pool/china_crop/NCP_winter_wheat_test/L8-124-35/LC08_L1TP_124035_20171106_20171121_01_T1_stacked_decloud.tif",
    #     "/home/tq/data_pool/china_crop/NCP_winter_wheat_test/L8-124-35/LC08_L1TP_124035_20171122_20171206_01_T1_stacked_decloud.tif",
    #     "/home/tq/data_pool/china_crop/NCP_winter_wheat_test/L8-124-35/LC08_L1TP_124035_20171208_20171223_01_T1_stacked_decloud.tif",
    # ]
    # [
    #     "/home/tq/data_pool/china_crop/NCP_winter_wheat_test/L8-122-37/LC08_L1TP_122037_20171108_20171121_01_T1_stacked_decloud.tif",
    #     "/home/tq/data_pool/china_crop/NCP_winter_wheat_test/L8-122-37/LC08_L1TP_122037_20171124_20171206_01_T1_stacked_decloud.tif",
    #     "/home/tq/data_pool/china_crop/NCP_winter_wheat_test/L8-122-37/LC08_L1TP_122037_20171210_20171223_01_T1_stacked_decloud.tif",
    # ]

    out_path = home_dir + "/data_pool/china_crop/NCP_winter_wheat_test/L8-out/test2.2/"
    vali_county = (
        "/home/tq/data_pool/china_crop/NCP_winter_wheat_test/shape/hebei-some-xians.shp"
    )

    ndvi_list = calc_ndvi(l8_list, out_path)
    # ndvi_list = glob.glob(out_path + "*ndvi.tif")
    # ndvi_list.sort()
    stat = calc_ndvi_and_ww(ndvi_list, out_path)

    # # count pixels ##useless in hebei
    # stat = cut_and_count_pixels(out_path + "ww.tif")

    # vali
    status = stat_area_zonal_county15(out_path + "ww-rso_sk.tif", vali_county)

    # # open result
    # ds = gdal.Open(out_path + "ww-clip.tif")
    # geo_trans = ds.GetGeoTransform()
    # w = ds.RasterXSize
    # h = ds.RasterYSize
    # proj = ds.GetProjection()
    # data = ds.GetRasterBand(1).ReadAsArray()

    # data1 = remove_small_blocks(data)

    # # build output path
    # outpath = out_path + "ww-rso.tif"
    # # write output into tiff file
    # out_ds = gdal.GetDriverByName("GTiff").Create(outpath, w, h, 1, gdal.GDT_Byte)
    # out_ds.GetRasterBand(1).WriteArray(data1)
    # out_ds.SetProjection(proj)
    # out_ds.SetGeoTransform(geo_trans)
    # out_ds.FlushCache()
    # out_ds = None

    print("fin")
