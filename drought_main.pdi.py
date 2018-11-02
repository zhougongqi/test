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
    PDI
    """
    outdir = "/home/tq/data_pool/dry/drought_index/"
    r_name = "/home/tq/data_pool/dry/sentienl2_L1/20180810/S2B_MSIL2A_20180810T024539_N0206_R132_T51TVG_20180810T053506.SAFE/GRANULE/L2A_T51TVG_A007451_20180810T025301/IMG_DATA/R10m/T51TVG_20180810T024539_B04_10m.jp2"
    nir_name = "/home/tq/data_pool/dry/sentienl2_L1/20180810/S2B_MSIL2A_20180810T024539_N0206_R132_T51TVG_20180810T053506.SAFE/GRANULE/L2A_T51TVG_A007451_20180810T025301/IMG_DATA/R10m/T51TVG_20180810T024539_B08_10m.jp2"
    top_n_pix = 25
    w_ratio = 1
    h_ratio = 1
    ndvi_thre = 0.4

    start_w = 0  # 3000
    start_h = 0  # 3000

    bname = os.path.basename(r_name)
    shortname, ext = os.path.splitext(bname)
    name_pieces = shortname.split("_")
    date_str = name_pieces[1][0:8]

    # open r band
    print("opening vi")
    ds = gdal.Open(r_name)
    r_width = ds.RasterXSize
    r_height = ds.RasterYSize
    r_geotrans = ds.GetGeoTransform()
    r_proj = ds.GetProjection()
    r_data = ds.ReadAsArray(
        start_w, start_h, int(r_width / w_ratio), int(r_height / h_ratio)
    )
    del ds
    h, w = r_data.shape
    print(r_data.shape)
    r0 = r_data
    r_data = r_data[0::20, 0::20]

    # open nir
    print("opening nir")
    ds = gdal.Open(nir_name)
    nir_width = ds.RasterXSize
    nir_height = ds.RasterYSize
    nir_geotrans = ds.GetGeoTransform()
    nir_proj = ds.GetProjection()
    nir_data = ds.ReadAsArray(
        start_w, start_h, int(nir_width / w_ratio), int(nir_height / h_ratio)
    )
    del ds
    print(nir_data.shape)
    nir0 = nir_data
    nir_data = nir_data[0::20, 0::20]

    assert r_height == nir_height, "shape not match, h"
    assert r_width == nir_width, "shape not match, w"

    # get vi-nir plot
    print("processing data")
    r_f = r_data.flatten()
    nir_f = nir_data.flatten()

    # remove invalid data
    ind_r_inv = np.where(r_f <= 0.1)
    r_f = np.delete(r_f, ind_r_inv, axis=0)
    nir_f = np.delete(nir_f, ind_r_inv, axis=0)

    ind_r_inv = np.where(r_f >= 10000)
    r_f = np.delete(r_f, ind_r_inv, axis=0)
    nir_f = np.delete(nir_f, ind_r_inv, axis=0)

    # print("plotting")
    # plt.figure(figsize=(8, 8))
    # plt.scatter(r_f, nir_f, color="gray", marker=".")
    # plt.show()

    # seperate 100 intervals
    nir_list = []
    for l in range(100):
        nir_list.append([])

    nir_top = []
    nir_bottom = []
    r_100 = []

    for p in range(len(r_f)):
        r = r_f[p]
        nir = nir_f[p]
        n = math.floor(r / 100)
        nir_list[n].append(nir)
        if n == 0:
            test = 1
        print_progress_bar(p, len(r_f))
    print("")

    #
    for l in range(100):
        if nir_list[l] is None:
            print_progress_bar(l + 1, 100)
            continue

        nir_list[l].sort()
        if len(nir_list[l]) >= top_n_pix:
            nir_bottom.append(get_average(nir_list[l][0:top_n_pix]))
            nir_top.append(get_average(nir_list[l][(-1 * top_n_pix) :]))
            r_100.append(l * 100 + 50)
        print_progress_bar(l + 1, 100)
    print("")

    print("plotting")
    r_f100 = [v * 10000 for v in r_f]
    plt.figure(figsize=(8, 8))
    plt.scatter(r_f, nir_f, color="gray", marker=".", alpha=0.05)
    for l in range(len(r_100)):
        plt.scatter(r_100[l], nir_top[l], color="red", marker="o")
        plt.scatter(r_100[l], nir_bottom[l], color="blue", marker="o")
    plt.show()

    # different from tvdi
    k_soil, b_soil = optimize.curve_fit(f_1, r_100, nir_bottom)[0]

    print("calc pdi")
    # pdi = r0
    # for row in range(h):
    #     for col in range(w):
    #         c1 = 1 / np.sqrt(1 + np.square(k_soil))
    #         c2 = r0[row, col] + k_soil * nir0[row, col]
    #         pdi[row, col] = c2 / c1
    c1 = 1 / np.sqrt(1 + np.square(k_soil))
    pdi = (r0 + k_soil * nir0) / c1

    ndvi = (nir0 - r0) / (nir0 + r0)
    ind_r_inv = np.where(ndvi > 1)
    ndvi[ind_r_inv] = 1
    ind_r_inv = np.where(ndvi < ndvi_thre)
    ndvi[ind_r_inv] = 0

    np.nan_to_num(ndvi)

    # save file
    print("calc pdi")
    outpath = outdir + "pdi" + date_str + ".tif"
    out_ds = gdal.GetDriverByName("GTiff").Create(outpath, w, h, 1, gdal.GDT_Float32)
    out_ds.SetProjection(r_proj)
    out_ds.SetGeoTransform(r_geotrans)
    out_ds.GetRasterBand(1).WriteArray(pdi)
    out_ds.FlushCache()

    outpath = outdir + "pdi" + date_str + "-ndvi-mask-" + str(ndvi_thre) + ".tif"
    out_ds = gdal.GetDriverByName("GTiff").Create(outpath, w, h, 1, gdal.GDT_Float32)
    out_ds.SetProjection(r_proj)
    out_ds.SetGeoTransform(r_geotrans)
    out_ds.GetRasterBand(1).WriteArray(ndvi)
    out_ds.FlushCache()

    print("fin")
