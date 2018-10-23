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

    outdir = "/home/tq/data_pool/dry/drought_index/"
    vi_name = "/home/tq/data_pool/dry/cliped_res/S2B_MSIL2A_20180810T024539_N0206_R132_T51TVG_20180810T053506_ndvi_cliped_region.tif"
    lst_name = "/home/tq/data_pool/dry/cliped_res/MOD11A2.A2018209.h27v04.006.2018218215735_LST_Day_1km_reprj_cliped_region.tif"
    top_n_pix = 100
    w_ratio = 1
    h_ratio = 1
    ndvi_thre = 0.4

    start_w = 0
    start_h = 0

    bname = os.path.basename(vi_name)
    shortname, ext = os.path.splitext(bname)
    name_pieces = shortname.split("_")
    date_str = name_pieces[6][0:8]

    # open vi
    print("opening vi")
    ds = gdal.Open(vi_name)
    vi_width = ds.RasterXSize
    vi_height = ds.RasterYSize
    vi_geotrans = ds.GetGeoTransform()
    vi_proj = ds.GetProjection()
    vi_data = ds.ReadAsArray(
        start_w, start_h, int(vi_width / w_ratio), int(vi_height / h_ratio)
    )
    del ds
    vi_data = vi_data[0, :, :].reshape(vi_height, vi_width)
    h, w = vi_data.shape
    print(vi_data.shape)
    ndvi0 = vi_data
    vi_data = vi_data[0::20, 0::20]

    # open lst
    print("opening lst")
    ds = gdal.Open(lst_name)
    lst_width = ds.RasterXSize
    lst_height = ds.RasterYSize
    lst_geotrans = ds.GetGeoTransform()
    lst_proj = ds.GetProjection()
    lst_data = ds.ReadAsArray(
        start_w, start_h, int(lst_width / w_ratio), int(lst_height / h_ratio)
    )
    del ds
    print(lst_data.shape)
    lst_data = lst_data * 0.02
    lst0 = lst_data
    lst_data = lst_data[0::20, 0::20]

    assert vi_height == lst_height, "shape not match, h"
    assert vi_width == lst_width, "shape not match, w"

    # get vi-lst plot
    print("processing data")
    vi_f = vi_data.flatten()
    lst_f = lst_data.flatten()

    # remove invalid data
    ind_vi_inv = np.where(vi_f <= 0.1)
    vi_f = np.delete(vi_f, ind_vi_inv, axis=0)
    lst_f = np.delete(lst_f, ind_vi_inv, axis=0)

    ind_vi_inv = np.where(vi_f > 1)
    vi_f = np.delete(vi_f, ind_vi_inv, axis=0)
    lst_f = np.delete(lst_f, ind_vi_inv, axis=0)

    ind_vi_inv = np.where(lst_f < 10)
    vi_f = np.delete(vi_f, ind_vi_inv, axis=0)
    lst_f = np.delete(lst_f, ind_vi_inv, axis=0)

    # print("plotting")
    # plt.figure(figsize=(8, 8))
    # plt.scatter(vi_f, lst_f, color="gray", marker=".")
    # plt.show()

    # seperate 100 intervals
    lst_list = []
    for l in range(100):
        lst_list.append([])

    lst_top = []
    lst_bottom = []
    vi_100 = []

    for p in range(len(vi_f)):
        vi = vi_f[p]
        lst = lst_f[p]
        n = math.floor(vi * 100)
        lst_list[n].append(lst)
        if n == 0:
            test = 1
        print_progress_bar(p, len(vi_f))
    print("")

    #
    for l in range(100):
        if lst_list[l] is None:
            print_progress_bar(l + 1, 100)
            continue

        lst_list[l].sort()
        if len(lst_list[l]) >= top_n_pix:
            lst_bottom.append(get_average(lst_list[l][0:top_n_pix]))
            lst_top.append(get_average(lst_list[l][(-1 * top_n_pix) :]))
            vi_100.append(l / 100 + 0.005)
        print_progress_bar(l + 1, 100)
    print("")

    print("plotting")
    vi_f100 = [v * 10000 for v in vi_f]
    plt.figure(figsize=(8, 8))
    plt.scatter(vi_f, lst_f, color="gray", marker=".", alpha=0.1)
    for l in range(len(vi_100)):
        plt.scatter(vi_100[l], lst_top[l], color="red", marker="o")
        plt.scatter(vi_100[l], lst_bottom[l], color="blue", marker="o")
    plt.show()
    vi_fit_list = np.arange(0, 100)

    k_dry, b_dry = optimize.curve_fit(f_1, vi_100[-25:], lst_top[-25:])[0]
    k_wet, b_wet = optimize.curve_fit(f_1, vi_100, lst_bottom)[0]

    tsmin = k_wet * ndvi0 + b_wet
    tsmax = k_dry * ndvi0 + b_dry
    print(k_dry, b_dry, k_wet, b_wet)
    tvdi = (lst0 - tsmin) / (tsmax - tsmin)

    k_dry = -20.46316917036919
    b_dry = 327.2212982580681
    k_wet = -2.84819330323533
    b_wet = 304.97250059024964

    tvdi[tvdi >= 1] = 1
    tvdi[tvdi <= 0] = 0

    ndvi_mask = ndvi0
    ind_r_inv = np.where(ndvi0 > ndvi_thre)
    ndvi_mask[ind_r_inv] = 1
    ind_r_inv = np.where(ndvi0 <= ndvi_thre)
    ndvi_mask[ind_r_inv] = 0

    tvdi_masked = tvdi * ndvi_mask

    # save file
    print("calc pdi")
    outpath = outdir + "tvdi" + date_str + "-masked.tif"
    out_ds = gdal.GetDriverByName("GTiff").Create(outpath, w, h, 1, gdal.GDT_Float32)
    out_ds.SetProjection(vi_proj)
    out_ds.SetGeoTransform(vi_geotrans)
    out_ds.GetRasterBand(1).WriteArray(tvdi_masked)
    out_ds.FlushCache()

    print("fin")
