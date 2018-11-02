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

    outdir = "/home/tq/data_pool/dry/drought_index2/"
    vi_name = "/home/tq/data_pool/dry/cliped_res/S2B_MSIL2A_20180810T024539_N0206_R132_T51TVG_20180810T053506_ndvi_cliped_region.tif"
    lst_name = "/home/tq/data_pool/dry/cliped_res/MOD11A2.A2018217.h27v04.006.2018227165657_LST_Day_1km_reprj_cliped_region.tif"

    vi_name17 = "/home/tq/data_pool/dry/cliped_res/S2B_MSIL2A_20170825T024539_N0205_R132_T51TCG_20170825T025536_ndvi_cliped_region.tif"
    lst_name17 = "/home/tq/data_pool/dry/cliped_res/MOD11A2.A2017217.h27v04.006.2017234151617_LST_Day_1km_reprj_cliped_region.tif"

    vi_name16 = "/home/tq/data_pool/dry/cliped_res/S2A_MSIL2A_20160822T023552_N0204_R089_T51TVG_20160822T024021_ndvi_cliped_region.tif"
    lst_name16 = "/home/tq/data_pool/dry/cliped_res/MOD11A2.A2016225.h27v04.006.2016244043626_LST_Day_1km_reprj_cliped_region.tif"

    top_n_pix = 20
    w_ratio = 1
    h_ratio = 1
    ndvi_thre = 0.4
    step = 50
    step_draw = 5

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
    vi_data = vi_data[0::step, 0::step]

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
    lst_data = lst_data[0::step, 0::step]

    #######################################################################
    # 2016 17
    # open vi
    print("opening vi 17")
    ds = gdal.Open(vi_name17)
    vi_data17 = ds.ReadAsArray(
        start_w, start_h, int(vi_width / w_ratio), int(vi_height / h_ratio)
    )
    del ds
    vi_data17 = vi_data17[0, :, :].reshape(vi_height, vi_width)
    vi_data17 = vi_data17[0::step, 0::step]

    # open lst
    print("opening lst 17")
    ds = gdal.Open(lst_name17)
    lst_data17 = ds.ReadAsArray(
        start_w, start_h, int(lst_width / w_ratio), int(lst_height / h_ratio)
    )
    del ds
    lst_data17 = lst_data17[0::step, 0::step] * 0.02

    print("opening vi 16")
    ds = gdal.Open(vi_name16)
    vi_data16 = ds.ReadAsArray(
        start_w, start_h, int(vi_width / w_ratio), int(vi_height / h_ratio)
    )
    del ds
    vi_data16 = vi_data16[0, :, :].reshape(vi_height, vi_width)
    vi_data16 = vi_data16[0::step, 0::step]

    # open lst
    print("opening lst 16")
    ds = gdal.Open(lst_name16)
    lst_data16 = ds.ReadAsArray(
        start_w, start_h, int(lst_width / w_ratio), int(lst_height / h_ratio)
    )
    del ds
    lst_data16 = lst_data16[0::step, 0::step] * 0.02

    #######################################################################

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

    # ----------------------------------------------------
    vi_f17 = vi_data17.flatten()
    lst_f17 = lst_data17.flatten()
    ind_vi_inv = np.where(vi_f17 <= 0.1)
    vi_f17 = np.delete(vi_f17, ind_vi_inv, axis=0)
    lst_f17 = np.delete(lst_f17, ind_vi_inv, axis=0)

    ind_vi_inv = np.where(vi_f17 > 1)
    vi_f17 = np.delete(vi_f17, ind_vi_inv, axis=0)
    lst_f17 = np.delete(lst_f17, ind_vi_inv, axis=0)

    ind_vi_inv = np.where(lst_f17 < 10)
    vi_f17 = np.delete(vi_f17, ind_vi_inv, axis=0)
    lst_f17 = np.delete(lst_f17, ind_vi_inv, axis=0)

    # ----------------------------------------------------
    vi_f16 = vi_data16.flatten()
    lst_f16 = lst_data16.flatten()
    ind_vi_inv = np.where(vi_f16 <= 0.1)
    vi_f16 = np.delete(vi_f16, ind_vi_inv, axis=0)
    lst_f16 = np.delete(lst_f16, ind_vi_inv, axis=0)

    ind_vi_inv = np.where(vi_f16 > 1)
    vi_f16 = np.delete(vi_f16, ind_vi_inv, axis=0)
    lst_f16 = np.delete(lst_f16, ind_vi_inv, axis=0)

    ind_vi_inv = np.where(lst_f16 < 10)
    vi_f16 = np.delete(vi_f16, ind_vi_inv, axis=0)
    lst_f16 = np.delete(lst_f16, ind_vi_inv, axis=0)

    # print("plotting")
    # plt.figure(figsize=(8, 8))
    # plt.scatter(vi_f, lst_f, color="gray", marker=".")
    # plt.show()

    # seperate 100 intervals
    lst_list = []
    for l in range(100):
        lst_list.append([])
    lst_list17 = []
    for l in range(100):
        lst_list17.append([])
    lst_list16 = []
    for l in range(100):
        lst_list16.append([])

    lst_top = []
    lst_bottom = []
    vi_100 = []

    lst_top17 = []
    lst_bottom17 = []
    vi_10017 = []

    lst_top16 = []
    lst_bottom16 = []
    vi_10016 = []

    for p in range(len(vi_f)):
        vi = vi_f[p]
        lst = lst_f[p]
        n = math.floor(vi * 100)
        lst_list[n].append(lst)
        if n == 0:
            test = 1
        print_progress_bar(p, len(vi_f))
    print("")

    for p in range(len(vi_f17)):
        vi = vi_f17[p]
        lst = lst_f17[p]
        n = math.floor(vi * 100)
        lst_list17[n].append(lst)
        if n == 0:
            test = 1
        print_progress_bar(p, len(vi_f17))
    print("")

    for p in range(len(vi_f16)):
        vi = vi_f16[p]
        lst = lst_f16[p]
        n = math.floor(vi * 100)
        lst_list16[n].append(lst)
        if n == 0:
            test = 1
        print_progress_bar(p, len(vi_f16))
    print("")

    # 18===================================================
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
    # 17===============================================
    for l in range(100):
        if lst_list17[l] is None:
            print_progress_bar(l + 1, 100)
            continue
        lst_list17[l].sort()
        if len(lst_list17[l]) >= top_n_pix:
            lst_bottom17.append(get_average(lst_list17[l][0:top_n_pix]))
            lst_top17.append(get_average(lst_list17[l][(-1 * top_n_pix) :]))
            vi_10017.append(l / 100 + 0.005)
        print_progress_bar(l + 1, 100)
    print("")
    # 16==============================================
    for l in range(100):
        if lst_list16[l] is None:
            print_progress_bar(l + 1, 100)
            continue
        lst_list16[l].sort()
        if len(lst_list16[l]) >= top_n_pix:
            lst_bottom16.append(get_average(lst_list16[l][0:top_n_pix]))
            lst_top16.append(get_average(lst_list16[l][(-1 * top_n_pix) :]))
            vi_10016.append(l / 100 + 0.005)
        print_progress_bar(l + 1, 100)
    print("")

    print("plotting")
    vi_f100 = [v * 10000 for v in vi_f]
    plt.figure(figsize=(8, 8))
    plt.scatter(
        vi_f[0::step_draw],
        lst_f[0::step_draw],
        color="gray",
        marker=".",
        alpha=0.3,
        label="scatter 2018",
    )
    plt.scatter(
        vi_f17[0::step_draw],
        lst_f17[0::step_draw],
        color="gray",
        marker="x",
        alpha=0.3,
        label="scatter 2017",
    )
    plt.scatter(
        vi_f16[0::step_draw],
        lst_f16[0::step_draw],
        color="gray",
        marker="*",
        alpha=0.3,
        label="scatter 2016",
    )
    for l in range(len(vi_100)):
        plt.scatter(vi_100[l], lst_top[l], color="blue", marker="o")
        plt.scatter(vi_100[l], lst_bottom[l], color="red", marker="s")
    plt.scatter(vi_100[l], lst_top[l], color="blue", marker="o", label="dry edge 2018")
    plt.scatter(
        vi_100[l], lst_bottom[l], color="red", marker="s", label="wet edge 2018"
    )
    for l in range(len(vi_10017)):
        plt.scatter(vi_10017[l], lst_top17[l], color="#0033ff", marker="o")
        plt.scatter(vi_10017[l], lst_bottom17[l], color="#ff3300", marker="s")
    plt.scatter(
        vi_10017[l], lst_top17[l], color="#0033ff", marker="o", label="dry edge 2017"
    )
    plt.scatter(
        vi_10017[l], lst_bottom17[l], color="#ff3300", marker="s", label="wet edge 2017"
    )
    for l in range(len(vi_10016)):
        plt.scatter(vi_10016[l], lst_top16[l], color="#0066ff", marker="o")
        plt.scatter(vi_10016[l], lst_bottom16[l], color="#ff6600", marker="s")
    plt.scatter(
        vi_10016[l], lst_top16[l], color="#0066ff", marker="o", label="dry edge 2016"
    )
    plt.scatter(
        vi_10016[l], lst_bottom16[l], color="#ff6600", marker="s", label="wet edge 2016"
    )
    plt.xlabel("NDVI")
    plt.ylabel("LST")
    plt.legend(loc="best")
    plt.show()
    vi_fit_list = np.arange(0, 100)

    k_dry, b_dry = optimize.curve_fit(f_1, vi_100[-25:], lst_top[-25:])[0]
    k_wet, b_wet = optimize.curve_fit(f_1, vi_100, lst_bottom)[0]
    print(k_dry, b_dry, k_wet, b_wet)

    k_dry = -20.46316917036919
    b_dry = 327.2212982580681
    k_wet = -1.870  # -2.84819330323533
    b_wet = 299.809  # 304.97250059024964

    tsmin = k_wet * ndvi0 + b_wet
    tsmax = k_dry * ndvi0 + b_dry

    tvdi = (lst0 - tsmin) / (tsmax - tsmin)

    tvdi[tvdi >= 1] = 1
    tvdi[tvdi <= 0] = 0

    ndvi_mask = ndvi0
    ind_r_inv = np.where(ndvi0 > ndvi_thre)
    ndvi_mask[ind_r_inv] = 1
    ind_r_inv = np.where(ndvi0 <= ndvi_thre)
    ndvi_mask[ind_r_inv] = 0

    np.nan_to_num(tvdi)

    tvdi_masked = tvdi * ndvi_mask
    tvdi_masked = tvdi_masked.astype(np.float)

    idx = np.where(tvdi_masked > 0)
    ttarea = len(idx[0])

    for fl in range(10):
        f = fl / 10
        ff = (fl + 1) / 10
        idx = np.where(tvdi_masked < f)
        idx2 = np.where(tvdi_masked <= ff)
        subarea = len(idx2[0]) - len(idx[0])
        print("[", f, ":", ff, "]", subarea, subarea / ttarea)

    # save file
    print("calc pdi")
    outpath = outdir + "tvdi" + date_str + "-masked2.tif"
    out_ds = gdal.GetDriverByName("GTiff").Create(outpath, w, h, 1, gdal.GDT_Float32)
    out_ds.SetProjection(vi_proj)
    out_ds.SetGeoTransform(vi_geotrans)
    out_ds.GetRasterBand(1).WriteArray(tvdi_masked)
    out_ds.FlushCache()
    out_ds = None
    print("fin")
