import os
import sys
import math
import glob
import pprint
import numpy as np
from osgeo import gdal, osr, gdalnumeric
import subprocess

home_dir = os.path.expanduser("~")


def get_landsat_by_pathrow(year: int, path: int, row: int, sensor: str):
    #

    search_str_base = "/home/tq/tq-data0ZZZ/landsat_sr/SSS/01/PPP/RRR/"

    pathstr = str(path).zfill(3)
    rowstr = str(row).zfill(3)
    yearstr = str(year)
    search_str_base = search_str_base.replace("PPP", pathstr)
    search_str_base = search_str_base.replace("RRR", rowstr)
    search_str_base = search_str_base.replace("SSS", sensor)
    llist = []

    for tqn in range(5):
        tqnstr = str(tqn + 1)

        search_str = search_str_base.replace("ZZZ", tqnstr)
        print("searching " + search_str)

        # begin search dir
        if os.path.exists(search_str):
            dir_list = os.listdir(search_str)
            for cur_file in dir_list:
                # get fullpath
                namepart = cur_file.split("_")
                datestr = namepart[3]
                yearstr_file = datestr[0:4]
                if yearstr_file != yearstr:
                    continue
                l8path = os.path.join(search_str, cur_file)
                if not l8path.endswith("/"):
                    l8path = l8path + "/"
                llist.append(l8path)
            # llist.extend(dir_list)
        else:
            continue

    return llist


def stack_landsat(l8list: str, out_path: str, sensor: str):
    """
    stack landsat bands to a multi-band file
    2, reform the path make them looks alright
    3, get bands from each l-8 paths, and stack them
    """
    work_path = out_path

    # 1 read list
    filelist = l8list
    print(filelist)

    # 2
    tmlist = []
    for f in filelist:
        if f.find(sensor) != -1:
            tmp = f.replace('"', "")
            tmp = tmp.replace(",", "")
            tmp = tmp.replace(" ", "")
            if tmp.endswith("/"):
                pass
            else:
                tmp = tmp + "/"
            tmlist.append(os.path.join(home_dir, tmp))
    print(tmlist)
    print("begin to stack")

    # 3
    for tm in tmlist:
        tm_shortname = tm.split("/")[-2]
        print(tm)
        band_list = get_bands_into_a_list(tm, "*sr_band*.tif")
        nbands = len(band_list)
        bn = 0

        # read the first one to get some paras
        ds = gdal.Open(band_list[0])
        geo_trans = ds.GetGeoTransform()
        w = ds.RasterXSize
        h = ds.RasterYSize
        img_shape = [h, w]
        proj = ds.GetProjection()
        data = ds.ReadAsArray()
        ds = None
        if "int8" in data.dtype.name:
            datatype = gdal.GDT_Byte
        elif "int16" in data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # build output path
        outpath = work_path + tm_shortname + "_stacked.tif"
        # write output into tiff file
        out_ds = gdal.GetDriverByName("GTiff").Create(outpath, w, h, nbands, datatype)

        for band in band_list:
            bn += 1
            print("  {}/{} bands".format(bn, nbands))
            # read files and stack them
            ds = gdal.Open(band)
            geo_trans = ds.GetGeoTransform()
            w = ds.RasterXSize
            h = ds.RasterYSize
            img_shape = [h, w]
            data = ds.ReadAsArray()
            # write
            out_ds.GetRasterBand(bn).WriteArray(data)
            ds = None
        out_ds.SetProjection(proj)
        out_ds.SetGeoTransform(geo_trans)
        out_ds.FlushCache()
        out_ds = None


def stack_landsat_decloud(l8list: str, out_path: str, sensor: str):
    """
    stack landsat bands to a multi-band file
    2, reform the path make them looks alright
    3, get bands from each l-8 paths, and stack them

    4, remove cloud pixels and replace them with fillvalue (0)
    """
    work_path = out_path
    if sensor == "LC08":
        clear_qa = [322]
    else:
        clear_qa = [66, 68, 130, 132]

    # 1 read list
    filelist = l8list
    print(filelist)

    # 2
    tmlist = []
    for f in filelist:
        if f.find(sensor) != -1:
            tmp = f.replace('"', "")
            tmp = tmp.replace(",", "")
            tmp = tmp.replace(" ", "")
            if tmp.endswith("/"):
                pass
            else:
                tmp = tmp + "/"
            tmlist.append(os.path.join(home_dir, tmp))
    print(tmlist)
    print("begin to stack")

    # 3
    for tm in tmlist:
        tm_shortname = tm.split("/")[-2]
        print(tm)
        band_list = get_bands_into_a_list(tm, "*sr_band*.tif")
        band_qa = get_bands_into_a_list(tm, "*pixel_qa.tif")[0]
        nbands = len(band_list)
        bn = 0

        # read the first one to get some paras
        ds = gdal.Open(band_list[0])
        geo_trans = ds.GetGeoTransform()
        w = ds.RasterXSize
        h = ds.RasterYSize
        img_shape = [h, w]
        proj = ds.GetProjection()
        data = ds.ReadAsArray()
        ds = None
        if "int8" in data.dtype.name:
            datatype = gdal.GDT_Byte
        elif "int16" in data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # read qa and get good quality pixels
        ds = gdal.Open(band_qa)
        qa_data = ds.ReadAsArray()
        if sensor == "LC08":
            qa_data[qa_data > 326] = 999
            qa_data[qa_data == 1] = 999
            qa_data[qa_data < 326] = 1  # cloud is 0
            qa_data[qa_data == 999] = 0
        else:  # landsat 7
            qa_data[qa_data == 1] = 200
            qa_data[qa_data == 66] = 1
            qa_data[qa_data == 68] = 1
            qa_data[qa_data == 130] = 1
            qa_data[qa_data == 132] = 1
            qa_data[qa_data >= 2] = 0  # cloud is 0

        # build output path
        outpath = work_path + tm_shortname + "_stacked_decloud.tif"
        # write output into tiff file
        out_ds = gdal.GetDriverByName("GTiff").Create(outpath, w, h, nbands, datatype)

        for band in band_list:
            bn += 1
            print("  {}/{} bands".format(bn, nbands))
            # read files and stack them
            ds = gdal.Open(band)
            geo_trans = ds.GetGeoTransform()
            w = ds.RasterXSize
            h = ds.RasterYSize
            img_shape = [h, w]
            data = ds.ReadAsArray()

            # delete cloud pixels
            # data1 = data * qa_data
            # data = data1.astype(np.byte)
            idx = np.where(qa_data == 0)
            data[idx] = 0
            # write
            out_ds.GetRasterBand(bn).WriteArray(data)
            ds = None
        out_ds.SetProjection(proj)
        out_ds.SetGeoTransform(geo_trans)
        out_ds.FlushCache()
        out_ds = None


def get_bands_into_a_list(img_path: str, expression: str):
    filelist = glob.glob(img_path + expression)
    filelist.sort()
    return filelist


if __name__ == "__main__":
    # winter wheat test in NCP hengshui around

    # get landsat 8 folder list by year, path and row
    # LE07 LC08
    l8list = get_landsat_by_pathrow(2018, 124, 35, "LC08")
    pprint.pprint(l8list)

    # get images stacked
    out_path = home_dir + "/data_pool/china_crop/NCP_winter_wheat_test/L8-124-35/"
    stack_landsat_decloud(l8list, out_path, "LC08")

    print("fin")
