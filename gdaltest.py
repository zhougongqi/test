import glob
import math
import logging
from osgeo import gdal
from skimage import filters
import numpy as np
import cv2
import matplotlib.pyplot as pyplot


class cwater:
    logger = logging.getLogger(__name__)

    def __init__(self, file_path):
        self.file_path = file_path
        gdal.AllRegister()

    def testwarp(self) -> bool:
        print("processing %s", self.file_path)
        tifpath = self.file_path
        Raster = gdal.Open(tifpath)
        Projection = Raster.GetProjectionRef()
        dem = gdal.Open(self.tar_file_path)
        Projectiondem = dem.GetProjectionRef()
        reproj = gdal.AutoCreateWarpedVRT()
        a = 1

    def run(self, zscale: int = 20, waterlevel: int = 67) -> bool:
        """
        zscale: scale coef when zooming is needed
        """
        print("running")
        try:
            dem_file_path = glob.glob(self.file_path + "/elevation.img")[0]
        except Exception as e:
            self.logger.debug("File not found! %s", e)
        Cwater_tif = dem_file_path.replace(".img", "_Cwater.tif")
        try:
            dem = gdal.Open(dem_file_path)
            dem_geo_trans = dem.GetGeoTransform()
            cols = dem.RasterXSize
            rows = dem.RasterYSize
            band = dem.GetRasterBand(1)
            mx_dem = band.ReadAsArray(0, 0, cols, rows)
        except Exception as e:
            self.logger.debug("Open file failed! %s", e)
            return False

        # pyplot.figure(num="fld", figsize=(8, 8))  # image show part
        # pyplot.imshow(mx_dem)
        # pyplot.show()  # image show part end

        seedlist = []
        seedlist.append([3187, 7408])
        seedlist.append([4184, 8865])
        seedlist.append([3711, 7071])
        seedlist.append([4020, 8448])
        seedlist.append([4208, 6402])
        seedlist.append([4282, 7746])
        seedlist.append([1788, 7683])
        seedlist.append([1547, 7827])
        seedlist.append([1724, 7752])
        seedlist.append([3550, 7828])
        # seedlist.append([3452, 4198])
        # seedlist.append([4095, 4609])
        seedlist = self.reverse_xy_in_list(seedlist)
        seedlistz = seedlist.copy()

        # downscaling img, z for zoomed
        demshape = list(mx_dem.shape)
        zshape = [int(i / zscale) for i in demshape]
        zshape = [zshape[1], zshape[0]]
        mxz_dem = cv2.resize(mx_dem, tuple(zshape))
        # print(mxz_dem.shape)

        seedlistz = [
            [int(seedlistz[i][j] / zscale) for j in range(len(seedlistz[i]))]
            for i in range(0, len(seedlistz))
        ]
        # mxzz_dem = cv2.resize(mxz_dem, tuple(demshape))
        maskz = self.gen_buff_resistance(mxz_dem, 500, seedlistz, 800.0, 1.3)

        mxz_dem = mxz_dem + maskz * 540
        fldz = self.region_grow(mxz_dem, waterlevel, seedlistz)
        # print(fldz.shape)
        demshape = [demshape[1], demshape[0]]

        fld = cv2.resize(fldz, tuple(demshape))
        fld = fld.astype(np.int8)
        # print(fld.shape)

        pyplot.figure(num="fld")  # image show part
        pyplot.subplot(1, 2, 1)
        pyplot.imshow(fldz)
        # pyplot.show()  # image show part end
        pyplot.subplot(1, 2, 2)
        pyplot.imshow(mx_dem)
        pyplot.show()  # image show part end

        try:
            outds = gdal.GetDriverByName("GTiff").Create(Cwater_tif, cols, rows, 1)
            outds.SetProjection(dem.GetProjection())
            outds.SetGeoTransform(dem_geo_trans)
            outds.GetRasterBand(1).WriteArray(fld, 0, 0)
            self.logger.info("Writing TIFF Done! %s", self.file_path)
            return True
        except Exception as e:
            self.logger.debug("Writing file failed! %s %s", self.file_path, e)
            return False

        return True

    def gen_buffer(self, img: np.ndarray, radius: int, slist: list) -> bytearray:
        """
        generate buffer from seed points in $slist with $radius
        will be replaced by gen_buffer_resistance
        """
        shape = img.shape
        out = np.zeros(shape, dtype=np.int8)
        slist0 = slist.copy()
        seed = slist0[0]
        r = int(radius * 1.414 / 2)

        # fill a circle around each seed with $radius
        while len(slist0) > 0:
            co = slist0.pop()
            x0 = co[0]
            y0 = co[1]
            # fill a max inner square
            out[x0 - r - 1 : x0 + r + 1, y0 - r - 1 : y0 + r + 1] = 1
            # loop 1/8 arc and fill*8
            for x in range(x0 - r, x0):
                for y in range(y0 - int(radius), y0 - r):
                    d = (x - x0 * 1.0) * (x - x0 * 1.0) + (y - y0 * 1.0) * (
                        y - y0 * 1.0
                    )
                    if d <= (radius * radius):
                        out[x, y] = 1
                        out[2 * x0 - x, y] = 1
                        out[2 * x0 - x, 2 * y0 - y] = 1
                        out[x, 2 * y0 - y] = 1
                        out[x0 - y0 + y, y0 - x0 + x] = 1
                        out[x0 - y0 + y, y0 + x0 - x] = 1
                        out[x0 + y0 - y, y0 + x0 - x] = 1
                        out[x0 + y0 - y, y0 - x0 + x] = 1

        print("is generating buffer")
        # pyplot.figure(num="test", figsize=(8, 8))  #image show part
        # pyplot.imshow(out)
        # pyplot.show()                              #image show part end
        return out

    def gen_buff_resistance(
        self, img: np.ndarray, radius: int, slist: list, scale0: float, scale1: float
    ) -> bytearray:
        """
        Function:
            generate a resistance buffer, which describes the resistance of water flowing to next pixel.
            the resistance at point p will increase when the distance increases between p and nearest seed points.
            the returned resistance buffer will be added to dem to simply rise the altitude as the resistance.
            then region-grow algorithm will be restrained by a higher altitude when it's far away from seed points.
        :param
            self: 
            img: input dem array
            radius: buffer radius, useless in this func
            slist: a list contains several seeds' coordinate
            scale0: scale factor controls the resistance radius, 100 for recommand default value
            scale1: scale factor controls the resistance increasing rate
        :return:
            out: an nd-array of the resistance buffer img, usually between 0 and 1
        """
        self.logger.info("Making resist-buffer...")  # print("making resist-buffer")
        shape = img.shape
        xmax = shape[0]
        ymax = shape[1]
        out = np.zeros(shape, dtype=float)
        dmax0 = (xmax * 1.0) * (xmax * 1.0) + (ymax * 1.0) * (ymax * 1.0)
        dmax0 = dmax0 / scale0
        slist0 = slist.copy()
        disarr = [0] * len(slist0)
        dislen = len(disarr)

        for i in range(0, xmax - 1):
            for j in range(0, ymax - 1):
                # if (i == 4603) and (j == 2505):
                #     a = 0
                for s in range(0, len(disarr)):
                    cor = slist0[s]
                    x = cor[0] * 1.0
                    y = cor[1] * 1.0
                    disarr[s] = (i - x) * (i - x) + (j - y) * (j - y)
                dmax = min(disarr)
                dd = dmax / dmax0
                # use logistic func to normalize the distance weight
                dd1 = 1 / (1 + math.exp(dd * (-1)))
                dd1 = dd1 * 2 - scale1
                if dd1 < 0:
                    dd1 = 0
                out[i, j] = dd1
            # print("%d" % i)
        return out

    def testfunc(self) -> int:
        return 1

    def region_grow(self, img: np.ndarray, thre: float, slist: list) -> bytearray:
        """
        Function:
            fill a region by growing from some seed points in slist,
            grow direction is 4-connection by default 
        :param
            self: 
            img: input image array
            thre: threshold when growing
            slist: a list contains several seeds' coordinate
        :return:
            out: an nd-array of the processed image
        """
        self.logger.info("Growing...")  # print("growing...")
        slist0 = slist.copy()
        shape = img.shape
        xmax = shape[0]
        ymax = shape[1]
        out = np.zeros(shape, dtype=np.float)
        dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        ndir = 4
        num = 0
        while len(slist0) > 0:
            co = slist0.pop()
            x0 = co[0]
            y0 = co[1]
            # out[x0 - 1 : x0 + 1, y0 - 1 : y0 + 1] = 3  # zzz
            out[x0, y0] = 1
            for i in range(0, ndir):
                ii = dirs[i]
                cor = [co[i] + ii[i] for i in range(len(ii))]  # list add each other
                # is out of edge
                if (
                    (cor[0] >= xmax)
                    or (cor[0] < 0)
                    or (cor[1] >= ymax)
                    or (cor[1] < 0)
                    or (img[cor[0], cor[1]] < -1000)
                ):
                    continue

                # spread to neighbor pixels
                if img[cor[0], cor[1]] <= thre:
                    if out[cor[0], cor[1]] == 0:
                        out[cor[0], cor[1]] = 1
                        num = num + 1
                        slist0.append(cor)
        return out

    def reverse_xy_in_list(self, ilist: list) -> list:
        # l = ilist.copy()
        l = [(ilist[i][1], ilist[i][0]) for i in range(0, len(ilist))]
        return l


if __name__ == "__main__":
    filepath = "/home/tq/zgq/zdata/flood/malaysia2018/s1/S1A_IW_GRDH_1SDV_20180105T220552_20180105T220619_020025_0221DE_0CC6_NR_Cal_Deb_ML_Spk_SRGR_TC.data"
    cw = cwater(filepath)
    cw.run()
