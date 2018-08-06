import time
import logging
from osgeo import gdal
import gdaltest as gl
import cv2


def run(para1: str, para2: str) -> (int):
    """
    test run func
    """
    print(para1 + para2)


if __name__ == "__main__":
    # print("helloworld!")
    flag = run("start", "-testing")
    g = gl.gdaltest(
        "/home/tq/zgq/zdata/flood/malaysia2018/miri_20171212/Gamma0_VH.img",
        "/home/tq/zgq/zdata/dem/miri.tif",
    )
    g.run(20)
