import os
import math
import sys
import logging
import subprocess
import numpy as np
from PIL import Image, ImageDraw
from osgeo import gdal, ogr
from osgeo import gdalnumeric


my_logger = logging.getLogger(__name__)
str_arrs = ["~", "/", "|", "\\"]


def image_to_array(i: Image) -> np.ndarray:
    """
    Converts a Python Imaging Library array to a
    gdalnumeric image.
    """
    a = gdalnumeric.fromstring(i.tobytes(), dtype=np.int32)
    a.shape = i.im.size[1], i.im.size[0]
    return a


def image_to_array_byte(i: Image) -> np.ndarray:
    """
    Converts a Python Imaging Library array to a
    gdalnumeric image.
    """
    a = gdalnumeric.fromstring(i.tobytes(), dtype=np.int8)
    a.shape = i.im.size[1], i.im.size[0]
    return a


def world_to_pixel(geo_matrix: tuple, x: int, y: int) -> tuple:
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    up_left_x = geo_matrix[0]
    up_left_y = geo_matrix[3]
    x_dist = geo_matrix[1]
    pixel = int((x - up_left_x) / x_dist)
    line = int((up_left_y - y) / x_dist)
    return pixel, line


def pixels_to_world(geo_matrix: tuple, pixel: int, line: int) -> tuple:
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the geo-location of a img pixel coordinate
    """
    up_left_x = geo_matrix[0]
    up_left_y = geo_matrix[3]
    x_dist = geo_matrix[1]
    x = pixel * x_dist + up_left_x
    y = up_left_y - line * x_dist
    return x, y


class Vividict(dict):
    def __missing__(self, key):
        # make multi-key assignment possible
        value = self[key] = type(self)()
        return value

    def walk(self):
        # flattened dict output
        for key, value in self.items():
            if isinstance(value, Vividict):
                for tup in value.walk():
                    yield (key,) + tup
            else:
                yield key, value


def calc_mask_by_shape(
    shape_path: str,
    geo_trans: list,
    img_shape: list,
    *,
    specified_field: str = "id",
    condition: list = None,
    mask_value: int = 1,
    flag_dlist: bool = False,
    field_strict: bool = False,
    pixel_size: int = -1,
    use_env: bool = False,
) -> (np.ndarray, int, list):
    """
    Function:
        this is a local test for blocking clip, by zgq
    Input:
        shape_path:  a string contains shapefile's path,
        geo_trans:   geo tansformation list, for calculating vector coordinates in
                    in raster file
        img_shape:   a 2-element list, for define the output tiff mask's shape
                    [x_size, y_size]

        *: optional parameters:
        specified_field:    specified field name that provides the value to be filled
                     into the mask.
        condition:  a list of condition for filtering data values in
                    $sp_field, None for default: means no condition;
                    all value in list will be processed to mask-making.
        mask_value:  default mask value for mask-making, 1 for default.
                    -1 for use value in $specified_field;
                    -2 for unique count value.
        flag_dlist: boolean, True for output a list contains all values in
                    $sp_field' False for nothing output
        field_strict: if $sp_field not found and $field_strict is true, a
                    exception will be thrown
        pixel_size:  int value of output raster mask pixel size,
                    -1 for same as geo_trans.
        use_env:    True: use envelope of selected polygon,
                    False: use envolope of img_shape.
    Output:
        mask: an int32 ndarray, 0 for background value,
                non-zero values for mask value, mask value can be any integer.
        npoly: num of polygons painted on the mask
        a list: polygon id list by order if $flag_dlist is true
                boundary cornor points [ulx, uly, lrx, lry] if $use_envi is true
                else None.
    """
    if condition is None:
        con = [-1]
    else:
        con = condition
    list_id = []
    sp_field = specified_field

    # create Image for painting
    try:
        if mask_value == -2:
            raster_poly = Image.new("I", (img_shape[0], img_shape[1]), 0)
        else:
            raster_poly = Image.new("L", (img_shape[0], img_shape[1]), 0)
        rasterize = ImageDraw.Draw(raster_poly)
    except Exception as e:
        my_logger.error("create image fail")
        return None, None, None
    shp_base_name = os.path.basename(shape_path)
    shp_name, extension = os.path.splitext(shp_base_name)

    # open vector shape_path
    try:
        shapef = ogr.Open(shape_path)
        lyr = shapef.GetLayer(0)
        poly = lyr.GetNextFeature()
        lyrdn = lyr.GetLayerDefn()
        maxpoly = lyr.GetFeatureCount()
    except Exception as e:
        my_logger.error("open shape file error: %s", shape_path)
        return None, None, None

    # check field existance
    flag = check_field_existance(shape_path, sp_field)
    if flag is False:
        my_logger.info("specified field -[" + sp_field + "]- not found in table!")
        if field_strict is True:
            my_logger.error(
                "specified field -[" + sp_field + "]- not found in table! skip!"
            )
            return None, None, None
        my_logger.info("using the first field instead, conditions are invalid now")
        sp_field = lyrdn.GetFieldDefn(0).GetName()  # set sp_field to first field
        con = [-1]

    npoly = 0
    while poly:
        # loop poly in lyr, draw ROIs in Image

        lb, numlb = get_field_value(poly, mask_value, sp_field, npoly, field_strict)

        if con[0] == -1:  # no condition
            pass
        else:
            if lb in con:  # proceed to next step
                pass
            else:
                poly = lyr.GetNextFeature()  # move on to next feature
                npoly = npoly + 1
                continue

        # read geometry
        list_id.append(numlb)
        geon = poly.GetGeometryRef()
        geon_type = geon.GetGeometryType()
        if geon.GetGeometryName() not in ["POLYGON", "MULTIPOLYGON"]:
            my_logger.error(
                "get_mask_by_shape(): This module can only load polygon/multipolygons"
            )
            return None, None, None
        # get envilope
        minX, maxX, minY, maxY = geon.GetEnvelope()
        ulX, ulY = world_to_pixel(geo_trans, minX, maxY)
        lrX, lrY = world_to_pixel(geo_trans, maxX, minY)
        bound = [ulX, ulY, lrX, lrY]

        # Create a new geomatrix for the image
        new_geo_trans = list(geo_trans)
        new_geo_trans[0] = minX
        new_geo_trans[3] = maxY

        if geon_type == 6 or geon.GetGeometryName() == "MULTIPOLYGON":
            # multipolygon
            for geon_part in geon:
                # loop each part polygon
                pixels = get_poly_pixels(geon_part, geo_trans, 0)
                rasterize.polygon(pixels, numlb)

        # for polygons
        if (
            geon.GetGeometryName() == "POLYGON" or geon_type == 3
        ):  # geon.GetGeometryType() == 3:
            pixels = get_poly_pixels(geon, geo_trans, 0)
            rasterize.polygon(pixels, numlb)

        # print progress bar
        npoly = npoly + 1
        print_progress_bar(npoly, maxpoly)
        poly = lyr.GetNextFeature()  # move on to next feature
        # end while

    lyr.ResetReading()
    if mask_value == -2:
        mask = image_to_array(raster_poly)
    else:
        mask = image_to_array_byte(raster_poly)

    n_ROI = len(mask[mask == 1])
    print(" ")
    print(n_ROI)
    gdal.ErrorReset()
    if flag_dlist:
        return mask, npoly, list_id
    else:
        return mask, None, None
    if use_env:
        return mask, None, bound


def check_field_existance(shp_path: str, specified_field: str):
    # check specified field existance
    shapef = ogr.Open(shp_path)
    lyr = shapef.GetLayer(0)
    lyrdn = lyr.GetLayerDefn()

    flag = False
    for i in range(lyrdn.GetFieldCount()):
        field_name = lyrdn.GetFieldDefn(i).GetName()
        if field_name == specified_field:
            flag = True
    return flag


def get_field_value(
    poly, mask_value: int, sp_field: str, npoly: int, field_strict: bool
):
    if mask_value == -1:
        # use field value(int) as mask value
        # if not int, use npoly + 1000 instead
        lb = poly.GetField(sp_field)
        if lb is None or type(lb) is str:
            lb = npoly + 1000
        if field_strict is True:
            assert lb < 1000, "value too large for this process"
        numlb = int(lb)
    elif mask_value == -2:
        # use unique value
        lb = npoly + 10000
        numlb = int(lb)
    else:
        # use specified value as mask value(common binary mask)
        numlb = mask_value
        lb = poly.GetField(sp_field)
        # lb = None
    return lb, numlb


def get_poly_pixels(geom, geo_trans: list, offset: int) -> list:
    geom0 = geom
    geom_type = geom.GetGeometryName()
    if geom_type == "LINEARRING":
        geom_t = ogr.Geometry(ogr.wkbPolygon)
        geom_t.AddGeometry(geom)
        geom0 = geom_t
    pts = geom0.GetGeometryRef(0)
    # may cause ERROR 6: Incompatible geometry for operation
    if pts is None:
        return None
    npp = pts.GetPointCount()
    points = []
    pixels = []
    if npp == 0:
        print("error in get_poly_pixels")
        return None
    for p in range(pts.GetPointCount()):
        points.append((pts.GetX(p), pts.GetY(p)))
    for p in points:
        x, y = world_to_pixel(geo_trans, p[0], p[1])
        pixels.append((x, y - offset))
        if x < 0 or y - offset < 0:
            # print("error x,y range", x, y - offset)
            pass
    return pixels


def print_progress_bar(now_pos: int, total_pos: int):
    n_sharp = math.floor(50 * now_pos / total_pos)
    n_space = 50 - n_sharp
    sys.stdout.write(
        str_arrs[int(now_pos) % 4]
        + "["
        + "#" * n_sharp
        + " " * n_space
        + "]"
        + "{:.2%}\r".format(now_pos / total_pos)
    )


def get_poly_dissolve(shp_path: str, specified_field: str, condition: list) -> str:
    """
    Function:
        extract conditioned polygons from shapefile and save them as a new shapefile,
        then use ogr2ogr dissolve all the polygons in new shapefile into one polygon,
        and save the result in a file ended with "_dissolve_all.shp"
    input :
        shp_path: original shape path , contains many polys
        specified_field: specified field to select certain polys
        con:      condition name list.
    output:
        out_shp_path: new shape,contains unioned one poly.

    example:
        field = "NAME_1"
        con = ["Aceh","Riau"]
        out_file = get_poly_dissolve(shp_file, field, con)
    """
    if condition is None:
        con = [-1]
    else:
        con = condition

    sp_field = specified_field
    out_shp_path = shp_path.replace(".shp", "_dissolved.shp")
    # out_shp_path = out_shp_path.replace("-", "_")
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.access(out_shp_path, os.F_OK):
        driver.DeleteDataSource(out_shp_path)

    # open shp file
    try:
        shapef = ogr.Open(shp_path)
        lyr = shapef.GetLayer(0)
        lyrdn = lyr.GetLayerDefn()
        spatial_ref = lyr.GetSpatialRef()
        maxpoly = lyr.GetFeatureCount()

        newds = driver.CreateDataSource(out_shp_path)
        layernew = newds.CreateLayer("copy", spatial_ref, ogr.wkbPolygon)
        # Add input Layer Fields to the output Layer if it is the one we want
        for i in range(0, lyrdn.GetFieldCount()):
            fieldDefn = lyrdn.GetFieldDefn(i)
            layernew.CreateField(fieldDefn)
        # Get the output Layer's Feature Definition
        layernew_dn = layernew.GetLayerDefn()
    except Exception as e:
        my_logger.error("open shape file error: {}".format(shp_path))
        return None

    # skip this func if there is only 1 poly in shp file
    if maxpoly == 1:
        return shp_path

    flag = False
    if specified_field is None:
        sp_field = lyrdn.GetFieldDefn(0).GetName()  # set sp_field to first field
        con = [-1]
    # check specified field existance
    for i in range(lyrdn.GetFieldCount()):
        field_name = lyrdn.GetFieldDefn(i).GetName()
        if field_name == sp_field:
            flag = True
            break
    if flag is False:
        my_logger.info("using the first field instead, conditions are invalid now")
        sp_field = lyrdn.GetFieldDefn(0).GetName()  # set sp_field to first field
        con = [-1]

    # loop each polygon in shp file, put selected polygons into one list
    npoly = 0
    lyr.ResetReading()
    for p0 in lyr:
        lb = p0.GetField(sp_field)
        print(npoly, ":", sp_field, lb)
        if con[0] == -1:  # no condition
            p_out = ogr.Feature(layernew_dn)
            npoly = npoly + 1
            for i in range(0, layernew_dn.GetFieldCount()):
                fieldDefn = layernew_dn.GetFieldDefn(i)
                p_out.SetField(layernew_dn.GetFieldDefn(i).GetNameRef(), p0.GetField(i))

            # Set geometry as centroid
            geom = p0.GetGeometryRef()
            p_out.SetGeometry(geom.Clone())
            # Add new feature to output Layer
            layernew.CreateFeature(p_out)
        else:
            if lb in con:  # add to newlayer
                p_out = ogr.Feature(layernew_dn)
                npoly = npoly + 1
                for i in range(0, layernew_dn.GetFieldCount()):
                    fieldDefn = layernew_dn.GetFieldDefn(i)
                    p_out.SetField(
                        layernew_dn.GetFieldDefn(i).GetNameRef(), p0.GetField(i)
                    )
                # Set geometry as centroid
                geom = p0.GetGeometryRef()
                p_out.SetGeometry(geom.Clone())
                # Add new feature to output Layer
                layernew.CreateFeature(p_out)
                continue
            else:
                npoly = npoly + 1
                continue
        p_out = None

    # Save and close DataSources
    shapef = None
    newds = None
    # destroy and flush

    # run ogr2ogr get features dissolved
    print(out_shp_path)
    outout_file = out_shp_path.replace(".shp", "_all.shp")
    base_name = os.path.basename(out_shp_path)
    short_name, ext = os.path.splitext(base_name)
    cmd_str = (
        "ogr2ogr "
        + outout_file
        + " "
        + out_shp_path
        + " -dialect sqlite -sql 'SELECT ST_Union(geometry) AS geometry FROM "
        + short_name
        + "'"
    )
    my_logger.info("cmd string is :" + cmd_str)
    process_status = subprocess.run(cmd_str, shell=True)
    if process_status.returncode != 0:
        my_logger.error("dissolve failed!")
        return None
    my_logger.info("finish poly dissolve")
    return outout_file


def calc_mask_by_shape_block(
    shape_path: str,
    geo_trans: list,
    img_shape: list,
    *,
    specified_field: str = "id",
    condition: list = None,
    mask_value: int = 1,
    flag_dlist: bool = False,
    field_strict: bool = False,
    pixel_size: int = -1,
    use_env: bool = False,
    n_block: int = 1,
):
    """
    Function:
        this is a local test for blocking clip, by zgq
    Input:
        shape_path:  a string contains shapefile's path,
        geo_trans:   geo tansformation list, for calculating vector coordinates in
                    in raster file
        img_shape:   a 2-element list, for define the output tiff mask's shape
                    [x_size, y_size]

        *: optional parameters:
        specified_field: specified field name that provides the value to be filled into
                    the mask.
        condition:  a list of condition for filtering data values in
                    $sp_field, None for default: means no condition;
                    all value in list will be processed to mask-making.
        mask_value:  default mask value for mask-making, 1 for default.
                    -1 for use value in $specified_field;
                    -2 for unique count value.
        flag_dlist: boolean, True for output a list contains all values in
                    $sp_field' False for nothing output
        field_strict: if $sp_field not found and $field_strict is true, a
                        exception will be thrown
        pixel_size:  int value of output raster mask pixel size,
                    -1 for same as geo_trans.
        use_env:    True: use envelope of selected polygon,
                    False: use envolope of img_shape.
        n_block:    num of blocks, 1 for default.
    Output:
        mask: an int32 ndarray, 0 for background value,
                non-zero values for mask value, mask value can be any integer.
        npoly: num of polygons painted on the mask
        a list: polygon id list by order if $flag_dlist is true
                boundary cornor points [ulx, uly, lrx, lry] if $use_envi is true
                else None.
    """
    if condition is None:
        con = [-1]
    else:
        con = condition
    list_id = []
    bound = None

    if n_block <= 0:
        my_logger.error("n_block must >= 1")
        return None, None, None
    if n_block == 1:
        my_logger.info("use no blocking method")
        # zzz
        return None, None, None

    # open vector shape_path
    try:
        shapef = ogr.Open(shape_path)
        lyr = shapef.GetLayer(0)
        poly = lyr.GetNextFeature()
        lyrdn = lyr.GetLayerDefn()
        maxpoly = lyr.GetFeatureCount()
    except Exception as e:
        my_logger.error("open shape file error: %s", shape_path)
        return None, None, None

    shp_base_name = os.path.basename(shape_path)
    shp_name, extension = os.path.splitext(shp_base_name)
    print(img_shape[1], img_shape[0])
    if mask_value == -2:
        mask = np.zeros((img_shape[1], img_shape[0]), dtype=np.int16)
    else:
        mask = np.zeros((img_shape[1], img_shape[0]), dtype=np.int8)
    sp_field = specified_field

    # check field existance
    flag = check_field_existance(shape_path, sp_field)
    if flag is False:
        my_logger.info("specified field -[" + sp_field + "]- not found in table!")
        if field_strict is True:
            my_logger.error(
                "specified field -[" + sp_field + "]- not found in table! skip!"
            )
            return None, None, None
        my_logger.info("using the first field instead, conditions are invalid now")
        sp_field = lyrdn.GetFieldDefn(0).GetName()  # set sp_field to first field
        con = [-1]

    # calc block parameters
    # get envilope
    img_ulX = 0
    img_ulY = 0
    img_lrX = img_shape[0]
    img_lrY = img_shape[1]
    img_minX, img_maxY = pixels_to_world(geo_trans, img_ulX, img_ulY)
    img_maxX, img_minY = pixels_to_world(geo_trans, img_lrX, img_lrY)

    # calc break points
    delta_Y = int(math.floor(img_lrY / n_block))
    perc = list(range(0, img_lrY + delta_Y, delta_Y))
    perc.pop()
    perc[-1] = img_lrY
    percXY_dic = Vividict()
    for i in range(n_block):
        # calculate cornor points
        percXY_dic[i]["ul"] = [img_ulX, perc[i]]  # upper left
        percXY_dic[i]["ur"] = [img_lrX, perc[i]]
        percXY_dic[i]["ll"] = [img_ulX, perc[i + 1]]
        percXY_dic[i]["lr"] = [img_lrX, perc[i + 1]]  # lower right
        # print(percXY_dic[i])
        # calculate geo coordinates
        geo_ulX, geo_ulY = pixels_to_world(
            geo_trans, percXY_dic[i]["ul"][0], percXY_dic[i]["ul"][1]
        )
        geo_urX, geo_urY = pixels_to_world(
            geo_trans, percXY_dic[i]["ur"][0], percXY_dic[i]["ur"][1]
        )
        geo_llX, geo_llY = pixels_to_world(
            geo_trans, percXY_dic[i]["ll"][0], percXY_dic[i]["ll"][1]
        )
        geo_lrX, geo_lrY = pixels_to_world(
            geo_trans, percXY_dic[i]["lr"][0], percXY_dic[i]["lr"][1]
        )
        # print("-" * 80)
        percXY_dic[i]["geo-ul"] = [geo_ulX, geo_ulY]  # upper left geo
        percXY_dic[i]["geo-ur"] = [geo_urX, geo_urY]
        percXY_dic[i]["geo-ll"] = [geo_llX, geo_llY]
        percXY_dic[i]["geo-lr"] = [geo_lrX, geo_lrY]  # lower right geo

    # get block start:
    for i in range(n_block):
        sub_x = percXY_dic[i]["ur"][0] - percXY_dic[i]["ul"][0]
        sub_y = percXY_dic[i]["ll"][1] - percXY_dic[i]["ul"][1]
        sub_shape = [sub_x, sub_y]
        # print(sub_shape)
        y_offset = perc[i]
        # make sub-square shapes
        geon_sq = create_poly_by_points(percXY_dic[i])

        # create Image-sub for painting
        try:
            if mask_value == -2:
                raster_poly = Image.new("I", (sub_shape[0], sub_shape[1]), 0)
            else:
                raster_poly = Image.new("L", (sub_shape[0], sub_shape[1]), 0)
            rasterize = ImageDraw.Draw(raster_poly)
        except Exception as e:
            my_logger.error("create image fail")
            return None, None, None

        lyr.ResetReading()
        # poly = lyr.GetNextFeature()
        npoly = 0
        npoly_con = 0
        for poly in lyr:
            # loop poly in lyr, draw ROIs in Image
            lb, numlb = get_field_value(poly, mask_value, sp_field, npoly, field_strict)
            npoly = npoly + 1
            if con[0] == -1:  # no condition
                pass
            else:
                if lb in con:  # proceed to next step
                    npoly_con += 1
                    pass
                else:
                    # poly = lyr.GetNextFeature()  # move on to next feature
                    continue
            # read geometry
            # print_progress_bar(maxpoly * (i) + npoly, maxpoly * n_block)
            # print(npoly, ":", maxpoly)
            list_id.append(numlb)
            geon = poly.GetGeometryRef()
            geon_type_name = geon.GetGeometryName()
            if geon.GetGeometryName() not in ["POLYGON", "MULTIPOLYGON"]:
                my_logger.error(
                    "get_mask_by_shape(): This module only load polygon/multipolygons"
                )
                return None, None, None
            # calculate if intersects
            if geon.Intersects(geon_sq):
                pass
            else:
                continue
            # get envilope
            minX, maxX, minY, maxY = geon.GetEnvelope()
            ulX, ulY = world_to_pixel(geo_trans, minX, maxY)
            lrX, lrY = world_to_pixel(geo_trans, maxX, minY)
            bound = [ulX, ulY, lrX, lrY, minX, maxY]

            # geon_sq = poly_sq.GetGeometryRef()
            geon_inter = get_intersection(geon, geon_sq, geon_type_name)
            if geon_inter is None:
                continue

            # draw on Image
            if geon_type_name == "MULTIPOLYGON":
                # multipolygon
                for geon_part in geon_inter:
                    # loop each part polygon
                    pixels = get_poly_pixels(geon_part, geo_trans, y_offset)
                    pixels = remove_dupli_pixels(pixels)
                    if pixels is not None:
                        if len(pixels) >= 3:
                            rasterize.polygon(pixels, numlb)
            if geon_type_name == "POLYGON":
                pixels = get_poly_pixels(geon_inter, geo_trans, y_offset)
                pixels = remove_dupli_pixels(pixels)
                if pixels is not None:
                    if len(pixels) >= 3:
                        rasterize.polygon(pixels, numlb)

            # rasterize.polygon(pixels, numlb)
            # print progress bar

            print_progress_bar(maxpoly * (i) + npoly, maxpoly * n_block)
            # poly = lyr.GetNextFeature()  # move on to next feature
            # geon_sq = None
            geon_inter = None
            # end while

        print_progress_bar(i + 1, n_block)
        # add rasterize to mask
        if mask_value == -2:
            mask_sub = image_to_array(raster_poly)
        else:
            mask_sub = image_to_array_byte(raster_poly)
        tmp_start = perc[i]
        tmp_end = perc[i + 1]
        mask[tmp_start:tmp_end, :] = mask_sub  # .swapaxes(0, 1)

        # pyplot.figure(num="test", figsize=(8, 8))  # image show part
        # p1 = pyplot.subplot(121)
        # p2 = pyplot.subplot(122)
        # p1.imshow(mask)
        # p2.imshow(mask_sub)
        # pyplot.show()  # image show part end
    print(" ")
    if flag_dlist:
        return mask, npoly, list_id
    elif use_env:
        if bound is None:
            my_logger.error(
                "get_mask_by_shape(): no valid state-id is found, check again"
            )
            return mask, None, None
        return mask, None, tuple(bound)
    else:
        return mask, None, None


def get_intersection(geom1, geom2, geom_type_name):
    """
    geom1: ori shape
    geom2: square shape
    geom_type_name: name of type of geom1
    """
    #
    if geom1.Intersects(geom2):
        geom_inter = geom1.Intersection(geom2)
        return geom_inter
    else:
        return None


def create_poly_by_points(point_dic: dict):
    # Create ring
    ring = ogr.Geometry(ogr.wkbLinearRing)
    x0 = point_dic["geo-ul"][0]
    y0 = point_dic["geo-ul"][1]
    for p in ["geo-ul", "geo-ur", "geo-lr", "geo-ll"]:
        x = point_dic[p][0]
        y = point_dic[p][1]
        ring.AddPoint_2D(x, y)
    ring.AddPoint_2D(x0, y0)  # close the ring

    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly


def get_pixels_from_geom(geon_type_name, geon, geo_trans, y_offset):
    if geon_type_name == "MULTIPOLYGON":
        # multipolygon
        for geon_part in geon:
            # loop each part polygon
            pixels = get_poly_pixels(geon_part, geo_trans, y_offset)
    if geon_type_name == "POLYGON":
        pixels = get_poly_pixels(geon, geo_trans, y_offset)
    return pixels


def remove_dupli_pixels(pixels: list):
    if pixels is None:
        return None
    pixels2 = []
    for item in pixels:
        if not item in pixels2:
            pixels2.append(item)
    return pixels2


if __name__ == "__main__":
    shp_file = "/home/tq/data_pool/zgq/vector/indonesia-main.shp"
    # con = [
    #     "Kalimantan Timur",
    #     "Kalimantan Utara",
    #     "Kalimantan Tengah",
    #     "Kalimantan Selatan",
    #     "Kalimantan Barat",
    # ]
    # field = "NAME_1"

    # out_file = get_poly_dissolve(shp_file, field, con)
    print("fin")
