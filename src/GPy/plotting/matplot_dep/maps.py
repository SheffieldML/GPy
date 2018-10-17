# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import numpy as np
try:
    from matplotlib import pyplot as pb
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    #from matplotlib import cm
    try:
        __IPYTHON__
        pb.ion()
    except NameError:
        pass
except:
    pass
import re

def plot(shape_records,facecolor='w',edgecolor='k',linewidths=.5, ax=None,xlims=None,ylims=None):
    """
    Plot the geometry of a shapefile

    :param shape_records: geometry and attributes list
    :type shape_records: ShapeRecord object (output of a shapeRecords() method)
    :param facecolor: color to be used to fill in polygons
    :param edgecolor: color to be used for lines
    :param ax: axes to plot on.
    :type ax: axes handle
    """
    #Axes handle
    if ax is None:
        fig     = pb.figure()
        ax      = fig.add_subplot(111)

    #Iterate over shape_records
    for srec in shape_records:
        points = np.vstack(srec.shape.points)
        sparts = srec.shape.parts
        par = list(sparts) + [points.shape[0]]

        polygs = []
        for pj in range(len(sparts)):
            polygs.append(Polygon(points[par[pj]:par[pj+1]]))
        ax.add_collection(PatchCollection(polygs,facecolor=facecolor,edgecolor=edgecolor, linewidths=linewidths))

    #Plot limits
    _box = np.vstack([srec.shape.bbox for srec in shape_records])
    minx,miny = np.min(_box[:,:2],0)
    maxx,maxy = np.max(_box[:,2:],0)

    if xlims is not None:
        minx,maxx = xlims
    if ylims is not None:
        miny,maxy = ylims
    ax.set_xlim(minx,maxx)
    ax.set_ylim(miny,maxy)


def string_match(sf,regex,field=2):
    """
    Return the geometry and attributes of a shapefile whose fields match a regular expression given

    :param sf: shapefile
    :type sf: shapefile object
    :regex: regular expression to match
    :type regex: string
    :field: field number to be matched with the regex
    :type field: integer
    """
    index = []
    shape_records = []
    for rec in enumerate(sf.shapeRecords()):
        m = re.search(regex,rec[1].record[field])
        if m is not None:
            index.append(rec[0])
            shape_records.append(rec[1])
    return index,shape_records

def bbox_match(sf,bbox,inside_only=True):
    """
    Return the geometry and attributes of a shapefile that lie within (or intersect) a bounding box

    :param sf: shapefile
    :type sf: shapefile object
    :param bbox: bounding box
    :type bbox: list of floats [x_min,y_min,x_max,y_max]
    :inside_only: True if the objects returned are those that lie within the bbox and False if the objects returned are any that intersect the bbox
    :type inside_only: Boolean
    """
    A,B,C,D = bbox
    index = []
    shape_records = []
    for rec in enumerate(sf.shapeRecords()):
        a,b,c,d = rec[1].shape.bbox
        if inside_only:
            if A <= a and B <= b and C >= c and D >= d:
                index.append(rec[0])
                shape_records.append(rec[1])
        else:
            cond1 = A <= a and B <= b and C >= a and D >= b
            cond2 = A <= c and B <= d and C >= c and D >= d
            cond3 = A <= a and D >= d and C >= a and B <= d
            cond4 = A <= c and D >= b and C >= c and B <= b
            cond5 = a <= C and b <= B and d >= D
            cond6 = c <= A and b <= B and d >= D
            cond7 = d <= B and a <= A and c >= C
            cond8 = b <= D and a <= A and c >= C
            if cond1 or cond2 or cond3 or cond4 or cond5 or cond6 or cond7 or cond8:
                index.append(rec[0])
                shape_records.append(rec[1])
    return index,shape_records


def plot_bbox(sf,bbox,inside_only=True):
    """
    Plot the geometry of a shapefile within a bbox

    :param sf: shapefile
    :type sf: shapefile object
    :param bbox: bounding box
    :type bbox: list of floats [x_min,y_min,x_max,y_max]
    :inside_only: True if the objects returned are those that lie within the bbox and False if the objects returned are any that intersect the bbox
    :type inside_only: Boolean
    """
    index,shape_records = bbox_match(sf,bbox,inside_only)
    A,B,C,D = bbox
    plot(shape_records,xlims=[bbox[0],bbox[2]],ylims=[bbox[1],bbox[3]])

def plot_string_match(sf,regex,field,**kwargs):
    """
    Plot the geometry of a shapefile whose fields match a regular expression given

    :param sf: shapefile
    :type sf: shapefile object
    :regex: regular expression to match
    :type regex: string
    :field: field number to be matched with the regex
    :type field: integer
    """
    index,shape_records = string_match(sf,regex,field)
    plot(shape_records,**kwargs)


def new_shape_string(sf,name,regex,field=2,type=None):
    import shapefile
    if type is None:
        type = shapefile.POINT
    newshp = shapefile.Writer(shapeType = sf.shapeType)
    newshp.autoBalance = 1

    index,shape_records = string_match(sf,regex,field)

    _fi = [sf.fields[j] for j in index]
    for f in _fi:
        newshp.field(name=f[0],fieldType=f[1],size=f[2],decimal=f[3])

    _shre = shape_records
    for sr in _shre:
        _points = []
        _parts = []
        for point in sr.shape.points:
            _points.append(point)
        _parts.append(_points)

        newshp.line(parts=_parts)
        newshp.records.append(sr.record)
        print(len(sr.record))

    newshp.save(name)
    print(index)

def apply_bbox(sf,ax):
    """
    Use bbox as xlim and ylim in ax
    """
    limits = sf.bbox
    xlim = limits[0],limits[2]
    ylim = limits[1],limits[3]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
