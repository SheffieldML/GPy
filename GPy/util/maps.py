import numpy as np
import pylab as pb
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
#from matplotlib import cm
import shapefile
import re


pb.ion()


def plot(sf,subset=None,facecolor='w',edgecolor='k',linewidths=.5, ax=None,xlims=None,ylims=None):
    records = [sf.records()[sj] for sj in subset]
    shapes = [sf.shapes()[sj] for sj in subset]
    N = len(shapes)

    if ax is None:
        fig     = pb.figure()
        ax      = fig.add_subplot(111)

    if subset is None:
        shape_records = sf.shapeRecords()
        #index = range(len(shape_records))
    else:
        index,shape_records = subset

    for srec in shape_records:
        #for srec in [sf.shapeRecords()[sj] for sj in subset]:
        points = np.vstack(srec.shape.points)
        sparts = srec.shape.parts
        par = list(sparts) + [points.shape[0]]

        polygs = []
        for pj in xrange(len(sparts)):
            polygs.append(Polygon(points[par[pj]:par[pj+1]]))
        ax.add_collection(PatchCollection(polygs,facecolor=facecolor,edgecolor=edgecolor, linewidths=linewidths))

    minx,miny,maxx,maxy = sf.bbox
    if xlims is not None:
        minx,maxx = xlims
    if ylims is not None:
        miny,maxy = ylims
    ax.set_xlim(minx,maxx)
    ax.set_ylim(miny,maxy)


def string_match(sf,regex,field=2):
    index = []
    shape_records = []
    for rec in enumerate(sf.records()):
        m = re.search(regex,rec[1][field])
        if m is not None:
            index.append(rec[0])
            shape_records.append(rec[1])
    return index,shape_records

def bbox_match(sf,bbox,exact=True):
    A,B,C,D = bbox
    index = []
    shape_records = []
    for rec in enumerate(sf.shapeRecords()):
        a,b,c,d = rec[1].shape.bbox
        if exact:
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


def plot_bbox(sf,bbox,exact=True):
    index,shape_records = bbox_match(sf,bbox,exact)
    A,B,C,D = bbox
    plot(sf,subset,xlims=[bbox[0],bbox[2]],ylims=[bbox[1],bbox[3]])

def plot_subset(sf,regex,field):
    index,shape_records = string_match(sf,regex,field)
    plot(sf,subset)


def new_shape_string(sf,regex,field=2,type=shapefile.POINT):
    new_shape = shapefile.Writer(shapeType=type)

    index,shape_records = string_match(sf,regex,field)
    #keep = []
    #for srec in [sf.shapeRecords()[sj] for sj in subset]:
    #    keep.append(srec)

    for sj,sr in zip(index,shape_records):
        new_shape._shapes.append(sf.shape(sj))
        new_shape.records.append(sr.record)
    stop
    return new_shape



def plot_shape(sf,facecolor='w',edgecolor='k',linewidths=.5, ax=None,bbox=True):
    #if isinstance(file,str):
    #    sf = shapefile.Reader(file)
    #else:
    #    sf = file
    records = sf.records()
    shapes = sf.shapes()
    N = len(shapes)

    if ax is None:
        fig     = pb.figure()
        ax      = fig.add_subplot(111)

    for srec in sf.shapeRecords():
        points = np.vstack(srec.shape.points)
        sparts = srec.shape.parts
        par = list(sparts) + [points.shape[0]]

        polygs = []
        for pj in xrange(len(sparts)):
            polygs.append(Polygon(points[par[pj]:par[pj+1]]))
        ax.add_collection(PatchCollection(polygs,facecolor=facecolor,edgecolor=edgecolor, linewidths=linewidths))

    if bbox:
        minx,miny,maxx,maxy = sf.bbox
        ax.set_xlim(minx,maxx)
        ax.set_ylim(miny,maxy)
