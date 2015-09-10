# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import matplotlib as mpl
from matplotlib import pyplot as pb
import sys
#sys.path.append('/home/james/mlprojects/sitran_cluster/')
#from switch_pylab_backend import *


#this stuff isn;t really Tango related: maybe it could be moved out? TODO
def removeRightTicks(ax=None):
    ax = ax or pb.gca()
    for i, line in enumerate(ax.get_yticklines()):
        if i%2 == 1:   # odd indices
            line.set_visible(False)
def removeUpperTicks(ax=None):
    ax = ax or pb.gca()
    for i, line in enumerate(ax.get_xticklines()):
        if i%2 == 1:   # odd indices
            line.set_visible(False)
def fewerXticks(ax=None,divideby=2):
    ax = ax or pb.gca()
    ax.set_xticks(ax.get_xticks()[::divideby])


colorsHex = {\
"Aluminium6":"#2e3436",\
"Aluminium5":"#555753",\
"Aluminium4":"#888a85",\
"Aluminium3":"#babdb6",\
"Aluminium2":"#d3d7cf",\
"Aluminium1":"#eeeeec",\
"lightPurple":"#ad7fa8",\
"mediumPurple":"#75507b",\
"darkPurple":"#5c3566",\
"lightBlue":"#729fcf",\
"mediumBlue":"#3465a4",\
"darkBlue": "#204a87",\
"lightGreen":"#8ae234",\
"mediumGreen":"#73d216",\
"darkGreen":"#4e9a06",\
"lightChocolate":"#e9b96e",\
"mediumChocolate":"#c17d11",\
"darkChocolate":"#8f5902",\
"lightRed":"#ef2929",\
"mediumRed":"#cc0000",\
"darkRed":"#a40000",\
"lightOrange":"#fcaf3e",\
"mediumOrange":"#f57900",\
"darkOrange":"#ce5c00",\
"lightButter":"#fce94f",\
"mediumButter":"#edd400",\
"darkButter":"#c4a000"}

darkList = [colorsHex['darkBlue'],colorsHex['darkRed'],colorsHex['darkGreen'], colorsHex['darkOrange'], colorsHex['darkButter'], colorsHex['darkPurple'], colorsHex['darkChocolate'], colorsHex['Aluminium6']]
mediumList = [colorsHex['mediumBlue'], colorsHex['mediumRed'],colorsHex['mediumGreen'], colorsHex['mediumOrange'], colorsHex['mediumButter'], colorsHex['mediumPurple'], colorsHex['mediumChocolate'], colorsHex['Aluminium5']]
lightList = [colorsHex['lightBlue'], colorsHex['lightRed'],colorsHex['lightGreen'], colorsHex['lightOrange'], colorsHex['lightButter'], colorsHex['lightPurple'], colorsHex['lightChocolate'], colorsHex['Aluminium4']]

def currentDark():
    return darkList[-1]
def currentMedium():
    return mediumList[-1]
def currentLight():
    return lightList[-1]

def nextDark():
    darkList.append(darkList.pop(0))
    return darkList[-1]
def nextMedium():
    mediumList.append(mediumList.pop(0))
    return mediumList[-1]
def nextLight():
    lightList.append(lightList.pop(0))
    return lightList[-1]

def reset():
    while not darkList[0]==colorsHex['darkBlue']:
        darkList.append(darkList.pop(0))
    while not mediumList[0]==colorsHex['mediumBlue']:
        mediumList.append(mediumList.pop(0))
    while not lightList[0]==colorsHex['lightBlue']:
        lightList.append(lightList.pop(0))

def setLightFigures():
    mpl.rcParams['axes.edgecolor']=colorsHex['Aluminium6']
    mpl.rcParams['axes.facecolor']=colorsHex['Aluminium2']
    mpl.rcParams['axes.labelcolor']=colorsHex['Aluminium6']
    mpl.rcParams['figure.edgecolor']=colorsHex['Aluminium6']
    mpl.rcParams['figure.facecolor']=colorsHex['Aluminium2']
    mpl.rcParams['grid.color']=colorsHex['Aluminium6']
    mpl.rcParams['savefig.edgecolor']=colorsHex['Aluminium2']
    mpl.rcParams['savefig.facecolor']=colorsHex['Aluminium2']
    mpl.rcParams['text.color']=colorsHex['Aluminium6']
    mpl.rcParams['xtick.color']=colorsHex['Aluminium6']
    mpl.rcParams['ytick.color']=colorsHex['Aluminium6']

def setDarkFigures():
    mpl.rcParams['axes.edgecolor']=colorsHex['Aluminium2']
    mpl.rcParams['axes.facecolor']=colorsHex['Aluminium6']
    mpl.rcParams['axes.labelcolor']=colorsHex['Aluminium2']
    mpl.rcParams['figure.edgecolor']=colorsHex['Aluminium2']
    mpl.rcParams['figure.facecolor']=colorsHex['Aluminium6']
    mpl.rcParams['grid.color']=colorsHex['Aluminium2']
    mpl.rcParams['savefig.edgecolor']=colorsHex['Aluminium6']
    mpl.rcParams['savefig.facecolor']=colorsHex['Aluminium6']
    mpl.rcParams['text.color']=colorsHex['Aluminium2']
    mpl.rcParams['xtick.color']=colorsHex['Aluminium2']
    mpl.rcParams['ytick.color']=colorsHex['Aluminium2']

def hex2rgb(hexcolor):
    hexcolor = [hexcolor[1+2*i:1+2*(i+1)] for i in range(3)]
    r,g,b = [int(n,16) for n in hexcolor]
    return (r,g,b)

colorsRGB = dict([(k,hex2rgb(i)) for k,i in colorsHex.items()])

cdict_RB = {'red' :((0.,colorsRGB['mediumRed'][0]/256.,colorsRGB['mediumRed'][0]/256.),
                     (.5,colorsRGB['mediumPurple'][0]/256.,colorsRGB['mediumPurple'][0]/256.),
                     (1.,colorsRGB['mediumBlue'][0]/256.,colorsRGB['mediumBlue'][0]/256.)),
            'green':((0.,colorsRGB['mediumRed'][1]/256.,colorsRGB['mediumRed'][1]/256.),
                     (.5,colorsRGB['mediumPurple'][1]/256.,colorsRGB['mediumPurple'][1]/256.),
                     (1.,colorsRGB['mediumBlue'][1]/256.,colorsRGB['mediumBlue'][1]/256.)),
            'blue':((0.,colorsRGB['mediumRed'][2]/256.,colorsRGB['mediumRed'][2]/256.),
                      (.5,colorsRGB['mediumPurple'][2]/256.,colorsRGB['mediumPurple'][2]/256.),
                      (1.,colorsRGB['mediumBlue'][2]/256.,colorsRGB['mediumBlue'][2]/256.))}

cdict_BGR = {'red' :((0.,colorsRGB['mediumBlue'][0]/256.,colorsRGB['mediumBlue'][0]/256.),
                     (.5,colorsRGB['mediumGreen'][0]/256.,colorsRGB['mediumGreen'][0]/256.),
                     (1.,colorsRGB['mediumRed'][0]/256.,colorsRGB['mediumRed'][0]/256.)),
            'green':((0.,colorsRGB['mediumBlue'][1]/256.,colorsRGB['mediumBlue'][1]/256.),
                     (.5,colorsRGB['mediumGreen'][1]/256.,colorsRGB['mediumGreen'][1]/256.),
                     (1.,colorsRGB['mediumRed'][1]/256.,colorsRGB['mediumRed'][1]/256.)),
            'blue':((0.,colorsRGB['mediumBlue'][2]/256.,colorsRGB['mediumBlue'][2]/256.),
                      (.5,colorsRGB['mediumGreen'][2]/256.,colorsRGB['mediumGreen'][2]/256.),
                      (1.,colorsRGB['mediumRed'][2]/256.,colorsRGB['mediumRed'][2]/256.))}


cdict_Alu = {'red' :((0./5,colorsRGB['Aluminium1'][0]/256.,colorsRGB['Aluminium1'][0]/256.),
                     (1./5,colorsRGB['Aluminium2'][0]/256.,colorsRGB['Aluminium2'][0]/256.),
                     (2./5,colorsRGB['Aluminium3'][0]/256.,colorsRGB['Aluminium3'][0]/256.),
                     (3./5,colorsRGB['Aluminium4'][0]/256.,colorsRGB['Aluminium4'][0]/256.),
                     (4./5,colorsRGB['Aluminium5'][0]/256.,colorsRGB['Aluminium5'][0]/256.),
                     (5./5,colorsRGB['Aluminium6'][0]/256.,colorsRGB['Aluminium6'][0]/256.)),
           'green' :((0./5,colorsRGB['Aluminium1'][1]/256.,colorsRGB['Aluminium1'][1]/256.),
                     (1./5,colorsRGB['Aluminium2'][1]/256.,colorsRGB['Aluminium2'][1]/256.),
                     (2./5,colorsRGB['Aluminium3'][1]/256.,colorsRGB['Aluminium3'][1]/256.),
                     (3./5,colorsRGB['Aluminium4'][1]/256.,colorsRGB['Aluminium4'][1]/256.),
                     (4./5,colorsRGB['Aluminium5'][1]/256.,colorsRGB['Aluminium5'][1]/256.),
                     (5./5,colorsRGB['Aluminium6'][1]/256.,colorsRGB['Aluminium6'][1]/256.)),
            'blue' :((0./5,colorsRGB['Aluminium1'][2]/256.,colorsRGB['Aluminium1'][2]/256.),
                     (1./5,colorsRGB['Aluminium2'][2]/256.,colorsRGB['Aluminium2'][2]/256.),
                     (2./5,colorsRGB['Aluminium3'][2]/256.,colorsRGB['Aluminium3'][2]/256.),
                     (3./5,colorsRGB['Aluminium4'][2]/256.,colorsRGB['Aluminium4'][2]/256.),
                     (4./5,colorsRGB['Aluminium5'][2]/256.,colorsRGB['Aluminium5'][2]/256.),
                     (5./5,colorsRGB['Aluminium6'][2]/256.,colorsRGB['Aluminium6'][2]/256.))}
# cmap_Alu = mpl.colors.LinearSegmentedColormap('TangoAluminium',cdict_Alu,256)
# cmap_BGR = mpl.colors.LinearSegmentedColormap('TangoRedBlue',cdict_BGR,256)
# cmap_RB = mpl.colors.LinearSegmentedColormap('TangoRedBlue',cdict_RB,256)
if __name__=='__main__':
    from matplotlib import pyplot as pb
    pb.figure()
    pb.pcolor(pb.rand(10,10),cmap=cmap_RB)
    pb.colorbar()
    pb.show()
