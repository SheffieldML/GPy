# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import matplotlib as mpl
import pylab as pb
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


coloursHex = {\
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

darkList = [coloursHex['darkBlue'],coloursHex['darkRed'],coloursHex['darkGreen'], coloursHex['darkOrange'], coloursHex['darkButter'], coloursHex['darkPurple'], coloursHex['darkChocolate'], coloursHex['Aluminium6']]
mediumList = [coloursHex['mediumBlue'], coloursHex['mediumRed'],coloursHex['mediumGreen'], coloursHex['mediumOrange'], coloursHex['mediumButter'], coloursHex['mediumPurple'], coloursHex['mediumChocolate'], coloursHex['Aluminium5']]
lightList = [coloursHex['lightBlue'], coloursHex['lightRed'],coloursHex['lightGreen'], coloursHex['lightOrange'], coloursHex['lightButter'], coloursHex['lightPurple'], coloursHex['lightChocolate'], coloursHex['Aluminium4']]

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
    while not darkList[0]==coloursHex['darkBlue']:
        darkList.append(darkList.pop(0))
    while not mediumList[0]==coloursHex['mediumBlue']:
        mediumList.append(mediumList.pop(0))
    while not lightList[0]==coloursHex['lightBlue']:
        lightList.append(lightList.pop(0))

def setLightFigures():
    mpl.rcParams['axes.edgecolor']=coloursHex['Aluminium6']
    mpl.rcParams['axes.facecolor']=coloursHex['Aluminium2']
    mpl.rcParams['axes.labelcolor']=coloursHex['Aluminium6']
    mpl.rcParams['figure.edgecolor']=coloursHex['Aluminium6']
    mpl.rcParams['figure.facecolor']=coloursHex['Aluminium2']
    mpl.rcParams['grid.color']=coloursHex['Aluminium6']
    mpl.rcParams['savefig.edgecolor']=coloursHex['Aluminium2']
    mpl.rcParams['savefig.facecolor']=coloursHex['Aluminium2']
    mpl.rcParams['text.color']=coloursHex['Aluminium6']
    mpl.rcParams['xtick.color']=coloursHex['Aluminium6']
    mpl.rcParams['ytick.color']=coloursHex['Aluminium6']

def setDarkFigures():
    mpl.rcParams['axes.edgecolor']=coloursHex['Aluminium2']
    mpl.rcParams['axes.facecolor']=coloursHex['Aluminium6']
    mpl.rcParams['axes.labelcolor']=coloursHex['Aluminium2']
    mpl.rcParams['figure.edgecolor']=coloursHex['Aluminium2']
    mpl.rcParams['figure.facecolor']=coloursHex['Aluminium6']
    mpl.rcParams['grid.color']=coloursHex['Aluminium2']
    mpl.rcParams['savefig.edgecolor']=coloursHex['Aluminium6']
    mpl.rcParams['savefig.facecolor']=coloursHex['Aluminium6']
    mpl.rcParams['text.color']=coloursHex['Aluminium2']
    mpl.rcParams['xtick.color']=coloursHex['Aluminium2']
    mpl.rcParams['ytick.color']=coloursHex['Aluminium2']

def hex2rgb(hexcolor):
    hexcolor = [hexcolor[1+2*i:1+2*(i+1)] for i in range(3)]
    r,g,b = [int(n,16) for n in hexcolor]
    return (r,g,b)

coloursRGB = dict([(k,hex2rgb(i)) for k,i in coloursHex.items()])

cdict_RB = {'red' :((0.,coloursRGB['mediumRed'][0]/256.,coloursRGB['mediumRed'][0]/256.),
                     (.5,coloursRGB['mediumPurple'][0]/256.,coloursRGB['mediumPurple'][0]/256.),
                     (1.,coloursRGB['mediumBlue'][0]/256.,coloursRGB['mediumBlue'][0]/256.)),
            'green':((0.,coloursRGB['mediumRed'][1]/256.,coloursRGB['mediumRed'][1]/256.),
                     (.5,coloursRGB['mediumPurple'][1]/256.,coloursRGB['mediumPurple'][1]/256.),
                     (1.,coloursRGB['mediumBlue'][1]/256.,coloursRGB['mediumBlue'][1]/256.)),
            'blue':((0.,coloursRGB['mediumRed'][2]/256.,coloursRGB['mediumRed'][2]/256.),
                      (.5,coloursRGB['mediumPurple'][2]/256.,coloursRGB['mediumPurple'][2]/256.),
                      (1.,coloursRGB['mediumBlue'][2]/256.,coloursRGB['mediumBlue'][2]/256.))}

cdict_BGR = {'red' :((0.,coloursRGB['mediumBlue'][0]/256.,coloursRGB['mediumBlue'][0]/256.),
                     (.5,coloursRGB['mediumGreen'][0]/256.,coloursRGB['mediumGreen'][0]/256.),
                     (1.,coloursRGB['mediumRed'][0]/256.,coloursRGB['mediumRed'][0]/256.)),
            'green':((0.,coloursRGB['mediumBlue'][1]/256.,coloursRGB['mediumBlue'][1]/256.),
                     (.5,coloursRGB['mediumGreen'][1]/256.,coloursRGB['mediumGreen'][1]/256.),
                     (1.,coloursRGB['mediumRed'][1]/256.,coloursRGB['mediumRed'][1]/256.)),
            'blue':((0.,coloursRGB['mediumBlue'][2]/256.,coloursRGB['mediumBlue'][2]/256.),
                      (.5,coloursRGB['mediumGreen'][2]/256.,coloursRGB['mediumGreen'][2]/256.),
                      (1.,coloursRGB['mediumRed'][2]/256.,coloursRGB['mediumRed'][2]/256.))}


cdict_Alu = {'red' :((0./5,coloursRGB['Aluminium1'][0]/256.,coloursRGB['Aluminium1'][0]/256.),
                     (1./5,coloursRGB['Aluminium2'][0]/256.,coloursRGB['Aluminium2'][0]/256.),
                     (2./5,coloursRGB['Aluminium3'][0]/256.,coloursRGB['Aluminium3'][0]/256.),
                     (3./5,coloursRGB['Aluminium4'][0]/256.,coloursRGB['Aluminium4'][0]/256.),
                     (4./5,coloursRGB['Aluminium5'][0]/256.,coloursRGB['Aluminium5'][0]/256.),
                     (5./5,coloursRGB['Aluminium6'][0]/256.,coloursRGB['Aluminium6'][0]/256.)),
           'green' :((0./5,coloursRGB['Aluminium1'][1]/256.,coloursRGB['Aluminium1'][1]/256.),
                     (1./5,coloursRGB['Aluminium2'][1]/256.,coloursRGB['Aluminium2'][1]/256.),
                     (2./5,coloursRGB['Aluminium3'][1]/256.,coloursRGB['Aluminium3'][1]/256.),
                     (3./5,coloursRGB['Aluminium4'][1]/256.,coloursRGB['Aluminium4'][1]/256.),
                     (4./5,coloursRGB['Aluminium5'][1]/256.,coloursRGB['Aluminium5'][1]/256.),
                     (5./5,coloursRGB['Aluminium6'][1]/256.,coloursRGB['Aluminium6'][1]/256.)),
            'blue' :((0./5,coloursRGB['Aluminium1'][2]/256.,coloursRGB['Aluminium1'][2]/256.),
                     (1./5,coloursRGB['Aluminium2'][2]/256.,coloursRGB['Aluminium2'][2]/256.),
                     (2./5,coloursRGB['Aluminium3'][2]/256.,coloursRGB['Aluminium3'][2]/256.),
                     (3./5,coloursRGB['Aluminium4'][2]/256.,coloursRGB['Aluminium4'][2]/256.),
                     (4./5,coloursRGB['Aluminium5'][2]/256.,coloursRGB['Aluminium5'][2]/256.),
                     (5./5,coloursRGB['Aluminium6'][2]/256.,coloursRGB['Aluminium6'][2]/256.))}
# cmap_Alu = mpl.colors.LinearSegmentedColormap('TangoAluminium',cdict_Alu,256)
# cmap_BGR = mpl.colors.LinearSegmentedColormap('TangoRedBlue',cdict_BGR,256)
# cmap_RB = mpl.colors.LinearSegmentedColormap('TangoRedBlue',cdict_RB,256)
if __name__=='__main__':
    import pylab as pb
    pb.figure()
    pb.pcolor(pb.rand(10,10),cmap=cmap_RB)
    pb.colorbar()
    pb.show()
