#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import os
import tempfile

# matplotlib/numpy compatibility
os.environ['HOME'] = tempfile.mkdtemp()
from matplotlib import use
use("Agg",warn=False)
from matplotlib import pyplot
import matplotlib.cm as cm

import ConfigParser
import argparse
import sys
import struct
import numpy
import math
import scipy.ndimage
import scipy.misc


# This array is convolved with the bead data in bfmask.bin to compute a sum in the range [0-100] of a 10x10 well area.
array_ten_by_ten =  numpy.ones((10,10))


def imresize(arr, size, interp='bilinear', mode=None):
    """Backported from scipy 0.11.0; not that complicated"""
    im = scipy.misc.toimage(arr, mode=mode)
    func = {'nearest':0,'bilinear':2,'bicubic':3,'cubic':3}
    size = size[1], size[0]
    imnew = im.resize(size, resample=func[interp])
    return scipy.misc.fromimage(imnew)


def calculate_scores(bfmask_data):
    """Compute an array whose values are the sum of a 10x10 well area around each well in bfmask_data.
    """
    return scipy.ndimage.correlate(bfmask_data, array_ten_by_ten, mode='reflect')


def reasonable_shrink(scores, bound=1000, threshold=2000):
    """Perform a smooth downsampling of the data if it is too large."""
    h, w = scores.shape
    largest = max((h, w))
    if largest > threshold:
        ratio = float(bound) / largest
        rh, rw = int(round(h * ratio)), int(round(w * ratio))
        return imresize(scores, (rh, rw), interp='bicubic', mode='F')
    else:
        return scores


def makeContourMap(arr, HEIGHT, WIDTH, outputId, maskId, plt_title, average, outputdir, barcodeId=-1, vmaxVal=100):
    scores = calculate_scores(arr)
    del arr
    scores = reasonable_shrink(scores)
    makeContourPlot(scores, average, HEIGHT, WIDTH, outputId, maskId, plt_title, outputdir, barcodeId, vmaxVal)
    makeRawDataPlot(scores, outputId, outputdir)
    makeFullBleed(scores, outputId, outputdir)


def getFormatForVal(value):
    if(numpy.isnan(value)): value = 0 #Value here doesn't really matter b/c printing nan will always result in -nan
    if(value==0): value=0.01
    precision = 2 - int(math.ceil(math.log10(value)))
    if(precision>4): precision=4;
    frmt = "%%0.%df" % precision
    return frmt


def getTicksForMaxVal (maxVal):
    #Note, function is only effective for maxVal range between 5 and 100
    ticksVal = [0,10,20,30,40,50,60,70,80,90,100]
    if(maxVal<=10): ticksVal = map(lambda x:x/10.0,ticksVal)
    elif(maxVal<=20): ticksVal = map(lambda x:x/5.0,ticksVal)
    elif(maxVal<=50): ticksVal = map(lambda x:x/2.0,ticksVal)
    return ticksVal


def autoGetVmaxFromAverage (average):
    vmaxVal = average*2
    return vmaxVal


def makeRawDataPlot(scores, outputId, outputdir):
    # Writes a png file containing only the data, i.e. no axes, labels, gridlines, ticks, etc.
    fig = pyplot.figure()
    ax1 = fig.add_subplot(111)
    ax1.xaxis.set_ticklabels([None])
    ax1.yaxis.set_ticklabels([None])
    ax1.xaxis.set_ticks([None])
    ax1.yaxis.set_ticks([None])
    ax1.imshow(scores, vmin=0, vmax=100, origin='lower', cmap=cm.jet)
    fig.savefig(outputdir+'/'+outputId+'_density_raw.png', dpi=20, transparent=True, bbox_inches='tight', pad_inches=0)
    print "Plot saved to %s" % outputId+'_density_raw.png'


def makeFullBleed(scores, outputId, outputdir):
    # Writes a png file containing only the data, i.e. no axes, labels, gridlines, ticks, etc.
    fig = pyplot.figure(figsize=(5,5))
    ax1 = fig.add_subplot(111)
    ax1.xaxis.set_ticklabels([None])
    ax1.yaxis.set_ticklabels([None])
    ax1.xaxis.set_ticks([None])
    ax1.yaxis.set_ticks([None])
    ax1.set_frame_on(False)
    ax1.imshow(scores,vmin=0, vmax=100, origin='lower', cmap=cm.jet)
    
    for size in (20, 70, 200, 1000):
        fig_path = os.path.join(outputdir, '%s_density_%d.png' % 
            (outputId, size))
        fig.savefig(fig_path, figsize=(5,5), dpi=size/3.875, transparent=True, 
            bbox_inches='tight', pad_inches=0)
        print("Full bleed plot saved to %s" % fig_path)


def makeContourPlot(scores, average, HEIGHT, WIDTH, outputId, maskId, plt_title, outputdir, barcodeId=-1, vmaxVal=100):
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(str(WIDTH) + ' wells')
    ax.set_ylabel(str(HEIGHT) + ' wells')
    if(barcodeId!=-1):
        if(barcodeId==0): maskId = "No Barcode Match,"
        else:             maskId = "Barcode Id %d," % barcodeId
    if plt_title != '':
        maskId = '%s\n%s' % (plt_title,maskId)
    ax.set_title('%s Loading Density (Avg ~ %0.f%%)' % (maskId, average))
    ax.autoscale_view()
    color_axis = ax.imshow(scores, vmin=0, vmax=vmaxVal, origin='lower', cmap=cm.jet)
    ticksVal = getTicksForMaxVal(vmaxVal)
    fig.colorbar(color_axis, format='%.0f %%', ticks=ticksVal)
    pngFn = outputdir+'/'+outputId+'_density_contour.png'
    fig.savefig(pngFn, bbox_inches='tight')
    print "Plot saved to", pngFn;


def extractMaskInfo(filename):
    """Read bfmask.bin and return an array of wells, value 1 for those with
    beads and 0 for empty wells.
    """
    with open(filename, 'rb') as f:
        h, w = struct.unpack_from('ii', f.read(8))
        wells_data = numpy.fromfile(f, dtype=numpy.int16, count=-1)
    ree = numpy.reshape(wells_data, (h, w))
    del wells_data
    # The second bit is a flag indicating loading
    ree &= 2
    # Shift to 1 for a loaded well and 0 for an empty well
    ree >>= 1
    return ree


def makeBarcodeArr(qBcId, row, col, bcIds, HEIGHT, WIDTH):
    # TODO: x/y are reversed from what is typical
    #print "makeBarcodeArr", qBcId, row, col, bcIds, HEIGHT, WIDTH
    counts = 0
    arr = numpy.zeros((HEIGHT,WIDTH))
    for i in range(len(row)):
        if qBcId == bcIds[i]:
            arr[row[i], col[i]] = 1
            counts += 1
    return arr, counts


def extractBarcodeMaskInfo(filePath):
    INPUTFILE=filePath
    print "Reading", INPUTFILE
    beadlist = numpy.loadtxt(INPUTFILE,dtype='int',comments='#')
    WIDTH=int(beadlist[0,0])
    HEIGHT=int(beadlist[0,1])
    bcbead_row = (beadlist[1:,0])  #really y, but is first column
    bcbead_col = (beadlist[1:,1])  #really x, but is the second column
    bcbead_bcIds = (beadlist[1:,2])
    
    unique_barcodeIds = dict.fromkeys(bcbead_bcIds).keys();
    print "Unique barcode ids found: ", unique_barcodeIds
    
    return unique_barcodeIds, HEIGHT, WIDTH, bcbead_row, bcbead_col, bcbead_bcIds


def genHeatmap(filePath, bfmaskstatspath, outputdir, plot_title):
    #
    #Called from TLScript.py
    #

    #provide loading density(average) and overwrite internal calculation
    statsparser = ConfigParser.RawConfigParser()
    statsparser.read(bfmaskstatspath)
    total_wells = float(statsparser.get('global', 'Total Wells'))
    excluded_wells = float(statsparser.get('global', 'Excluded Wells'))
    bead_wells = float(statsparser.get('global', 'Bead Wells'))
    average = 100*bead_wells/(total_wells-excluded_wells)

    print total_wells, excluded_wells, bead_wells, average

    beadarr = extractMaskInfo(filePath)
    HEIGHT, WIDTH = beadarr.shape
    # The fact that these are fixed is a clue that we might be able to remove them
    outputId = "Bead"
    maskId = ""
    makeContourMap(beadarr, HEIGHT, WIDTH, outputId, maskId, plot_title, average, outputdir)


#MaskBead.mask contains all well coordinations with Beads (Bead Wells = NUMBER), see bfmask.stats

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('maskfile', default='bfmask.bin', help='e.g. bfmask.bin')
    parser.add_argument('bfmask', default='bfmask.stats', help='e.g. bfmask.stats')
    parser.add_argument('plt_title', default='title', help='e.g. FOZ-223')
    args = parser.parse_args()

    genHeatmap(args.maskfile, args.bfmask, "./", args.plt_title)
