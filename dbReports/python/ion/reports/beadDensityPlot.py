# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import argparse
import sys
import pylab
import numpy
import math
from matplotlib import *
from matplotlib.ticker import *

INCREMENT = 10
SIZE = 20
rowSize=0
colSize=0

def getAreaScore(row,column,arr, HEIGHT, WIDTH):
    tempscore = 0
    if (row * INCREMENT + SIZE/2) < HEIGHT:
        rowSize = range((row * INCREMENT + (SIZE/2)))
        rowend = (row * INCREMENT + (SIZE/2))
    elif (row * INCREMENT + SIZE/2) >= HEIGHT:
        rowSize = range(HEIGHT)
        rowend = HEIGHT
    if (column * INCREMENT + SIZE/2) < WIDTH:
        colSize = range((column * INCREMENT + (SIZE/2)))
        colend = (column * INCREMENT + (SIZE/2))
    elif (column * INCREMENT + SIZE/2) >= WIDTH:
        colSize = range(WIDTH)
        colend = WIDTH
    if (row * INCREMENT - (SIZE/2)) < 0:
        rowStart = 0
    else:
        rowStart = (row * INCREMENT - (SIZE/2))
    if (column * INCREMENT - (SIZE/2)) < 0:
        colStart = 0
    else:
        colStart = (column * INCREMENT - (SIZE/2))
    for i in rowSize[rowStart:]:
        for j in colSize[colStart:]:
            if arr[i,j] == 1:
                tempscore += 1
    sizecol = colend - colStart
    sizerow = rowend - rowStart
    size = sizecol * sizerow
    return tempscore,size

def makeContourMap(arr, HEIGHT, WIDTH, outputId, maskId, barcodeId=-1, vmaxVal=100):    
    score,scores,average = calculateScores(arr, HEIGHT, WIDTH)
    makeContourPlot(score, scores, average, HEIGHT, WIDTH, outputId, maskId, barcodeId, vmaxVal)
    makeRawDataPlot(scores, outputId)

def calculateScores(arr, HEIGHT, WIDTH):
    rowlen,collen = arr.shape
    scores = pylab.zeros(((rowlen/INCREMENT),(collen/INCREMENT)))
    score = []
    for row in range(rowlen/INCREMENT):
        for column in range(collen/INCREMENT):            
            keypassed,size = getAreaScore(row,column,arr, HEIGHT, WIDTH)
            scores[row,column] = round(float(keypassed)/float(size)*100,2)
            if keypassed > 2:
                score.append(round(float(keypassed)/float(size)*100,2))         
            
            #scores[0,0] = 0
            #scores[HEIGHT/INCREMENT -1,WIDTH/INCREMENT -1] = 100
    print scores
    
    flattened = []
    for i in score:
        flattened.append(i)
    flattened = filter(lambda x: x != 0.0, flattened)    
    average=pylab.average(flattened)
    
    return score, scores, average

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

def makeRawDataPlot(scores, outputId):
    # Writes a png file containing only the data, i.e. no axes, labels, gridlines, ticks, etc.
    pylab.jet()
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)
    ax1.xaxis.set_ticklabels([None])
    ax1.yaxis.set_ticklabels([None])
    ax1.xaxis.set_ticks([None])
    ax1.yaxis.set_ticks([None])
    ax1.imshow(scores,vmin=0, vmax=100, origin='lower')
    pylab.savefig(outputId+'_density_raw.png', dpi=20, transparent=True, bbox_inches='tight', pad_inches=0)
    print "Plot saved to %s" % outputId+'_density_raw.png'
    
def makeContourPlot(score, scores, average, HEIGHT, WIDTH, outputId, maskId, barcodeId=-1, vmaxVal=100):
    pylab.bone()
    #majorFormatter = FormatStrFormatter('%.f %%')
    #ax = pylab.gca()
    #ax.xaxis.set_major_formatter(majorFormatter)
    
    pylab.figure()
    ax = pylab.gca()
    ax.set_xlabel('<--- Width = '+str(WIDTH)+' wells --->')
    ax.set_ylabel('<--- Height = '+str(HEIGHT)+' wells --->')
    ax.set_yticks([0,HEIGHT/INCREMENT])
    ax.set_xticks([0,WIDTH/INCREMENT])
    ax.autoscale_view()
    pylab.jet()
    #pylab.contourf(scores, 40,origin="lower")
    
    if vmaxVal=='auto':
        vmaxVal = autoGetVmaxFromAverage(average)
    
    pylab.imshow(scores,vmin=0, vmax=vmaxVal, origin='lower')
    pylab.vmin = 0.0
    pylab.vmax = 100.0
    ticksVal = getTicksForMaxVal(vmaxVal)
    pylab.colorbar(format='%.0f %%',ticks=ticksVal)
    
    string_value1 = getFormatForVal(average) % average
    if(barcodeId!=-1):
        if(barcodeId==0): maskId = "No Barcode Match,"
        else:             maskId = "Barcode Id %d," % barcodeId
    pylab.title(maskId+' Loading Density (Avg ~ '+string_value1+'%)')
    pylab.axis('scaled')
    pylab.axis([0,WIDTH/INCREMENT-1,0,HEIGHT/INCREMENT-1])
    pngFn = outputId+'_density_contour.png'
    pylab.savefig(pngFn)
    print "Plot saved to", pngFn;
    #pylab.show()  #Do we want this within a pipeline

def makeKeypassArr(row,col, HEIGHT, WIDTH):
    arr = numpy.zeros((HEIGHT,WIDTH))
    temp = 0
    for i in range(len(row)):
        arr[row[i], col[i]] = 1  # TODO: x/y are reversed from what is typical
        temp +=1
    return arr,temp

def extractMaskInfo(filePath):
    INPUTFILE=filePath
    print "Reading", INPUTFILE
    #Current mlab doc indicates that numpy.loadtxt should be used instead of mlab.load
    beadlist = numpy.loadtxt(INPUTFILE,dtype='int',comments='#')
    WIDTH=int(beadlist[0,0])
    HEIGHT=int(beadlist[0,1])
    beadx = (beadlist[1:,0]) #TODO: x/y are reveresed from what is typical
    beady = (beadlist[1:,1])
    beadarr = makeKeypassArr(beadx,beady, HEIGHT, WIDTH)[0]
    
    # Assume the filename is the same as output by the BeadmaskParse tool
    # "MaskBead.mask" - strip off the first 4 characters, then strip off the last 5 characters
    INPUTFILE = INPUTFILE.strip().split('/')[-1]
    maskId=INPUTFILE[4:-5]
    outputId=maskId
    if maskId == "Bead":
        maskId = ""
    return beadarr, HEIGHT, WIDTH, outputId, maskId

def makeBarcodeArr(qBcId, row, col, bcIds, HEIGHT, WIDTH):
    # TODO: x/y are reversed from what is typical
    #print "makeBarcodeArr", qBcId, row, col, bcIds, HEIGHT, WIDTH
    counts = 0;
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

def genHeatmapBarcode(barcodeMaskFilePath,HEIGHT=False, WIDTH=False, regMaskFilename=''):
    unique_barcodeIds, HEIGHT2, WIDTH2, bcbead_row, bcbead_col,bcbead_bcIds = \
        extractBarcodeMaskInfo(barcodeMaskFilePath)
    
    #If present checks consistency in the dimensions between the masks
    if(HEIGHT and WIDTH):
        if (HEIGHT!=HEIGHT2) or (WIDTH!=WIDTH2):
            print "ERROR, '%s' does not have the same dimensions as '%s" % (regMaskFilename, barcodeMaskFilePath)
            sys.exit(1)
    
    #Calculate barcode specific density arrays
    scoreArr = []
    barcodeCountArr = []
    maxAverage = 0.01;
    for barcodeId in unique_barcodeIds: #[6,7,8,12,13,14,15,16] #
        print "Working on barcodeId %d" % barcodeId
        barcodeArr, counts = makeBarcodeArr(barcodeId,bcbead_row,bcbead_col,bcbead_bcIds,HEIGHT,WIDTH)
        print "Found %d wells having barcodeId %d" % (counts, barcodeId)
        barcodeCountArr.append((barcodeId,counts));
        score,scores,average = calculateScores(barcodeArr, HEIGHT, WIDTH)
        if(not numpy.isnan(average)):
            if(average>maxAverage): maxAverage = average
        scoreArr.append((score,scores,average,barcodeId))
    
    #Making actual plots from data
    for (score,scores,average,barcodeId) in scoreArr:
        outputFilenamePrefix = "bcId%d" % barcodeId
        if(not numpy.isnan(average)):
            makeContourPlot(score, scores, average, HEIGHT, WIDTH, outputFilenamePrefix, "", barcodeId,
                            autoGetVmaxFromAverage(maxAverage))
    return barcodeCountArr

def genHeatmap(filePath, barcodeMaskFilePath=""):
    #
    #Called from TLScript.py
    #
    pylab.clf()
    beadarr, HEIGHT, WIDTH, outputId, maskId = extractMaskInfo(filePath)
    makeContourMap(beadarr, HEIGHT, WIDTH, outputId, maskId)
    
    # If this has barcode information, then split it up.
    if barcodeMaskFilePath:
        return genHeatmapBarcode(barcodeMaskFilePath,HEIGHT,WIDTH, filePath)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('maskfile', default='MaskBead.mask', help='e.g. MaskBead.mask')
    parser.add_argument('barcodefile', nargs='?', help='e.g. barcode.mask')
    args = parser.parse_args()

    genHeatmap(args.maskfile, args.barcodefile)
