#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from matplotlib import use
use("Agg")
import pylab 
import sys
import matplotlib
import matplotlib.pyplot
import math

def findMode(a):
    num = len(a)

    # find max val
    maxVal = 0
    for val in a:
        if val > maxVal:
            maxVal = val

    # clamp to reasonable range
    if maxVal > 4096:
        maxVal = 4096

    # make bin array
    bins = [0]*(int(maxVal+1))

    # fill bins
    for val in a:
        ival = int(val)
        if ival > maxVal:
            ival = maxVal
        bins[ival] = bins[ival] + 1

    # find bigest bin
    maxBin = bins[0]
    maxBinVal = 0
    for val in range(0, int(maxVal)):
        if bins[val] > maxBinVal:
            maxBinVal = bins[val]
            maxBin = val
 
    return maxBin

def trimmedReadLenHisto(readLenFileName, readLenHistoFileName):
    f = open(readLenFileName)
    title = f.readline()
    data = f.readlines()
    f.close()
    title=title.strip('\n')
    titlelist=title.split('\t')
    column = -1
    for b in range(0,len(titlelist)): 
        if titlelist[b] == 'trimLen':
            column=b
    if column == -1:
        print "ERROR: trimmedReadLenHisto: %s doesn't contain a valid title" % readLenFileName
    a = [float(i.strip().split('\t')[column]) for i in data]
    # establish a dynamic X axis scale based on data size
    mode = findMode(a)
    graph_max_x = math.trunc(mode/100.0+0.5) * 100 + 100
    if graph_max_x < 400:
        graph_max_x = 400
    pylab.clf()
    pylab.gcf().set_size_inches((8.0,4.0))
    pylab.gcf().set_size_inches((8,4))
    pylab.hist(a, bins=graph_max_x, range=(0,graph_max_x),color='#2D4782',histtype='stepfilled')
    pylab.title('Read Length Histogram')
    pylab.xlabel('Read Length')
    pylab.ylabel('Count')
    pylab.savefig(readLenHistoFileName)

if __name__=='__main__':
    readLenFileName = sys.argv[1]
    readLenHistoFileName = sys.argv[2]
    trimmedReadLenHisto(readLenFileName, readLenHistoFileName)
