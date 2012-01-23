#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from matplotlib import use
use("Agg")
import pylab 
import sys
import matplotlib
import matplotlib.pyplot

def trimmedReadLenHisto(readLenFileName, readLenHistoFileName):
    pylab.clf()
    pylab.gcf().set_size_inches((8.0,4.0))
    f = open(readLenFileName)
    title = f.readline()
    data = f.readlines()
    f.close()
    pylab.gcf().set_size_inches((8,4))
    title=title.strip('\n')
    title=title.split('\t')
    for b in range(0,len(title)): 
        if title[b] == 'trimLen':
            column=b
    a = [float(i.strip().split('\t')[column]) for i in data]
    pylab.hist(a, bins=400, range=(0,400),color='indigo',histtype='stepfilled')
    pylab.title('Read Length Histogram')
    pylab.xlabel('Read Length')
    pylab.ylabel('Count')
    pylab.savefig(readLenHistoFileName)

if __name__=='__main__':
    readLenFileName = sys.argv[1]
    readLenHistoFileName = sys.argv[2]
    trimmedReadLenHisto(readLenFileName, readLenHistoFileName)
