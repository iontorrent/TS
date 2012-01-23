# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
from operator import itemgetter
from subprocess import *
import numpy 
import pylab
from ion.reports.plotters import plotters
from os import path
from scipy import stats, signal
import random

def makeRawPlot(data, labels, flowOrder):
    title = None
    if labels == 'TCAG':
        title = 'Library'
    elif labels == 'ATCG':
        title = 'Test Fragment'
    trace = {}
    for line in data:
        t = line.strip().split(" ")
        fTrace = [float(i) for i in t[1:]]
        trace[t[0]] = fTrace
        
    toPlot = []
    for k in flowOrder:
        if k != "G":
            toPlot.append(trace[k])
    expected = [1,1,1]
    tracePlot = plotters.Iontrace(flowOrder, expected, toPlot, title="Consensus Key 1-Mer - %s" % title)
    tracePlot.render()
    pylab.savefig(path.join(os.getcwd(), 'iontrace_%s' % title))

def start(data, labels):
    makeRawPlot(data, labels, 'TCAG')    

if __name__=="__main__":
    import sys
    f = open(sys.argv[1])
    data = f.readlines()
    f.close()
    exp = sys.argv[1].strip().split("_")
    print exp
    labels = exp[-1].split(".")
    makeRawPlot(data, labels[0], 'TACG')
