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
import traceback

def parseLog(logText):
    metrics ={}
    lineDict = {}
    currentKey = None
    ###Get Headings for Cafie Params
    for line in logText:
        if '#' not in line:
            name = line.strip().split("=")
            key = name[0].strip()
            value = name[1].strip()
            if key == "TF":
                if key not in metrics:
                    metrics[value] = {}
                    currentKey = value
                    lineDict = {}
            elif 'System' in line:
                if 'System' not in metrics:
                    metrics['System'] = {}
                    currentKey = 'System'
                    lineDict = {}
                lineDict[key]=value
                metrics[currentKey]=lineDict
            else:        
                lineDict[key] = value
                metrics[currentKey]=lineDict
    return metrics

def genCafieIonograms(data,floworder):
    for tf,metrics in data.iteritems():
        if tf != 'System':
            title = str(tf)
            aveI = metrics['Avg Ionogram']
            aveI = aveI.split(" ")
            aveI = [float(i) for i in aveI]
    
            corrI = metrics['Corrected Avg Ionogram']
            corrI = corrI.split(" ")
            corrI = [float(i) for i in corrI]
            ionogramDict = {'Average Raw Ionogram': aveI, 'Average Corrected Ionogram': corrI}
            for k,v in ionogramDict.iteritems():
                corrIonogram = plotters.IonogramJMR(floworder, v[:len(v)],v,k)
                corrIonogram.render()
                pylab.savefig(path.join(os.getcwd(), "%s_%s.png" % (k, title)))
        
def generateMetrics(fileIn,floworder):
    f = open(fileIn, 'r')
    cafieLog = f.readlines()
    f.close()
    data = parseLog(cafieLog)
    try:
        genCafieIonograms(data,floworder)
    except:
        print traceback.format_exc()
    return data
    
if __name__=='__main__':
    import sys
    floworder = sys.argv[2]
    f = open(sys.argv[1], 'r')
    cafieLog = f.readlines()
    f.close()
    data = parseLog(cafieLog)
    print data
    genCafieIonograms(data,floworder)
    
    
    
    
    
    
