# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
from operator import itemgetter

def parseLog(logText):
    metrics ={}
    ###Get Headings for Process Params
    for line in logText:
        if '=' in line:
            name = line.strip().split("=")
            key = name[0].strip()
            value = name[1].strip()                
            metrics[key]=value
    return metrics

def generateMetrics(fileIn):
    f = open(fileIn, 'r')
    processParams = f.readlines()
    f.close()
    data =  parseLog(processParams)
    return data 

if __name__=='__main__':
    f = open(sys.argv[1], 'r')
    processParams = f.readlines()
    f.close()
    data = parseLog(processParams)
    print data
    
