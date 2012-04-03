#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
import parseTFstats
from matplotlib import use
use("Agg")

from operator import itemgetter
from subprocess import *
import numpy 
import pylab
from ion.reports.plotters import plotters
from os import path
from scipy import stats, signal
#import uploadMetrics
from colorify import *


#def hpError(metrics):
#    for k,v in metrics.iteritems():
#        if k != 'LIB':
#            title = str(k)
#            e = v['Per HP accuracy'].strip().split(',')
#            correct = {}
#            counts = {}
#            for i in e:
#                k = i.split(':')
#                key = k[0].strip()
#                inp = k[1].split("/")
#                corr = inp[0].strip()
#                cnt = inp[1].strip()
#                if int(cnt) !=0:
#                    correct[int(key)] = float(corr)
#                    counts[int(key)] = float(cnt)
#                else:
#                    correct[int(key)] = float(corr)
#                    counts[int(key)] = 1
#            erPlot = plotters.HomopolymerErrorPlot(correct, counts=counts)
#            erPlot.render()
#            pylab.savefig(path.join(os.getcwd(), "perHPAccuracy_%s.png" % title))
        
def Q10(metrics):
    for k,v in metrics.iteritems():
        if k != 'LIB':
            title = str(k)
            e = v['Q10'].strip()
            s = v['TF Seq'][:100]
            qLi = e.split(" ")
            qInt = [int(i) for i in qLi]
            qplot = plotters.QPlot2(qInt[:100], q=10, expected=s)
            qplot.render()
            pylab.savefig(path.join(os.getcwd(), "Q10_%s.png" % title)) 
            pylab.clf()
        
def Q17(metrics):
    for k,v in metrics.iteritems():
        if k != 'LIB':
            title = str(k)
            e = v['Q17'].strip()
            s = v['TF Seq'][:100]
            qLi = e.split(" ")
            qInt = [int(i) for i in qLi]
            qplot = plotters.QPlot2(qInt[:100], q=17, expected=s)
            qplot.render()
            pylab.savefig(path.join(os.getcwd(), "Q17_%s.png" % title)) 
            pylab.clf()
        
#def matchMisMatch(metrics):
#    for k,v in metrics.iteritems():
#        if k != 'LIB':
#            title = str(k)
#            e = v['Match-Mismatch'].strip()
#            qLi = e.split(" ")
#            qInt = [int(i) for i in qLi]
#            xaxis = [i for i in range(len(qInt))]   
#            ymax = max(qInt) + 10
#            xlen = len(qInt) + 10
#            pylab.gcf().set_size_inches((8,4))
#            pylab.bar(xaxis,qInt, facecolor = "blue", align = 'center')
#            pylab.xlabel("Correct Bases - Incorrect Bases")
#            pylab.title("Match-Mismatch Bases")
#            pylab.axis([0,80,0,ymax])
#            pylab.savefig(path.join(os.getcwd(),"Match-Mismatch_%s.png" % title))
#            pylab.clf()
         
#def overlapHist(metrics, histType):
#    for k,v in metrics.iteritems():
#        if k != 'LIB':
#            title = str(k)
#            e = v[histType].strip().split(',')
#            pylab.clf()
#            color = ['r','y','g','b','k']#,'c','m']
#            pylab.xlabel("Signal")
#            pylab.title("%s_%s" % (histType,title))
#            pylab.gcf().set_size_inches((8,4))
#            count = 0
#            for i,c in zip(e,color):
#                data = i.strip().split(" ")
#                dList = [int(i) for i in data]
#                xaxis = [i for i in range(len(dList))]  
#                ymax = max(dList) + 10
#                xlen = len(dList) + 10
#                pylab.axis([0,xlen,0,ymax])
#                pylab.bar(xaxis,dList, color=c, linewidth=0,alpha=.5, align='center', label="%s-mer"%count)
#                count+=1
#            pylab.legend()
#            pylab.savefig(path.join(os.getcwd(), "%s_%s.png" % (histType,title)))
#            pylab.clf()

#def hpSNR(metrics, charType):
#    for k,v in metrics.iteritems():
#        if k != 'LIB':
#            title = str(k)
#            e = v[charType].strip().split(',')
#            hp = []
#            snr = []
#            for i in e:
#                k = i.split(':')
#                key = k[0].strip()
#                value = k[1].strip()
#                hp.append(key)
#                snr.append(float(value))
#            pylab.clf()
#            pylab.xlabel("HP")
#            pylab.ylabel("SNR")
#            pylab.title("%s_%s" % (charType,title))
#            pylab.gcf().set_size_inches((8,4))
#            xaxis = [i for i in range(len(hp))]  
#            ymax = int(max(snr)) + 1
#            xlen = int(len(hp)) + 1
#            pylab.axis([0,xlen,0,ymax])
#            pylab.xticks(range(len(hp)))
#            pylab.bar(xaxis, snr, width=.5, linewidth=0, color='r', alpha=.5, align='center')
#        
#        #pylab.legend()
#            pylab.savefig(path.join(os.getcwd(), "%s_%s.png" % (charType,title)))
#            pylab.clf() 

def ionograms(metrics):
    '''generate ionograms from tf information'''
    for k,v in metrics.iteritems():
        if k != 'LIB':
            title = str(k)
            savePaths = []
            alignhtml = []
            for i in range(10):
                key = "Top " + str(i+1)
                if key in v:
                    index = i+1
                    e = v[key]
                    splitE = e.split(',')
                    row = splitE[0]
                    col = splitE[1]
                    pre = splitE[2]
                    post = splitE[3]
                    ideal = splitE[4]
                    bars = splitE[5]
                    called = splitE[6]
                    preA = [float(i) for i in pre.strip().split(' ')]
                    postA = [float(i) for i in post.strip().split(' ')]
                    rowCol = "(%s,%s)" % (row,col)
                    
                    header = title + " Pre-Corrected " + rowCol
                    preIonogram = plotters.IonogramJMR('TACG', preA[:len(preA)], preA[:len(preA)], header)
                    preIonogram.render()
                    preSavePath = "Pre-Corrected_%s_%s.png" % (index,title)
                    pylab.savefig(preSavePath)
                    
                    header = title + " Post-Corrected " + rowCol
                    preIonogram = plotters.IonogramJMR('TACG', postA[:len(postA)], postA[:len(postA)], header)
                    preIonogram.render()
                    postSavePath ="Post-Corrected_%s_%s.png" % (index,title) 
                    pylab.savefig(postSavePath)
                    savePaths.append([preSavePath, postSavePath])
                    alignment = [called, bars, ideal]
                    ainter = []
                    for count, string in enumerate(alignment):
                        inter = []
                        if count != 2:
                            for i in string:
                                inter.append(colorify(i.upper()))
                        else:
                            for i in string:
                                inter.append(i.upper())
                        ainter.append(inter)
                    alignhtml.append(ainter)
            if len(savePaths) > 0:
                generateWeb(savePaths, title, alignhtml)
    
#def sigOverlap(data, histType):
#    type = {'Raw signal overlap':'Pre Corrected',
#            'Corrected signal overlap':'Post Corrected'}
#    for k,v in data.iteritems():
#        if k != 'LIB':
#            title = str(k)
#            e = v[histType].strip().split(',')
#            toPlot = {}
#            for n,hp in enumerate(e):
#                if hp != '' and hp != ' ':
#                    d = hp.strip().split(" ") 
#                    if len(d)>1: # we don't want to plot anything that is only at zero
#                        toPlot[int(n)] = [float(i) for i in d]
#            p = plotters.SignalOverlap(toPlot, title="%s Signal Resolution" % (type[histType],))
#            p.render()
#            pylab.savefig(path.join(os.getcwd(), "%s_%s.png" % (histType,title)))

#def sigDist(data):
#    for k,v in data.iteritems():
#        if k != 'LIB':
#            data = v['TransferPlot'].strip().split(',')          
#            datAr = []
#            datDict = {}
#            for d in data:
#                line = d.strip().split(" ")
#                key = int(line[0].strip())
#                datDict[key] = ""
#                ave = line[1].strip()
#                std = line[2].strip()
#                datAr.append((float(ave),float(std)))
#            p = plotters.TransferPlot(datDict,datAr)
#            p.render()
#            pylab.savefig("Transfer_Plot_%s" % k)


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

def generateGraphics(metricsdata):
    Q10(metricsdata)
    Q17(metricsdata)
    ionograms(metricsdata)
    
if __name__=='__main__':

    data = generateMetricsData(sys.argv[1])
    generateGraphics(metricsdata)
    genCafieIonograms(data,"TACG")
