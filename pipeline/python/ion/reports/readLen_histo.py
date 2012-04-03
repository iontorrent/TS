#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
# readLen_histo.py: Generates Read Length histograms per barcode
#

from matplotlib import use
use("Agg")
import pylab
import matplotlib
import matplotlib.pyplot
import os
from subprocess import call
from glob import iglob
import numpy as np
import matplotlib.cm as cm

from matplotlib import rcParams
rcParams['font.size'] = 10
rcParams['legend.labelspacing'] = 0.5
rcParams['axes.titlesize'] = 12
rcParams['legend.markerscale'] = 10


def run_SFFSummary(sff_file, bId):
    try:
        com = "SFFSummary -o quality.%s.summary --sff-file %s --read-length 50,100,150 --min-length 0,0,0 --qual 0,17,20 -d readLen_%s.txt" % (bId, sff_file, bId)
        print "sffsummary" , com
        ret = call(com,shell=True)
        if int(ret)!=0 and STATUS==None:
            STATUS='ERROR'
    except:
        print 'Failed SFFSumary'

def run_barcodeSplit():
    pass


def plot_separate_histo(data, seq):
    #pylab.clf()
    #pylab.gcf().set_size_inches((8.0,4.0))
    pylab.hist(data, bins=400, range=(0,400),color='indigo',histtype='step')
    pylab.title('Read Length Histogram  Barcode Sequence %s' % seq)
    pylab.xlabel('Read Length')
    pylab.ylabel('Count')
    pylab.savefig("%s_readLenHisto.png" % seq)


def upper_readLenHisto(analysisPath,seq):
    x=[] #x is the list for plaotting histograms
    pylab.clf()
    #pylab.gcf().set_size_inches((8.0,4.0))
    f = open(analysisPath)
    title = f.readline()
    data = f.readlines()
    f.close()
    #pylab.gcf().set_size_inches((8,4))
    title=title.strip('\n')
    title=title.split('\t')
    for b in range(0,len(title)):
        if title[b] == 'trimLen':
            column=b
    a = [float(i.strip().split('\t')[column]) for i in data]
    return a

def plot_one_histo(data, sequence, x_maximum):
    #x is a list
    a = np.arange(0.0, len(data))
    #pylab.hist(hist['Data'], bins=600, range=(0,600), histtype='step', align='mid', label=hist['Sequence']) 
    j=0
    y_maximum = 0
    for i in data:
        c = cm.hot(a[j]/(len(data)+1), 1)
        n, bins, patches = pylab.hist(i, bins=600, range=(0,600), histtype='step', label=str(sequence[j]), color=c)
        if max(n) > y_maximum:
            y_maximum = max(n)
        #pylab.title('Read Length Histogram  Barcode Sequence %s' % seq)
        j=j+1
    pylab.axis([0, x_maximum+100, 0, y_maximum+100])
    pylab.title("Read Length Histogram for Barcodes")
    pylab.legend(loc="upper right", ncol=1, labelspacing=0.01)
    pylab.xlabel('Read Length')
    pylab.ylabel('Count')
    pylab.savefig("ion_readLenHisto.png")
    pylab.grid(True)


if __name__=='__main__':
    #Run barcodeSplit to produce the .sff files per barcode. May not be neccessary
    # First get all the .sff files per barcode
    barcodes=[]
    with open('barcodeList.txt','r') as file2: #Should be compatible with old and new barcodeList.txt
         lines = file2.readlines()
         for line in lines:
            if line.startswith("barcode "): 
                barcodes.append(line.split(',')[1])  
    
    #Now grep for .sff files
    os.system('ls *sff > sff_file_list')
    file3=open('sff_file_list','r')
    
    for line in file3:   
 	   if line.split('_R')[0] in barcodes:
               run_SFFSummary(line.split('\n')[0], line.split('_R')[0])
           if line.split('_')[0] == "nomatch":
               run_SFFSummary(line.split('\n')[0], line.split('_R')[0])
    file3.close()
    data=[]
    s=[] 
    x_maximum = 0             
    #now glob for all the readLen_* files produced and call the readHisto method as below
    for readLen in iglob("readLen_*.txt"):
         sequence=readLen.split('readLen_')[1].split('.txt')[0]
         b = upper_readLenHisto(readLen.split('\n')[0], sequence)
         print "Plotting histogram for sequence - %s" % sequence
         plot_separate_histo(b,sequence)
         if max(b) > x_maximum:
             x_maximum = max(b)
         data.append(b)
         s.append(sequence) 
    
    print "Plotting one histogram"
    plot_one_histo(data, s, x_maximum)

