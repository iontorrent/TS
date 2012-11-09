#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import os

from matplotlib import use
use("Agg",warn=False)
import matplotlib.pyplot as plt
from ion.utils.blockprocessing import printtime


def read_length_sparkline(readlen_txt_path, sparkline_path, max_length):
    
    if not os.path.exists(readlen_txt_path):
        printtime("ERROR: %s does not exist" % readlen_txt_path)
        return
    
    
    histogram_x = range(0,max_length,5)
    num_bins = len(histogram_x)
    histogram_y = [0] * num_bins
    
    try:
        f = open(readlen_txt_path,'r')
        f.readline()
        for line in f:
            current_read = int(line.split()[1])
            current_bin = min(current_read/5,num_bins-1)
            histogram_y[current_bin] += 1
        f.close()
        
    except:
        printtime("ERROR: problem parsing %s" % readlen_txt_path)
        traceback.print_exc()
        return


    max_y = max(histogram_y)
    max_y = max(max_y,1)
    
    fig = plt.figure(figsize=(3,0.3),dpi=100)
    ax = fig.add_subplot(111,frame_on=False,xticks=[],yticks=[],position=[0,0,1,1])
    ax.bar(histogram_x,histogram_y,width=6.5, color="#2D4782",linewidth=0, zorder=2)

    for idx in range(0,max_length,50):
        label_bottom = str(idx)
        ax.text(idx,max_y*0.70,label_bottom,horizontalalignment='center',verticalalignment='center',
                fontsize=8, zorder=1)
        ax.axvline(x=idx,color='#D0D0D0',ymax=0.5, zorder=0)
        ax.axvline(x=idx,color='#D0D0D0',ymin=0.9, zorder=0)

    ax.set_ylim(0,max_y)
    ax.set_xlim(-10,max_length)
    fig.patch.set_alpha(0.0)
    plt.savefig(sparkline_path)



def read_length_histogram(readlen_txt_path, histogram_path, max_length):

    #graph_max_x = math.trunc(mode/100.0+0.5) * 100 + 100
    #if graph_max_x < 400:
    #    graph_max_x = 400

    
    if not os.path.exists(readlen_txt_path):
        printtime("ERROR: %s does not exist" % readlen_txt_path)
        return
    
    
    histogram_x = range(0,max_length,1)
    num_bins = len(histogram_x)
    histogram_y = [0] * num_bins
    
    try:
        f = open(readlen_txt_path,'r')
        f.readline()
        for line in f:
            current_read = int(line.split()[1])
            current_bin = min(current_read,num_bins-1)
            if (current_read < num_bins):
                histogram_y[current_bin] += 1
        f.close()
        
    except:
        printtime("ERROR: problem parsing %s" % readlen_txt_path)
        traceback.print_exc()
        return

    max_y = max(histogram_y)
    max_y = max(max_y,1)
    
    fig = plt.figure(figsize=(4,3.5),dpi=100)
    ax = fig.add_subplot(111,frame_on=False,yticks=[],position=[0,0.15,1,0.88])
    ax.bar(histogram_x,histogram_y,width=2.5, color="#2D4782",linewidth=0, zorder=2)


    ax.set_ylim(0,1.2*max_y)
    ax.set_xlim(-5,max_length+15)
    plt.xlabel("Read Length")
    fig.patch.set_alpha(0.0)
    plt.savefig(histogram_path)


if __name__=="__main__":
    
    read_length_histogram('readLen.txt','readLenHisto.png',400)

