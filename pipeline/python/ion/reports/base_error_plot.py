#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import os
import json
import traceback

from matplotlib import use
use("Agg",warn=False)
import matplotlib.pyplot as plt
from ion.utils.blockprocessing import printtime


def generate_base_error_plot(alignStats,plotPng, graph_max_x):
    
    if not os.path.exists(alignStats):
        printtime("ERROR: %s does not exist" % alignStats)
        return
    
    try:
        f = open(alignStats,'r')
        alignStats_json = json.load(f);
        f.close()
        read_length = alignStats_json["read_length"]
        n_err_at_position = alignStats_json["n_err_at_position"]
        aligned = alignStats_json["aligned"]       
    except:
        printtime("ERROR: problem parsing %s" % alignStats)
        traceback.print_exc()
        return

    accuracy = []
    reads = []

    for i, base in enumerate(read_length):
        if aligned[i] > 1000:
            accuracy.append( 100 * ( 1-float(n_err_at_position[i]) / float(aligned[i]) ) )
            reads.append(base)


    fig = plt.figure(figsize=(4,4),dpi=100)
    ax = fig.add_subplot(111,frame_on=False)

    max_x = 1
    if len(reads) > 0:
        max_x = max(reads)
    #max_x = min(max_x,graph_max_x)
    max_x = graph_max_x

    plt.plot(reads, accuracy, linewidth=3.0, color="#2D4782")
    plt.axis([0, max_x, 90 , 100.9])


    plt.xlabel('Position in Read')

    plt.ylabel("Accuracy at Position")
    
    #ax.set_xlim(0,read_length[-1])
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    fig.patch.set_alpha(0.0)
    plt.savefig(plotPng)




def generate_alignment_rate_plot(alignStats, readlen_txt_path, plotPng,graph_max_x):
    
    if not os.path.exists(alignStats):
        printtime("ERROR: %s does not exist" % alignStats)
        return
    
    def intWithCommas(x):
        if type(x) not in [type(0), type(0L)]:
            raise TypeError("Parameter must be an integer.")
        if x < 0:
            return '-' + intWithCommas(-x)
        result = ''
        while x >= 1000:
            x, r = divmod(x, 1000)
            result = ",%03d%s" % (r, result)
        return "%d%s" % (x, result)
    
    try:
        f = open(alignStats,'r')
        alignStats_json = json.load(f);
        f.close()
        read_length = alignStats_json["read_length"]
        #nread = alignStats_json["nread"]
        aligned = alignStats_json["aligned"]       
    except:
        printtime("ERROR: problem parsing %s" % alignStats)
        traceback.print_exc()
        return

    if not os.path.exists(readlen_txt_path):
        printtime("ERROR: %s does not exist" % readlen_txt_path)
        return
    
    
    histogram_x = range(0,graph_max_x)
    histogram_y = [0] * graph_max_x
    
    try:
        f = open(readlen_txt_path,'r')
        f.readline()
        for line in f:
            current_read = int(line.split()[1])
            current_bin = min(current_read,graph_max_x-1)
            histogram_y[current_bin] += 1
        f.close()
        
    except:
        printtime("ERROR: problem parsing %s" % readlen_txt_path)
        traceback.print_exc()
        return


    for idx in range(graph_max_x-1,0,-1):
        histogram_y[idx-1] += histogram_y[idx]
    
    nread = histogram_y[1:]

    fig = plt.figure(figsize=(4,3),dpi=100)
    ax = fig.add_subplot(111,frame_on=False,yticks=[], position=[0.1,0.15,0.8,0.89])

    max_x = 1
    if len(read_length) > 0:
        max_x = max(read_length)
    max_x = min(max_x,graph_max_x)

    max_y = max(nread)

    plt.fill_between(histogram_x[1:], nread, color="#808080", zorder=1)
    plt.fill_between(read_length, aligned, color="#2D4782", zorder=2)

    plt.xlabel('Position in Read')
    plt.ylabel('Reads')

    map_percent = int(round(100.0 * float(alignStats_json["total_mapped_target_bases"]) 
                            / float(sum(nread))))
    unmap_percent = 100 - map_percent

    color_blue = "#2D4782"
    color_gray = "#808080"
    fontsize_big = 15
    fontsize_small = 10
    fontsize_medium = 8

    ax.text(0.8*max_x,0.95*max_y,'Aligned Bases',horizontalalignment='center',verticalalignment='center', fontsize=fontsize_small, zorder=4, color=color_blue,weight='bold',stretch='condensed')
    ax.text(0.8*max_x,1.05*max_y,' %d%%'%map_percent,horizontalalignment='center',verticalalignment='center', fontsize=fontsize_big, zorder=4, color=color_blue,weight='bold',stretch='condensed')
    ax.text(0.8*max_x,0.7*max_y,'Unaligned',horizontalalignment='center',verticalalignment='center',
            fontsize=fontsize_small, zorder=4, color=color_gray,weight='bold',stretch='condensed')
    ax.text(0.8*max_x,0.8*max_y,' %d%%'%unmap_percent,horizontalalignment='center',verticalalignment='center', fontsize=fontsize_big, zorder=4, color=color_gray,weight='bold',stretch='condensed')

    ax.text(-0.06*max_x,1.02*max_y,intWithCommas(max_y),horizontalalignment='left',verticalalignment='bottom',  zorder=4, color="black")
       
    ax.set_xlim(0,max_x)
    ax.set_ylim(0,1.2*max_y)
    #plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    fig.patch.set_alpha(0.0)
    plt.savefig(plotPng)


if __name__=="__main__":
    
    generate_base_error_plot('alignStats_err.json','base_error_plot.png',300)
    generate_alignment_rate_plot('alignStats_err.json','readLen.txt','alignment_rate_plot.png',300)
