#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import os
import json

from matplotlib import use
use("Agg",warn=False)
import matplotlib.pyplot as plt
from ion.utils.blockprocessing import printtime


def generate_quality_histogram(basecaller_json_path,quality_histogram_path):
    
    if not os.path.exists(basecaller_json_path):
        printtime("ERROR: %s does not exist" % basecaller_json_path)
        return
    
    try:
        f = open(basecaller_json_path,'r')
        basecaller_json = json.load(f);
        f.close()
        qv_histogram = basecaller_json["Filtering"]["qv_histogram"]
        
    except:
        printtime("ERROR: problem parsing %s" % basecaller_json_path)
        traceback.print_exc()
        return

    sum_total = float(sum(qv_histogram))
    percent_0_5 = 100.0 * sum(qv_histogram[0:5]) / sum_total
    percent_5_10 = 100.0 * sum(qv_histogram[5:10]) / sum_total
    percent_10_15 = 100.0 * sum(qv_histogram[10:15]) / sum_total
    percent_15_20 = 100.0 * sum(qv_histogram[15:20]) / sum_total
    percent_20 = 100.0 * sum(qv_histogram[20:]) / sum_total

    graph_x = [0,5,10,15,20]
    graph_y = [percent_0_5,percent_5_10,percent_10_15,percent_15_20,percent_20]

    max_y = max(graph_y)
    
    ticklabels = ['0-4','5-9','10-14','15-19','20+']

    fig = plt.figure(figsize=(4,4),dpi=100)
    ax = fig.add_subplot(111,frame_on=False,xticks=[],yticks=[],position=[.1,0.1,1,0.9])
    ax.bar(graph_x,graph_y,width=4.8, color="#2D4782",linewidth=0)

    for idx in range(5):
        label_bottom = ticklabels[idx]
        label_top = '%1.0f%%' % graph_y[idx]
        ax.text(idx*5 + 2.5,-max_y*0.04,label_bottom,horizontalalignment='center',verticalalignment='top',
                fontsize=12)
        ax.text(idx*5 + 2.5,max_y*0.06+graph_y[idx],label_top,horizontalalignment='center',verticalalignment='bottom',
                fontsize=12)
    
    plt.xlabel("Base Quality")
    
    ax.set_xlim(0,34.8)
    ax.set_ylim(-0.1*max_y,1.2*max_y)
    fig.patch.set_alpha(0.0)
    plt.savefig(quality_histogram_path)

if __name__=="__main__":
    
    generate_quality_histogram('BaseCaller.json','quality_histogram.png')

