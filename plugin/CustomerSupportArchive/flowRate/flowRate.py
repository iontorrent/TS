#!/usr/bin/python
# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import shutil
import textwrap
import json
import re
import csv
from subprocess import *
from ion.plugin import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def image_link(imgpath, width = 50):
    ''' Returns code for displaying an image also as a link '''
    text = '<a href="%s"><img src="%s" width="%d%%" /></a>' % ( imgpath, imgpath , width )
    return text

class flowRate():
    version = "1.6.6"
    allow_autorun = True # if true, no additional user input
    depends = []    
    def launch(self):
        """ main """
        print "running the flowRate plugin."

        self.results_dir = os.environ['TSP_FILEPATH_PLUGIN_DIR']
        self.analysis_dir = os.environ['ANALYSIS_DIR']
        self.raw_data_dir = os.environ['RAW_DATA_DIR'] 

        if os.path.exists(os.path.join(self.raw_data_dir, 'explog_final.txt')) or os.path.exists(os.path.join(self.analysis_dir, 'explog_final.txt')):
            if os.path.exists(os.path.join(self.raw_data_dir, 'explog_final.txt')):
                f, r = self.getFR(os.path.join(self.raw_data_dir, 'explog_final.txt'))
            else:
                f, r = self.getFR(os.path.join(self.analysis_dir, 'explog_final.txt'))
            if len(r) == 0 or len(f) == 0:
                print  "Cannot parse explog_final.txt in analysis directory!"
                sys.exit(1)
            else:
                fig = plt.figure()
                ax = plt.subplot(111)
                ax.plot(f, r, 'o-')
                plt.title(self.analysis_dir.split('/')[-1] + ' - flow rate plot')
                plt.ylabel('Flow Rate')
                plt.xlabel('Flow #')
                fig.savefig(os.path.join(self.results_dir, 'flowRate.png'))

                with open(os.path.join(self.results_dir, 'flowRate.csv'), 'wb') as myfile:
                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    wr.writerow(r)
                    wr.writerow(f)
                self.write_html()
        else:
            print "Cannot find explog_final.txt in analysis directory!"
            sys.exit(1)
        sys.exit(0) 

    def write_html(self):
        """ Creates html and block html files """
        html = textwrap.dedent('''\
        <html>
        <head><title>LibPrepLog</title></head>
        <body>
        <style type="text/css">div.scroll {max-height:800px; overflow-y:auto; overflow-x:auto; border: 0px solid black; padding: 5px 5px 0px 25px; float:left; }</style>
        <style type="text/css">div.plots {max-height:850px; overflow-y:hidden; overflow-x:hidden; padding: 0px; }</style>
        <style type="text/css">tr.shade {background-color: #eee; }</style>
        <style type="text/css">td.top {vertical-align:top; }</style>
        <style>table.link {border: 1px solid black; border-collapse: collapse; padding: 0px; table-layout: fixed; text-align: center; vertical-align:middle; }</style>
        <style>th.link, td.link {border: 1px solid black; }</style>
        ''' )
        width = 33
        html += '<table cellspacing="0" width="100%%"><tr><td width="70%%">'
        html += '<div class="plots">'
        html += '<table cellspacing="0" width="100%%">' 
        html += '<tr>'
        html += '<td width="%s%%">%s</td>' % ( width , image_link( 'flowRate.png' ))
        html += '</tr>'
        html += '</div></td></tr></table><br>'
        html += '<h3><b>flowRate</b></h3>'
        html += '<table width="100%%" cellspacing="0" border="0">'
        html += '<td><a href="flowRate.csv" download>flowRate </a></td>'
        html += '</div></td></tr></table><br><hr>'
        with open( os.path.join( self.results_dir , 'flowRate_block.html' ) , 'w' ) as f:
            f.write( html )

    def getFR(self, fname):
        flows = []
        rates = []
        with open(fname, 'r') as f:
            found = False
            for line in f:
                if 'ExperimentInfoLog:' in line:
                    try:
                        while True:
                            lineCurr = f.next()
                            preflowID = re.search('prerun_(.*).dat',lineCurr)
                            if preflowID:
                                found = True
                                flowRate = re.search('FR=(.*), ',lineCurr)
                                if flowRate:
                                    rates.append(float(flowRate.group(1)))
                                    flows.append(int(preflowID.group(1)) - 8)
                            flowID = re.search('acq_(.*).dat',lineCurr)
                            if flowID and found:
                                flowRate = re.search('FR=(.*), ',lineCurr)
                                if flowRate:
                                    rates.append(float(flowRate.group(1)))
                                    flows.append(int(flowID.group(1)))
                    except(StopIteration):
                        print " "
        return flows, rates

if __name__ == "__main__": 
    fR = flowRate()
    fR.launch()