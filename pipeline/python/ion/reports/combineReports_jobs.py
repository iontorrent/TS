#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import argparse
import traceback
import subprocess
import os
import json

from ion.utils.blockprocessing import merge_bam_files
from ion.utils.compress import make_zip
os.environ['MPLCONFIGDIR'] = '/tmp'
from ion.utils import ionstats, ionstats_plots

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--add-file', dest='files', action='append', default=[], help="list of files to process")
    parser.add_argument('-m', '--merge-bams', dest='merge_out', action='store', default = "", help='merge bam files')
    parser.add_argument('-d', '--mark-duplicates', dest='duplicates', action='store_true', default = False, help='mark duplicates')
    parser.add_argument('-a', '--align-stats', dest='align_stats', action='store', default = "", help='generate alignment stats')
    parser.add_argument('-g', '--genomeinfo', dest='genomeinfo', action='store', default = "", help='genome info file for alignment stats')
    parser.add_argument('-p', '--merge-plots', dest='merge_plots', action='store_true', default = "", help='generate report plots')
    parser.add_argument('-z', '--zip', dest='zip', action='store', default = "", help='zip input files')

    args = parser.parse_args()
    
    
    if args.merge_out and len(args.files) > 1:   
       # Merge BAM files 
       outputBAM = args.merge_out
       print "Merging bam files to %s, mark duplicates is %s" % (outputBAM, args.duplicates)
       try:
          merge_bam_files(args.files, outputBAM, outputBAM.replace('.bam','.bam.bai'), args.duplicates)
       except:
          traceback.print_exc()
       # generate ionstats files from merged BAM
       graph_max_x = 400
       if outputBAM == 'rawlib.bam':
          ionstats_file = 'ionstats_alignment.json'
       else:
          ionstats_file = outputBAM.split('.bam')[0] + '.ionstats_alignment.json'
       ionstats.generate_ionstats_alignment(outputBAM, ionstats_file, graph_max_x)
       
    if args.merge_plots:          
        print "Generating plots for merged report"
        ionstats_file = 'ionstats_alignment.json'
        
        try:       
            stats = json.load(open(ionstats_file))
            l = stats['full']['max_read_length']
            graph_max_x = int(round(l + 49, -2))
            
            # Make alignment_rate_plot.png and base_error_plot.png
            ionstats_plots.alignment_rate_plot2(ionstats_file, 'alignment_rate_plot.png', int(graph_max_x))
            ionstats_plots.base_error_plot(ionstats_file, 'base_error_plot.png', int(graph_max_x))
        except:
            traceback.print_exc()
      
    if args.zip and len(args.files) > 1: 
       # zip barcoded files
       zipname = args.zip
       print "Zip merged barcode files to %s" % zipname
       for filename in args.files:                      
         if os.path.exists(filename):
            try:
                make_zip(zipname, filename, arcname=filename)
            except:
                print("ERROR: zip target: %s" % filename)
                traceback.print_exc()

