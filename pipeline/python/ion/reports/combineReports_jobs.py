#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import argparse
import traceback
import subprocess
import os

from ion.utils.blockprocessing import merge_bam_files
from ion.utils.compress import make_zip
os.environ['MPLCONFIGDIR'] = '/tmp'
from ion.reports import  base_error_plot

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
       merge_bam_files(args.files, outputBAM, outputBAM.replace('.bam','.bam.bai'), args.duplicates)


    if args.align_stats:
       # Call alignStats on merged bam file       
       inputBAM = args.align_stats    
       print "Running alignStats on %s" % inputBAM
       
       cmd = "alignStats"
       
       if '_rawlib.bam' in inputBAM:
          bcid = inputBAM.rstrip('_rawlib.bam')          
          cmd += " -o %s" % bcid
          # make alignment_BC.summary links to BC.alignment.summary output of alignStats
          os.symlink('%s.alignment.summary' % bcid, 'alignment_%s.summary' % bcid)  
       
       if args.genomeinfo:
          cmd += " --genomeinfo %s" % args.genomeinfo
       
       cmd += " --infile %s" % inputBAM
       cmd += " --qScores 7,10,17,20,30,47"
       cmd += " --alignSummaryFilterLen 20"
       cmd += " --alignSummaryMaxLen  400"
       cmd += " --errTableMaxLen 400"
       cmd += " --outputDir %s" % './'  
       
               
       print("DEBUG: Calling '%s'" % cmd)
       try:  
         subprocess.call(cmd,shell=True)      
       except:
         traceback.print_exc() 
    
    if args.merge_plots and len(args.files) > 1:          
        print "Generating plots for merged report"
        dirs = [os.path.dirname(filepath) for filepath in args.files]

        # merge readLen.txt        
        readlen_file_list = [os.path.join(dir,'basecaller_results','readLen.txt') for dir in dirs]
        l = []
        try:            
            rl_out = open("readLen.txt",'w')
            rl_out.write("read\ttrimLen\n")
            for readlen_file in readlen_file_list:
                if os.path.exists(readlen_file):                    
                    rl_in = open(readlen_file,'r')
                    first_line = rl_in.readline()
                    for line in rl_in:
                        l.append(int(line.split()[1]))
                        rl_out.write(line)
                    rl_in.close()
                else:
                    print("ERROR: skipped %s" % readlen_file)
            rl_out.close()
        except:
            traceback.print_exc()
            
        # Make plots for merged Report
        ALIGNMENT_RESULTS = '.'
        graph_max_x = round(max(l),-2) if round(max(l),-2) < 400 else 400
        try:
            base_error_plot.generate_base_error_plot(
                os.path.join(ALIGNMENT_RESULTS,'alignStats_err.json'),
                os.path.join(ALIGNMENT_RESULTS,'base_error_plot.png'),int(graph_max_x))
            base_error_plot.generate_alignment_rate_plot(
                os.path.join(ALIGNMENT_RESULTS,'alignStats_err.json'),
                os.path.join('.','readLen.txt'),
                os.path.join(ALIGNMENT_RESULTS,'alignment_rate_plot.png'),int(graph_max_x))            
            print("Base error plot has been created successfully")
        except:
            print("ERROR: Failed to generate base error plot")
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
        
        
    
         
