#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Loads Fullchip analysis pipeline
Uses output from there to generate charts and graphs and dumps to current directory
Adds key metrics to database(not implemented yet)
"""

__version__ = filter(str.isdigit, "$Revision: 20865 $")

import os
import tempfile

# matplotlib/numpy compatibility
os.environ['HOME'] = tempfile.mkdtemp()
from matplotlib import use
use("Agg")

import datetime
import subprocess
import sys
import time
from collections import deque
from urlparse import urlunsplit

# Import analysis-specific packages. We try to follow the from A import B
# convention when B is a module in package A, or import A when
# A is a module.
from ion.reports import libGraphs
from ion.reports.plotters import *
from ion.utils.aggregate_alignment import *


import math
from scipy import cluster, signal, linalg
import traceback
import json

import dateutil.parser

#NB: this should be r-factored out!
def printtime(message, *args):
    if args:
        message = message % args
    print "[ " + time.strftime('%X') + " ] " + message

def makeAlignGraphs():
    ###################################################
    # Make histograms graphs from tmap output         #
    ###################################################
    qualdict = {}
    qualdict['Q10'] = 'red'
    qualdict['Q17'] = 'yellow'
    qualdict['Q20'] = 'green'
    qualdict['Q47'] = 'blue'
    for qual in qualdict.keys():
        dpath = qual + ".histo.dat"
        if os.path.exists(dpath):
            try:
                data = libGraphs.parseFile(dpath)
                it = libGraphs.QXXAlign(data, "Filtered "+qual+" Read Length", 'Count',
                                        "Filtered "+qual+" Read Length", qualdict[qual], "Filtered_Alignments_"+qual+".png")
                it.plot()
            except:
                traceback.print_exc()

def align_full_chip_core(libsff, libKey, tfKey, floworder, fastqName, align_full, graph_max_x, do_barcode, make_align_graphs, sam_parsed, DIR_BC_FILES, env, outputdir):
    #collect all the meta data for the SAM file
    SAM_META = {}

    # id - this hash comes from the fastq file
    try:
        #read the first line of the fastq file
        fastqFile = open(fastqName,'r')
        id_hash = fastqFile.readline()
        fastqFile.close()

        #now just pull out the hash
        id_hash = id_hash[1:6]

        SAM_META['ID'] = id_hash

    except IOError:
        printtime("Could not read fastq file.  The ID for the SAM file could not be found.")

    # sm - name for reads - project name
    SAM_META['SM'] = env['project']

    # lb - library name
    SAM_META['LB'] = env['libraryName']

    # pu - the platform unit
    SAM_META['PU'] = "PGM/" + env['chipType'].replace('"',"")

    SAM_META['PL'] = "IONTORRENT"

    #TODO: do not assume localhost.  Find the name of the masternode
    try:
        #this will get the exp data from the database
        exp_json = json.loads(env['exp_json'])

        # ds - the "notes", only the alphanumeric and space characters.
        SAM_META['DS'] = ''.join(ch for ch in exp_json['notes'] if ch.isalnum() or ch == " ")

        # dt - the run date
        exp_log_json = json.loads(exp_json['log'])
        iso_exp_time = exp_log_json['start_time']

        #convert to ISO time
        iso_exp_time = dateutil.parser.parse(iso_exp_time)

        SAM_META['DT'] = iso_exp_time.isoformat()

        #the site name should be here, also remove spaces
        site_name = env['site_name']
        site_name = ''.join(ch for ch in site_name if ch.isalnum() )
        SAM_META['CN'] = site_name

        env['flows'] = exp_json['flows']

    except:
        printtime("There was an error getting the site name, because the Torrent Browser could not be contacted")
        traceback.print_exc()

    #Now build the SAM meta data arg string
    aligner_opts_rg= '--aligner-opts-rg "'
    aligner_opts_extra = ''
    if sam_parsed:
        aligner_opts_extra += ' -p 1'
    if env['aligner_opts_extra']:
        print '  found extra alignment options: "%s"' % env['aligner_opts_extra']
        aligner_opts_extra = ' --aligner-opts-extra "'
        aligner_opts_extra += env['aligner_opts_extra'] + '"'
    first = True
    for key, value in SAM_META.items():
        if value:
            sam_arg =  r'-R \"'
            end =  r'\"'

            sam_arg = sam_arg + key + ":" + value + end

            if first:
                aligner_opts_rg = aligner_opts_rg + sam_arg
                first = False
            else:
                aligner_opts_rg = aligner_opts_rg + " " + sam_arg

    #add the trailing quote
    aligner_opts_rg = aligner_opts_rg + '"'

    if 0 < graph_max_x:
        # establish the read-length histogram range by using the simple rule: 0.6 * num-flows
        flowsUsed = 0
    	try:
            flowsUsed = int(env['flows'])
        except:
            flowsUsed = 400
        graph_max_x = 100 * math.trunc((0.6 * flowsUsed + 99)/100.0)
    if graph_max_x < 400:
        graph_max_x = 400

    #-----------------------------------
    # DEFAULT SINGLE SFF/FASTQ BEHAVIOR - (Runs for barcoded runs too)
    #-----------------------------------
    if (align_full):
        #If a full align is forced add a '--align-all-reads' flag
        com = "alignmentQC.pl"
        com += " --logfile %s" % os.path.join(outputdir,"alignmentQC_out.txt")
        com += " --output-dir %s" % outputdir
        com += " --input %s" % libsff
        com += " --genome %s" % env["libraryName"]
        com += " --max-plot-read-len %s" % graph_max_x
        com += " --align-all-reads"
        com += " %s %s" % (aligner_opts_rg,aligner_opts_extra)
        com += " >> ReportLog.html 2>&1"
    else:
        # Add -p 1 to enable default.sam file generation
        com = "alignmentQC.pl"
        com += " --logfile %s" % os.path.join(outputdir,"alignmentQC_out.txt")
        com += " --output-dir %s" % outputdir
        com += " --input %s" % libsff
        com += " --genome %s" % env["libraryName"]
        com += " --max-plot-read-len %s" % graph_max_x
        com += " %s %s" % (aligner_opts_rg,aligner_opts_extra)
        com += " >> ReportLog.html 2>&1"

    try:
        printtime("Alignment QC command line:\n%s" % com)
        retcode = subprocess.call(com, shell=True)
        if retcode != 0:
            printtime("alignmentQC failed, return code: %d" % retcode)
            alignError = open("alignment.error", "w")
            alignError.write('alignmentQC returned with error code: ')
            alignError.write(str(retcode))
            alignError.close()
    except OSError:
        printtime('Alignment Failed to start')
        alignError = open("alignment.error", "w")
        alignError.write(str(traceback.format_exc()))
        alignError.close()
        traceback.print_exc()
    if make_align_graphs:
        makeAlignGraphs()

    #--------------------------------------------
    # BARCODE HANDLING BEHAVIOR (Multiple FASTQ)
    #--------------------------------------------
    if env['barcodeId'] and True == do_barcode:
        printtime("Renaming non-barcoded alignment results to 'comprehensive'")
        files = [ 'alignment.summary',
                  'alignmentQC_out.txt',
                  'alignTable.txt',
                ]
        for fname in files:
            try:
                if os.path.exists(fname):
                    os.rename(fname, fname + ".comprehensive")
            except:
                printtime('error renaming')
                traceback.print_exc()
        # Only make the graphs from the alignment of comprehensive fastq file
	if make_align_graphs:
            makeAlignGraphs()

        printtime("STARTING BARCODE ALIGNMENTS")
        if not os.path.exists(DIR_BC_FILES):
            os.mkdir(DIR_BC_FILES)

        barcodeList = parse_bcfile('barcodeList.txt')

        align_full = True
        for bcid in (x['id_str'] for x in barcodeList):
            sffName = "%s_%s_%s.sff" % (bcid, env['expName'], env['resultsName'])
            if not os.path.exists(sffName):
                printtime("No barcode SFF file found for '%s'" % bcid)
                continue
            if (align_full):
                printtime("Align All Reads")
                #If a full align is forced add a '--align-all-reads' flag
                com = "alignmentQC.pl" 
                com += " --logfile %s" % os.path.join(outputdir,"alignmentQC_out.txt")
                com += " --output-dir %s" % outputdir
                com += " --input %s" % sffName
                com += " --genome %s" % env["libraryName"]
                com += " --max-plot-read-len %s" % graph_max_x
                com += " --align-all-reads"
                com += " %s %s" % (aligner_opts_rg, aligner_opts_extra)
                com += " >> ReportLog.html 2>&1" 
            else:
                printtime("Align Subset of Reads")
                # Add -p 1 to enable default.sam file generation
                com = "alignmentQC.pl" 
                com += " --logfile %s" % os.path.join(outputdir,"alignmentQC_out.txt")
                com += " --output-dir %s" % outputdir
                com += " --input %s" % sffName
                com += " --genome %s" % env["libraryName"]
                com += " --max-plot-read-len %s" % graph_max_x
                com += " %s %s" % (aligner_opts_rg, aligner_opts_extra)
                com += " >> ReportLog.html 2>&1"
            try:
                printtime("Alignment QC command line:\n%s" % com)
                retcode = subprocess.call(com, shell=True)
                if retcode != 0:
                    printtime("alignmentQC failed, return code: %d" % retcode)
                    alignError = open("alignment.error", "a")
                    alignError.write(com)
                    alignError.write(': \nalignmentQC returned with error code: ')
                    alignError.write(str(retcode))
                    alignError.close()
            except OSError:
                printtime('Alignment Failed to start')
                alignError = open("alignment.error", "a")
                alignError.write(str(traceback.format_exc()))
                alignError.close()
                traceback.print_exc()

            #rename each output file based on barcode found in fastq filename
            #but ignore the comprehensive fastq output files
            if os.path.exists('alignment.summary'):
                try:
                    fname='alignment_%s.summary' % bcid
                    os.rename('alignment.summary', fname)
                    os.rename(fname,os.path.join(DIR_BC_FILES,fname))
                    fname='alignmentQC_out_%s.txt' % bcid
                    os.rename('alignmentQC_out.txt', fname)
                    os.rename(fname,os.path.join(DIR_BC_FILES,fname))
                    fname='alignTable_%s.txt' % bcid
                    os.rename('alignTable.txt', fname)
                    os.rename(fname,os.path.join(DIR_BC_FILES,fname))

                    #move fastq, sff, bam, bai files
                    extlist = ['fastq','sff','bam','bam.bai']
                    for ext in extlist:
                        bcfile = "%s_%s_%s.%s" % (bcid,env['expName'], env['resultsName'],ext)
                        if os.path.isfile(bcfile):
                            os.rename(bcfile,os.path.join(DIR_BC_FILES,bcfile))
                except:
                    printtime('error renaming')
                    traceback.print_exc()
        #rename comprehensive results back to default names
        files = [ 'alignment.summary',
                  'alignmentQC_out.txt',
                  'alignTable.txt',
                ]
        for fname in files:
            if os.path.exists(fname + '.comprehensive'):
                os.rename(fname + '.comprehensive', fname)

        aggregate_alignment (DIR_BC_FILES,'barcodeList.txt')

def align_full_chip(libsff, libKey, tfKey, floworder, fastqName, align_full, DIR_BC_FILES, env, outputdir):
    #call whole chip metrics script (as os.system because all output to file)
    if os.path.exists('alignment.summary'):
        try:
            os.rename('alignment.summary', 'alignment.summary_%s' % datetime.datetime.now())
        except:
            printtime('error renaming')
            traceback.print_exc()
    align_full_chip_core(libsff, libKey, tfKey, floworder, fastqName, align_full, 1, True, True, False, DIR_BC_FILES, env, outputdir)

