#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

__version__ = filter(str.isdigit, "$Revision$")

from ion.utils.blockprocessing import printtime
from ion.utils.blockprocessing import isbadblock
from ion.utils.blockprocessing import MyConfigParser
from ion.utils import blockprocessing

import traceback
import datetime


import os
import numpy
import ConfigParser
import tempfile
import shutil
# matplotlib/numpy compatibility
os.environ['HOME'] = tempfile.mkdtemp()
from matplotlib import use
use("Agg")

import subprocess
import sys
import time
from collections import deque
from urlparse import urlunsplit

from ion.reports import libGraphs
from ion.reports.plotters import *
from ion.utils.aggregate_alignment import *


import math
from scipy import cluster, signal, linalg

import dateutil.parser



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
                it = libGraphs.QXXAlign(data,
                                        "Filtered "+qual+" Read Length",
                                        'Count',
                                        "Filtered "+qual+" Read Length",
                                        qualdict[qual],
                                        "Filtered_Alignments_"+qual+".png")
                it.plot()
            except:
                traceback.print_exc()

def createSAMMETA(fastqName, sampleName, libraryName, chipType, site_name, notes, start_time):

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

    except:
        printtime("ERROR: Could not read fastq file '%s'.  The ID for the SAM file could not be found." % fastqName)
        raise

    # sm - Sample. Use pool name where a pool is being sequenced.
    if sampleName:
        SAM_META['SM'] = sampleName
    else:
        SAM_META['SM'] = "unknown"
        printtime("WARNING: sample name not set. SM tag set to 'unknown'")

    # lb - library name
    if libraryName:
        SAM_META['LB'] = libraryName
    else:
        SAM_META['LB'] = "unknown"
        printtime("WARNING: library name not set. LB tag set to 'unknown'")

    # pu - the platform unit
    SAM_META['PU'] = "PGM/" + chipType.replace('"',"")

    SAM_META['PL'] = "IONTORRENT"

    # ds - the "notes", only the alphanumeric and space characters.
    SAM_META['DS'] = ''.join(ch for ch in notes if ch.isalnum() or ch == " ")

    # dt - the run date
    iso_exp_time = start_time

    #convert to ISO time
    iso_exp_time = dateutil.parser.parse(iso_exp_time)

    SAM_META['DT'] = iso_exp_time.isoformat()

    #the site name should be here, also remove spaces
    site_name = ''.join(ch for ch in site_name if ch.isalnum() )
    SAM_META['CN'] = site_name

    return SAM_META


def align_full_chip(
    SAM_META,
    libsff_path,
    align_full,
    graph_max_x,
    do_barcode,
    make_align_graphs,
    sam_parsed,
    bidirectional,
    DIR_BC_FILES,
    libraryName,
    flows,
    barcodeId,
    opts_extra,
    outputdir):

    printtime("sam_parsed is %s" % sam_parsed)

    #Now build the SAM meta data arg string
    aligner_opts_rg= '--aligner-opts-rg "'
    aligner_opts_extra = ''
    additional_aligner_opts = ''
    if sam_parsed:
        additional_aligner_opts += ' -p 1'
    if bidirectional:
        additional_aligner_opts += ' --bidirectional'
    if opts_extra:
        print '  found extra alignment options: "%s"' % opts_extra
        aligner_opts_extra = ' --aligner-opts-extra "'
        aligner_opts_extra += opts_extra + '"'
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
            flowsUsed = int(flows)
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
        com += " --input %s" % libsff_path
        com += " --genome %s" % libraryName
        com += " --max-plot-read-len %s" % graph_max_x
        com += " --align-all-reads"
        com += " %s" % (additional_aligner_opts)
        com += " %s %s" % (aligner_opts_rg,aligner_opts_extra)
        com += " >> ReportLog.html 2>&1"
    else:
        com = "alignmentQC.pl"
        com += " --logfile %s" % os.path.join(outputdir,"alignmentQC_out.txt")
        com += " --output-dir %s" % outputdir
        com += " --input %s" % libsff_path
        com += " --genome %s" % libraryName
        com += " --max-plot-read-len %s" % graph_max_x
        com += " %s" % (additional_aligner_opts)
        com += " %s %s" % (aligner_opts_rg,aligner_opts_extra)
        com += " >> ReportLog.html 2>&1"

    try:
        printtime("Alignment QC command line:\n%s" % com)
        retcode = subprocess.call(com, shell=True)
        blockprocessing.add_status("alignmentQC.pl", retcode)
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
    if barcodeId and do_barcode:
        printtime("Renaming non-barcoded alignment results to 'comprehensive'")
        files = [ 'alignment.summary',
                  'alignmentQC_out.txt',
                  'alignTable.txt',
                ]
        for fname in files:
            try:
                #if os.path.exists(fname):
                #   os.rename(fname, fname + ".comprehensive")
                shutil.copyfile(fname, fname + ".comprehensive")
            except:
                printtime('ERROR copying %s' % fname)
                traceback.print_exc()

        printtime("STARTING BARCODE ALIGNMENTS")
        
        barcodelist_path = 'barcodeList.txt'
        if not os.path.exists(barcodelist_path):
            barcodelist_path = '../barcodeList.txt'
        if not os.path.exists(barcodelist_path):
            barcodelist_path = '../../barcodeList.txt'
        if not os.path.exists(barcodelist_path):
            printtime('ERROR: barcodeList.txt not found')
        barcodeList = parse_bcfile(barcodelist_path)

        align_full = True
        top_dir = os.getcwd()
        try:
            os.chdir(DIR_BC_FILES)
            printtime('DEBUG changing to %s for barcodes alignment' % DIR_BC_FILES)
        except:
            printtime('ERROR missing %s folder' % DIR_BC_FILES)
            
        for bcid in (x['id_str'] for x in barcodeList):
            (head,tail) = os.path.split(libsff_path)
            sffName = os.path.join(head,"%s_%s" % (bcid, tail))
            if os.path.exists(sffName):
                printtime("Barcode processing for '%s': %s" % (bcid, sffName))
            else:
                printtime("No barcode SFF file found for '%s': %s" % (bcid, sffName))
                continue
            if (align_full):
                printtime("Align All Reads")
                #If a full align is forced add a '--align-all-reads' flag
                com = "alignmentQC.pl" 
                com += " --logfile %s" % os.path.join(outputdir,"alignmentQC_out.txt")
                com += " --output-dir %s" % outputdir
                com += " --input %s" % sffName
                com += " --genome %s" % libraryName
                com += " --max-plot-read-len %s" % graph_max_x
                com += " --align-all-reads"
                com += " %s" % (additional_aligner_opts)
                com += " %s %s" % (aligner_opts_rg, aligner_opts_extra)
                com += " >> ReportLog.html 2>&1" 
            else:
                printtime("Align Subset of Reads")
                com = "alignmentQC.pl" 
                com += " --logfile %s" % os.path.join(outputdir,"alignmentQC_out.txt")
                com += " --output-dir %s" % outputdir
                com += " --input %s" % sffName
                com += " --genome %s" % libraryName
                com += " --max-plot-read-len %s" % graph_max_x
                com += " %s" % (additional_aligner_opts)
                com += " %s %s" % (aligner_opts_rg, aligner_opts_extra)
                com += " >> ReportLog.html 2>&1"
            try:
                printtime("Alignment QC command line:\n%s" % com)
                retcode = subprocess.call(com, shell=True)
                blockprocessing.add_status("alignmentQC.pl", retcode)
                if retcode != 0:
                    printtime("alignmentQC failed, return code: %d" % retcode)
                    alignError = open("alignment.error", "a")
                    alignError.write(com)
                    alignError.write(': \nalignmentQC returned with error code: ')
                    alignError.write(str(retcode))
                    alignError.close()
            except:
                printtime('ERROR: Alignment Failed to start')
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
    #                os.rename(fname,os.path.join(DIR_BC_FILES,fname))
                    fname='alignmentQC_out_%s.txt' % bcid
                    os.rename('alignmentQC_out.txt', fname)
    #                os.rename(fname,os.path.join(DIR_BC_FILES,fname))
                    fname='alignTable_%s.txt' % bcid
                    os.rename('alignTable.txt', fname)
    #                os.rename(fname,os.path.join(DIR_BC_FILES,fname))
                    
                except:
                    printtime('error renaming')
                    traceback.print_exc()
                    
        os.chdir(top_dir)     

        #rename comprehensive results back to default names
        for fname in files:
            #if os.path.exists(fname + '.comprehensive'):
            #    os.rename(fname + '.comprehensive', fname)
            try:
                shutil.copyfile(fname + '.comprehensive', fname)
            except:
                printtime('ERROR copying %s' % fname + '.comprehensive')
                traceback.print_exc()
                
        aggregate_alignment (DIR_BC_FILES,barcodelist_path)


def alignment(libsff_path,
              fastq_path,
              align_full,
              DIR_BC_FILES,
              libraryName,
              sample,
              chipType,
              site_name,
              flows,
              notes,
              barcodeId,
              aligner_opts_extra,
              start_time,
              ALIGNMENT_RESULTS,
              bidirectional,
              sam_parsed):

    printtime("Attempt to align")

    do_barcode = True

    #TODO
    if os.path.exists('alignment.summary'):
        try:
            os.rename('alignment.summary', 'alignment.summary_%s' % datetime.datetime.now())
        except:
            printtime('error renaming')
            traceback.print_exc()

    try:
        sammeta = createSAMMETA(
            fastq_path,
            sample,
            libraryName,
            chipType,
            site_name,
            notes,
            start_time)

        align_full_chip(
            sammeta,
            libsff_path,
            align_full, 1, do_barcode, True, sam_parsed,
            bidirectional,
            DIR_BC_FILES,
            libraryName,
            flows,
            barcodeId,
            aligner_opts_extra,
            ALIGNMENT_RESULTS)

    except Exception:
        printtime("ERROR: Alignment Failed")
        alignError = open("alignment.error", "w")
        alignError.write(str(traceback.format_exc()))
        alignError.close()
        traceback.print_exc()

    printtime("**** Alignment completed ****")

def mergeAlignmentResults(dirs, env, ALIGNMENT_RESULTS):

    ############################################
    # Merge individual alignment.summary files #
    ############################################
    printtime("Merging individual alignment.summary files")

    config_out = ConfigParser.RawConfigParser()
    config_out.optionxform = str # don't convert to lowercase
    config_out.add_section('global')

    quallist = ['Q7', 'Q10', 'Q17', 'Q20', 'Q47']
    bplist = [50, 100, 150, 200, 250, 300, 350, 400]

    fixedkeys = [ 'Genome', 'Genome Version', 'Index Version', 'Genomesize' ]

    numberkeys = ['Total number of Reads',
                  'Filtered Mapped Bases in Q7 Alignments',
                  'Filtered Mapped Bases in Q10 Alignments',
                  'Filtered Mapped Bases in Q17 Alignments',
                  'Filtered Mapped Bases in Q20 Alignments',
                  'Filtered Mapped Bases in Q47 Alignments',
                  'Filtered Q7 Alignments',
                  'Filtered Q10 Alignments',
                  'Filtered Q17 Alignments',
                  'Filtered Q20 Alignments',
                  'Filtered Q47 Alignments']

    for q in quallist:
        for bp in bplist:
            numberkeys.append('Filtered %s%s Reads' % (bp, q))

    maxkeys = ['Filtered Q7 Longest Alignment',
               'Filtered Q10 Longest Alignment',
               'Filtered Q17 Longest Alignment',
               'Filtered Q20 Longest Alignment',
               'Filtered Q47 Longest Alignment']

    meankeys = ['Filtered Q7 Mean Alignment Length',
                'Filtered Q10 Mean Alignment Length',
                'Filtered Q17 Mean Alignment Length',
                'Filtered Q20 Mean Alignment Length',
                'Filtered Q47 Mean Alignment Length',
                'Filtered Q7 Coverage Percentage',
                'Filtered Q10 Coverage Percentage',
                'Filtered Q17 Coverage Percentage',
                'Filtered Q20 Coverage Percentage',
                'Filtered Q47 Coverage Percentage',
                'Filtered Q7 Mean Coverage Depth',
                'Filtered Q10 Mean Coverage Depth',
                'Filtered Q17 Mean Coverage Depth',
                'Filtered Q20 Mean Coverage Depth',
                'Filtered Q47 Mean Coverage Depth']

    # init
    for key in fixedkeys:
        value_out = 'unknown'
        config_out.set('global', key, value_out)
    for key in numberkeys:
        value_out = 0
        config_out.set('global', key, int(value_out))
    for key in maxkeys:
        value_out = 0
        config_out.set('global', key, int(value_out))
    for key in meankeys:
        value_out = 0
        config_out.set('global', key, float(value_out))

    config_in = MyConfigParser()
    config_in.optionxform = str # don't convert to lowercase
    for i,subdir in enumerate(dirs):
        if isbadblock(subdir, "Merging alignment.summary"):
            continue
        alignmentfile=os.path.join(subdir, 'alignment.summary')
        if os.path.exists(alignmentfile):
            config_in.read(os.path.join(alignmentfile))

            for key in numberkeys:
                value_in = config_in.get('global',key)
                value_out = config_out.get('global', key)
                config_out.set('global', key, int(value_in) + int(value_out))
            for key in maxkeys:
                value_in = config_in.get('global',key)
                value_out = config_out.get('global', key)
                config_out.set('global', key, max(int(value_in),int(value_out)))
            for key in fixedkeys:
                value_in = config_in.get('global',key)
                value_out = config_out.get('global',key)
                #todo
                config_out.set('global', key, value_in)
            for key in meankeys:
                value_in = config_in.get('global',key)
                value_out = config_out.get('global', key)
                config_out.set('global', key, float(value_out)+float(value_in)/len(dirs))

         #              'Filtered Q17 Mean Coverage Depth' = 
         #                  'Filtered Mapped Bases in Q17 Alignments' / 'Genomesize';

        else:
            printtime("ERROR: skipped %s" % alignmentfile)


    with open('alignment.summary.merged', 'wb') as configfile:
        config_out.write(configfile)

    r = subprocess.call(["ln", "-s", os.path.join(ALIGNMENT_RESULTS,"alignment.summary.merged"), os.path.join(ALIGNMENT_RESULTS,"alignment.summary")])

    #########################################
    # Merge individual alignTable.txt files #
    #########################################
    printtime("Merging individual alignTable.txt files")

    table = 0
    header = None
    for subdir in dirs:
        if isbadblock(subdir, "Merging alignTable.txt"):
            continue
        alignTableFile = os.path.join(subdir,'alignTable.txt')
        if os.path.exists(alignTableFile):
            if header is None:
                header = numpy.loadtxt(alignTableFile, dtype='string', comments='#')
            table += numpy.loadtxt(alignTableFile, dtype='int', comments='#',skiprows=1)
        else:
            printtime("ERROR: skipped %s" % alignTableFile)
    #fix first column
    table[:,0] = (header[1:,0])
    f_handle = open('alignTable.txt.merged', 'w')
    numpy.savetxt(f_handle, header[0][None], fmt='%s', delimiter='\t')
    numpy.savetxt(f_handle, table, fmt='%i', delimiter='\t')
    f_handle.close()

    r = subprocess.call(["ln", "-s", os.path.join(ALIGNMENT_RESULTS,"alignTable.txt.merged"), os.path.join(ALIGNMENT_RESULTS,"alignTable.txt")])


    #############################################
    # Merge alignment.summary (json)            #
    #############################################
    printtime("Merging  alignment.summary (json)")
    try:
        cmd = 'merge_alignment.summary.py'
        for subdir in dirs:
            if isbadblock(subdir, "Merging alignment.summary (json)"):
                continue
            alignmentfile=os.path.join(subdir, 'alignment.summary')
            if os.path.exists(alignmentfile):
                cmd = cmd + ' %s' % alignmentfile
            else:
                printtime("ERROR: skipped %s" % alignmentfile)
        cmd = cmd + ' > alignment.summary.json'
        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd,shell=True)
    except:
        printtime("Merging alignment.summary (json) failed")


    #############################################
    # Merge alignTable.txt (json)               #
    #############################################
    printtime("Merging alignTable.txt (json)")
    try:
        cmd = 'merge_alignTable.py'
        for subdir in dirs:
            if isbadblock(subdir, "Merging alignTable.txt (json)"):
                continue
            alignstatsfile=os.path.join(subdir, 'alignTable.txt')
            if os.path.exists(alignstatsfile):
                cmd = cmd + ' %s' % alignstatsfile
            else:
                printtime("ERROR: skipped %s" % alignstatsfile)
        cmd = cmd + ' > alignTable.txt.json'
        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd,shell=True)
    except:
        printtime("Merging alignTable.txt (json) failed")


    #############################################
    # Merge individual block bam files   #
    #############################################
    printtime("Merging bam files")
    try:
#        cmd = 'picard-tools MergeSamFiles'
        cmd = 'java -Xmx8g -jar /opt/picard/picard-tools-current/MergeSamFiles.jar'
        for subdir in dirs:
            if isbadblock(subdir, "Merging bam files"):
                continue
            bamfile = os.path.join(ALIGNMENT_RESULTS, subdir, "rawlib.bam")
            if os.path.exists(bamfile):
                cmd = cmd + ' I=%s' % bamfile
            else:
                printtime("ERROR: skipped %s" % bamfile)
        cmd = cmd + ' O=%s/%s_%s.bam' % (ALIGNMENT_RESULTS, env['expName'], env['resultsName'])
        cmd = cmd + ' ASSUME_SORTED=true'
        cmd = cmd + ' CREATE_INDEX=true'
        cmd = cmd + ' USE_THREADING=true'
        cmd = cmd + ' VALIDATION_STRINGENCY=LENIENT'
        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd,shell=True)
    except:
        printtime("bam file merge failed")

    try:
        srcbaifilepath = '%s/%s_%s.bai' % (ALIGNMENT_RESULTS, env['expName'], env['resultsName'])
        dstbaifilepath = '%s/%s_%s.bam.bai' % (ALIGNMENT_RESULTS, env['expName'], env['resultsName'])
        if os.path.exists(srcbaifilepath):
            os.rename(srcbaifilepath, dstbaifilepath)
        else:
            printtime("ERROR: %s doesn't exists" % srcbaifilepath)
    except:
        traceback.print_exc()

    #remove symbolic links
    os.remove("alignment.summary")
    os.remove("alignTable.txt")

    ##################################################
    #Call alignStats on merged bam file              #
    ##################################################
    printtime("Call alignStats on merged bam file")

    try:
        cmd = "alignStats -i %s/%s_%s.bam" % (ALIGNMENT_RESULTS, env['expName'], env['resultsName'])
        cmd = cmd + " -g /results/referenceLibrary/%s/%s/%s.info.txt" % (env["tmap_version"],env["libraryName"], env["libraryName"])
        cmd = cmd + " -n 12 -l 20 -m 400 -q 7,10,17,20,47 -s 0 -a alignTable.txt"
        cmd = cmd + " --outputDir %s" % ALIGNMENT_RESULTS
        cmd = cmd + " 2>> " + os.path.join(ALIGNMENT_RESULTS, "alignStats_out.txt")
        printtime("DEBUG: Calling '%s'" % cmd)
        os.system(cmd)
    except:
        printtime("alignStats failed")
