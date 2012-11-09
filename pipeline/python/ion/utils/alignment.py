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
use("Agg",warn=False)

import subprocess
import sys
import time
import json
from collections import deque
from urlparse import urlunsplit

from ion.reports import libGraphs
from ion.reports.plotters import *
from ion.utils.aggregate_alignment import *
from ion.reports import  base_error_plot

import math

import dateutil.parser



def makeAlignGraphs():
    ###################################################
    # Make histograms graphs from tmap output         #
    ###################################################
    qualdict = {}
    qualdict['Q10'] = 'red'
    qualdict['Q17'] = 'yellow'
    qualdict['Q20'] = 'green'
    qualdict['Q30'] = 'blue'
    qualdict['Q47'] = 'purple'
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

def createSAMMETA(runID, sampleName, libraryName, chipType, site_name, notes, start_time):

    #collect all the meta data for the SAM file
    SAM_META = {}

    # id
    if runID:
        SAM_META['ID'] = runID
    else:
        SAM_META['ID'] = 'FALSE'
        printtime("WARNING: runID not set. ID tag set to 'FALSE'")

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
    make_align_graphs,
    sam_parsed,
    bidirectional,
    libraryName,
    flows,
    opts_extra,
    mark_duplicates,
    outputdir,
    outBaseName=''
    ):

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
    try:
        com = "alignmentQC.pl"
        if (align_full):
            com += " --align-all-reads"
        com += " --logfile %s" % os.path.join(outputdir,"alignmentQC_out.txt")
        com += " --output-dir %s" % outputdir
        com += " --input %s" % libsff_path
        com += " --genome %s" % libraryName
        com += " --max-plot-read-len %s" % graph_max_x
        if len(outBaseName) > 1:
            com += " --out-base-name %s" % outBaseName
        else:
            com += " %s" % (additional_aligner_opts)
            com += " %s %s" % (aligner_opts_rg,aligner_opts_extra)
        if mark_duplicates:
            com += " --mark-duplicates"
        com += " >> %s 2>&1" % os.path.join(outputdir, 'alignment.log')

        printtime("Alignment QC command line:\n%s" % com)
        retcode = subprocess.call(com, shell=True)
        blockprocessing.add_status("alignmentQC.pl", retcode)
        if retcode != 0:
            printtime("alignmentQC failed, return code: %d" % retcode)
            alignError = open("alignment.error", "w")
            alignError.write(com)
            alignError.write('alignmentQC returned with error code: ')
            alignError.write(str(retcode))
            alignError.close()
    except OSError:
        printtime('ERROR: Alignment Failed to start')
        alignError = open("alignment.error", "w")
        alignError.write(str(traceback.format_exc()))
        alignError.close()
        traceback.print_exc()
    if make_align_graphs:
        makeAlignGraphs()


def alignment(libsff_path,
              runID,
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
              mark_duplicates,
              start_time,
              ALIGNMENT_RESULTS,
              bidirectional,
              sam_parsed,
              outBaseName=''
              ):

    printtime("Attempt to align")


    #TODO
    if os.path.exists('alignment.summary'):
        try:
            os.rename('alignment.summary', 'alignment.summary_%s' % datetime.datetime.now())
        except:
            printtime('error renaming')
            traceback.print_exc()

    try:
        sammeta = createSAMMETA(
            runID,
            sample,
            libraryName,
            chipType,
            site_name,
            notes,
            start_time)

        align_full_chip(
            sammeta,
            libsff_path,
            align_full, 1, True, sam_parsed,
            bidirectional,
            libraryName,
            flows,
            aligner_opts_extra,
            mark_duplicates,
            ALIGNMENT_RESULTS,
            outBaseName)

        if barcodeId != '':
            align_barcodes(
                sammeta,
                libsff_path,
                align_full,
                sam_parsed,
                bidirectional,
                libraryName,
                DIR_BC_FILES,
                flows,
                aligner_opts_extra,
                mark_duplicates,
                ALIGNMENT_RESULTS,
                outBaseName)

    except Exception:
        printtime("ERROR: Alignment Failed")
        alignError = open("alignment.error", "w")
        alignError.write(str(traceback.format_exc()))
        alignError.close()
        traceback.print_exc()

    printtime("**** Alignment completed ****")


def align_barcodes(
            sammeta,
            libsff_path,
            align_full,
            sam_parsed,
            bidirectional,
            libraryName,
            DIR_BC_FILES,
            flows,
            aligner_opts_extra,
            mark_duplicates,
            ALIGNMENT_RESULTS,
            outBaseName=''
            ):
    printtime("Renaming non-barcoded alignment results to 'comprehensive'")
    files = [ 'alignment.summary',
              'alignmentQC_out.txt',
              'alignTable.txt',
            ]
    for fname in files:
        if os.path.exists(fname):
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
        barcodelist_path = '../../../barcodeList.txt'
    if not os.path.exists(barcodelist_path):
        barcodelist_path = '../../../../barcodeList.txt'
    if not os.path.exists(barcodelist_path):
        printtime('ERROR: barcodeList.txt not found')
    barcodeList = parse_bcfile(barcodelist_path)

    align_full = True

    (head,tail) = os.path.split(libsff_path)
    for bcid in (x['id_str'] for x in barcodeList):
        sffName = os.path.join(DIR_BC_FILES,"%s_%s" % (bcid, tail))
        print "sffName: "+sffName
        if os.path.exists(sffName):
            printtime("Barcode processing for '%s': %s" % (bcid, sffName))
        else:
            printtime("No barcode SFF file found for '%s': %s" % (bcid, sffName))
            continue

        align_full_chip(
                sammeta,
                sffName,
                align_full,
                1,
                True,
                sam_parsed,
                bidirectional,
                libraryName,
                flows,
                aligner_opts_extra,
                mark_duplicates,
                ALIGNMENT_RESULTS,
                outBaseName)

        #rename each output file based on barcode found in fastq filename
        #but ignore the comprehensive fastq output files
        printtime("Barcode processing, rename")
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

    #rename comprehensive results back to default names
    for fname in files:
        if os.path.exists(fname + '.comprehensive'):
            #    os.rename(fname + '.comprehensive', fname)
            try:
                shutil.copyfile(fname + '.comprehensive', fname)
            except:
                printtime('ERROR copying %s' % fname + '.comprehensive')
                traceback.print_exc()

    printtime("Barcode processing, aggreagate")
    aggregate_alignment ("./",barcodelist_path)



def alignment_unmapped_bam(
        BASECALLER_RESULTS,
        ALIGNMENT_RESULTS,
        align_full,
        libraryName,
        flows,
        aligner_opts_extra,
        mark_duplicates,
        bidirectional,
        sam_parsed):

    printtime("Attempt to align")

    datasets_basecaller_path = os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json")

    if not os.path.exists(datasets_basecaller_path):
        printtime("ERROR: %s does not exist" % datasets_basecaller_path)
        open('badblock.txt', 'w').close()
        return
    datasets_basecaller = {}
    try:
        f = open(datasets_basecaller_path,'r')
        datasets_basecaller = json.load(f);
        f.close()
    except:
        printtime("ERROR: problem parsing %s" % datasets_basecaller_path)
        traceback.print_exc()
        open('badblock.txt', 'w').close()
        return

    # establish the read-length histogram range by using the simple rule:
    # 0.7 * num_flows rounded up to a next multiple of 50
    try:
        graph_max_x = int(50 * math.ceil(0.014 * int(flows)))
    except:
        graph_max_x = 400

    

    input_prefix_list = []

    for dataset in datasets_basecaller["datasets"]:
        if not os.path.exists(os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])):
            continue
    
        #-----------------------------------
        # DEFAULT SINGLE SFF/FASTQ BEHAVIOR - (Runs for barcoded runs too)
        #-----------------------------------
        try:
            com = "alignmentQC.pl"
            if align_full:
                com += " --align-all-reads"
            com += " --logfile %s" % os.path.join(ALIGNMENT_RESULTS,dataset['file_prefix']+'.alignmentQC_out.txt')
            com += " --output-dir %s" % ALIGNMENT_RESULTS
            com += " --out-base-name %s" % dataset['file_prefix']
            com += " --input %s" % os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])
            com += " --genome %s" % libraryName
            #com += " --max-plot-read-len %s" % str(int(graph_max_x))
            com += " --max-plot-read-len %s" % str(int(400))
            if sam_parsed:
                com  += ' -p 1'
            if bidirectional:
                com += ' --bidirectional'
            if aligner_opts_extra:
                com += ' --aligner-opts-extra "%s"' % aligner_opts_extra
            if mark_duplicates:
                com += ' --mark-duplicates'
            com += " >> %s 2>&1" % os.path.join(ALIGNMENT_RESULTS, 'alignment.log')


            printtime("Alignment QC command line:\n%s" % com)
            retcode = subprocess.call(com, shell=True)
            blockprocessing.add_status("alignmentQC.pl", retcode)
            if retcode != 0:
                printtime("alignmentQC failed, return code: %d" % retcode)
                alignError = open("alignment.error", "w")
                alignError.write(com)
                alignError.write('alignmentQC returned with error code: ')
                alignError.write(str(retcode))
                alignError.close()
        except OSError:
            printtime('ERROR: Alignment Failed to start')
            alignError = open("alignment.error", "w")
            alignError.write(str(traceback.format_exc()))
            alignError.close()
            traceback.print_exc()
            
        printtime("Barcode processing, rename")
        if os.path.exists(os.path.join(ALIGNMENT_RESULTS,'alignment.summary')):
            try:
                fname=os.path.join(ALIGNMENT_RESULTS,dataset['file_prefix']+'.alignment.summary')
                os.rename(os.path.join(ALIGNMENT_RESULTS,'alignment.summary'), fname)
                fname=os.path.join(ALIGNMENT_RESULTS,dataset['file_prefix']+'.alignTable.txt')
                os.rename(os.path.join(ALIGNMENT_RESULTS,'alignStats_err.txt'), fname)
            except:
                printtime('error renaming')
                traceback.print_exc()
        

    alignment_post_processing(BASECALLER_RESULTS, ALIGNMENT_RESULTS, flows, mark_duplicates, False)

    printtime("**** Alignment completed ****")



def alignment_post_processing(
        BASECALLER_RESULTS,
        ALIGNMENT_RESULTS,
        flows,
        mark_duplicates,
        force_alignstats):


    datasets_basecaller = {}
    try:
        f = open(os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json"),'r')
        datasets_basecaller = json.load(f);
        f.close()
    except:
        printtime("ERROR: problem parsing %s" % os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json"))
        traceback.print_exc()
        return

    try:
        graph_max_x = int(50 * math.ceil(0.014 * int(flows)))
    except:
        graph_max_x = 400

    

    input_prefix_list = []

    for dataset in datasets_basecaller["datasets"]:
        if not os.path.exists(os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])):
            continue

        printtime("Barcode processing, rename")
        src = os.path.join(ALIGNMENT_RESULTS,dataset['file_prefix']+'.alignment.summary')
        if os.path.exists(src):
            input_prefix_list.append(os.path.join(ALIGNMENT_RESULTS,dataset['file_prefix']+'.'))
            #terrible hack to make aggregate_alignment happy
            X_name = 'nomatch'
            read_group = dataset['read_groups'][0]
            if 'barcode_name' in datasets_basecaller['read_groups'][read_group]:
                X_name = datasets_basecaller['read_groups'][read_group]['barcode_name']
            dst = os.path.join(ALIGNMENT_RESULTS, 'alignment_%s.summary' % X_name)
            try:
                os.symlink(os.path.relpath(src,os.path.dirname(dst)),dst)
            except:
                printtime("ERROR: Unable to symlink '%s' to '%s'" % (src, dst))

        printtime("Creating legacy name links")
        if dataset.has_key('legacy_prefix'):
            link_src = [
                os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.bam'),
                os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.bam.bai')]
            link_dst = [
                os.path.join(ALIGNMENT_RESULTS, os.path.basename(dataset['legacy_prefix'])+'.bam'),
                os.path.join(ALIGNMENT_RESULTS, os.path.basename(dataset['legacy_prefix'])+'.bam.bai')]
            for (src,dst) in zip(link_src,link_dst):
                try:
                    os.symlink(os.path.relpath(src,os.path.dirname(dst)),dst)
                except:
                    printtime("ERROR: Unable to symlink '%s' to '%s'" % (src, dst))

    # Special legacy post-processing.
    # Generate merged rawlib.basecaller.bam and rawlib.sff on barcoded runs

    composite_bam_filename = os.path.join(ALIGNMENT_RESULTS,'rawlib.bam')
    if not os.path.exists(composite_bam_filename):

        bam_file_list = []
        for dataset in datasets_basecaller["datasets"]:
            bam_name = os.path.join(ALIGNMENT_RESULTS,os.path.basename(dataset['file_prefix'])+'.bam')
            if os.path.exists(bam_name):
                bam_file_list.append(bam_name)

        blockprocessing.merge_bam_files(bam_file_list,composite_bam_filename,composite_bam_filename+'.bai',mark_duplicates)
        force_alignstats = True

    if force_alignstats:        
        ## Generate data for error plot for barcoded run from composite bam
        printtime("Call alignStats to generate raw accuracy")
        try:
            cmd = "alignStats"
            cmd += " -n 12"
            cmd += " --alignSummaryFile alignStats_err.txt"
            cmd += " --alignSummaryJsonFile alignStats_err.json"
            cmd += " --alignSummaryMinLen  1"
            #cmd += " --alignSummaryMaxLen  %s" % str(int(graph_max_x))
            cmd += " --alignSummaryMaxLen  %s" % str(int(400))
            cmd += " --alignSummaryLenStep 1"
            cmd += " --alignSummaryMaxErr  10"
            cmd += " --infile %s" % composite_bam_filename
            cmd = cmd + " --outputDir %s" % ALIGNMENT_RESULTS
            printtime("DEBUG: Calling '%s'" % cmd)
            os.system(cmd)
        except:
            printtime("alignStats failed")


    mergeAlignStatsResults(input_prefix_list,ALIGNMENT_RESULTS+"/")

    try:
        base_error_plot.generate_base_error_plot(
            os.path.join(ALIGNMENT_RESULTS,'alignStats_err.json'),
            os.path.join(ALIGNMENT_RESULTS,'base_error_plot.png'),int(graph_max_x))
        base_error_plot.generate_alignment_rate_plot(
            os.path.join(ALIGNMENT_RESULTS,'alignStats_err.json'),
            os.path.join(BASECALLER_RESULTS,'readLen.txt'),
            os.path.join(ALIGNMENT_RESULTS,'alignment_rate_plot.png'),int(graph_max_x))

        # Create aligned histogram plot
        
        # Create AQ20 plot
        
        printtime("Base error plot has been created successfully")
    except:
        printtime("ERROR: Failed to generate base error plot")
        traceback.print_exc()

    # Generate alignment_barcode_summary.csv
    barcodelist_path = 'barcodeList.txt'
    if not os.path.exists(barcodelist_path):
        barcodelist_path = '../barcodeList.txt'
    if not os.path.exists(barcodelist_path):
        barcodelist_path = '../../barcodeList.txt'
    if not os.path.exists(barcodelist_path):
        barcodelist_path = '../../../barcodeList.txt'
    if not os.path.exists(barcodelist_path):
        barcodelist_path = '../../../../barcodeList.txt'
    if os.path.exists(barcodelist_path):
        printtime("Barcode processing, aggregate")
        aggregate_alignment ("./",barcodelist_path)

    # These graphs are likely obsolete
    makeAlignGraphs()





def mergeAlignStatsResults(input_prefix_list,output_prefix):

    ############################################
    # Merge individual alignment.summary files #
    ############################################
    printtime("Merging individual alignment.summary files")

    config_out = ConfigParser.RawConfigParser()
    config_out.optionxform = str # don't convert to lowercase
    config_out.add_section('global')

    quallist = ['Q7', 'Q10', 'Q17', 'Q20', 'Q30', 'Q47']
    bplist = [50, 100, 150, 200, 250, 300, 350, 400]

    fixedkeys = [ 'Genome', 'Genome Version', 'Index Version', 'Genomesize' ]

    numberkeys = ['Total number of Reads',
                  'Filtered Mapped Bases in Q7 Alignments',
                  'Filtered Mapped Bases in Q10 Alignments',
                  'Filtered Mapped Bases in Q17 Alignments',
                  'Filtered Mapped Bases in Q20 Alignments',
                  'Filtered Mapped Bases in Q30 Alignments',
                  'Filtered Mapped Bases in Q47 Alignments',
                  'Filtered Q7 Alignments',
                  'Filtered Q10 Alignments',
                  'Filtered Q17 Alignments',
                  'Filtered Q20 Alignments',
                  'Filtered Q30 Alignments',
                  'Filtered Q47 Alignments']

    for q in quallist:
        for bp in bplist:
            numberkeys.append('Filtered %s%s Reads' % (bp, q))

    maxkeys = ['Filtered Q7 Longest Alignment',
               'Filtered Q10 Longest Alignment',
               'Filtered Q17 Longest Alignment',
               'Filtered Q20 Longest Alignment',
               'Filtered Q30 Longest Alignment',
               'Filtered Q47 Longest Alignment']

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

    config_in = MyConfigParser()
    config_in.optionxform = str # don't convert to lowercase
    for input_prefix in input_prefix_list:
        alignmentfile = input_prefix + 'alignment.summary'
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

        else:
            printtime("ERROR: skipped %s" % alignmentfile)

    # Regenerate trickier alignment.summary metrics

    for qual in quallist:
        try:
            q_bases = config_out.get('global','Filtered Mapped Bases in %s Alignments' % qual)
            q_reads = config_out.get('global','Filtered %s Alignments' % qual)

            q_readlen = 0
            if q_reads > 0:
                q_readlen = q_bases / q_reads
            config_out.set('global','Filtered %s Mean Alignment Length' % qual, q_readlen)

            genomesize = float(config_out.get('global','Genomesize'))
            q_coverage = 0.0
            if genomesize > 0:
                q_coverage = q_bases / genomesize
            config_out.set('global','Filtered %s Mean Coverage Depth' % qual, '%1.1f' % q_coverage)

            # Not mergeable at this point
            config_out.set('global','Filtered %s Coverage Percentage' % qual, 'N/A')
           
        except:
            pass
    

    with open(output_prefix + 'alignment.summary', 'wb') as configfile:
        config_out.write(configfile)


    #########################################
    # Merge individual alignTable.txt files #
    #########################################
    printtime("Merging individual alignTable.txt files")
    
    table = 0
    header = None
    for input_prefix in input_prefix_list:
        alignTableFile = input_prefix + 'alignTable.txt'
        if os.path.exists(alignTableFile):
            if header is None:
                header = numpy.loadtxt(alignTableFile, dtype='string', comments='#')
            table += numpy.loadtxt(alignTableFile, dtype='int', comments='#',skiprows=1)
        else:
            printtime("ERROR: skipped %s" % alignTableFile)
    #fix first column
    if header is not None:
        table[:,0] = (header[1:,0])
        f_handle = open(output_prefix+ 'alignTable.txt', 'w')
        numpy.savetxt(f_handle, header[0][None], fmt='%s', delimiter='\t')
        numpy.savetxt(f_handle, table, fmt='%i', delimiter='\t')
        f_handle.close()




def merge_alignment_stats(dirs, BASECALLER_RESULTS, ALIGNMENT_RESULTS, flows):
    
    datasets_json = {}
    try:
        f = open(os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json"),'r')
        datasets_json = json.load(f);
        f.close()
    except:
        printtime("ERROR: problem parsing %s" % os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json"))
        traceback.print_exc()
        return
    
    for dataset in datasets_json['datasets']:

        # What needs merging:
        #  - alignment.summary
        #  - alignTable.txt
        # Some time in the future:
        #  - alignStats_err.json

        # Merge alignStats metrics
        try:
            input_prefix_list = [os.path.join(dir,ALIGNMENT_RESULTS, dataset['file_prefix']+'.') for dir in dirs]
            input_prefix_list = [prefix for prefix in input_prefix_list if os.path.exists(prefix+'alignment.summary')]
            composite_prefix = os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.')
            if input_prefix_list:
                mergeAlignStatsResults(input_prefix_list,composite_prefix)
            else:
                printtime("Nothing to merge: "+dataset['file_prefix'])
        except:
            printtime("ERROR: merging %s stats unsuccessful" % (dataset['file_prefix']+'.bam'))
    

    datasets_basecaller = {}
    try:
        f = open(os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json"),'r')
        datasets_basecaller = json.load(f);
        f.close()
    except:
        printtime("ERROR: problem parsing %s" % os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json"))
        traceback.print_exc()
        return

    try:
        graph_max_x = int(50 * math.ceil(0.014 * int(flows)))
    except:
        graph_max_x = 400

    

    input_prefix_list = []

    for dataset in datasets_basecaller["datasets"]:
        printtime("Barcode processing, rename")
        src = os.path.join(ALIGNMENT_RESULTS,dataset['file_prefix']+'.alignment.summary')
        if os.path.exists(src):
            input_prefix_list.append(os.path.join(ALIGNMENT_RESULTS,dataset['file_prefix']+'.'))
            #terrible hack to make aggregate_alignment happy
            X_name = 'nomatch'
            read_group = dataset['read_groups'][0]
            if 'barcode_name' in datasets_basecaller['read_groups'][read_group]:
                X_name = datasets_basecaller['read_groups'][read_group]['barcode_name']
            dst = os.path.join(ALIGNMENT_RESULTS, 'alignment_%s.summary' % X_name)
            try:
                os.symlink(os.path.relpath(src,os.path.dirname(dst)),dst)
            except:
                printtime("ERROR: Unable to symlink '%s' to '%s'" % (src, dst))


    # Merge alignStats_err.json right here!

    merged_align_stats = {}
    align_stats_num_bases = 400
    for dir in dirs:
        current_align_stats = {}
        try:
            f = open(os.path.join(dir,ALIGNMENT_RESULTS,'alignStats_err.json'),'r')
            current_align_stats = json.load(f);
            f.close()
        except:
            printtime("Merge alignStats_err.json: skipping %s" % os.path.join(dir,ALIGNMENT_RESULTS,'alignStats_err.json'))
            continue
        
        if not merged_align_stats:
            merged_align_stats = current_align_stats
            align_stats_num_bases = len(merged_align_stats.get("read_length",[]))
            continue
        
        for idx in range(align_stats_num_bases):
            merged_align_stats['nread'][idx] += current_align_stats['nread'][idx]
            merged_align_stats['unaligned'][idx] += current_align_stats['unaligned'][idx]
            merged_align_stats['filtered'][idx] += current_align_stats['filtered'][idx]
            merged_align_stats['clipped'][idx] += current_align_stats['clipped'][idx]
            merged_align_stats['aligned'][idx] += current_align_stats['aligned'][idx]
            merged_align_stats['n_err_at_position'][idx] += current_align_stats['n_err_at_position'][idx]
            merged_align_stats['cum_aligned'][idx] += current_align_stats['cum_aligned'][idx]
            merged_align_stats['cum_err_at_position'][idx] += current_align_stats['cum_err_at_position'][idx]

        merged_align_stats['accuracy_total_bases'] += current_align_stats['accuracy_total_bases']
        merged_align_stats['accuracy_total_errors'] += current_align_stats['accuracy_total_errors']
        merged_align_stats['total_mapped_target_bases'] += current_align_stats['total_mapped_target_bases']
        merged_align_stats['total_mapped_reads'] += current_align_stats['total_mapped_reads']
            
        
    try:
        f = open(os.path.join(ALIGNMENT_RESULTS,'alignStats_err.json'),"w")
        json.dump(merged_align_stats, f, indent=4)
        f.close()
    except:
        printtime("ERROR; Failed to write merged alignStats_err.json")
        traceback.print_exc()
        
        
        
    mergeAlignStatsResults(input_prefix_list,ALIGNMENT_RESULTS+"/")

    try:
        base_error_plot.generate_base_error_plot(
            os.path.join(ALIGNMENT_RESULTS,'alignStats_err.json'),
            os.path.join(ALIGNMENT_RESULTS,'base_error_plot.png'),int(graph_max_x))
        base_error_plot.generate_alignment_rate_plot(
            os.path.join(ALIGNMENT_RESULTS,'alignStats_err.json'),
            os.path.join(BASECALLER_RESULTS,'readLen.txt'),
            os.path.join(ALIGNMENT_RESULTS,'alignment_rate_plot.png'),int(graph_max_x))

        
        printtime("Base error plot has been created successfully")
    except:
        printtime("ERROR: Failed to generate base error plot")
        traceback.print_exc()

    # Generate alignment_barcode_summary.csv
    barcodelist_path = 'barcodeList.txt'
    if not os.path.exists(barcodelist_path):
        barcodelist_path = '../barcodeList.txt'
    if not os.path.exists(barcodelist_path):
        barcodelist_path = '../../barcodeList.txt'
    if not os.path.exists(barcodelist_path):
        barcodelist_path = '../../../barcodeList.txt'
    if not os.path.exists(barcodelist_path):
        barcodelist_path = '../../../../barcodeList.txt'
    if os.path.exists(barcodelist_path):
        printtime("Barcode processing, aggregate")
        aggregate_alignment ("./",barcodelist_path)




def merge_alignment_bigdata(dirs, BASECALLER_RESULTS, ALIGNMENT_RESULTS, mark_duplicates):
    
    datasets_json = {}
    try:
        f = open(os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json"),'r')
        datasets_json = json.load(f);
        f.close()
    except:
        printtime("ERROR: problem parsing %s" % os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json"))
        traceback.print_exc()
        return
    
    for dataset in datasets_json['datasets']:
        # Merge BAMs
        try:
            block_bam_list = [os.path.join(dir,ALIGNMENT_RESULTS, dataset['file_prefix']+'.bam') for dir in dirs]
            block_bam_list = [block_bam_filename for block_bam_filename in block_bam_list if os.path.exists(block_bam_filename)]
            composite_bam_filename = os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.bam')
            if block_bam_list:
                blockprocessing.merge_bam_files(block_bam_list,composite_bam_filename,composite_bam_filename+'.bai',mark_duplicates)
        except:
            printtime("ERROR: merging %s unsuccessful" % (dataset['file_prefix']+'.bam'))

        printtime("Creating legacy name links")
        if dataset.has_key('legacy_prefix'):
            link_src = [
                os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.bam'),
                os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.bam.bai')]
            link_dst = [
                os.path.join(ALIGNMENT_RESULTS, os.path.basename(dataset['legacy_prefix'])+'.bam'),
                os.path.join(ALIGNMENT_RESULTS, os.path.basename(dataset['legacy_prefix'])+'.bam.bai')]
            for (src,dst) in zip(link_src,link_dst):
                try:
                    os.symlink(os.path.relpath(src,os.path.dirname(dst)),dst)
                except:
                    printtime("ERROR: Unable to symlink '%s' to '%s'" % (src, dst))


