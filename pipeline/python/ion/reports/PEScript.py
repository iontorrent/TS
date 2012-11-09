# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import xmlrpclib
import argparse
import subprocess
import sys
import os
import socket
import shutil
import traceback
import datetime
import json

from ion.utils import blockprocessing
from ion.utils import explogparser
from ion.utils import sigproc
from ion.utils import basecaller
from ion.utils import alignment
from ion.utils.compress import make_zip
from ion.utils.blockprocessing import printtime
from ion.utils.blockprocessing import parse_metrics

from urlparse import urlunsplit


# import /etc/torrentserver/cluster_settings.py, provides JOBSERVER_HOST, JOBSERVER_PORT
import sys
sys.path.append('/etc')
from torrentserver.cluster_settings import *


def copyfile(filename):
    try:
        (head, tail) = os.path.split(filename)
        try:
            os.mkdir(os.path.join(outdir,head))
        except:
            pass

        fwd_filename = os.path.join(forwarddir,filename)
        if os.path.exists(fwd_filename):
            shutil.copy(fwd_filename, os.path.join(outdir,head,"fwd_"+tail))
        else:
            printtime("ERROR: %s doesn't exist" % fwd_filename)

        rev_filename = os.path.join(reversedir,filename)
        if os.path.exists(rev_filename):
            shutil.copy(rev_filename, os.path.join(outdir,head,"rev_"+tail))
        else:
            printtime("ERROR: %s doesn't exist" % rev_filename)
    except:
        printtime('ERROR: copying file %s' % filename)
        traceback.print_exc()


def PE_set_value(key):
    if forward_env[key] == reverse_env[key]:
        return forward_env[key]
    else:
        printtime("ERROR forward run %s (%s) doesn't match reverse run %s (%s)"
                  % (key, forward_env[key], key, reverse_env[key]) )
        return 'unknown'


def make_expDat(fwd, rev):
    fwd_f = open(fwd,'r')
    fwd_dat = fwd_f.readlines()
    rev_f = open(rev,'r')
    rev_dat = rev_f.readlines()
        
    meta = "Run Name = %s\n" % (env['expName'])
    meta+= "Analysis Date = %s\n" % (datetime.date.today())
    meta+= "Analysis Name = %s\n" % (env['resultsName'])
    
    fwd_uniq = ''
    rev_uniq = ''        
    for f,r in zip(fwd_dat,rev_dat):
      if f == r and 'Analysis Date' not in f:        
        meta+= f
      else:
        fwd_uniq += f
        rev_uniq += r          
        
    f = open('expMeta.dat','w')
    f.write(meta)    
    f.close()
    f = open('fwd_expMeta.dat.uniq','w')
    f.write(fwd_uniq)    
    f.close()  
    f = open('rev_expMeta.dat.uniq','w')
    f.write(rev_uniq)    
    f.close() 
      
    fwd_f.close()
    rev_f.close()


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('-f', '--forward', required=False, dest='forwarddir', default='.', help='forward run report')
    parser.add_argument('-r', '--reverse', required=False, dest='reversedir', default='.', help='reverse run report')
    parser.add_argument('-o', '--output', required=False, dest='outdir', default='.', help='output directory')
    parser.add_argument('param', default='paramfile', help='param file')

    args = parser.parse_args()

    args.verbose = True
    if args.verbose:
        print "args:",args

    blockprocessing.printheader()
    env,warn = explogparser.getparameter()
    print warn

    #-------------------------------------------------------------
    # Update Report Status to 'Started'
    #-------------------------------------------------------------
    try:
        jobserver = xmlrpclib.ServerProxy("http://%s:%d" % (JOBSERVER_HOST, JOBSERVER_PORT), verbose=False, allow_none=True)
        debugging_cwd = os.getcwd()
    except:
        traceback.print_exc()
    
    def set_result_status(status):
        try:
            primary_key_file = os.path.join(os.getcwd(),'primary.key')
            jobserver.updatestatus(primary_key_file, status, True)
            printtime("PEStatus %s\tpid %d\tpk file %s started in %s" % 
                (status, os.getpid(), primary_key_file, debugging_cwd))
        except:
            traceback.print_exc()

    set_result_status('Started')


    if args.verbose:
        print env['pe_forward']
        print env['pe_reverse']

    pluginbasefolder = 'plugin_out'

    blockprocessing.initTLReport(pluginbasefolder)

    blockprocessing.write_version()

    if env['pe_forward'] == '' or env['pe_reverse'] == '':
        sys.exit(1)

    forwarddir = env['pe_forward']
    reversedir = env['pe_reverse']
    outdir = "."

    # create links to the contributing reports before creating the paired end report.
    # this makes sure that the contributing reports are accessible from the paired end report in case the sffcorrection failes.
    try:
        os.symlink(forwarddir,"fwd_folder")
        os.symlink(reversedir,"rev_folder")
    except:
        printtime(traceback.format_exc())

    forward_env,warn = explogparser.getparameter(os.path.join(forwarddir,"ion_params_00.json"))
    reverse_env,warn = explogparser.getparameter(os.path.join(reversedir,"ion_params_00.json"))

    libraryName        = PE_set_value('libraryName')
    sample             = PE_set_value('sample')
    chipType           = PE_set_value('chipType')
    site_name          = PE_set_value('site_name')
    flows              = PE_set_value('flows')
    barcodeId          = PE_set_value('barcodeId')
    aligner_opts_extra = PE_set_value('aligner_opts_extra')
    mark_duplicates    = PE_set_value('mark_duplicates')
    notes              = forward_env['notes']
    start_time         = forward_env['start_time']
    align_full         = True
    DIR_BC_FILES       = "bc" #todo
    sam_parsed         = True

    refLocation = "/results/referenceLibrary/tmap-f3"
    refGenome = "%s/%s/%s%s"%(refLocation,libraryName,libraryName, ".fasta")

    #TODO
    #shutil.copy(os.path.join(forwarddir,"processParameters.txt"), os.path.join(outdir,"processParameters.txt"))
    #Reference Genome Information
    shutil.copy(os.path.join(forwarddir,"alignment.summary"), os.path.join(outdir,"alignment.summary"))

    fwd_rev_files = [
        "version.txt",
        "expMeta.dat",
        "sigproc_results/bfmask.stats",
        "sigproc_results/Bead_density_contour.png",
        "basecaller_results/BaseCaller.json",
        "basecaller_results/quality.summary",
        "alignTable.txt",
        "alignment.summary",
        "sigproc_results/processParameters.txt",
        "raw_peak_signal"
    ]
    for filename in fwd_rev_files:
        copyfile(filename)
 
    forward_sff  = "%s/%s_%s.sff" % (forward_env['BASECALLER_RESULTS'], forward_env['expName'], forward_env['resultsName'])
    reverse_sff  = "%s/%s_%s.sff" % (reverse_env['BASECALLER_RESULTS'], reverse_env['expName'], reverse_env['resultsName'])

    forward_sff_path = os.path.join(forwarddir,forward_sff)
    reverse_sff_path = os.path.join(reversedir,reverse_sff)


    #mapped bam files are processed to generate strand bias files
    forward_mapped_bam  = "%s_%s.bam" % (forward_env['expName'], forward_env['resultsName'])
    reverse_mapped_bam  = "%s_%s.bam" % (reverse_env['expName'], reverse_env['resultsName'])
    forward_mapped_bam_bai  = "%s_%s.bam.bai" % (forward_env['expName'], forward_env['resultsName'])
    reverse_mapped_bam_bai  = "%s_%s.bam.bai" % (reverse_env['expName'], reverse_env['resultsName'])

    forward_mapped_bam_path = os.path.join(forwarddir,forward_mapped_bam)
    reverse_mapped_bam_path = os.path.join(reversedir,reverse_mapped_bam)
    forward_mapped_bam_bai_path = os.path.join(forwarddir,forward_mapped_bam_bai)
    reverse_mapped_bam_bai_path = os.path.join(reversedir,reverse_mapped_bam_bai)

    #strandBiasExtractor
    try:
        com = "strandBiasExtractor"
        com += " -n 8"
        com += " -r %s" % (refGenome)
        com += " -c 0.1"
        com += " -b %s" % (forward_mapped_bam_path)
        com += " -i %s" % (forward_mapped_bam_bai_path)
        com += " -o ./"
        com += " -f ./forward_strand_bias.txt"
        printtime("DEBUG: Calling '%s'" % com)
        ret = subprocess.call(com,shell=True)
        if ret:
            raise BaseException("strandBiasExtractor returned with error code %s" % ret)
    except:
        traceback.print_exc()
        jobserver.updatestatus(os.path.join(os.getcwd(),'primary.key'),'Error',True)
        printtime("ERROR running strandBiasExtractor for forward run")
        #sys.exit(1)

    try:
        com = "strandBiasExtractor"
        com += " -n 8"
        com += " -c 0.1"
        com += " -r %s" % (refGenome)
        com += " -b %s" % (reverse_mapped_bam_path)
        com += " -i %s" % (reverse_mapped_bam_bai_path)
        com += " -o ./"
        com += " -f ./reverse_strand_bias.txt"
        printtime("DEBUG: Calling '%s'" % com)
        ret = subprocess.call(com,shell=True)
        if ret:
            raise BaseException("strandBiasExtractor returned with error code %s" % ret)
    except:
        traceback.print_exc()
        jobserver.updatestatus(os.path.join(os.getcwd(),'primary.key'),'Error',True)
        printtime("ERROR running strandBiasExtractor for reverse run")
        #sys.exit(1)


    forward_bam  = "%s/%s_%s.basecaller.bam" % (forward_env['BASECALLER_RESULTS'], forward_env['expName'], forward_env['resultsName'])
    reverse_bam  = "%s/%s_%s.basecaller.bam" % (reverse_env['BASECALLER_RESULTS'], reverse_env['expName'], reverse_env['resultsName'])

    forward_bam_path = os.path.join(forwarddir,forward_bam)
    reverse_bam_path = os.path.join(reversedir,reverse_bam)

    f = open('progress.txt','w')
    f.write('wellfinding = green\n')
    f.write('signalprocessing = green\n')
    f.write('basecalling = green\n')
    f.write('sffread = grey\n')
    f.write('alignment = grey')
    f.close()

    basename = os.path.join(outdir,"%s_%s" % (env['expName'], env['resultsName']))
    
    try:
        com = "PairedEndErrorCorrection"
        com += " -n 8"
        com += " -s %s.bam" % (basename)
        com += " %s" % (forward_bam_path)
        com += " %s" % (reverse_bam_path)
        com += " -b ./forward_strand_bias.txt"
        com += " -c ./reverse_strand_bias.txt"
        printtime("DEBUG: Calling '%s'" % com)
        ret = subprocess.call(com,shell=True)
        if ret:
            raise BaseException("PairedEndErrorCorrection returned with error code %s" % ret)
    except:
        traceback.print_exc()
        set_result_status('Error')
        printtime("ERROR running PairedEndErrorCorrection, abort report generation")
        sys.exit(1)

    # merge raw_peak_signal
    try:
        if os.path.exists("fwd_raw_peak_signal") and os.path.exists("rev_raw_peak_signal"):
            fwd_keyPeak = parse_metrics("fwd_raw_peak_signal")
            rev_keyPeak = parse_metrics("rev_raw_peak_signal")
            keyPeak = {}
            keyPeak['Test Fragment'] = 0
            keyPeak['Library'] = (int(fwd_keyPeak['Library']) +int(rev_keyPeak['Library']))/2
        else:
            keyPeak = {}
            keyPeak['Test Fragment'] = 0
            keyPeak['Library'] = 0
        f = open('raw_peak_signal','w')
        f.write('Test Fragment = %s\n' % keyPeak['Test Fragment'])
        f.write('Library = %s\n' % keyPeak['Library'])
        f.close()
    except:
        printtime(traceback.format_exc())

    set_result_status('Alignment')
    f = open('progress.txt','w')
    f.write('wellfinding = green\n')
    f.write('signalprocessing = green\n')
    f.write('basecalling = green\n')
    f.write('sffread = yellow\n')
    f.write('alignment = yellow')
    f.close()


    top_dir = os.getcwd()

    for status in ["Paired_Fwd","Paired_Rev","Singleton_Fwd","Singleton_Rev","corrected"]:

        print "status:"+status

        bam_path = basename+"_"+status+".bam"
        sff_path = basename+"_"+status+".sff"

        try:
            com = "bam2sff"
            com += " -o %s" % sff_path
            com += " %s" % bam_path

            printtime("DEBUG: Calling '%s'" % com)
            ret = subprocess.call(com,shell=True)
        except:
            printtime('ERROR: Failed to convert bam ' + str(bam_path) + ' to sff')
            continue


    for status in ["Paired_Fwd","Paired_Rev","Singleton_Fwd","Singleton_Rev","corrected"]:
        if not os.path.exists(os.path.join(top_dir,status)):
            os.makedirs(os.path.join(top_dir,status))
        os.chdir(os.path.join(top_dir,status))
        print "status:"+status

        sff_path = os.path.join("..",basename+"_"+status+".sff")
        fastq_path = os.path.join(basename+"_"+status+".fastq")

        try:
            com = "SFFRead"
            com += " -q %s" % fastq_path
            com += " %s" % sff_path

            printtime("DEBUG: Calling '%s'" % com)
            ret = subprocess.call(com,shell=True)
        except:
            printtime('ERROR: Failed to convert SFF ' + str(sff_path) + ' to fastq')
            continue

        try:
            com = "SFFSummary"
            com += " -o %s" % os.path.join('.', 'quality.summary')
            com += " --sff-file %s" % sff_path
            com += " --read-length 50,100,150"
            com += " --min-length 0,0,0"
            com += " --qual 0,17,20"
            com += " -d %s" % os.path.join('.', 'readLen.txt')

            printtime("DEBUG: Calling '%s'" % com)
            ret = subprocess.call(com,shell=True)
        except:
            printtime('ERROR: Failed SFFSummary')

        printtime('alignment')

        if status == 'corrected':
           bidirectional = True
        else:
           bidirectional = False

        alignment.alignment(
            sff_path,
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
            mark_duplicates,
            start_time,
            env['ALIGNMENT_RESULTS'],
            bidirectional,
            sam_parsed
            )

        try:
            quality = os.path.join('.',"quality.summary")
            shutil.copy(quality, "../"+status+".quality.summary")
        except:
            printtime("ERROR: %s doesn't exist" % quality)
            pass
        shutil.copy("alignTable.txt", "../"+status+".alignTable.txt")
        shutil.copy("alignment.summary", "../"+status+".alignment.summary")

        make_zip(sff_path+'.zip',sff_path,arcname=sff_path)
        make_zip(fastq_path+'.zip',fastq_path,arcname=fastq_path)

    os.chdir(top_dir)

    # plugin framework expects the sff file in the env['BASECALLER_RESULTS'] subdirectory
    union_sff = os.path.join(env['BASECALLER_RESULTS'],basename+".sff")
    union_fastq = os.path.join(env['BASECALLER_RESULTS'],basename+".fastq")

    # merge sff files
    try:
        com = "SFFMerge"
        com += " -r"
        com += " -o %s" % union_sff
        for status in ["Paired_Fwd","Paired_Rev","Singleton_Fwd","Singleton_Rev","corrected"]:
            sff = basename+"_"+status+".sff"
            if os.path.exists(sff):
                printtime("DEBUG: %s exists" % sff)
                com += " %s" % sff
            else:
                printtime("ERROR: %s is missing" % sff)

        printtime("DEBUG: Calling '%s'" % com)
        #TODO,error handling for missing sff files
        ret = subprocess.call(com,shell=True)
        if ret:
            raise BaseException("SFFMerge returned with error code %s" % ret)
    except:
        jobserver.updatestatus(os.path.join(os.getcwd(),'primary.key'),'Error',True)
        printtime('ERROR: Failed to create union sff file, abort report generation')
        sys.exit(1)

    try:
        com = "SFFRead"
        com += " -q %s" % union_fastq
        com += " %s" % union_sff

        printtime("DEBUG: Calling '%s'" % com)
        ret = subprocess.call(com,shell=True)
    except:
        printtime('ERROR: Failed to convert SFF ' + str(union_sff) + ' to fastq')

    try:
        com = "SFFSummary"
        com += " -o %s" % os.path.join(env['BASECALLER_RESULTS'], 'quality.summary')
        com += " --sff-file %s" % union_sff
        com += " --read-length 50,100,150"
        com += " --min-length 0,0,0"
        com += " --qual 0,17,20"
        com += " -d %s" % os.path.join(env['BASECALLER_RESULTS'], 'readLen.txt')

        printtime("DEBUG: Calling '%s'" % com)
        ret = subprocess.call(com,shell=True)
    except:
        printtime('ERROR: Failed SFFSummary')

    printtime('alignment')

    bidirectional = False
    alignment.alignment(
        union_sff,
        union_fastq,
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
        env['ALIGNMENT_RESULTS'],
        bidirectional,
        sam_parsed
        )

    make_zip(union_sff+'.zip',union_sff,arcname=union_sff)
    make_zip(union_fastq+'.zip',union_fastq,arcname=union_fastq)



    # Get Reverse/Forward files for Default_Report File Links
    for ext in [".tf.sff.zip",".sff.zip",".fastq.zip",".bam",".bam.bai"]:
      try:
        os.symlink(forward_sff_path.replace(".sff",ext),basename+"_forward"+ext)
        os.symlink(reverse_sff_path.replace(".sff",ext),basename+"_reverse"+ext)
      except:
        printtime(traceback.format_exc()) 
    
    # These settings do not reflect reality of block processing but are intended
    # to 'clean up' the progress indicators only.
    # create analysis progress bar file
    f = open('progress.txt','w')
    f.write('wellfinding = green\n')
    f.write('signalprocessing = green\n')
    f.write('basecalling = green\n')
    f.write('sffread = green\n')
    f.write('alignment = green')
    f.close()

    #basecaller_results/quality.summary
    try:
        union_quality_metrics = parse_metrics('basecaller_results/quality.summary')
        totalbasesunion       = union_quality_metrics['Number of Bases at Q0']
        totalq17basesunion    = union_quality_metrics['Number of Bases at Q17']
        totalq20basesunion    = union_quality_metrics['Number of Bases at Q20']
        meanlengthunion       = union_quality_metrics['Mean Read Length at Q0']
        totalreadsbasesunion  = union_quality_metrics['Number of Reads at Q0']
    except:
        totalbasesunion       = 0
        totalq17basesunion    = 0
        totalq20basesunion    = 0
        meanlengthunion       = 0
        totalreadsbasesunion  = 0
        printtime(traceback.format_exc()) 

    #Singleton_Fwd.quality.summary
    try:
        unpairedfwd_quality_metrics = parse_metrics('Singleton_Fwd.quality.summary')
        totalbasesunpairedfwd       = unpairedfwd_quality_metrics['Number of Bases at Q0']
        totalq17basesunpairedfwd    = unpairedfwd_quality_metrics['Number of Bases at Q17']
        totalq20basesunpairedfwd    = unpairedfwd_quality_metrics['Number of Bases at Q20']
        meanlengthunpairedfwd       = unpairedfwd_quality_metrics['Mean Read Length at Q0']
        totalreadsbasesunpairedfwd  = unpairedfwd_quality_metrics['Number of Reads at Q0']
    except:
        totalbasesunpairedfwd       = 0
        totalq17basesunpairedfwd    = 0
        totalq20basesunpairedfwd    = 0
        meanlengthunpairedfwd       = 0
        totalreadsbasesunpairedfwd  = 0
        printtime(traceback.format_exc()) 

    #Singleton_Rev.quality.summary
    try:
        unpairedrev_quality_metrics = parse_metrics('Singleton_Rev.quality.summary')
        totalbasesunpairedrev       = unpairedrev_quality_metrics['Number of Bases at Q0']
        totalq17basesunpairedrev    = unpairedrev_quality_metrics['Number of Bases at Q17']
        totalq20basesunpairedrev    = unpairedrev_quality_metrics['Number of Bases at Q20']
        meanlengthunpairedrev       = unpairedrev_quality_metrics['Mean Read Length at Q0']
        totalreadsbasesunpairedrev  = unpairedrev_quality_metrics['Number of Reads at Q0']
    except:
        totalbasesunpairedrev       = 0
        totalq17basesunpairedrev    = 0
        totalq20basesunpairedrev    = 0
        meanlengthunpairedrev       = 0
        totalreadsbasesunpairedrev  = 0
        printtime(traceback.format_exc()) 

    #corrected.quality.summary
    try:
        corrected_quality_metrics = parse_metrics('corrected.quality.summary')
        totalbasescorrected       = corrected_quality_metrics['Number of Bases at Q0']
        totalq17basescorrected    = corrected_quality_metrics['Number of Bases at Q17']
        totalq20basescorrected    = corrected_quality_metrics['Number of Bases at Q20']
        meanlengthcorrected       = corrected_quality_metrics['Mean Read Length at Q0']
        totalreadsbasescorrected  = corrected_quality_metrics['Number of Reads at Q0']
    except:
        totalbasescorrected       = 0
        totalq17basescorrected    = 0
        totalq20basescorrected    = 0
        meanlengthcorrected       = 0
        totalreadsbasescorrected  = 0
        printtime(traceback.format_exc()) 

    try:
        fwd_quality_metrics = parse_metrics('basecaller_results/fwd_quality.summary')
        pairedfwd_quality_metrics = parse_metrics('Paired_Fwd.quality.summary')

        pairingrate = (float(pairedfwd_quality_metrics['Number of Reads at Q0']) \
                     + float(corrected_quality_metrics['Number of Reads at Q0']) ) \
                     / float(fwd_quality_metrics['Number of Reads at Q0'])
    except:
        pairingrate = 0.0
        printtime(traceback.format_exc())

    try:
        fwdandrevcorrected = float(totalreadsbasescorrected) / float(fwd_quality_metrics['Number of Reads at Q0'])
    except:
        fwdandrevcorrected = 0.0
        printtime(traceback.format_exc())

    try:
        fwdandrevuncorrected = float(pairedfwd_quality_metrics['Number of Reads at Q0']) / float(fwd_quality_metrics['Number of Reads at Q0'])
    except:
        fwdandrevuncorrected = 0.0
        printtime(traceback.format_exc())

    try:
        fwdnotrev = float(unpairedfwd_quality_metrics['Number of Reads at Q0']) / float(fwd_quality_metrics['Number of Reads at Q0'])
    except:
        fwdnotrev = 0.0
        printtime(traceback.format_exc())


    try:
        # create pe.json
        # 'Pairing rate is defined as the percentage of forward reads that have a reverse read pair'
        dataset = {
            'pairingrate' : pairingrate,
            'fwdandrevcorrected' : fwdandrevcorrected,
            'fwdandrevuncorrected' : fwdandrevuncorrected,
            'fwdnotrev' : fwdnotrev,

            'totalbasesunion' : totalbasesunion,
            'totalbasescorrected' : totalbasescorrected,
            'totalbasesunpairedfwd' : totalbasesunpairedfwd,
            'totalbasesunpairedrev' : totalbasesunpairedrev,

            'totalq17basesunion' : totalq17basesunion,
            'totalq17basesunpairedrev' : totalq17basesunpairedrev,
            'totalq17basesunpairedfwd' : totalq17basesunpairedfwd,
            'totalq17basescorrected' : totalq17basescorrected,

            'totalq20basesunion' : totalq20basesunion,
            'totalq20basescorrected' : totalq20basescorrected,
            'totalq20basesunpairedfwd' : totalq20basesunpairedfwd,
            'totalq20basesunpairedrev' : totalq20basesunpairedrev,

            'totalreadsbasesunion' : totalreadsbasesunion,
            'totalreadsbasesunpairedrev' : totalreadsbasesunpairedrev,
            'totalreadsbasesunpairedfwd' : totalreadsbasesunpairedfwd,
            'totalreadsbasescorrected' : totalreadsbasescorrected,

            'meanlengthunion' : meanlengthunion,
            'meanlengthcorrected' : meanlengthcorrected,
            'meanlengthunpairedfwd' : meanlengthunpairedfwd,
            'meanlengthunpairedrev' : meanlengthunpairedrev
        }
        f = open("pe.json","w")
        json.dump(dataset, f, indent=4)
        f.close()
    except:
        printtime("Error, unable to create pe.json")
        traceback.print_exc()


    print "jobserver.uploadmetrics(...)"
    mycwd = os.getcwd()
    STATUS = "Completed"
    reportLink = True
    ret_message = jobserver.uploadmetrics(
        "",#    os.path.join(mycwd,tfmapperstats_outputfile),
        os.path.join(mycwd,"processParameters.txt"),
        "",#     os.path.join(mycwd,beadPath),
        "",#     filterPath,
        os.path.join(mycwd,"alignment.summary"),
        os.path.join(mycwd,"raw_peak_signal"),
        os.path.join(mycwd,"quality.summary"),
        "",#     os.path.join(mycwd,BaseCallerJsonPath),
        os.path.join(mycwd,'pe.json'),
        os.path.join(mycwd,'primary.key'),
        os.path.join(mycwd,'uploadStatus'),
        STATUS,
        reportLink)
    print "jobserver.uploadmetrics returned: "+str(ret_message)


    try:
        primary_key = open("primary.key").readline()
        primary_key = primary_key.split(" = ")
        env['primary_key'] = primary_key[1]
        printtime(env['primary_key'])
    except:
        printtime("Error, unable to get the primary key")

    url_root = os.path.join(env['url_path'],os.path.basename(os.getcwd()))

    plugin_set = set()
    plugin_set.add('1_Torrent_Accuracy')
    blockprocessing.run_selective_plugins(plugin_set, env, pluginbasefolder, url_root)

#    getExpLogMsgs(env)
    printtime("Run Complete")
    sys.exit(0)
