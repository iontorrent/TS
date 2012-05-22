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

from ion.utils import blockprocessing
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
        fwd_filename = os.path.join(forwarddir,filename)
        if os.path.exists(fwd_filename):
            shutil.copy(fwd_filename, os.path.join(outdir,"fwd_"+filename))
        else:
            printtime("ERROR: %s doesn't exist" % fwd_filename)

        rev_filename = os.path.join(reversedir,filename)
        if os.path.exists(rev_filename):
            shutil.copy(rev_filename, os.path.join(outdir,"rev_"+filename))
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


    try:
        jobserver = xmlrpclib.ServerProxy("http://%s:%d" % (JOBSERVER_HOST, JOBSERVER_PORT), verbose=False, allow_none=True)
    except (socket.error, xmlrpclib.Fault):
        traceback.print_exc()

    blockprocessing.printheader()
    env = blockprocessing.getparameter()


    try:
        jobserver.updatestatus(os.path.join(os.getcwd(),'primary.key'),'Started',True)
    except:
        traceback.print_exc()


    if args.verbose:
        print env['pe_forward']
        print env['pe_reverse']

    blockprocessing.initreports(env['SIGPROC_RESULTS'], env['BASECALLER_RESULTS'], env['ALIGNMENT_RESULTS'])
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

    forward_env = blockprocessing.getparameter(os.path.join(forwarddir,"ion_params_00.json"))
    reverse_env = blockprocessing.getparameter(os.path.join(reversedir,"ion_params_00.json"))

    libraryName        = PE_set_value('libraryName')
    sample             = PE_set_value('sample')
    chipType           = PE_set_value('chipType')
    site_name          = PE_set_value('site_name')
    flows              = PE_set_value('flows')
    barcodeId          = PE_set_value('barcodeId')
    aligner_opts_extra = PE_set_value('aligner_opts_extra')
    notes              = forward_env['notes']
    start_time         = forward_env['start_time']
    align_full         = True
    DIR_BC_FILES       = "bc" #todo
    sam_parsed         = True


    #TODO
    shutil.copy(os.path.join(forwarddir,"processParameters.txt"), os.path.join(outdir,"processParameters.txt"))
    #Reference Genome Information
    shutil.copy(os.path.join(forwarddir,"alignment.summary"), os.path.join(outdir,"alignment.summary"))

    fwd_rev_files = []
    fwd_rev_files.append("version.txt")
    fwd_rev_files.append("expMeta.dat")
    fwd_rev_files.append("bfmask.stats")
    fwd_rev_files.append("Bead_density_contour.png")
    fwd_rev_files.append("beadSummary.filtered.txt")
    fwd_rev_files.append("quality.summary")
    fwd_rev_files.append("alignTable.txt")
    fwd_rev_files.append("alignment.summary")
    fwd_rev_files.append("processParameters.txt")
    fwd_rev_files.append("raw_peak_signal")
    for filename in fwd_rev_files:
        copyfile(filename)
 
    forward_sff  = "%s/%s_%s.sff" % (forward_env['BASECALLER_RESULTS'], forward_env['expName'], forward_env['resultsName'])
    reverse_sff  = "%s/%s_%s.sff" % (reverse_env['BASECALLER_RESULTS'], reverse_env['expName'], reverse_env['resultsName'])

    forward_sff_path = os.path.join(forwarddir,forward_sff)
    reverse_sff_path = os.path.join(reversedir,reverse_sff)

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
        com += " -s %s.sff" % (basename)
        com += " %s" % (forward_sff_path)
        com += " %s" % (reverse_sff_path)
        printtime("DEBUG: Calling '%s'" % com)
        ret = subprocess.call(com,shell=True)
        if ret:
            raise BaseException("PairedEndErrorCorrection returned with error code %s" % ret)
    except:
        traceback.print_exc()
        jobserver.updatestatus(os.path.join(os.getcwd(),'primary.key'),'Error',True)
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


    f = open('progress.txt','w')
    f.write('wellfinding = green\n')
    f.write('signalprocessing = green\n')
    f.write('basecalling = green\n')
    f.write('sffread = yellow\n')
    f.write('alignment = yellow')
    f.close()


    top_dir = os.getcwd()
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
            com += " -o %s" % os.path.join(env['BASECALLER_RESULTS'], 'quality.summary')
            com += " --sff-file %s" % sff_path
            com += " --read-length 50,100,150"
            com += " --min-length 0,0,0"
            com += " --qual 0,17,20"
            com += " -d %s" % os.path.join(env['BASECALLER_RESULTS'], 'readLen.txt')

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
            start_time,
            env['ALIGNMENT_RESULTS'],
            bidirectional,
            sam_parsed
            )

        shutil.copy("alignment.summary", "../"+status+".alignment.summary")
        shutil.copy("quality.summary", "../"+status+".quality.summary")
        shutil.copy("alignTable.txt", "../"+status+".alignTable.txt")
        
        make_zip(sff_path+'.zip',sff_path)
        make_zip(fastq_path+'.zip',fastq_path)

    os.chdir(top_dir)


    union_sff = basename+".sff"
    union_fastq = basename+".fastq"
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
        start_time,
        env['ALIGNMENT_RESULTS'],
        bidirectional,
        sam_parsed
        )

    make_zip(union_sff+'.zip',union_sff)
    make_zip(union_fastq+'.zip',union_fastq)



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
        os.path.join(mycwd,'primary.key'),
        os.path.join(mycwd,'uploadStatus'),
        STATUS,
        reportLink)
    print "jobserver.uploadmetrics returned: "+str(ret_message)


    basefolder = 'plugin_out'

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
    blockprocessing.run_selective_plugins(plugin_set, env, basefolder, url_root)

#    getExpLogMsgs(env)
    printtime("Run Complete")
    sys.exit(0)
