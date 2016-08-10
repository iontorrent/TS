#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

__version__ = filter(str.isdigit, "$Revision$")

import os
import sys
import ConfigParser
import StringIO
import datetime
import shutil
import socket
import subprocess
import time
import traceback
import json

sys.path.append('/opt/ion/')


class MyConfigParser(ConfigParser.RawConfigParser):

    def read(self, filename):
        try:
            text = open(filename).read()
        except IOError:
            pass
        else:
            afile = StringIO.StringIO("[global]\n" + text)
            self.readfp(afile, filename)


def printtime(message, *args):
    if args:
        message = message % args
    print "[ " + time.strftime('%a %Y-%m-%d %X %Z') + " ] " + message
    sys.stdout.flush()
    sys.stderr.flush()


def write_version():
    a = subprocess.Popen('ion_versionCheck.py --ion', shell=True, stdout=subprocess.PIPE)
    ret = a.stdout.readlines()
    f = open('version.txt', 'w')
    for i in ret[:len(ret)-1]:
#    for i in ret:
        f.write(i)
    f.close()


def parse_metrics(fileIn):
    """Takes a text file where a '=' is the delimter
    in a key value pair and return a python dict of those values """

    f = open(fileIn, 'r')
    data = f.readlines()
    f.close()
    ret = {}
    for line in data:
        l = line.strip().split('=')
        key = l[0].strip()
        value = l[-1].strip()
        ret[key] = value
    return ret


def create_index_file(composite_bam_filepath, composite_bai_filepath):
    try:
        # TODO: samtools 1.2: Samtools-htslib-API: bam_index_build2() not yet implemented
        # cmd = 'samtools index %s %s' % (composite_bam_filepath,composite_bai_filepath)
        cmd = 'samtools index %s' % (composite_bam_filepath)
        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd, shell=True)
    except:
        printtime("index creation failed: %s %s" % (composite_bam_filepath, composite_bai_filepath))
        traceback.print_exc()


def isbadblock(blockdir, message):
    # process blockstatus.txt
    # printtime("WARNING: %s: skipped %s" % (message,blockdir))
    return False


def get_datasets_basecaller(BASECALLER_RESULTS, datasets_basecaller_path=None):

    if datasets_basecaller_path == None:
        datasets_basecaller_path = os.path.join(BASECALLER_RESULTS, "datasets_basecaller.json")

    if not os.path.exists(datasets_basecaller_path):
        printtime("ERROR: %s does not exist" % datasets_basecaller_path)
        raise Exception("ERROR: %s does not exist" % datasets_basecaller_path)

    datasets_basecaller = {}
    try:
        f = open(datasets_basecaller_path, 'r')
        datasets_basecaller = json.load(f);
        f.close()
    except:
        printtime("ERROR: problem parsing %s" % datasets_basecaller_path)
        raise Exception("ERROR: problem parsing %s" % datasets_basecaller_path)
    return datasets_basecaller


def printheader():
    # 
    # Print nice header information                        #
    # 
    python_data = [sys.executable, sys.version, sys.platform, socket.gethostname(),
                   str(os.getpid()), os.getcwd(),
                   os.environ.get("JOB_ID", '[Stand-alone]'),
                   os.environ.get("JOB_NAME", '[Stand-alone]'),
                   datetime.datetime.now().strftime("%H:%M:%S %b/%d/%Y")
                   ]
    python_data_labels = ["Python Executable", "Python Version", "Platform",
                          "Hostname", "PID", "Working Directory", "Job ID",
                          "Job Name", "Start Time"]
    _MARGINS = 4
    _TABSIZE = 4
    _max_sum = max(map(lambda (a, b): len(a) + len(b), zip(python_data, python_data_labels)))
    _info_width = _max_sum + _MARGINS + _TABSIZE
    print('*'*_info_width)
    for d, l in zip(python_data, python_data_labels):
        spacer = ' '*(_max_sum - (len(l) + len(d)) + _TABSIZE)
        print('* %s%s%s *' % (str(l), spacer, str(d).replace('\n', ' ')))
    print('*'*_info_width)

    sys.stdout.flush()
    sys.stderr.flush()


def merge_bam_files(bamfilelist, composite_bam_filepath, composite_bai_filepath, mark_duplicates, method="samtools"):

    if method == 'samtools':
        merge_bam_files_samtools(bamfilelist, composite_bam_filepath, composite_bai_filepath, mark_duplicates)

    if method == 'picard':
        merge_bam_files_picard(bamfilelist, composite_bam_filepath, composite_bai_filepath, mark_duplicates)


def extract_and_merge_bam_header(bamfilelist, composite_bam_filepath):

    try:
        for bamfile in bamfilelist:
            cmd = 'samtools view -H %s > %s.header.sam' % (bamfile, bamfile,)
            printtime("DEBUG: Calling '%s'" % cmd)
            subprocess.call(cmd, shell=True)

        cmd = 'java -Xmx8g -jar /opt/picard/picard-tools-current/picard.jar MergeSamFiles'
        for bamfile in bamfilelist:
            cmd = cmd + ' I=%s.header.sam' % bamfile
        cmd = cmd + ' O=%s.header.sam' % (composite_bam_filepath)
        cmd = cmd + ' VERBOSITY=WARNING'  # suppress INFO on stderr
        cmd = cmd + ' QUIET=true'  # suppress job-summary on stderr
        cmd = cmd + ' VALIDATION_STRINGENCY=SILENT'
        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd, shell=True)
    except:
        printtime("bam header merge failed")
        traceback.print_exc()
        return 1


def merge_bam_files_samtools(bamfilelist, composite_bam_filepath, composite_bai_filepath, mark_duplicates):

    try:
        extract_and_merge_bam_header(bamfilelist, composite_bam_filepath)
        if len(bamfilelist) == 1:
            # Usage: samtools reheader <in.header.sam> <in.bam>
            if mark_duplicates:
                cmd = 'samtools reheader %s.header.sam %s' % (composite_bam_filepath, bamfilelist[0])
            else:
                cmd = 'samtools reheader %s.header.sam %s > %s' % (composite_bam_filepath, bamfilelist[0], composite_bam_filepath)
        else:
            # Usage: samtools merge [-nr] [-h inh.sam] <out.bam> <in1.bam> <in2.bam> [...]
            cmd = 'samtools merge -l1 -@8'
            if mark_duplicates:
                cmd += ' - '
            else:
                cmd += ' %s' % (composite_bam_filepath)
            for bamfile in bamfilelist:
                cmd += ' %s' % bamfile
            cmd += ' -h %s.header.sam' % composite_bam_filepath

        if mark_duplicates:
            json_name = ('BamDuplicates.%s.json') % (os.path.normpath(composite_bam_filepath)) if os.path.normpath(composite_bam_filepath) != 'rawlib.bam' else 'BamDuplicates.json'
            cmd += ' | BamDuplicates -i stdin -o %s -j %s' % (composite_bam_filepath, json_name)

        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd, shell=True)

        if composite_bai_filepath:
            create_index_file(composite_bam_filepath, composite_bai_filepath)
    except:
        printtime("bam file merge failed")
        traceback.print_exc()
        return 1


def merge_bam_files_picard(bamfilelist, composite_bam_filepath, composite_bai_filepath, mark_duplicates):

    try:
#        cmd = 'picard-tools MergeSamFiles'
        if mark_duplicates:
            cmd = 'java -Xmx8g -jar /usr/local/bin/MarkDuplicates.jar M=%s.markduplicates.metrics.txt' % composite_bam_filepath
        else:
            cmd = 'java -Xmx8g -jar /opt/picard/picard-tools-current/picard.jar MergeSamFiles'

        for bamfile in bamfilelist:
            cmd = cmd + ' I=%s' % bamfile
        cmd = cmd + ' O=%s' % (composite_bam_filepath)
        cmd = cmd + ' ASSUME_SORTED=true'
        if composite_bai_filepath:
            cmd = cmd + ' CREATE_INDEX=true'
        cmd = cmd + ' USE_THREADING=true'
        cmd = cmd + ' VERBOSITY=WARNING'  # suppress INFO on stderr
        cmd = cmd + ' QUIET=true'  # suppress job-summary on stderr
        cmd = cmd + ' VALIDATION_STRINGENCY=SILENT'
        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd, shell=True)
    except:
        printtime("bam file merge failed")
        traceback.print_exc()
        return 1

    try:
        if composite_bai_filepath:
            if not os.path.exists(composite_bai_filepath):
                # picard is using .bai , we want .bam.bai
                srcbaifilepath = composite_bam_filepath.replace(".bam", ".bai")
                if os.path.exists(srcbaifilepath):
                    os.rename(srcbaifilepath, composite_bai_filepath)
                else:
                    printtime("ERROR: %s doesn't exists" % srcbaifilepath)
    except:
        traceback.print_exc()
        return 1


def remove_unneeded_block_files(blockdirs):
    return
    for blockdir in blockdirs:
        try:
            bamfile = os.path.join(blockdir, 'basecaller_results', 'rawlib.basecaller.bam')
            if os.path.exists(bamfile):
                os.remove(bamfile)

            recalibration_dir = os.path.join(blockdir, 'basecaller_results', 'recalibration')
            shutil.rmtree(recalibration_dir, ignore_errors=True)
        except:
            printtime("remove unneeded block files failed")
            traceback.print_exc()


def bam2fastq_command(BAMName, FASTQName):
    com = "java -Xmx8g -jar /opt/picard/picard-tools-current/picard.jar SamToFastq"
    com += " I=%s" % BAMName
    com += " F=%s" % FASTQName
    return com

'''
def merge_raw_key_signals(filelist,composite_file):

    mergedKeyPeak = {}
    mergedKeyPeak['Test Fragment'] = 0
    mergedKeyPeak['Library'] = 0

    N = 0
    merged_key_signal_sum = 0
    for xfile in filelist:
        try:
            keyPeak = parse_metrics(xfile)
            library_key_signal = int(keyPeak['Library'])
            merged_key_signal_sum += library_key_signal
            N += 1
        except:
            printtime(traceback.format_exc())
            continue
    if N > 0:
        mergedKeyPeak['Library'] = merged_key_signal_sum/N

    try:
        f = open(composite_file,'w')
        f.write('Test Fragment = %s\n' % mergedKeyPeak['Test Fragment'])
        f.write('Library = %s\n' % mergedKeyPeak['Library'])
        f.close()
    except:
        printtime(traceback.format_exc())

    return 0
'''


def merge_bams_one_dataset(dirs, BASECALLER_RESULTS, ALIGNMENT_RESULTS, dataset, reference, filtered, mark_duplicates):

        try:
            if reference and not filtered:
                bamdir = ALIGNMENT_RESULTS
                bamfile = dataset['file_prefix']+'.bam'
            else:
                bamdir = BASECALLER_RESULTS
                bamfile = dataset['file_prefix']+'.basecaller.bam'
#                bamfile = dataset['basecaller_bam']
            block_bam_list = [os.path.join(blockdir, bamdir, bamfile) for blockdir in dirs]
            block_bam_list = [block_bam_filename for block_bam_filename in block_bam_list if os.path.exists(block_bam_filename)]
            composite_bam_filepath = os.path.join(bamdir, bamfile)
            if block_bam_list:
                if reference and not filtered:
                    composite_bai_filepath = composite_bam_filepath+'.bai'
                    merge_bam_files(block_bam_list, composite_bam_filepath, composite_bai_filepath, mark_duplicates)
                else:
                    composite_bai_filepath = ""
                    merge_bam_files(block_bam_list, composite_bam_filepath, composite_bai_filepath, mark_duplicates=False, method='samtools')

        except:
            print traceback.format_exc()
            printtime("ERROR: merging %s unsuccessful" % bamfile)


def merge_bams(dirs, BASECALLER_RESULTS, ALIGNMENT_RESULTS, basecaller_datasets, mark_duplicates):

    for dataset in basecaller_datasets['datasets']:

        try:
            read_group = dataset['read_groups'][0]
            reference = basecaller_datasets['read_groups'][read_group]['reference']

            filtered = False  # True
            for rg_name in dataset["read_groups"]:
                if not basecaller_datasets["read_groups"][rg_name].get('filtered', False):
                    filtered = False

            if reference and not filtered:
                bamdir = ALIGNMENT_RESULTS
                bamfile = dataset['file_prefix']+'.bam'
            else:
                bamdir = BASECALLER_RESULTS
                bamfile = dataset['basecaller_bam']
            block_bam_list = [os.path.join(blockdir, bamdir, bamfile) for blockdir in dirs]
            block_bam_list = [block_bam_filename for block_bam_filename in block_bam_list if os.path.exists(block_bam_filename)]
            composite_bam_filepath = os.path.join(bamdir, bamfile)
            if block_bam_list:
                if reference and not filtered:
                    composite_bai_filepath = composite_bam_filepath+'.bai'
                    merge_bam_files(block_bam_list, composite_bam_filepath, composite_bai_filepath, mark_duplicates)
                else:
                    composite_bai_filepath = ""
                    merge_bam_files(block_bam_list, composite_bam_filepath, composite_bai_filepath, mark_duplicates=False, method='samtools')

        except:
            print traceback.format_exc()
            printtime("ERROR: merging %s unsuccessful" % bamfile)


def merge_unmapped_bams(dirs, BASECALLER_RESULTS, basecaller_datasets, method):

    for dataset in basecaller_datasets['datasets']:

        try:
            bamdir = BASECALLER_RESULTS
            bamfile = dataset['basecaller_bam']
            block_bam_list = [os.path.join(blockdir, bamdir, bamfile) for blockdir in dirs]
            block_bam_list = [block_bam_filename for block_bam_filename in block_bam_list if os.path.exists(block_bam_filename)]
            composite_bam_filepath = os.path.join(bamdir, bamfile)
            if block_bam_list:
                composite_bai_filepath = ""
                mark_duplicates = False
                merge_bam_files(block_bam_list, composite_bam_filepath, composite_bai_filepath, mark_duplicates, method)
        except:
            traceback.print_exc()
            printtime("ERROR: merging %s unsuccessful" % bamfile)

    printtime("Finished merging basecaller BAM files")


def merge_barcoded_alignment_bams(ALIGNMENT_RESULTS, basecaller_datasets, method):

    try:
        composite_bam_filename = os.path.join(ALIGNMENT_RESULTS, 'rawlib.bam')

        bam_file_list = []
        for dataset in basecaller_datasets["datasets"]:
            bam_name = os.path.join(ALIGNMENT_RESULTS, os.path.basename(dataset['file_prefix'])+'.bam')
            if os.path.exists(bam_name):
                bam_file_list.append(bam_name)
            else:
                printtime("WARNING: exclude %s from merging into %s" % (bam_name, composite_bam_filename))

        composite_bai_filename = composite_bam_filename+'.bai'
        mark_duplicates = False
        merge_bam_files(bam_file_list, composite_bam_filename, composite_bai_filename, mark_duplicates, method)
    except:
        traceback.print_exc()
        printtime("ERROR: Generate merged %s on barcoded run failed" % composite_bam_filename)

    printtime("Finished barcode merging of %s" % ALIGNMENT_RESULTS)
