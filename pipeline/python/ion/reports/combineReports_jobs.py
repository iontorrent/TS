#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import argparse
import traceback
import subprocess
import os
import json
import socket
import re

from ion.utils.blockprocessing import printtime as _printtime
from ion.utils.compress import make_zip
os.environ['MPLCONFIGDIR'] = '/tmp'
from ion.utils import ionstats_plots


def printtime(message, *args):
    try:
        _printtime(message, *args)
    except:
        pass


def merge_bam_files(bamfilelist, composite_bam_filepath, mark_duplicates, new_sample_name=''):

    composite_bai_filepath = composite_bam_filepath.replace('.bam', '.bam.bai')
    composite_header_filepath = composite_bam_filepath + '.header.sam'

    try:
        # generate file headers and merge them using picard tools
        for bamfile in bamfilelist:
            cmd = 'samtools view -H %s > %s.header.sam' % (bamfile, bamfile,)
            printtime("DEBUG: Calling '%s'" % cmd)
            subprocess.call(cmd, shell=True)

        cmd = 'java -Xmx8g -jar /opt/picard/picard-tools-current/picard.jar MergeSamFiles'
        for bamfile in bamfilelist:
            cmd = cmd + ' I=%s.header.sam' % bamfile
        cmd = cmd + ' O=%s' % composite_header_filepath
        cmd = cmd + ' VERBOSITY=WARNING'  # suppress INFO on stderr
        cmd = cmd + ' QUIET=true'  # suppress job-summary on stderr
        cmd = cmd + ' VALIDATION_STRINGENCY=SILENT'
        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd, shell=True)

        # overwrite sample names in header SM tag
        if new_sample_name:
            header = []
            with open(composite_header_filepath, 'r') as f:
                for line in f.readlines():
                    if line.startswith('@RG'):
                        line = re.sub('SM:(.+?)\t', 'SM:%s\t' % new_sample_name, line)
                    header.append(line)
            with open(composite_header_filepath, 'w') as f:
                f.writelines(header)
    except:
        traceback.print_exc()

    try:
        # do BAM files merge
        cmd = 'samtools merge -l1 -@8 -c'
        if mark_duplicates:
            cmd += ' - '
        else:
            cmd += ' %s' % (composite_bam_filepath)

        for bamfile in bamfilelist:
            cmd += ' %s' % bamfile
        cmd += ' -h %s' % composite_header_filepath

        if mark_duplicates:
            json_name = ('BamDuplicates.%s.json') % (os.path.normpath(composite_bam_filepath)) if os.path.normpath(composite_bam_filepath) != 'rawlib.bam' else 'BamDuplicates.json'
            cmd += ' | BamDuplicates -i stdin -o %s -j %s' % (composite_bam_filepath, json_name)

        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd, shell=True)
    except:
        traceback.print_exc()

    try:
        if composite_bai_filepath:
            # TODO: samtools 1.2: Samtools-htslib-API: bam_index_build2() not yet implemented
            # cmd = 'samtools index %s %s' % (composite_bam_filepath,composite_bai_filepath)
            cmd = 'samtools index %s' % (composite_bam_filepath)
            printtime("DEBUG: Calling '%s'" % cmd)
            subprocess.call(cmd, shell=True)
    except:
        traceback.print_exc()


def generate_ionstats(bam_filename, ionstats_filename, ionstats_h5_filename, histogram_length):
    try:
        com = "ionstats alignment"
        com += " -i %s" % (bam_filename)
        com += " -o %s" % (ionstats_filename)
        com += " -h %d" % (int(histogram_length))
        com += " --output-h5 %s" % ionstats_h5_filename
        printtime("DEBUG: Calling '%s'" % com)
        subprocess.call(com, shell=True)
    except:
        printtime('Failed generating ionstats')
        traceback.print_exc()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--add-file', dest='files', action='append', default=[], help="list of files to process")
    parser.add_argument('-m', '--merge-bams', dest='merge_out', action='store', default="", help='merge bam files')
    parser.add_argument('-d', '--mark-duplicates', dest='duplicates', action='store_true', default=False, help='mark duplicates')
    parser.add_argument('-a', '--align-stats', dest='align_stats', action='store_true', default="", help='generate alignment stats')
    parser.add_argument('-p', '--merge-plots', dest='merge_plots', action='store_true', default="", help='generate report plots')
    parser.add_argument('-z', '--zip', dest='zip', action='store', default="", help='zip input files')
    parser.add_argument('-s', '--new-sample-name', dest='sample', action='store', default="", help='overwrite BAM header sample name')

    args = parser.parse_args()

    printtime("DEBUG: CA job running on %s." % socket.gethostname())

    if args.merge_out and len(args.files) > 1:
        # Merge BAM files
        outputBAM = args.merge_out
        printtime("Merging bam files to %s, mark duplicates is %s" % (outputBAM, args.duplicates))
        try:
            merge_bam_files(args.files, outputBAM, args.duplicates, args.sample)
        except:
            traceback.print_exc()

    if args.align_stats and len(args.files) > 0:
        # generate ionstats files from merged BAMs
        printtime("Generating alignment stats for %s" % ', '.join(args.files))
        graph_max_x = 400
        for bamfile in args.files:
            if bamfile == 'rawlib.bam':
                ionstats_file = 'ionstats_alignment.json'
                error_summary_file = 'ionstats_error_summary.h5'
            else:
                ionstats_file = bamfile.split('.bam')[0] + '.ionstats_alignment.json'
                error_summary_file = bamfile.split('.bam')[0] + '.ionstats_error_summary.h5'

            generate_ionstats(bamfile, ionstats_file, error_summary_file, graph_max_x)

    if args.merge_plots:
        printtime("Generating plots for merged report")
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
        printtime("Zip merged barcode files to %s" % zipname)
        for filename in args.files:
            if os.path.exists(filename):
                try:
                    make_zip(zipname, filename, arcname=filename)
                except:
                    print("ERROR: zip target: %s" % filename)
                    traceback.print_exc()

    printtime("DEBUG: CA job done.")
