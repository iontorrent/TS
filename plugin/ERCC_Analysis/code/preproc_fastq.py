# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
#pre-process mapped bam files by filtering for base quality and then for length

from __future__ import division
import os
import sys
from itertools import izip
import traceback
import pysam

reverse_enumerate = lambda l: izip(xrange(len(l)-1, -1, -1), reversed(l))

def window_mean_quality_score(ascii_string):
    mean_quality_score = 0
    for char in ascii_string:
        mean_quality_score += (ord(char) - 33)
    mean_quality_score = mean_quality_score / len(ascii_string)
    return mean_quality_score

def base_quality_trim(base_seq, quality_metric_seq, threshold, min_length):
    base_seq_length = len(base_seq)
    if base_seq_length < min_length:
        return ''
    else:
        window_size = int(base_seq_length / 10)
    new_base_seq = []
    for position, base in reverse_enumerate(base_seq):
        if (position + window_size) >= base_seq_length:
            window_end = base_seq_length
        else:
            window_end = position + window_size
        if (window_mean_quality_score(quality_metric_seq[position:window_end]) >= threshold):
            new_base_seq = base_seq[:position]
            break
        else:
            continue
    if (len(new_base_seq) >= min_length):
        return new_base_seq
    else:
        return ''


def process_read(read,threshold,min_length):
    # backup read.qual, pysam removes it once I set red.seq
    read_qual = read.qual

    read.seq = base_quality_trim(read.seq, read.qual, threshold, min_length)
    if len(read.seq)>0:
        read.qual = read_qual[0:len(read.seq)]


def bam_preproc(path_to_input_bam, path_to_output_file, threshold, min_length):

    input_bam = pysam.Samfile(path_to_input_bam, mode="rb",check_header=False,check_sq=False)
    #output_bam = pysam.Samfile(path_to_output_file, mode="wb",template=input_bam,check_header=False,check_sq=False)
    output_fastq = open(path_to_output_file, 'w')
    for x in input_bam.fetch(until_eof=True):
        process_read(x,threshold,min_length)
        #output_bam.write(x)
        output_fastq.write("@%s\n%s\n+\n%s\n" % (x.qname,x.seq,x.qual))
    input_bam.close()
    #output_bam.close()
    output_fastq.close()

if __name__=='__main__':
    cwd = os.getcwd() # current working directory
    PATH_TO_BAM = sys.argv[1]
    print 'PATH_TO_BAM',PATH_TO_BAM
    PATH_TO_BAM.replace(" ","") #remove white space
    BARCODING_USED = sys.argv[2]
    print 'BARCODING_USED',BARCODING_USED
    RESULTS_DIR = sys.argv[3]
    print 'RESULTS_DIR',RESULTS_DIR
    version_txt = open(RESULTS_DIR+'/../../version.txt', 'r')
    for line in version_txt:
        x = line.split("=")
        if x[0] == "Torrent_Suite":
            version = x[1]
            break
    try:
        version_nbr = float(version[0:3])
    except:
        version_nbr = 0
    print 'version_nbr',version_nbr
    INPUT_BAM = 'none found'
    if BARCODING_USED == 'Y':
        INPUT_BAM = PATH_TO_BAM
        filepath = PATH_TO_BAM.split('/')
        print 'filepath',filepath
        file_prefix = filepath[-1]
        file_prefix = file_prefix.rstrip()
        print 'file_prefix',file_prefix
        part_to_trim = -1-len(file_prefix)-1-len(filepath[-2])-1-len(filepath[-3])
        print 'part_to_trim',part_to_trim
        # e.g. part_to_trim = length of '/plugin_out/ERCC_Analysis_barcoding_out/IonXpress_001'
        run_dir = PATH_TO_BAM[:part_to_trim]
        print 'run_dir',run_dir
        os.chdir(run_dir)
        for file in os.listdir('.'):
            if file.startswith(file_prefix) and file.endswith('.bam'):
                INPUT_BAM = os.path.abspath(file)
                if os.path.islink(INPUT_BAM): #we want the real thing, not a symlink
                    INPUT_BAM = os.path.realpath(INPUT_BAM)
                    if os.path.exists(INPUT_BAM):
                        print 'it really exists'
                        pass
                    else:
                        INPUT_BAM = 'none found'
                if INPUT_BAM != 'none found':
                    break
        if INPUT_BAM == 'none found':
            os.chdir(run_dir+'/basecaller_results')
            for file in os.listdir('.'):
                if file.startswith(file_prefix) and file.endswith('.bam'):
                    INPUT_BAM = run_dir+'basecaller_results/'+file
                    if os.path.islink(INPUT_BAM): #we want the real thing, not a symlink
                        INPUT_BAM = os.path.realpath(INPUT_BAM)
                        if os.path.exists(INPUT_BAM):
                            print 'it really exists'
                            pass
                        else:
                            INPUT_BAM = 'none found'
                    if INPUT_BAM != 'none found':
                        break
    elif version_nbr < 3.4: # no barcoding
        INPUT_BAM = PATH_TO_BAM
        filepath = PATH_TO_BAM.split('/')
        print 'filepath',filepath
        file_name = filepath[-1]
        print 'file_name',file_name
        run_dir = ''
        for chunk in filepath:
            if chunk not in ['',' ','.']:
                run_dir += '/'
                run_dir += chunk
            elif run_dir == '': #nothing found yet
                continue
            else:
                break
        print 'run_dir',run_dir
        INPUT_BAM = 'none found'
        os.chdir(run_dir)
        for file in os.listdir('.'):
            if file == file_name:
                if os.path.islink(file):
                    continue
                else:
                    INPUT_BAM = os.path.abspath(file)
                    break
        if INPUT_BAM == 'none found':
            file_fixes = file_name.split('.')
            file_prefix = file_fixes[0]
            file_suffix = file_fixes[-1]
            os.chdir(run_dir+'/basecaller_results')
            for file in os.listdir('.'):
                if file.startswith(file_prefix) and file.endswith(file_suffix):
                    print 'found it in basecaller_results'
                    INPUT_BAM = run_dir+'/basecaller_results/'+file
    else: # run in 3.4 or later, no barcoding
        INPUT_BAM = PATH_TO_BAM
    print 'INPUT_BAM',INPUT_BAM
    if INPUT_BAM != 'none found':
        print 'writing filtered.fastq at',RESULTS_DIR
        bam_preproc(INPUT_BAM, RESULTS_DIR+'/filtered.fastq',15,20)
    os.chdir(cwd)