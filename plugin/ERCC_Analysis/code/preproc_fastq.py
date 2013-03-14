# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
#pre-process mapped bam files by filtering for base quality and then for length

from __future__ import division
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
