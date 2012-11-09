# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
#pre-process fastq files by filtering for base quality and then for length
from __future__ import division
from itertools import izip

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

def process_4line_block(four_lines,threshold,min_length):
    output_lines = ['','','','']
    base_seq = four_lines[1]
    quality_metric_seq = four_lines[3]
    output_lines[1] = base_quality_trim(base_seq, quality_metric_seq, threshold, min_length)+ '\n'
    if (len(output_lines[1])>1):
        output_lines[0] = four_lines[0]
        output_lines[2] = four_lines[2]
        output_lines[3] = quality_metric_seq[0:(len(output_lines[1])-1)]
 	if (output_lines[3][-1] != "\n"): #carriage return was trimmed by above line
 	    output_lines[3] = output_lines[3] + "\n"
        return ''.join(output_lines)

def fastq_preproc(path_to_fastq_input,path_to_fastq_output,threshold, min_length):
    input_file = open(path_to_fastq_input,'r')
    output_file = open(path_to_fastq_output,'w')
    lines_from_input = []
    no_more_to_read = False
    while True:
        if ((no_more_to_read == False) and (len(lines_from_input) < 4)):
            new_lines_read = input_file.readlines(5000)
            if (len(new_lines_read)==0):
		no_more_to_read = True
            for new_line in new_lines_read:
                lines_from_input.append(new_line)
        if (len(lines_from_input)>=4):
            try:
                output_file.write(process_4line_block(lines_from_input[0:4],threshold,min_length))
            except (TypeError): #process_4line_block didn't return anything, perhaps bcs less than min_length
                pass
            for i in range(4):
                lines_from_input.pop(0)
        if (no_more_to_read and (len(lines_from_input)<4)):
            break
    #for output_block in process_4line_blocks(input_file,threshold, min_length):
    #    output_file.write(output_block)
    input_file.close()
    output_file.close()
