# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from __future__ import division
from __future__ import print_function
import sys
import re

def parse_transcript_line(line,fwd_only=False):
    parsed_line = line.split()
    if parsed_line[0][0] == '@':
        return '@', 'na' #header line or other non-data
    elif fwd_only and parsed_line[1] != '0': #something other than primary mapping, forward strand
        return '*', 'na'
    elif int(parsed_line[4]) <= 50: #low mapping quality
        return '*', 'na'
    else:
        return parsed_line[2], int(parsed_line[4])

def process_sam_file(path_to_raw_sam_file,path_to_filtered_sam_file,transcript_names,fwd_only=False):
    result_counts = {}
    all_ercc_counts = 0
    total_counts = 0
    total_mapqs = {}
    mean_mapqs = {}
    filtered_sam_file = open(path_to_filtered_sam_file,'w')
    #todo: check if path_to_raw_sam_file is a valid file name
    with open(path_to_raw_sam_file) as sam_file:
        for line in sam_file:
            transcript_name, transcript_mapq = parse_transcript_line(line,fwd_only)
            # if transcript_name == '*':
            if transcript_name == '@':
                filtered_sam_file.write(line)
            # GDM: changed so only ERCCs in the provided list are counted
            #elif re.search('^ERCC-\d+$',transcript_name) == None:
            elif not transcript_name in transcript_names:
                total_counts += 1
            elif transcript_name in result_counts:
                result_counts[transcript_name] += 1
                total_mapqs[transcript_name] += transcript_mapq
                total_counts += 1
                all_ercc_counts += 1
                mean_mapqs[transcript_name] = total_mapqs[transcript_name]/result_counts[transcript_name]
                filtered_sam_file.write(line)
            else:
                result_counts[transcript_name] = 1
                total_mapqs[transcript_name] = transcript_mapq
                total_counts += 1
                all_ercc_counts += 1
                mean_mapqs[transcript_name] = total_mapqs[transcript_name]
                filtered_sam_file.write(line)
    return result_counts, all_ercc_counts, total_counts, mean_mapqs

def write_output_counts_file(path_to_raw_sam_file,path_to_filtered_sam_file,path_to_output_counts_file,transcript_names,fwd_only=False):
    result_counts, all_ercc_counts, total_counts, mean_mapqs = process_sam_file(path_to_raw_sam_file,path_to_filtered_sam_file,transcript_names,fwd_only)
    #todo: check that result_counts is a valid result
    #todo: check that the path_to_output_counts_file is valid
    with open(path_to_output_counts_file,'w') as fout:
        for transcript_name in transcript_names:
            if transcript_name in result_counts:
                line_to_write = transcript_name + '\t' + str(result_counts[transcript_name]) + '\n'
                fout.write(line_to_write)
            else:
                line_to_write = transcript_name + '\t' + '0' + '\n'
                fout.write(line_to_write)
    return result_counts, all_ercc_counts, total_counts, mean_mapqs
