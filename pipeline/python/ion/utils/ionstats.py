#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved


import subprocess
import json
import traceback

from ion.utils.blockprocessing import printtime


''' Invoke ionstats basecaller to generate alignment-independent metrics for unmapped BAM files'''

def generate_ionstats_basecaller(unmapped_bam_filenames, ionstats_basecaller_filename, histogram_length):

    try:
        com = "ionstats basecaller"
        com += " -i %s" % (unmapped_bam_filenames[0])
        for unmapped_bam_filename in unmapped_bam_filenames[1:]:
            com += ",%s" % (unmapped_bam_filename)
        com += " -o %s" % (ionstats_basecaller_filename)
        com += " -h %d" % (int(histogram_length))
        printtime("DEBUG: Calling '%s'" % com)
        subprocess.call(com,shell=True)
    except:
        printtime('Failed ionstats basecaller')
        traceback.print_exc()


''' Invoke ionstats alignment to generate alignment-based metrics for a mapped BAM files'''

def generate_ionstats_alignment(bam_filenames, ionstats_alignment_filename, ionstats_alignment_h5_filename, basecaller_json, histogram_length):

    try:
        com = "ionstats alignment"
        com += " -i %s" % (bam_filenames[0])
        for bam_filename in bam_filenames[1:]:
            com += ",%s" % (bam_filename)
        com += " -o %s" % (ionstats_alignment_filename)
        com += " -h %d" % (int(histogram_length))

        if basecaller_json:
            block_col_offset = basecaller_json["BaseCaller"]['block_col_offset']
            block_row_offset = basecaller_json["BaseCaller"]['block_row_offset']
            block_col_size   = basecaller_json["BaseCaller"]['block_col_size']
            block_row_size   = basecaller_json["BaseCaller"]['block_row_size']
            subregion_col_size, subregion_row_size = generate_ionstats_subregion_dims(block_col_size, block_row_size)

            com += " --evaluate-hp true"
            com += " --output-h5 %s" % ionstats_alignment_h5_filename
            com += " --chip-origin %s,%s" % (block_col_offset, block_row_offset)
            com += " --chip-dim %s,%s" % (block_col_size, block_row_size)
            com += " --subregion-dim %s,%s" % (subregion_col_size, subregion_row_size)

        printtime("DEBUG: Calling '%s'" % com)
        subprocess.call(com,shell=True)
    except:
        printtime('Failed ionstats alignment')
        traceback.print_exc()


''' Invoke ionstats tf to generate test fragment statistics from a BAM mapped to TF reference '''

def generate_ionstats_tf(tf_bam_filename, tfref_fasta_filename, ionstats_tf_filename):
    
    try:
        com = "ionstats tf"
        com += " -i %s" % (tf_bam_filename)
        com += " -r %s" % (tfref_fasta_filename)
        com += " -o %s" % (ionstats_tf_filename)
        printtime("DEBUG: Calling '%s'" % com)
        subprocess.call(com,shell=True)
    except:
        printtime('Failed ionstats tf')
        traceback.print_exc()

''' Invoke ionstats reduce to combine multiple ionstats json files by merging the metrics '''

def reduce_stats (input_filename_list, output_filename):

    try:
        #need to copy, cannot index an iterator
        copy_input_filename_list = list(input_filename_list)
        length=len(copy_input_filename_list)

        # process file list in smaller intervalls
        size = 100
        i=0
        while (i<length):
            if i+size<length:
                input_files = copy_input_filename_list[i:i+size]
                output_file = output_filename+"."+str(i+size)
            else:
                input_files = copy_input_filename_list[i:length]
                output_file = output_filename
            # add results from earlier iterations
            if i>0:
                input_files=input_files+[output_filename+"."+str(i)]
            i=i+size

            com = "ionstats reduce"
            com += " -o %s" % (output_file)
            com += " " + " ".join(input_files)
            printtime("DEBUG: Calling '%s'" % com)
            subprocess.call(com,shell=True)
    except:
        printtime('Failed ionstats reduce')
        traceback.print_exc()


def reduce_stats_h5 (input_filename_list, output_filename):

    try:
        #need to copy, cannot index an iterator
        copy_input_filename_list = list(input_filename_list)
        length=len(copy_input_filename_list)

        # process file list in smaller intervalls
        size = 100
        i=0
        while (i<length):
            if i+size<length:
                input_files = copy_input_filename_list[i:i+size]
                output_file = output_filename+"."+str(i+size)
            else:
                input_files = copy_input_filename_list[i:length]
                output_file = output_filename
            # add results from earlier iterations
            if i>0:
                input_files=input_files+[output_filename+"."+str(i)]
            i=i+size

            com = "ionstats reduce-h5"
            com += " -o %s" % (output_file)
            com += " " + " ".join(input_files)
            printtime("DEBUG: Calling '%s'" % com)
            subprocess.call(com,shell=True)
    except:
        printtime('Failed ionstats reduce-h5')
        traceback.print_exc()


''' Use ionstats_quality.json file to generate legacy files: quality.summary '''

def generate_legacy_basecaller_files (ionstats_basecaller_filename, legacy_filename_prefix):

    try:
        f = open(ionstats_basecaller_filename,'r')
        ionstats_basecaller = json.load(f);
        f.close()
    except:
        printtime('Failed to load %s' % (ionstats_basecaller_filename))
        traceback.print_exc()
        return

    # Generate quality.summary (based on quality.json content
    
    quality_summary_filename = legacy_filename_prefix+'quality.summary'
    try:
        f = open(quality_summary_filename,'w')
        f.write("[global]\n")

        #f.write("Number of Bases at Q0 = %d\n" % ionstats_basecaller['full']['num_bases'])
        f.write("Number of Bases at Q0 = %d\n" % sum(ionstats_basecaller['qv_histogram']))
        f.write("Number of Reads at Q0 = %d\n" % ionstats_basecaller['full']['num_reads'])
        f.write("Max Read Length at Q0 = %d\n" % ionstats_basecaller['full']['max_read_length'])
        f.write("Mean Read Length at Q0 = %d\n" % ionstats_basecaller['full']['mean_read_length'])
        read_length_histogram = ionstats_basecaller['full']['read_length_histogram']
        if len(read_length_histogram) > 50:
            f.write("Number of 50BP Reads at Q0 = %d\n" % sum(read_length_histogram[50:]))
        if len(read_length_histogram) > 100:
            f.write("Number of 100BP Reads at Q0 = %d\n" % sum(read_length_histogram[100:]))
        if len(read_length_histogram) > 150:
            f.write("Number of 150BP Reads at Q0 = %d\n" % sum(read_length_histogram[150:]))
        
        #f.write("Number of Bases at Q17 = %d\n" % ionstats_basecaller['Q17']['num_bases'])
        f.write("Number of Bases at Q17 = %d\n" % sum(ionstats_basecaller['qv_histogram'][17:]))
        f.write("Number of Reads at Q17 = %d\n" % ionstats_basecaller['Q17']['num_reads'])
        f.write("Max Read Length at Q17 = %d\n" % ionstats_basecaller['Q17']['max_read_length'])
        f.write("Mean Read Length at Q17 = %d\n" % ionstats_basecaller['Q17']['mean_read_length'])
        read_length_histogram = ionstats_basecaller['Q17']['read_length_histogram']
        if len(read_length_histogram) > 50:
            f.write("Number of 50BP Reads at Q17 = %d\n" % sum(read_length_histogram[50:]))
        if len(read_length_histogram) > 100:
            f.write("Number of 100BP Reads at Q17 = %d\n" % sum(read_length_histogram[100:]))
        if len(read_length_histogram) > 150:
            f.write("Number of 150BP Reads at Q17 = %d\n" % sum(read_length_histogram[150:]))

        #f.write("Number of Bases at Q20 = %d\n" % ionstats_basecaller['Q20']['num_bases'])
        f.write("Number of Bases at Q20 = %d\n" % sum(ionstats_basecaller['qv_histogram'][20:]))
        f.write("Number of Reads at Q20 = %d\n" % ionstats_basecaller['Q20']['num_reads'])
        f.write("Max Read Length at Q20 = %d\n" % ionstats_basecaller['Q20']['max_read_length'])
        f.write("Mean Read Length at Q20 = %d\n" % ionstats_basecaller['Q20']['mean_read_length'])
        read_length_histogram = ionstats_basecaller['Q20']['read_length_histogram']
        if len(read_length_histogram) > 50:
            f.write("Number of 50BP Reads at Q20 = %d\n" % sum(read_length_histogram[50:]))
        if len(read_length_histogram) > 100:
            f.write("Number of 100BP Reads at Q20 = %d\n" % sum(read_length_histogram[100:]))
        if len(read_length_histogram) > 150:
            f.write("Number of 150BP Reads at Q20 = %d\n" % sum(read_length_histogram[150:]))

        read_length_histogram = ionstats_basecaller['full']['read_length_histogram']
        if len(read_length_histogram) > 50:
            f.write("Number of 50BP Reads = %d\n" % sum(read_length_histogram[50:]))
        if len(read_length_histogram) > 100:
            f.write("Number of 100BP Reads = %d\n" % sum(read_length_histogram[100:]))
        if len(read_length_histogram) > 150:
            f.write("Number of 150BP Reads = %d\n" % sum(read_length_histogram[150:]))

        f.write("System SNR = %1.1f\n" % ionstats_basecaller['system_snr']) # this metric is obsolete
        
        f.close()
    except:
        printtime('Failed to generate %s' % (quality_summary_filename))
        traceback.print_exc()
        


''' Allow for specification of subregion sizes that depend on the chip/block dimensions '''

def generate_ionstats_subregion_dims(block_col_size, block_row_size):

    try:
        subregion_col_size = 92
        subregion_row_size = 74
        if (block_col_size == 1200 and block_row_size == 800): # Proton thumbnail
            subregion_col_size = 50
            subregion_row_size = 50
        elif ((block_col_size == 15456 and block_row_size == 10656) or (block_col_size == 1288 and block_row_size == 1332)): # P1
            subregion_col_size = 184
            subregion_row_size = 148
        elif ((block_col_size ==  7680 and block_row_size ==  5312) or (block_col_size ==  640 and block_row_size ==  664)): # P0
            subregion_col_size = 80
            subregion_row_size = 83
        elif (block_col_size == 3392 and block_row_size == 3792): # 318
            subregion_col_size = 53
            subregion_row_size = 48
        elif (block_col_size == 3392 and block_row_size == 2120): # 316v2
            subregion_col_size = 53
            subregion_row_size = 53
        elif (block_col_size == 2736 and block_row_size == 2640): # 316
            subregion_col_size = 48
            subregion_row_size = 48
        elif (block_col_size == 1280 and block_row_size == 1152): # 314
            subregion_col_size = 40
            subregion_row_size = 48
        return(subregion_col_size, subregion_row_size)
    except:
        printtime('Failed to generate subregion dims from input %s,%s' % (block_col_size, block_row_size))
        traceback.print_exc()




''' Use ionstats_tf.json file to generate legacy files: TFStats.json '''

def generate_legacy_tf_files (ionstats_tf_filename, tfstats_json_filename):

    try:
        f = open(ionstats_tf_filename,'r')
        ionstats_tf = json.load(f);
        f.close()
        
        tfstats_json = {}
        for tf_name,tf_data in ionstats_tf['results_by_tf'].iteritems():
            
            tfstats_json[tf_name] = {
                'TF Name' : tf_name,
                'TF Seq' : tf_data['sequence'],
                'Num' : tf_data['full']['num_reads'],
                'System SNR' : tf_data['system_snr'],
                'Per HP accuracy NUM' : tf_data['hp_accuracy_numerator'],
                'Per HP accuracy DEN' : tf_data['hp_accuracy_denominator'],
                'Q10' : tf_data['AQ10']['read_length_histogram'],
                'Q17' : tf_data['AQ17']['read_length_histogram'],
                'Q10 Mean' : tf_data['AQ10']['mean_read_length'],
                'Q17 Mean' : tf_data['AQ17']['mean_read_length'],
                '50Q10' : sum(tf_data['AQ10']['read_length_histogram'][50:]),
                '50Q17' : sum(tf_data['AQ17']['read_length_histogram'][50:]),
                '100Q17' : sum(tf_data['AQ17']['read_length_histogram'][100:]),
            }

            
        f = open(tfstats_json_filename,'w')
        f.write(json.dumps(tfstats_json, indent=4))
        f.close()

    except:
        printtime('Failed to generate %s' % (tfstats_json_filename))
        traceback.print_exc()


