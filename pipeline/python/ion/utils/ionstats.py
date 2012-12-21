#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved


import subprocess
import json
import traceback

from ion.utils.blockprocessing import printtime


''' Invoke ionstats quality to generate alignment-independent metrics for a single unmapped BAM '''

def generate_ionstats_basecaller(unmapped_bam_filename,ionstats_basecaller_filename,histogram_length):
    
    try:
        com = "ionstats basecaller"
        com += " -i %s" % (unmapped_bam_filename)
        com += " -o %s" % (ionstats_basecaller_filename)
        com += " -h %d" % (int(histogram_length))
        printtime("DEBUG: Calling '%s'" % com)
        subprocess.call(com,shell=True)
    except:
        printtime('Failed ionstats basecaller')
        traceback.print_exc()

''' Invoke ionstats quality to generate alignment-independent metrics for a single unmapped BAM '''

def generate_ionstats_tf(tf_bam_filename,tfref_fasta_filename,ionstats_tf_filename):
    
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
        com = "ionstats reduce"
        com += " -o %s" % (output_filename)
        com += " " + " ".join(input_filename_list)
        printtime("DEBUG: Calling '%s'" % com)
        subprocess.call(com,shell=True)
    except:
        printtime('Failed ionstats reduce')
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
            }

            
        f = open(tfstats_json_filename,'w')
        f.write(json.dumps(tfstats_json, indent=4))
        f.close()

    except:
        printtime('Failed to generate %s' % (tfstats_json_filename))
        traceback.print_exc()


if __name__=="__main__":
    
    #generate_legacy_basecaller_files ('ionstats_basecaller.json', '')
    generate_legacy_tf_files ('ionstats_tf.json', 'TFStats2.json')

    








