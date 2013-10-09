#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# FastqCreator plugin
import os
import glob
import json
import traceback
import pysam
import subprocess
from ion.utils import blockprocessing
from ion.plugin import *

# DNA base complements
COMPLEMENT = {'A': 'T',
              'T': 'A',
              'C': 'G',
              'G': 'C',
              'N': 'N'}

def reverse_complement(sequence):
    return ''.join(COMPLEMENT[b] for b in sequence[::-1])

class FastqCreator(IonPlugin):
    version = "3.6.0-r%s" % filter(str.isdigit,"$Revision: 57238 $")
    runtypes = [ RunType.COMPOSITE, RunType.THUMB, RunType.FULLCHIP ]

    def bam2fastq(self, bam_filename_list, fastq_filename):
        try:
            with open(fastq_filename, 'w') as fastq_file:

                for bam_file in bam_filename_list:
                    if os.path.exists(bam_file):
                        try:
                            samfile = pysam.Samfile(bam_file, mode="rb",check_header=False,check_sq=False)
                            for x in samfile.fetch(until_eof=True):
                                if x.is_reverse:
                                    qual = x.qual[::-1]
                                    seq = reverse_complement(x.seq)
                                else:
                                    qual = x.qual
                                    seq = x.seq
                                fastq_file.write("@%s\n%s\n+\n%s\n" % (x.qname,seq,qual))
                            samfile.close()
                        except:
                            traceback.print_exc()
        except:
            traceback.print_exc()

    def bam2fastq_picard(self, bam_filename_list, fastq_filename):
        try:
            com = blockprocessing.bam2fastq_command(bam_filename_list[0],fastq_filename)
            ret = subprocess.call(com,shell=True)
        except:
            traceback.print_exc()

    def launch(self, data=None):

        with open('startplugin.json', 'r') as fh:
            spj = json.load(fh)
            net_location = spj['runinfo']['net_location']
            basecaller_dir = spj['runinfo']['basecaller_dir']
            alignment_dir = spj['runinfo']['alignment_dir']
            barcodeId = spj['runinfo']['barcodeId']
            runtype = spj['runplugin']['run_type']

        url_path = os.getenv('TSP_URLPATH_PLUGIN_DIR','./')
        print url_path
        output_stem = os.getenv('TSP_FILEPATH_OUTPUT_STEM','unknown')
        print "TSP_FILEPATH_OUTPUT_STEM: %s" % output_stem

        reference_path = os.getenv('TSP_FILEPATH_GENOME_FASTA','')

        with open(os.path.join(basecaller_dir, "datasets_basecaller.json"),'r') as f:
            datasets_basecaller = json.load(f);

        for dataset in datasets_basecaller['datasets']:
            print dataset

            # input
            bam_list = []
            if reference_path != '':
                # don't use aligned bam files (TS-6279) or reverse-complemented it
                bam = os.path.join(alignment_dir, dataset['file_prefix']+'.bam')
            else:
                bam = os.path.join(basecaller_dir, dataset['file_prefix']+'.basecaller.bam')
            print bam
            if os.path.exists(bam):
                bam_list.append(bam)

            # output
            if barcodeId:
                dataset['fastq'] = dataset['file_prefix'].rstrip('_rawlib')+'_'+output_stem+'.fastq'
            else:
                dataset['fastq'] = output_stem+'.fastq'

            if len(bam_list) == 0:
                print 'WARNING: missing input file(s) for %s' % dataset['fastq']
                continue

            try:
                #use pysam only for unmapped bam files
                #self.bam2fastq_pysam(bam_list,dataset['fastq'])
                self.bam2fastq_picard(bam_list,dataset['fastq'])
            except:
                traceback.print_exc()
		
        return True

    def report(self):
        output = {
            'sections': {
                'title': 'FastqCreator',
                'type': 'html',
                'content': '<p>FastqCreator util</p>',
            },
        }
        return output

    def metrics(self):
        """ Write result.json metrics """
        return { 'blocks': 96 }

# dev use only - makes testing easier
if __name__ == "__main__": PluginCLI(FastqCreator())
