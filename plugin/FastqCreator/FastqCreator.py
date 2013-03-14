#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# FastqCreator plugin
import os
import glob
import json
import traceback
import pysam
from ion.plugin import *


class FastqCreator(IonPlugin):
    version = "3.4.49884"
    runtypes = [ RunType.COMPOSITE, RunType.THUMB, RunType.FULLCHIP ]

    def bam2fastq(self, bam_filename_list, fastq_filename):
        try:
            with open(fastq_filename, 'w') as fastq_file:

                for bam_file in bam_filename_list:
                    if os.path.exists(bam_file):
                        try:
                            samfile = pysam.Samfile(bam_file, mode="rb",check_header=False,check_sq=False)
                            for x in samfile.fetch(until_eof=True):
                                fastq_file.write("@%s\n%s\n+\n%s\n" % (x.qname,x.seq,x.qual))
                            samfile.close()
                        except:
                            traceback.print_exc()
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
                self.bam2fastq(bam_list,dataset['fastq'])
            except:
                traceback.print_exc()


        with open('FastqCreator_block.html','w') as f:
            f.write('<html><body>To download: "Right Click" -> "Save Link As..."<br>\n')

            for fastq_file in glob.glob('*.fastq'):
                size = os.path.getsize(fastq_file)/1000
                f.write('<a href="%s">%s</a> %sK<br>\n' % (os.path.join(net_location, url_path, fastq_file), fastq_file, size))
            f.write('</body></html>\n')

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
