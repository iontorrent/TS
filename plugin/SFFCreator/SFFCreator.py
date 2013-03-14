#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# SFFCreator plugin
import os
import sys
import glob
import json
import subprocess
import traceback
from ion.plugin import *


class SFFCreator(IonPlugin):
    version = "3.4.49884"
    runtypes = [ RunType.COMPOSITE, RunType.THUMB, RunType.FULLCHIP ]


    def bam2sff_command(self, bam_file_list, sff_file):
        com = "bam2sff"
        com += " -o %s"  % sff_file
        for bam_file in bam_file_list:
            if os.path.exists(bam_file):
                com += " %s" % bam_file
        return com

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
                dataset['sff'] = dataset['file_prefix'].rstrip('_rawlib')+'_'+output_stem+'.sff'
            else:
                dataset['sff'] = output_stem+'.sff'

            if len(bam_list) == 0:
                print 'WARNING: missing input files for %s' % dataset['sff']
                continue

            try:
                com = self.bam2sff_command(bam_list,dataset['sff'])
                print com
                ret = subprocess.call(com,shell=True)
            except:
                traceback.print_exc()


        with open('SFFCreator_block.html','w') as f:
            f.write('<html><body>To download: "Right Click" -> "Save Link As..."<br>\n')

            for sff_file in glob.glob('*.sff'):
                size = os.path.getsize(sff_file)/1000
                f.write('<a href="%s">%s</a> %sK<br>\n' % (os.path.join(net_location, url_path, sff_file), sff_file, size))
            f.write('</body></html>\n')

        return True

    def report(self):
        output = {
            'sections': {
                'title': 'SFFCreator',
                'type': 'html',
                'content': '<p>SFFCreator util</p>',
            },
        }
        return output

    def metrics(self):
        """ Write result.json metrics """
        return { 'blocks': 96 }

# dev use only - makes testing easier
if __name__ == "__main__": PluginCLI(SFFCreator())
