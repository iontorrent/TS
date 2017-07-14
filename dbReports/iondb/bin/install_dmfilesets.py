#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
#
# This is probably going to be a messy file to maintain.
# One pointer is that whenever making changes to the include list of SIG,BASE,OUT,
# always update the EXCLUDE list of INTR.
#
# The file_selector function will use the following regular expression to find files:
#    re.compile(r'(%s/)(%s)' % (start_dir,pattern))
# where start_dir is either the fullpath to the raw data directory or the results directory.
#
# Some additional tips:
# For a complete filename, anywhere in the directory tree, put the entire filename: ie, "barcodeList.txt"
# To match files ONLY in top level directory, ".[^/]*?arcodeList.txt" - need to remove at least one char from beginning and prepend the funny stuff.
# r'.[^/]*?' will match all characters except forward slash.
#
# There is a special code hack that filters onboard_results when dealing with a thumbnail dataset in dmactions._file_selector().
#
import sys
import traceback
from iondb.bin import djangoinit
from iondb import settings
from iondb.rundb import models
from iondb.rundb.data import dmactions_types
#pylint: disable=W1401
DM_FILE_SETS = [
    {
        'type': dmactions_types.SIG,
        'description': 'Required input files for signal processing',
        'include': [
            '.[^/]*\.dat',
            'X\d+_Y\d+.*\.dat',
            'thumbnail.*\.dat',
            'thumbnail/explog_final.txt',
            'thumbnail/explog_final.json',
            #'.[^/]*?\.txt',    need to be specific
            '.[^/]*?alW1Step_explog\.txt',
            '.[^/]*?hecksum_status\.txt',
            '.[^/]*?xplog\.txt',
            '.[^/]*?xplog\.json',
            '.[^/]*?xplog_final\.txt',
            '.[^/]*?xplog_final\.json',
            '.[^/]*?nitLog.*?txt',
            '.[^/]*?nitValsW.*?txt',
            '.[^/]*?awInit\.txt',
            '.[^/]*?cript_.*?txt',
            '.[^/]*?curves\.txt',
            '.[^/]*?ettling_coeff_vals\.txt',
            '.[^/]*?ross_talk_vectors\.txt',
            '.[^/]*?\.log',
            'Controller',
            'DataCollect\.config',
            'debug',
            'Gain.lsr',
            'thumbnail/Gain.lsr',
            # not required, but want to clean these up if exist
            ],
        'exclude':[
            '.[^/]*?onboard_results.*?',
            '.*?expMeta\.dat',
            '.*?histo\.dat',
            ],
        'keepwith':{
            dmactions_types.BASE: [
                '.[^/]*?xplog\.txt',
                '.[^/]*?xplog\.json',
            ],
        },
        'version': settings.RELVERSION,
        'auto_trigger_age': '21',
        'auto_trigger_usage': '90',
        'auto_action': 'OFF',
        'del_empty_dir': True,
    },
    {
        'type': dmactions_types.BASE,
        'description': 'Required input files for basecalling',
        'include': [
            'sigproc_results/1\.wells',
            'sigproc_results/.[^/]*?\.bin',
            'sigproc_results/bfmask\.stats',
            'sigproc_results/analysis\.bfmask\.stats',
            'sigproc_results/analysis_return_code\.txt',
            'sigproc_results/avgNukeTrace_ATCG\.txt',
            'sigproc_results/avgNukeTrace_TCAG\.txt',
            'sigproc_results/Bead_density_.*?\.png',
            'sigproc_results/sigproc\.log',
            'sigproc_results/processParameters\.txt',
            'onboard_results/sigproc_results/.*?block_.*?/1\.wells',
            'onboard_results/sigproc_results/.*?block_.*?/.[^/]*?\.bin',
            'onboard_results/sigproc_results/.*?block_.*?/bfmask\.stats',
            'onboard_results/sigproc_results/.*?block_.*?/sigproc\.log',
            'onboard_results/sigproc_results/.*?block_.*?/processParameters\.txt',
            'onboard_results/sigproc_results/.*?block_.*?/analysis\.bfmask\.stats',
            'onboard_results/sigproc_results/.*?block_.*?/analysis_return_code\.txt',
            'onboard_results/sigproc_results/.*?block_.*?/avgNukeTrace_ATCG\.txt',
            'onboard_results/sigproc_results/.*?block_.*?/avgNukeTrace_TCAG\.txt',
            'onboard_results/sigproc_results/.*?block_.*?/Bead_density_.*?\.png',
            'onboard_results/sigproc_results/analysis\.bfmask\.stats',
            'onboard_results/sigproc_results/analysis_return_code\.txt',
            'onboard_results/sigproc_results/avgNukeTrace_ATCG\.txt',
            'onboard_results/sigproc_results/avgNukeTrace_TCAG\.txt',
            'onboard_results/sigproc_results/Bead_density_.*?\.png',
            '.[^/]*?xplog\.txt',
            '.[^/]*?xplog\.json',
            ],
        'exclude':[
            ],
        'keepwith':{
            dmactions_types.OUT: [
                'sigproc_results/analysis\.bfmask\.stats',
                'onboard_results/sigproc_results/analysis\.bfmask\.stats',
                'sigproc_results/Bead_density_.*?\.png',
            ],
            dmactions_types.SIG:[
                '.[^/]*?xplog\.txt',
                '.[^/]*?xplog\.json',
            ],
        },
        'version': settings.RELVERSION,
        'auto_trigger_age': '60',
        'auto_trigger_usage': '90',
        'auto_action': 'OFF',
        'del_empty_dir': True,
    },
    {
        'type': dmactions_types.OUT,
        'description': 'Report rendering, deliverables, plugins output',
        'include': [
            'alignment.*?\.summary',
            'rawlib.alignment.summary',
            '.[^/]*?efault_Report.php',
            '.[^/]*?arsefiles.php',
            'barcodeList\.txt',
            'primary.key',
            '.[^/]*?\.histo.dat',
            'expMeta.dat',
            'ion_analysis_00.py',
            'ion_params_00.json',
            'ionstats_alignment.json',
            'alignment_barcode_summary\.csv',
            'raw_peak_signal',
            '.[^/]*?\.bam',
            '.[^/]*?\.bai',
            'pgm_logs.zip',
            '.[^/]*?\.png',
            'plugin_out/.*?',
            '.[^/]*?\.fasta',
            '.[^/]*?\.fai',
            'bam.header.sam',
            'drmaa_stderr_block.txt',
            'drmaa_stdout_block.txt',
            'drmaa_stdout.txt',
            'version.txt',
            'basecaller_results/.[^/]*?\.json',
            'basecaller_results/.[^/]*?\.png',
            'basecaller_results/quality\.summary',
            'basecaller_results/basecaller\.log',
            'basecaller_results/.[^/]*\.bam',
            'sigproc_results/.[^/]*?\.json',
            'sigproc_results/analysis.bfmask.stats',
            'onboard_results/sigproc_results/analysis\.bfmask\.stats',
            'sigproc_results/Bead_density_.*?\.png',
            'bc_files/.*?',
            'bc_filtered/.*?',
            'download_links/.*?',
            '.[^/]*?eport_layout\.json',
            'CA_barcode_summary.json',
            '.*NucStep/.*',
            '.*dcOffset/.*',
            'status.txt',
            ],
        'exclude':[
            '.*?filtered.untrimmed.*?',
            '.*?filtered.trimmed.*?',
            '.*?unfiltered.untrimmed.*?',
            '.*?unfiltered.trimmed.*?',
            ],
        'keepwith':{
            dmactions_types.BASE: [
                'sigproc_results/analysis.bfmask.stats',
                'onboard_results/sigproc_results/analysis\.bfmask\.stats',
                'sigproc_results/Bead_density_.*?\.png',
            ],
        },
        'version': settings.RELVERSION,
        'auto_trigger_age': '180',
        'auto_trigger_usage': '90',
        'auto_action': 'OFF',
        'del_empty_dir': True,
    },
    {
        'type': dmactions_types.INTR,
        'description': 'Files used for debugging only',
        'include': [
            '.*?',
            '.*?bcfiles',
            '.*?bc_filtered',
            '.*?filtered.trimmed',
            '.*?filtered.untrimmed',
            '.*?unfiltered.trimmed',
            '.*?unfiltered.untrimmed',
            '.*?block_.*?/.*?',
            '.*?onboard_results.*?',
            # from sigproc directory
            'jpg/.*?',
            '.[^/]*?\.jpg',
            '.[^/]*?hipCalImage\.bmp\.bz2',
            'InitRawTrace0.png',
            ],
        'exclude':[
            # Specific Files
            '.[^/]*?support\.zip',
            'report.pdf',
            'backupPDF.pdf',
            '.[^/]*?\-full\.pdf',
            'pgm_logs.zip',
            # From other categories' include list
            # Signal Processing category
            '.[^/]*\.dat',
            'X\d+_Y\d+.*\.dat',
            'thumbnail.*\.dat',
            'thumbnail/explog_final.txt',
            'thumbnail/explog_final.json',
            '.*?CalW1Step_explog\.txt',
            '.*?checksum_status\.txt',
            '.*?explog\.txt',
            '.*?explog\.json',
            '.*?explog_final\.txt',
            '.*?explog_final\.json',
            '.*?InitLog.*?txt',
            '.*?InitValsW.*?txt',
            '.*?RawInit\.txt',
            '.*?Script_.*?\.txt',
            '.*?scurves\.txt',
            '.*?settling_coeff_vals\.txt',
            '.*?cross_talk_vectors\.txt',
            'status.txt',
            '.[^/]*?\.log',
            'Controller',
            'DataCollect\.config',
            'debug',
            'Gain.lsr',
            'thumbnail/Gain.lsr',
            # not required, but want to clean these up if exist
            'jpg/.*?',
            '.[^/]*?\.jpg',
            '.[^/]*?hipCalImage\.bmp\.bz2',
            # Basecaller category
            'sigproc_results/1\.wells',
            'sigproc_results/.[^/]*?\.bin',
            'sigproc_results/bfmask\.stats',
            'sigproc_results/analysis\.bfmask\.stats',
            'sigproc_results/analysis_return_code\.txt',
            'sigproc_results/avgNukeTrace_ATCG\.txt',
            'sigproc_results/avgNukeTrace_TCAG\.txt',
            'sigproc_results/Bead_density_.*?\.png',
            'sigproc_results/sigproc\.log',
            'sigproc_results/processParameters\.txt',
            'onboard_results/sigproc_results/.*?block_.*?/1\.wells',
            'onboard_results/sigproc_results/.*?block_.*?/.[^/]*?\.bin',
            'onboard_results/sigproc_results/.*?block_.*?/bfmask\.stats',
            'onboard_results/sigproc_results/.*?block_.*?/sigproc\.log',
            'onboard_results/sigproc_results/.*?block_.*?/processParameters\.txt',
            'onboard_results/sigproc_results/.*?block_.*?/analysis\.bfmask\.stats',
            'onboard_results/sigproc_results/.*?block_.*?/analysis_return_code\.txt',
            'onboard_results/sigproc_results/.*?block_.*?/avgNukeTrace_ATCG\.txt',
            'onboard_results/sigproc_results/.*?block_.*?/avgNukeTrace_TCAG\.txt',
            'onboard_results/sigproc_results/.*?block_.*?/Bead_density_.*?\.png',
            'onboard_results/sigproc_results/analysis\.bfmask\.stats',
            'onboard_results/sigproc_results/analysis_return_code\.txt',
            'onboard_results/sigproc_results/avgNukeTrace_ATCG\.txt',
            'onboard_results/sigproc_results/avgNukeTrace_TCAG\.txt',
            'onboard_results/sigproc_results/Bead_density_.*?\.png',
            # Report Output category
            'alignment.*?\.summary',
            'rawlib.alignment.summary',
            '.[^/]*?efault_Report.php',
            '.[^/]*?arsefiles.php',
            'barcodeList\.txt',
            'primary.key',
            '.[^/]*?\.histo.dat',
            'expMeta.dat',
            'ion_analysis_00.py',
            'ion_params_00.json',
            'ionstats_alignment.json',
            'alignment_barcode_summary\.csv',
            'raw_peak_signal',
            '.[^/]*?\.bam',
            '.[^/]*?\.bai',
            'pgm_logs.zip',
            '.[^/]*?\.png',
            'plugin_out/.*?',
            '.[^/]*?\.fasta',
            '.[^/]*?\.fai',
            'bam.header.sam',
            'drmaa_stderr_block.txt',
            'drmaa_stdout_block.txt',
            'drmaa_stdout.txt',
            'version.txt',
            'basecaller_results/.[^/]*?\.json',
            'basecaller_results/.[^/]*?\.png',
            'basecaller_results/basecaller\.log',
            'basecaller_results/quality\.summary',
            'basecaller_results/.[^/]*\.bam',
            'sigproc_results/.[^/]*?\.json',
            'sigproc_results/analysis.bfmask.stats',
            'sigproc_results/Bead_density_.*?\.png',
            'bc_files/.*?',
            'bc_filtered/.*?',
            'download_links/.*?',
            '.[^/]*?eport_layout\.json',
            'CA_barcode_summary.json',
            '.*NucStep/.*',
            '.*dcOffset/.*',
            ],
        'keepwith':{
            dmactions_types.BASE: [
                'sigproc_results/analysis.bfmask.stats',
                'sigproc_results/Bead_density_.*?\.png',
            ],
            dmactions_types.OUT:[
                'sigproc_results/analysis.bfmask.stats',
                'sigproc_results/Bead_density_.*?\.png',
                'plugin_out.*?',
                'pdf.*?',
            ],
        },
        'version': settings.RELVERSION,
        'auto_trigger_age': '7',
        'auto_trigger_usage': '20',
        'auto_action': 'OFF',
        'del_empty_dir': True,
    },

    # Filesets for reports created with TS2.2
    {
        'type': dmactions_types.SIG,
        'description': 'Required input files for signal processing',
        'include': [
            '.[^/]*\.dat',
            'X\d+_Y\d+.*\.dat',
            'thumbnail.*\.dat',
            #'.[^/]*?\.txt',    need to be specific
            '.[^/]*?alW1Step_explog\.txt',
            '.[^/]*?hecksum_status\.txt',
            '.[^/]*?xplog\.txt',
            '.[^/]*?xplog_final\.txt',
            '.[^/]*?nitLog.*?txt',
            '.[^/]*?nitValsW.*?txt',
            '.[^/]*?awInit\.txt',
            '.[^/]*?cript_.*?txt',
            '.[^/]*?curves\.txt',
            '.[^/]*?ettling_coeff_vals\.txt',
            '.[^/]*?\.log',
            'Controller',
            'DataCollect\.config',
            'debug',
            # not required, but want to clean these up if exist
            ],
        'exclude':[
            '.[^/]*?onboard_results.*?',
            '.*?expMeta\.dat',
            '.*?histo\.dat',
            ],
        'keepwith':{},
        'version': '2.2',
        'auto_trigger_age': '21',
        'auto_trigger_usage': '90',
        'auto_action': 'OFF',
        'del_empty_dir': True,
    },
    {
        'type': dmactions_types.BASE,
        'description': 'Required input files for basecalling',
        'include': [
            '1\.wells',
            '.[^/]*?\.bin',
            'bfmask\.stats',
            'analysis\.bfmask\.stats',
            'analysis_return_code\.txt',
            'avgNukeTrace_ATCG\.txt',
            'avgNukeTrace_TCAG\.txt',
            'Bead_density_.*?\.png',
            'sigproc\.log',
            'processParameters\.txt',
            ],
        'exclude':[
            ],
        'keepwith':{
            dmactions_types.OUT: [
                'analysis\.bfmask\.stats',
                'Bead_density_.*?\.png',
            ],
        },
        'version': '2.2',
        'auto_trigger_age': '60',
        'auto_trigger_usage': '90',
        'auto_action': 'OFF',
        'del_empty_dir': True,
    },
    {
        'type': dmactions_types.OUT,
        'description': 'Report rendering, deliverables, plugins output',
        'include': [
            'alignment.*?\.summary',
            'rawlib.alignment.summary',
            '.[^/]*?\.php',
            'barcodeList\.txt',
            'primary.key',
            '.[^/]*?\.histo.dat',
            'expMeta.dat',
            'ion_analysis_00.py',
            'ion_params_00.json',
            'raw_peak_signal',
            '.[^/]*?\.bam',
            '.[^/]*?\.bai',
            'pgm_logs.zip',
            '.[^/]*?\.png',
            'plugin_out/.*?',
            '.[^/]*?\.fasta',
            '.[^/]*?\.fai',
            'bam.header.sam',
            'drmaa_stderr_block.txt',
            'drmaa_stdout_block.txt',
            'drmaa_stdout.txt',
            'version.txt',
            '.[^/]*?\.json',
            'analysis.bfmask.stats',
            'Bead_density_.*?\.png',
            'bc_files/.*?',
            'bc_filtered/.*?',
            'download_links/.*?',
            'ReportLog.html',
            'alignTable.txt',
            'quality\.summary',
            'beadSummary.filtered.txt',
            '.[^/]*?\.sff\.zip',
            '.[^/]*?\.fastq\.zip',
            ],
        'exclude':[
            '.*?filtered.untrimmed.*?',
            '.*?filtered.trimmed.*?',
            '.*?unfiltered.untrimmed.*?',
            '.*?unfiltered.trimmed.*?',
            ],
        'keepwith':{
            dmactions_types.BASE: [
                'analysis.bfmask.stats',
                'Bead_density_.*?\.png',
            ],
        },
        'version': '2.2',
        'auto_trigger_age': '180',
        'auto_trigger_usage': '90',
        'auto_action': 'OFF',
        'del_empty_dir': True,
    },
    {
        'type': dmactions_types.INTR,
        'description': 'Files used for debugging only',
        'include': [
            '.*?',
            '.*?bcfiles',
            '.*?bc_filtered',
            '.*?filtered.trimmed',
            '.*?filtered.untrimmed',
            '.*?unfiltered.trimmed',
            '.*?unfiltered.untrimmed',
            '.*?block_.*?/.*?',
            '.*?onboard_results.*?',
            # from sigproc directory
            'jpg/.*?',
            '.[^/]*?\.jpg',
            'InitRawTrace0.png',
            ],
        'exclude':[
            # Specific Files
            '.[^/]*?support\.zip',
            'report.pdf',
            'backupPDF.pdf',
            '.[^/]*?\-full\.pdf',
            'pgm_logs.zip',
            # From other categories' include list
            # Signal Processing category
            '.[^/]*\.dat',
            'X\d+_Y\d+.*\.dat',
            'thumbnail.*\.dat',
            '.*?CalW1Step_explog\.txt',
            '.*?checksum_status\.txt',
            '.*?explog\.txt',
            '.*?explog_final\.txt',
            '.*?InitLog.*?txt',
            '.*?InitValsW.*?txt',
            '.*?RawInit\.txt',
            '.*?Script_.*?\.txt',
            '.*?scurves\.txt',
            '.*?settling_coeff_vals\.txt',
            '.*?\.log',
            'Controller',
            'DataCollect\.config',
            'debug',
            # not required, but want to clean these up if exist
            'jpg/.*?',
            '.[^/]*?\.jpg',
            # Basecaller category
            '1\.wells',
            '.[^/]*?\.bin',
            'bfmask\.stats',
            'analysis\.bfmask\.stats',
            'analysis_return_code\.txt',
            'avgNukeTrace_ATCG\.txt',
            'avgNukeTrace_TCAG\.txt',
            'Bead_density_.*?\.png',
            'sigproc\.log',
            'processParameters\.txt',
            # Report Output category
            'alignment.*?\.summary',
            'rawlib.alignment.summary',
            '.[^/]*?\.php',
            'barcodeList\.txt',
            'primary.key',
            '.[^/]*?\.histo.dat',
            'expMeta.dat',
            'ion_analysis_00.py',
            'ion_params_00.json',
            'raw_peak_signal',
            '.[^/]*?\.bam',
            '.[^/]*?\.bai',
            'pgm_logs.zip',
            '.[^/]*?\.png',
            'plugin_out/.*?',
            '.[^/]*?\.fasta',
            '.[^/]*?\.fai',
            'bam.header.sam',
            'drmaa_stderr_block.txt',
            'drmaa_stdout_block.txt',
            'drmaa_stdout.txt',
            'version.txt',
            '.[^/]*?\.json',
            'analysis.bfmask.stats',
            'Bead_density_.*?\.png',
            'bc_files/.*?',
            'bc_filtered/.*?',
            'download_links/.*?',
            'ReportLog.html',
            'alignTable.txt',
            'quality\.summary',
            'beadSummary.filtered.txt',
            '.[^/]*?\.sff\.zip',
            '.[^/]*?\.fastq\.zip',
            ],
        'keepwith':{
            dmactions_types.BASE: [
                'analysis.bfmask.stats',
                'Bead_density_.*?\.png',
            ],
            dmactions_types.OUT:[
                'analysis.bfmask.stats',
                'Bead_density_.*?\.png',
                'plugin_out.*?',
                'pdf.*?',
            ],
        },
        'version': '2.2',
        'auto_trigger_age': '7',
        'auto_trigger_usage': '20',
        'auto_action': 'OFF',
        'del_empty_dir': True,
    },
]


def main():

    for dmfileset in DM_FILE_SETS:
        try:
            dmfileset_obj, created = models.DMFileSet.objects.get_or_create(
                type=dmfileset['type'], version=dmfileset['version'])
            if created:
                # New object at this version
                # apply these values to new object
                dmfileset_obj.auto_trigger_age = dmfileset['auto_trigger_age']
                dmfileset_obj.auto_trigger_usage = dmfileset['auto_trigger_usage']
                dmfileset_obj.auto_action = dmfileset['auto_action']
                dmfileset_obj.description = dmfileset['description']
                dmfileset_obj.include = dmfileset['include']
                dmfileset_obj.exclude = dmfileset['exclude']
                dmfileset_obj.keepwith = dmfileset['keepwith']

                # Check for previous version objects
                try:
                    # Apply previous version object's values for auto age and usage
                    olddm = models.DMFileSet.objects.filter(
                        type=dmfileset['type']).exclude(version=dmfileset['version'])
                    olddm = olddm[0]
                    dmfileset_obj.auto_trigger_age = olddm.auto_trigger_age
                    dmfileset_obj.auto_trigger_usage = olddm.auto_trigger_usage
                    dmfileset_obj.auto_action = olddm.auto_action
                    dmfileset_obj.backup_directory = olddm.backup_directory
                    dmfileset_obj.bandwidth_limit = olddm.bandwidth_limit
                except IndexError:
                    pass
                except:
                    traceback.print_exc()
            else:
                # Object already exists at this version
                # update everything except auto age and usage and action
                dmfileset_obj.description = dmfileset['description']
                dmfileset_obj.include = dmfileset['include']
                dmfileset_obj.exclude = dmfileset['exclude']
                dmfileset_obj.keepwith = dmfileset['keepwith']

            dmfileset_obj.save()

        except:
            traceback.print_exc()
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
