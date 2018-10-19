#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

# Create a Multi-report by merging multiple input results.

import subprocess
import traceback
import os
import socket
import xmlrpclib
import json
import shutil
import time
import datetime
from glob import glob

import sys
sys.path.append('/etc')
from torrentserver.cluster_settings import *

from ion.utils.blockprocessing import printheader, printtime
from ion.utils.blockprocessing import write_version
from ion.utils import ionstats

NUM_BARCODE_JOBS = 4


def submit_job(script, args, sge_queue='all.q', hold_jid=None):
    cwd = os.getcwd()
    # SGE
    jt_nativeSpecification = "-pe ion_pe 1 -q " + sge_queue

    printtime("Use " + sge_queue)

    jt_remoteCommand = "python"
    jt_workingDirectory = cwd
    jt_outputPath = ":" + "%s/drmaa_stdout_block.txt" % cwd
    jt_errorPath = ":" + "%s/drmaa_stderr_block.txt" % cwd
    jt_args = [script] + args
    jt_joinFiles = False

    if hold_jid != None and len(hold_jid) > 0:
        jt_nativeSpecification += " -hold_jid "
        for holdjobid in hold_jid:
            jt_nativeSpecification += "%s," % holdjobid

    try:
        jobid = jobserver.submitjob(
            jt_nativeSpecification,
            jt_remoteCommand,
            jt_workingDirectory,
            jt_outputPath,
            jt_errorPath,
            jt_args,
            jt_joinFiles)
        return jobid

    except:
        traceback.print_exc()
        printtime("FAILED submitting %s job" % script)
        sys.exit()


def wait_on_jobs(jobIds, jobName, status="Processing", max_running_jobs=0):
    try:
        jobserver.updatestatus(primary_key_file, status, True)
    except:
        traceback.print_exc()

    # wait for job to finish
    while len(jobIds) > max_running_jobs:
        printtime("waiting for %s job(s) to finish ..." % jobName)
        for jobid in jobIds:
            try:
                jobstatus = jobserver.jobstatus(jobid)
            except:
                traceback.print_exc()
                continue

            if jobstatus == 'done' or jobstatus == 'failed' or jobstatus == "DRMAA BUG":
                printtime("DEBUG: Job %s has ended with status %s" % (str(jobid), jobstatus))
                jobIds.remove(jobid)

        time.sleep(20)


def get_parent_barcode_files(parent_folder, datasets_path, barcodeSet):
    # try to get barcode names from datasets json, fallback on globbing for older reports
    datasetsFile = os.path.join(parent_folder, datasets_path)
    barcode_bams = []
    try:
        with open(datasetsFile, 'r') as f:
            datasets_json = json.loads(f.read())
        for dataset in datasets_json.get("datasets", []):
            bamfile = os.path.join(parent_folder, dataset["file_prefix"]+'.bam')
            if os.path.exists(bamfile):
                barcode_bams.append(bamfile)
            elif 'legacy_prefix' in dataset.keys():
                old_bamfile = os.path.join(parent_folder, dataset["legacy_prefix"]+'.bam')
                if os.path.exists(old_bamfile):
                    barcode_bams.append(old_bamfile)
    except:
        pass

    if len(barcode_bams) == 0:
        printtime("DEBUG: no barcoded files found from %s" % datasetsFile)
        barcode_bams = glob(os.path.join(parent_folder, barcodeSet+'*_rawlib.bam'))
        if os.path.exists(os.path.join(parent_folder, 'nomatch_rawlib.bam')):
            barcode_bams.append(os.path.join(parent_folder, 'nomatch_rawlib.bam'))
        barcode_bams.sort()

    printtime("DEBUG: found %i barcodes in %s" % (len(barcode_bams), parent_folder))
    return barcode_bams


def barcode_report_stats(barcode_names):
    CA_barcodes_json = []
    ionstats_file_list = []
    printtime("DEBUG: creating CA_barcode_summary.json")

    for bcname in sorted(barcode_names):
        ionstats_file = bcname + '_rawlib.ionstats_alignment.json'
        barcode_json = {"barcode_name": bcname, "AQ7_num_bases": 0, "full_num_reads": 0, "AQ7_mean_read_length": 0}
        try:
            stats = json.load(open(ionstats_file))
            for key in stats.keys():
                if key in ['AQ7', 'AQ10', 'AQ17', 'AQ20', 'AQ30', 'AQ47', 'full', 'aligned']:
                    barcode_json.update({
                        key + "_max_read_length": stats[key].get("max_read_length"),
                        key + "_mean_read_length": stats[key].get("mean_read_length"),
                        key + "_num_bases": stats[key].get("num_bases"),
                        key + "_num_reads": stats[key].get("num_reads")
                    })
            ionstats_file_list.append(ionstats_file)
        except:
            printtime("DEBUG: error reading ionstats from %s" % ionstats_file)
            traceback.print_exc()

        if bcname == 'nomatch':
            CA_barcodes_json.insert(0, barcode_json)
        else:
            CA_barcodes_json.append(barcode_json)

    with open('CA_barcode_summary.json', 'w') as f:
        f.write(json.dumps(CA_barcodes_json, indent=2))

    # generate merged ionstats_alignment.json
    if not os.path.exists('ionstats_alignment.json'):
        ionstats.reduce_stats(ionstats_file_list, 'ionstats_alignment.json')


def generate_datasets_pipeline_json(barcodeSet, found_barcode_names, barcodeSet_Info, sample, barcodeSamples, runID):
    # Note this writes a file with some but not all of regular report's datasets_pipeline.json parameters

    datasets_json_path = "datasets_pipeline.json"

    datasets = {
        "meta": {
            "format_name": "Dataset Map",
            "format_version": "1.0",
            "generated_by": "combineReports.py",
            "creation_date": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        },
        "datasets": [],
        "read_groups": {}
    }

    no_barcode_group = runID+".nomatch" if barcodeSet else runID

    datasets["datasets"].append({
        "dataset_name": sample + "/No_barcode_match" if barcodeSet else sample,
        "file_prefix": "nomatch_rawlib" if barcodeSet else "rawlib",
        "read_groups": [no_barcode_group, ]
    })

    datasets["read_groups"][no_barcode_group] = {
        "index": 0,
        "sample": sample,
    }

    if barcodeSet:
        datasets["barcode_config"] = {"barcode_id": barcodeSet}
        try:
            for bcname in sorted(found_barcode_names):

                if bcname == 'nomatch':
                    continue

                bcsample = [k for k, v in barcodeSamples.items() if bcname in v.get('barcodes', [])]
                bcsample = bcsample[0] if len(bcsample) == 1 else 'none'

                datasets["datasets"].append({
                    "dataset_name": bcsample + "/" + bcname,
                    "file_prefix": '%s_rawlib' % bcname,
                    "read_groups": [runID+"."+bcname, ]
                })

                datasets["read_groups"][runID+"."+bcname] = {
                    "barcode_name": bcname,
                    "barcode_sequence": barcodeSet_Info[bcname]['sequence'],
                    "barcode_adapter": barcodeSet_Info[bcname]['adapter'],
                    "index": barcodeSet_Info[bcname]['index'],
                    "sample": bcsample,
                }
        except:
            traceback.print_exc()

    with open(datasets_json_path, "w") as f:
        json.dump(datasets, f, indent=4)

    return datasets


def generate_datasets_basecaller_json(datasets, parentBAMs, barcodeSet):
    # Note this writes a file with some but not all of regular report's datasets_basecaller.json parameters

    datasets_path = "basecaller_results/datasets_basecaller.json"
    basecaller_results = "basecaller_results"

    readgroups_init = {
        "read_count": 0,
        "Q20_bases": 0,
        "total_bases": 0,
        "filtered": [],
        "reference": "",
    }
    non_barcoded_file_prefix = "rawlib"
    by_file_prefix = {}

    for bamfile in parentBAMs:
        # read parent datasets files and merge relevant fields
        parent_folder = os.path.dirname(bamfile)
        datasetsFile = os.path.join(parent_folder, datasets_path)
        try:
            with open(datasetsFile) as f:
                parent_datasets_json = json.load(f)

            for parent_read_group in parent_datasets_json['read_groups'].values():
                if barcodeSet:
                    prefix = '%s_rawlib' % parent_read_group.get('barcode_name', 'nomatch')
                else:
                    prefix = non_barcoded_file_prefix

                if prefix not in by_file_prefix:
                    by_file_prefix[prefix] = dict(readgroups_init)

                by_file_prefix[prefix]['read_count'] += parent_read_group['read_count']
                by_file_prefix[prefix]['Q20_bases'] += parent_read_group['Q20_bases']
                by_file_prefix[prefix]['total_bases'] += parent_read_group['total_bases']

                if parent_read_group.get('reference'):
                    by_file_prefix[prefix]['reference'] = parent_read_group['reference']
                by_file_prefix[prefix]['filtered'].append(parent_read_group.get('filtered', False))
        except:
            printtime("DEBUG: unable to update datasets_basecaller.json from %s" % datasetsFile)
            traceback.print_exc()
            break
    else:
        # now update the combined datasets
        for merged_dataset in datasets['datasets']:
            prefix = merged_dataset['file_prefix']
            if prefix in by_file_prefix:
                merged_dataset['read_count'] = by_file_prefix[prefix]['read_count']
                read_group_key = merged_dataset['read_groups'][0]

                merged_read_group = datasets['read_groups'][read_group_key]
                merged_read_group['read_count'] = by_file_prefix[prefix]['read_count']
                merged_read_group['Q20_bases'] = by_file_prefix[prefix]['Q20_bases']
                merged_read_group['total_bases'] = by_file_prefix[prefix]['total_bases']
                merged_read_group['reference'] = by_file_prefix[prefix]['reference']
                merged_read_group['filtered'] = all(by_file_prefix[prefix]['filtered'])

    os.mkdir(basecaller_results)
    with open(datasets_path, "w") as f:
        json.dump(datasets, f, indent=4)


def find_barcodes_to_process(parentBAMs, barcodeSet):
    # get barcode files to process
    barcode_files = {}
    barcodeSet_Info = None
    datasets_path = 'basecaller_results/datasets_basecaller.json'
    barcodelist_path = 'barcodeList.txt'

    if not barcodeSet:
        return barcodeSet, barcode_files, barcodeSet_Info

    for bamfile in parentBAMs:
        parent_folder = os.path.dirname(bamfile)
        if os.path.exists(os.path.join(parent_folder, barcodelist_path)):
            bcList_file = os.path.join(parent_folder, barcodelist_path)
            bcSetName_new = open(bcList_file, 'r').readline().split('file_id')[1].strip()
            if barcodeSet != bcSetName_new:
                printtime("Warning: different barcode sets: %s and %s" % (barcodeSet, bcSetName_new))

            if not barcodeSet_Info:
                barcodeSet_Info = {'nomatch': {'index': 0}}
                try:
                    with open(bcList_file, 'r') as f:
                        for line in f.readlines():
                            if line.startswith('barcode'):
                                splitline = line.split(',')
                                name = splitline[1]
                                barcodeSet_Info[name] = {
                                    'index': splitline[0].split()[1],
                                    'sequence': splitline[2],
                                    'adapter': splitline[3]
                                }
                except:
                    traceback.print_exc()

            # get barcode BAM files
            barcode_bams = get_parent_barcode_files(parent_folder, datasets_path, barcodeSet)

            for bc_path in barcode_bams:
                try:
                    bcname = [name for name in sorted(barcodeSet_Info.keys(), reverse=True) if os.path.basename(bc_path).startswith(name)][0]
                except:
                    bcname = 'unknown'

                if bcname not in barcode_files:
                    barcode_files[bcname] = {
                        'count': 0,
                        'bcfiles_to_merge': []
                    }
                barcode_files[bcname]['filename'] = bcname + '_rawlib.bam'
                barcode_files[bcname]['count'] += 1
                barcode_files[bcname]['bcfiles_to_merge'].append(bc_path)

    if barcodeSet:
        try:
            shutil.copy(bcList_file, barcodelist_path)
        except:
            traceback.print_exc()

    return barcodeSet, barcode_files, barcodeSet_Info


if __name__ == '__main__':

    with open('ion_params_00.json', 'r') as f:
        env = json.loads(f.read())

    parentBAMs = env['parentBAMs']
    mark_duplicates = env['mark_duplicates']
    override_samples = env.get('override_samples', False)
    sample = env.get('sample') or 'none'
    barcodeSamples = env['experimentAnalysisSettings'].get('barcodedSamples', '{}')

    barcodeSet = env['barcodeId']
    runID = env.get('runid', 'ABCDE')

    from distutils.sysconfig import get_python_lib
    script = os.path.join(get_python_lib(), 'ion', 'reports', 'combineReports_jobs.py')

    try:
        jobserver = xmlrpclib.ServerProxy("http://%s:%d" % (JOBSERVER_HOST, JOBSERVER_PORT), verbose=False, allow_none=True)
    except (socket.error, xmlrpclib.Fault):
        traceback.print_exc()

    printheader()
    primary_key_file = os.path.join(os.getcwd(), 'primary.key')

    try:
        jobserver.updatestatus(primary_key_file, 'Started', True)
    except:
        traceback.print_exc()

    # Software version
    write_version()

    #  *** Barcodes ***
    barcodeSet, barcode_files, barcodeSet_Info = find_barcodes_to_process(parentBAMs, barcodeSet)

    if barcodeSet:

        bc_jobs = []
        # zipname = '_'+ env['resultsName']+ '.barcode.bam.zip'
        # zip_args = ['--zip', zipname]
        stats_args = ['--align-stats']

        # launch merge jobs, one per barcode
        for bcname, barcode_file_dict  in barcode_files.iteritems():
            filename = barcode_file_dict['filename']
            jobId = ""
            if barcode_file_dict['count'] > 1:
                printtime("DEBUG: merge barcode %s" % bcname)
                merge_args = ['--merge-bams', filename]

                if mark_duplicates:
                    merge_args.append('--mark-duplicates')

                if override_samples:
                    bcsample = [k for k, v in barcodeSamples.items() if bcname in v.get('barcodes', [])]
                    bcsample = bcsample[0] if len(bcsample) == 1 else 'none'
                    merge_args += ['--new-sample-name', bcsample]

                for bam in barcode_file_dict['bcfiles_to_merge']:
                    merge_args.append('--add-file')
                    merge_args.append(bam)
                jobId = submit_job(script, merge_args, 'plugin.q')
                bc_jobs.append(jobId)
                printtime("DEBUG: Submitted %s job %s" % ('merge barcodes', jobId))

                # limit number of parallel jobs
                if NUM_BARCODE_JOBS and (len(bc_jobs) > NUM_BARCODE_JOBS-1):
                    wait_on_jobs(bc_jobs, 'barcode', 'Processing barcodes', NUM_BARCODE_JOBS-1)

            else:
                printtime("DEBUG: copy barcode %s" % bcname)
                file_to_copy = barcode_file_dict['bcfiles_to_merge'][0]
                shutil.copy(file_to_copy, filename)
                if os.path.exists(file_to_copy + '.bai'):
                    shutil.copy(file_to_copy + '.bai', filename + '.bai')

            stats_args.append('--add-file')
            stats_args.append(filename)

            # add bam files to be zipped
            # zip_args.append('--add-file')
            # zip_args.append(filename)

        # zip barcoded files
        # jobId = submit_job(script, zip_args, 'all.q', bc_jobs)
        # printtime("DEBUG: Submitted %s job %s" % ('zip barcodes', jobId))

        wait_on_jobs(bc_jobs, 'barcode', 'Processing barcodes')

        # generate barcoded ionstats json files
        jobId = submit_job(script, stats_args, 'all.q', bc_jobs)
        printtime("DEBUG: Submitted %s job %s" % ('barcode stats', jobId))
        wait_on_jobs([jobId], 'barcode stats', 'Processing barcodes')

    # *** END Barcodes ***

    # merge BAM files
    bamfile = 'rawlib.bam'
    printtime("Merging bam files")
    merge_args = ['--merge-bams', bamfile]
    if mark_duplicates:
        merge_args.append('--mark-duplicates')

    if override_samples:
        if barcodeSet:
            bam_files_to_merge = [barcode_file_dict['filename'] for barcode_file_dict in barcode_files.values()]
        else:
            bam_files_to_merge = parentBAMs
            merge_args += ['--new-sample-name', sample]
    else:
        bam_files_to_merge = parentBAMs

    file_args = []
    for bam in bam_files_to_merge:
        if not os.path.exists(bam):
            # compatibility: try expName_resultName.bam
            parent_path = os.path.dirname(bam)
            with open(os.path.join(parent_path, 'ion_params_00.json'), 'r') as f:
                parent_env = json.loads(f.read())
            bam = "%s_%s.bam" % (parent_env.get('expName', ''), parent_env.get('resultsName', ''))
            bam = os.path.join(parent_path, bam)
        if not os.path.exists(bam):
            printtime("WARNING: Unable to find BAM file to merge in %s" % parent_path)
            continue
        file_args.append('--add-file')
        file_args.append(bam)
        # print 'BAM file %s' % bam
    merge_args += file_args

    jobId = submit_job(script, merge_args)
    printtime("DEBUG: Submitted %s job %s" % ('BAM merge', jobId))
    wait_on_jobs([jobId], 'BAM merge', 'Merging BAM files')

    # generate ionstats json file
    if os.path.exists(bamfile):
        stats_args = ['--align-stats', '--add-file', bamfile]
        jobId = submit_job(script, stats_args, 'all.q')
        printtime("DEBUG: Submitted %s job %s" % ('BAM alignment stats', jobId))
        wait_on_jobs([jobId], 'BAM alignment stats', 'Merging BAM files')

    # Generate files needed to display Report
    if barcodeSet:
        barcode_report_stats(barcode_files.keys())

    # Generate files needed by Plugins
    try:
        datasets = generate_datasets_pipeline_json(barcodeSet, barcode_files.keys(), barcodeSet_Info, sample, barcodeSamples, runID)
        generate_datasets_basecaller_json(datasets, parentBAMs, barcodeSet)
    except:
        traceback.print_exc()

    jobId = submit_job(script, ['--merge-plots'])
    printtime("DEBUG: Submitted %s job %s" % ('MergePlots', jobId))
    wait_on_jobs([jobId], 'mergePlots', 'Generating Alignment plots')

    # make downloadable BAM filenames
    mycwd = os.getcwd()
    download_links = 'download_links'
    newname = env['run_name'] + '_' + env['resultsName']
    try:
        os.mkdir(download_links)
        filename = os.path.join(mycwd, bamfile)
        os.symlink(filename, os.path.join(download_links, newname+'.bam'))
        os.symlink(filename+'.bai', os.path.join(download_links, newname + '.bam.bai'))
        # barcodes:
        if barcodeSet:
            # os.symlink(os.path.join(mycwd, zipname), os.path.join(download_links, newname+'.barcode.bam.zip'))
            for bcname in barcode_files.keys():
                filename = os.path.join(mycwd, bcname+'_rawlib.bam')
                os.symlink(filename, os.path.join(download_links, newname+'.'+bcname+'_rawlib.bam'))
                os.symlink(filename+'.bai', os.path.join(download_links, newname+'.'+bcname+'_rawlib.bam.bai'))
    except:
        traceback.print_exc()

    if os.path.exists(bamfile):
        status = 'Completed'
    elif barcodeSet:
        status = 'Completed'  # for barcoded data rawlib.bam may not get created
    else:
        status = 'Error'

    # Upload metrics
    try:
        ret_message = jobserver.uploadmetrics(
            '',
            '',
            '',
            os.path.join(mycwd, 'ionstats_alignment.json'),
            os.path.join(mycwd, 'ion_params_00.json'),
            '',
            '',
            '',
            os.path.join(mycwd, 'primary.key'),
            os.path.join(mycwd, 'uploadStatus'),
            status,
            True,
            mycwd)
    except:
        traceback.print_exc()

    # copy files for report
    os.umask(0002)
    TMPL_DIR = '/usr/share/ion/web'
    templates = [
        # DIRECTORY, SOURCE_FILE, DEST_FILE or None for same as SOURCE
        (TMPL_DIR, "report_layout.json", None),
        (TMPL_DIR, "parsefiles.php", None),
        (TMPL_DIR, "combinedReport.php", "Default_Report.php",),  # Renamed during copy
    ]
    for (d, s, f) in templates:
        if not f: f = s
        # If owner is different copy fails - unless file is removed first
        if os.access(f, os.F_OK):
            os.remove(f)
        shutil.copy(os.path.join(d, s), f)

    # create plugin folder
    basefolder = 'plugin_out'
    if not os.path.isdir(basefolder):
        oldmask = os.umask(0000)  # grant write permission to plugin user
        os.mkdir(basefolder)
        os.umask(oldmask)

    try:
        jobserver.updatestatus(primary_key_file, status, True)
    except:
        traceback.print_exc()

    printtime("combineReports Done")
    sys.exit(0)
