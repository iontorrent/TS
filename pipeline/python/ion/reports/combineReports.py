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
from glob import glob

import sys
sys.path.append('/etc')
from torrentserver.cluster_settings import *

from ion.utils.blockprocessing import printheader, printtime
from ion.utils.blockprocessing import write_version
from ion.utils import ionstats

NUM_BARCODE_JOBS = 4

def submit_job(script, args, sge_queue = 'all.q', hold_jid = None):
    cwd = os.getcwd()
    #SGE    
    jt_nativeSpecification = "-pe ion_pe 1 -q " + sge_queue

    printtime("Use "+ sge_queue)

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

def wait_on_jobs(jobIds, jobName, status = "Processing", max_running_jobs = 0):
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
                
            if jobstatus=='done' or jobstatus=='failed' or jobstatus=="DRMAA BUG":
                printtime("DEBUG: Job %s has ended with status %s" % (str(jobid),jobstatus))
                jobIds.remove(jobid)
                
        time.sleep(20)

def get_barcode_files(parent_folder, datasets_path, bcSetName):
    # try to get barcode names from datasets json, fallback on globbing for older reports
    datasetsFile = os.path.join(parent_folder,datasets_path)
    barcode_bams = []
    try:
        with open(datasetsFile, 'r') as f:
            datasets_json = json.loads(f.read())
        for dataset in datasets_json.get("datasets",[]):
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
        barcode_bams = glob( os.path.join(parent_folder, bcSetName+'*_rawlib.bam') )
        barcode_bams.append( os.path.join(parent_folder, 'nomatch_rawlib.bam') )    
        barcode_bams.sort()
        
    printtime("DEBUG: found %i barcodes in %s" % (len(barcode_bams), parent_folder) )
    return barcode_bams

def barcode_report_stats(barcode_names):
    CA_barcodes_json = []
    ionstats_file_list = []
    printtime("DEBUG: creating CA_barcode_summary.json")

    for bcname in barcode_names:
        ionstats_file = bcname + '_rawlib.ionstats_alignment.json'
        barcode_json = {"barcode_name": bcname, "AQ7_num_bases":0, "full_num_reads":0, "AQ7_mean_read_length":0}
        try:
            stats = json.load(open(ionstats_file))
            for key in stats.keys():
                if key in ['AQ7', 'AQ10', 'AQ17', 'AQ20', 'AQ30', 'AQ47', 'full', 'aligned']:
                    barcode_json.update({
                        key+ "_max_read_length": stats[key].get("max_read_length"),
                        key+ "_mean_read_length": stats[key].get("mean_read_length"),
                        key+ "_num_bases": stats[key].get("num_bases"),
                        key+ "_num_reads": stats[key].get("num_reads")
                    })
            ionstats_file_list.append(ionstats_file)
        except:
            printtime("DEBUG: error reading ionstats from %s" % ionstats_file)
            traceback.print_exc()

        if bcname == 'nomatch':
            CA_barcodes_json.insert(0, barcode_json)
        else:
            CA_barcodes_json.append(barcode_json)

    with open('CA_barcode_summary.json','w') as f:
        f.write(json.dumps(CA_barcodes_json, indent=2))
    
    # generate merged ionstats_alignment.json
    if not os.path.exists('ionstats_alignment.json'):
        ionstats.reduce_stats(ionstats_file_list,'ionstats_alignment.json')

if __name__ == '__main__':
  
    with open('ion_params_00.json', 'r') as f:
        env = json.loads(f.read())
    
    parentBAMs = env['parentBAMs']
    mark_duplicates = env['mark_duplicates']
    override_samples = env.get('override_samples', False)
    sample = env.get('sample') or 'None'
    barcodeSamples = json.loads(env.get('barcodeSamples','{}'))
    
    from distutils.sysconfig import get_python_lib
    script = os.path.join(get_python_lib(), 'ion', 'reports', 'combineReports_jobs.py')
    
    try:
        jobserver = xmlrpclib.ServerProxy("http://%s:%d" % (JOBSERVER_HOST, JOBSERVER_PORT), verbose=False, allow_none=True)
    except (socket.error, xmlrpclib.Fault):
        traceback.print_exc()
    
    printheader()
    primary_key_file = os.path.join(os.getcwd(),'primary.key')
    
    try:
        jobserver.updatestatus(primary_key_file,'Started',True)
    except:
        traceback.print_exc()
    
    # Software version
    write_version()
    
    #  *** Barcodes ***
    do_barcodes = False
    barcodelist_path = 'barcodeList.txt'
    datasets_path = 'basecaller_results/datasets_basecaller.json'
    bcSetName = ""
    csvfilename = 'alignment_barcode_summary.csv'

    # get barcode files to process
    barcode_files = {}
    for bamfile in parentBAMs:
        parent_folder = os.path.dirname(bamfile)
        if os.path.exists(os.path.join(parent_folder,barcodelist_path)):
            do_barcodes = True
            bcList_file = os.path.join(parent_folder,barcodelist_path)
            bcSetName_new = open(bcList_file, 'r').readline().split()[1]
            
            # merge barcodes only if they are from the same barcode set
            if not bcSetName:
                bcSetName = bcSetName_new
                with open(bcList_file, 'r') as f:
                    barcode_names = [line.split(',')[1] for line in f.readlines() if line.startswith('barcode')]
                    barcode_names.append('nomatch')
            elif bcSetName != bcSetName_new:
                do_barcodes = False
                printtime("ERROR: unable to merge different barcode sets: %s and %s" % (bcSetName, bcSetName_new))
                break
                
            # get barcode BAM files
            barcode_bams = get_barcode_files(parent_folder, datasets_path, bcSetName)
            for bc_path in barcode_bams:
                try:
                    bcname = [name for name in barcode_names if os.path.basename(bc_path).startswith(name)][0]
                except:
                    bcname = 'unknown'
                
                if bcname not in barcode_files:
                    barcode_files[bcname] = {
                        'count': 0,
                        'bcfiles_to_merge' : []
                    }
                barcode_files[bcname]['filename'] = bcname + '_rawlib.bam'
                barcode_files[bcname]['count'] += 1
                barcode_files[bcname]['bcfiles_to_merge'].append(bc_path)
  
    if do_barcodes:
        try:
            shutil.copy(bcList_file, barcodelist_path)
        except:  
            traceback.print_exc()
        
        bc_jobs = []    
        #zipname = '_'+ env['resultsName']+ '.barcode.bam.zip'
        #zip_args = ['--zip', zipname]
        stats_args = ['--align-stats']
        
        # launch merge jobs, one per barcode
        for bcname,barcode_file_dict  in barcode_files.iteritems():
            filename = barcode_file_dict['filename']
            jobId = ""
            if barcode_file_dict['count'] > 1:
                printtime("DEBUG: merge barcode %s" % bcname)
                merge_args = ['--merge-bams', filename]
                
                if mark_duplicates:
                    merge_args.append('--mark-duplicates')
                
                if override_samples:
                    bcsample = [k for k,v in barcodeSamples.items() if bcname in v.get('barcodes',[])]
                    bcsample = bcsample[0] if len(bcsample) == 1 else 'None'
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
                shutil.copy(barcode_file_dict['bcfiles_to_merge'][0], filename)
                if os.path.exists(barcode_file_dict['bcfiles_to_merge'][0]+'.bai'):
                    shutil.copy(barcode_file_dict['bcfiles_to_merge'][0]+'.bai', filename+'.bai')
            
            stats_args.append('--add-file')
            stats_args.append(filename)
            
            # add bam files to be zipped                       
            #zip_args.append('--add-file')
            #zip_args.append(filename)
      
        # zip barcoded files        
        #jobId = submit_job(script, zip_args, 'all.q', bc_jobs)  
        #printtime("DEBUG: Submitted %s job %s" % ('zip barcodes', jobId))
        
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
        if do_barcodes:
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
            bam = "%s_%s.bam" % (parent_env.get('expName',''), parent_env.get('resultsName',''))
            bam = os.path.join(parent_path, bam)
        if not os.path.exists(bam):
            printtime("WARNING: Unable to find BAM file to merge in %s" % parent_path)
            continue
        file_args.append('--add-file')
        file_args.append(bam)
        #print 'BAM file %s' % bam
    merge_args += file_args
    
    jobId = submit_job(script, merge_args)  
    printtime("DEBUG: Submitted %s job %s" % ('BAM merge', jobId))
    wait_on_jobs([jobId], 'BAM merge', 'Merging BAM files')
    
    # generate ionstats json file
    if os.path.exists(bamfile):
        stats_args = ['--align-stats','--add-file',bamfile]
        jobId = submit_job(script, stats_args, 'all.q')
        printtime("DEBUG: Submitted %s job %s" % ('BAM alignment stats', jobId))
        wait_on_jobs([jobId], 'BAM alignment stats', 'Merging BAM files')
  
    # Generate files needed to display Report
    if do_barcodes:
        barcode_report_stats(sorted(barcode_files.keys()))
    jobId = submit_job(script, ['--merge-plots']) 
    printtime("DEBUG: Submitted %s job %s" % ('MergePlots', jobId))  
    wait_on_jobs([jobId], 'mergePlots', 'Generating Alignment plots')
      
    # make downloadable BAM filenames
    mycwd = os.getcwd()
    download_links = 'download_links'
    newname = env['run_name'] + '_'+ env['resultsName']
    try:
        os.mkdir(download_links)
        filename = os.path.join(mycwd, bamfile)
        os.symlink(filename, os.path.join(download_links, newname+'.bam'))
        os.symlink(filename+'.bai', os.path.join(download_links, newname + '.bam.bai'))
        # barcodes:
        if do_barcodes:
            #os.symlink(os.path.join(mycwd, zipname), os.path.join(download_links, newname+'.barcode.bam.zip'))
            for bcname in barcode_files.keys():
                filename = os.path.join(mycwd, bcname+'_rawlib.bam')
                os.symlink(filename, os.path.join(download_links, newname+'.'+bcname+'_rawlib.bam' ) )
                os.symlink(filename+'.bai', os.path.join(download_links, newname+'.'+bcname+'_rawlib.bam.bai' ) ) 
    except:
        traceback.print_exc()
  
    if os.path.exists(bamfile):
        status = 'Completed' 
    elif do_barcodes:      
        status = 'Completed' # for barcoded Proton data rawlib.bam may not get created
    else:
        status = 'Error'
  
    # Upload metrics
    try:
        ret_message = jobserver.uploadmetrics(
            '',
            '',
            '',
            os.path.join(mycwd,'ionstats_alignment.json'),
            os.path.join(mycwd,'ion_params_00.json'),
            '',
            '',
            '',
            os.path.join(mycwd,'primary.key'),
            os.path.join(mycwd,'uploadStatus'),
            status,
            True,
            mycwd)
    except:
        traceback.print_exc()
  
    # copy php script
    os.umask(0002)
    TMPL_DIR = '/usr/share/ion/web/db/writers'
  
    templates = [
      # DIRECTORY, SOURCE_FILE, DEST_FILE or None for same as SOURCE
        (TMPL_DIR, "report_layout.json", None),
        (TMPL_DIR, "parsefiles.php", None),
        (TMPL_DIR, "combinedReport.php", "Default_Report.php",), ## Renamed during copy
      ]
    for (d,s,f) in templates:
        if not f: f=s
        # If owner is different copy fails - unless file is removed first
        if os.access(f, os.F_OK):
            os.remove(f)
        shutil.copy(os.path.join(d,s), f)
    
    # create plugin folder
    basefolder = 'plugin_out'
    if not os.path.isdir(basefolder):
        oldmask = os.umask(0000)   #grant write permission to plugin user
        os.mkdir(basefolder)
        os.umask(oldmask)
    
    try:
        jobserver.updatestatus(primary_key_file,status,True)
    except:
        traceback.print_exc()  
      
    printtime("combineReports Done")
    sys.exit(0)
  
