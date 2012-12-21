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
from ion.utils.aggregate_alignment import aggregate_alignment

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

def wait_on_jobs(jobIds, jobName, status = "Processing"):
    try:
      jobserver.updatestatus(primary_key_file, status, True)
    except:
      traceback.print_exc()

    # wait for job to finish
    while len(jobIds) > 0:  
        for jid in jobIds:
            try:
                jobstatus = jobserver.jobstatus(jobId)
            except:
                traceback.print_exc()
                continue
                
            if jobstatus=='done' or jobstatus=='failed' or jobstatus=="DRMAA BUG":
                printtime("DEBUG: Job %s has ended with status %s" % (str(jid),jobstatus))
                jobIds.remove(jid)
                
        printtime("waiting for %s job(s) to finish ..." % jobName)    
        time.sleep(10)    

def get_barcode_files(parent_folder, datasets_path, bcSetName):
    # try to get barcode names from datasets json, fallback on globbing for older reports
    datasetsFile = os.path.join(parent_folder,datasets_path)
    barcode_bams = []
    try:
        with open(datasetsFile, 'r') as f:
            datasets_json = json.loads(f.read())
        for dataset in datasets_json.get("datasets",[]):
            bamfile = os.path.join(parent_folder, dataset["legacy_prefix"]+'.bam')
            if os.path.exists(bamfile):
                barcode_bams.append(bamfile)
    except:
        pass  
    
    if len(barcode_bams) == 0:
        barcode_bams = glob( os.path.join(parent_folder, bcSetName+'*_rawlib.bam') )
        barcode_bams.append( os.path.join(parent_folder, 'nomatch_rawlib.bam') )    
        barcode_bams.sort()
        
    printtime("DEBUG: found %i barcodes in %s" % (len(barcode_bams), parent_folder) )
    return barcode_bams

if __name__ == '__main__':
  
  with open('ion_params_00.json', 'r') as f:
    env = json.loads(f.read())
  
  script = '/usr/lib/python2.6/dist-packages/ion/reports/combineReports_jobs.py'
  
  try:
    jobserver = xmlrpclib.ServerProxy("http://%s:%d" % (JOBSERVER_HOST, JOBSERVER_PORT), verbose=False, allow_none=True)
  except (socket.error, xmlrpclib.Fault):
    traceback.print_exc()
  
  printheader()
  status = 'Error'
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
  bcfile_count = {}
  bcfile_files = {}
  
  # get barcoded files to process  
  for bamfile in env['parentBAMs']:
    parent_folder = os.path.dirname(bamfile)
    if os.path.exists(os.path.join(parent_folder,barcodelist_path)): 
      do_barcodes = True      
      bcList_file = os.path.join(parent_folder,barcodelist_path)      
      bcSetName_new = open(bcList_file, 'r').readline().split()[1]
      
      # merge barcodes only if they are from the same barcode set
      if not bcSetName:
          bcSetName = bcSetName_new
      elif bcSetName != bcSetName_new:
          do_barcodes = False
          break
          
      # get barcode files          
      barcode_bams = get_barcode_files(parent_folder, datasets_path, bcSetName)
      for bc in barcode_bams:
          bcname = os.path.basename(bc)
          bcfile_count[bcname] = bcfile_count.get(bcname, 0) + 1 
          bcfile_files[bcname] = bcfile_files.get(bcname, [])+ [bc]

  if do_barcodes:
    try:
      shutil.copy(bcList_file, barcodelist_path)
    except:  
      traceback.print_exc()
    
    bc_jobs = []    
    zipname = '_'+ env['resultsName']+ '.barcode.bam.zip'
    zip_args = ['--zip', zipname]    
    # launch merge and alignstats jobs, one per barcode  
    for filename in bcfile_count.keys():        
        jobId = ""
        if bcfile_count[filename] > 1:
            printtime("DEBUG: merge barcode %s" % filename)
            merge_args = ['--merge-bams', filename]
            if env['mark_duplicates']:
                merge_args.append('--mark-duplicates')
            for bam in bcfile_files[filename]:
                merge_args.append('--add-file')
                merge_args.append(bam)
            jobId = submit_job(script, merge_args, 'plugin.q')  
            printtime("DEBUG: Submitted %s job %s" % ('merge barcodes', jobId))        
        else:          
            printtime("DEBUG: copy barcode %s" % filename)
            shutil.copy(bcfile_files[filename][0], filename)
        
        # add bam files to be zipped                       
        zip_args.append('--add-file')
        zip_args.append(filename)
        
        # create barcode alignment summary files
        if jobId:
            jobId = submit_job(script, ['--align-stats', filename, '--genomeinfo', env['genomeinfo']], 'plugin.q', [jobId])
        else:
            jobId = submit_job(script, ['--align-stats', filename, '--genomeinfo', env['genomeinfo']], 'plugin.q')    
        printtime("DEBUG: Submitted %s job %s for %s" % ('alignStats', jobId, filename))
        bc_jobs.append(jobId) 

    # zip barcoded files        
    jobId = submit_job(script, zip_args, 'all.q', bc_jobs)  
    printtime("DEBUG: Submitted %s job %s" % ('zip barcodes', jobId))
    
    # TODO: could launch non-barcoded merge before waiting
    wait_on_jobs(bc_jobs, 'barcode', 'Processing barcodes')
        
    # create barcode csv 
    printtime("DEBUG: creating alignment_barcode_summary.csv")
    aggregate_alignment('./', barcodelist_path)
  
  # *** END Barcodes ***
  
  
  # merge BAM files
  bamfile = 'rawlib.bam'
  printtime("Merging bam files")
#  merge_bam_files(env['parentBAMs'], bamfile, bamfile.replace('.bam','.bam.bai'), '')
  merge_args = ['--merge-bams', bamfile]  
  if env['mark_duplicates']:
      merge_args.append('--mark-duplicates')
  file_args = []
  for bam in env['parentBAMs']:
      file_args.append('--add-file')
      file_args.append(bam)
  merge_args += file_args
  jobId = submit_job(script, merge_args)  
  printtime("DEBUG: Submitted %s job %s" % ('BAM merge', jobId))

  wait_on_jobs([jobId], 'BAM merge', 'Merging BAM files')  

  # Call alignStats on merged bam file 
  jobId = submit_job(script, ['--align-stats', bamfile, '--genomeinfo', env['genomeinfo']])
  printtime("DEBUG: Submitted %s job %s" % ('alignStats', jobId))
  wait_on_jobs([jobId], 'alignStats', 'Generating Alignment metrics')
  
  # Generate files needed to display Report
  jobId = submit_job(script, ['--merge-plots'] + file_args) 
  printtime("DEBUG: Submitted %s job %s" % ('MergePlots', jobId))  
  wait_on_jobs([jobId], 'mergePlots', 'Generating Alignment plots')
    
  status = 'Completed'
  
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
    for bcfile in bcfile_count.keys():
        filename = os.path.join(mycwd, bcfile)
        os.symlink(filename, os.path.join(download_links, newname+'.'+bcfile ) )
        os.symlink(filename+'.bai', os.path.join(download_links, newname+'.'+bcfile+'.bai' ) ) 
  except:
    traceback.print_exc()

  # upload metrics
  printtime("Upload metrics.")
  reportLink = True
  ret_message = jobserver.uploadmetrics(
        "",#    os.path.join(mycwd,tfmapperstats_outputfile),
        "",#    os.path.join(mycwd,"processParameters.txt"),
        "",#    os.path.join(mycwd,beadPath),
        "",#    filterPath,
        os.path.join(mycwd,"alignment.summary"),
        "",#    os.path.join(mycwd,"raw_peak_signal"),
        "",#    os.path.join(mycwd,"quality.summary"),
        "",#    os.path.join(mycwd,BaseCallerJsonPath),
        os.path.join(mycwd,'primary.key'),
        os.path.join(mycwd,'uploadStatus'),
        status,
        reportLink)
#  print "jobserver.uploadmetrics returned: "+str(ret_message)
  
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
  
