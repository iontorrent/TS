#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import os
import hashlib
import shutil
import time
import ConfigParser
import sys
import argparse
import traceback
import subprocess
import shlex
import random
import fnmatch
import glob
import re

import json
import string
import httplib
import base64

from collections import deque

from Queue import Queue
from threading import Thread


from ion.utils.explogparser import load_log
from ion.utils.explogparser import parse_log
from ion.utils import explogparser


import logging
import logging.handlers


class Worker(Thread):
    """Thread executing tasks from a given tasks queue"""
    def __init__(self, id, tasks, pool ,logger):
        Thread.__init__(self)
        self.tasks = tasks
        self.pool = pool
        self.logger = logger
        self.id = id
        self.daemon = True
        self.start()

    def run(self):
        logger.info('thread running %s' % self.id)
        while True:

            block = self.tasks.get()

            if 'justBeadFind' in block.command:
                self.pool.beadfind_counter += 1
            if 'Analysis' in block.command:
                self.pool.analysis_counter += 1


            block.status = "processing"
            logger.info("%s: T1: %s" % (self.id, block) )
            starttime = time.localtime()

            try:
                if not os.path.exists( block.sigproc_results_path_tmp ):
                    os.makedirs( block.sigproc_results_path_tmp )
                    logger.debug('mkdir %s' % block.sigproc_results_path_tmp)

                try:
                    if 'justBeadFind' in block.command and block.run.exp_ecc_enabled:
                        outfile = open(os.path.join(block.sigproc_results_path_tmp,'ecc.log'), "a")
                        ecc_command="/usr/share/ion/oia/ecc.py -i %s -o %s" % (block.dat_path, block.sigproc_results_path_tmp)
                        args = shlex.split(ecc_command.encode('utf8'))
                        logger.info("%s: run process: %s" % (self.id, args) )
                        p = subprocess.Popen(args, stdout=outfile, stderr=subprocess.STDOUT)
                        # add popen process to block
                        block.process = p
                        ret = p.wait()
                except:
                    logger.error(traceback.format_exc())               
                    pass

                logger.info("%s: run process: %s" % (self.id, block) )

                # don't use shell=True , otherwise child process cannot be killed
                outfile = open(os.path.join(block.sigproc_results_path_tmp,'sigproc.log'), "a")
                args = shlex.split(block.command.encode('utf8'))
                my_environment=os.environ.copy()
                if not "/usr/local/bin" in my_environment['PATH']:
                    my_environment['PATH'] = my_environment['PATH']+":/usr/local/bin"
                #logger.info(my_environment)
                p = subprocess.Popen(args, stdout=outfile, stderr=subprocess.STDOUT, env=my_environment)
                # add popen process to block
                block.process = p
                ret = p.wait()

                # error generation
                #rand = random.randint(0,99)
                #if rand < 2:
                #    ret = 777
                #else:
                #    ret = 0

                block.ret = ret
            except:
                logger.error(traceback.format_exc())
                block.ret = 666

            try:
                f = open(os.path.join(block.sigproc_results_path_tmp,'analysis_return_code.txt'), 'w')
                f.write(str(block.ret))
                f.close()
            except:
                logger.error('%s failed to write return code' % block.name)
                logger.error(traceback.format_exc())
                pass

            block.status = "processed"
            self.tasks.task_done()


            if 'justBeadFind' in block.command:
                self.pool.beadfind_counter -= 1
            if 'Analysis' in block.command:
                self.pool.analysis_counter -= 1

            stoptime = time.localtime()
            block.jobtiming.append((starttime,stoptime,block.info,block.ret,self.id))
            logger.info("%s: T2: %s" % (self.id, block) )


class ThreadPool:
    """Pool of threads consuming tasks from a queue"""
    def __init__(self, num_threads, logger):
        self.tasks = Queue(num_threads) # limit the queue to the number of threads
        self.beadfind_counter = 0
        self.analysis_counter = 0

        for threadid in range(num_threads): Worker(threadid, self.tasks, self, logger)

    def add_task(self, block):
        """Add a task to the queue"""
        #print 'add task', block
        self.tasks.put(block)

    def wait_completion(self):
        """Wait for completion of all the tasks in the queue"""
        self.tasks.join()


def run_a_shell_process(command):
    #command = "echo "+command
    #print command
    #time.sleep(30)
    ret = subprocess.call(command, shell=True)
    return ret


def getSeparatorCommand(config,block):
    #command = "strace -o %s/strace.log %s" % (block.sigproc_results_path_tmp,block.run.exp_beadfindArgs)
    command = "nice -n 1 %s" % block.run.exp_beadfindArgs
    command += " --local-wells-file off"
    command += " --beadfind-num-threads %s" % config.get('global','nb_beadfind_threads')
    command += " --no-subdir"
    command += " --output-dir=%s" % block.sigproc_results_path_tmp
    command += " --librarykey=%s" % block.run.libraryKey
    command += " --tfkey=%s" % block.run.tfKey
    command += " %s" % block.dat_path
    logger.debug('beadfindArgs(PR):"%s"' % command)
    return command


def getAnalysisCommand(config,block):
    command = "nice -n 1 %s" % block.run.exp_analysisArgs
    command += " --numcputhreads %s" % config.get('global','nb_analysis_threads')
    command += " --threaded-file-access"
    command += " --local-wells-file off"
    if block.flow_start != 0:
        command += " --restart-from step.%s" % (block.flow_start-1)
    if block.flow_end != block.flows_total-1:
        command += " --restart-next step.%s" % (block.flow_end)
    command += " --flowlimit %s" % block.flows_total
    command += " --start-flow-plus-interval %s,%s" % (block.flow_start, block.flow_end-block.flow_start+1)
    command += " --no-subdir"
    command += " --output-dir=%s" % block.sigproc_results_path_tmp
    command += " --librarykey=%s" % block.run.libraryKey
    command += " --tfkey=%s" % block.run.tfKey
    command += " %s" % block.dat_path
    logger.debug('analysisArgs(PR):"%s"' % command)
    return command


def Transfer(run_name, directory_to_transfer, file_to_transfer):
    if directory_to_transfer and file_to_transfer:
        args = ['/software/cmdControl', 'datamngt', 'WellsTrans', run_name, directory_to_transfer, file_to_transfer]
    elif directory_to_transfer and not file_to_transfer:
        args = ['/software/cmdControl', 'datamngt', 'WellsTrans', run_name, directory_to_transfer]
    else:
        logger.error("TransferBlock: no directory or file specified")
        return 1

    logger.info(args)
    p = subprocess.Popen(args, stdout=subprocess.PIPE)
    out, err = p.communicate()
    logger.debug("out:%s,err:%s" % (out,err))
    if 'done' in out:
       return 0
    else:
       return 1

def instrument_busy():
    return os.path.exists("/software/config/CurExperiment")


def get_memory():
    try:
        args = shlex.split("/usr/bin/nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits".encode('utf8'))
        p = subprocess.Popen(args, stdout=subprocess.PIPE)
        out, err = p.communicate()
        mem=int(out)
        if mem < 2000 or mem > 7000:
            mem=4000
    except:
        mem=4000
    return mem

'''
# 6 ana jobs are too much for GPU # find . -name sigproc.log | xargs grep -i StreamResources
# sigproc.log:CUDA 0 StreamManager: No StreamResources could be aquired! retry pending. jobs will he handled by CPU for now!
'''
def get_number_max_analysis_jobs():
    nb_jobs = get_memory()/1000
    return nb_jobs

def get_run_list(config):

    tmp_list = []

    tmp_run_list = fnmatch.filter(os.listdir(config.get('global','results')), "R_*")
    for run_name in tmp_run_list:
        run = Run(run_name,config)
        if run:
            tmp_list.append(run)

    return tmp_list

def print_config(config,logger):
    DEBUG = True
    #DEBUG = False
    if DEBUG:
        sections = config.sections()
        for section in sections:
            logger.info('section: %s' % section)
            options = config.options(section)
            for option in options:
                logger.info(option + ": " + config.get(section,option))

class Run:
    def __init__(self,name,config):

        self.name = name
        self.dat_path = os.path.join(config.get('global','results'), name)
        self.sigproc_results_path = os.path.join(self.dat_path, "onboard_results", "sigproc_results")
        self.status = 'new'
        self.last_flow = -1

        explog_file = os.path.join(self.dat_path, "explog.txt")
        if not os.path.exists(explog_file):
            raise Exception("%s doesn't exist" % explog_file)

        try:
            self.explog_file = explog_file

            # parse explog.txt
            (head,tail) = os.path.split(explog_file)
            explogtext = load_log(head,tail)

            self.explogdict = parse_log(explogtext)
            self.exp_flows = int(self.explogdict["flows"])
            self.exp_oninstranalysis = 'yes' in self.explogdict.get('oninstranalysis','no')
            self.exp_usesynchdats = 'yes' in self.explogdict.get('use_synchdats','no')
            self.exp_oia_during_run = 'yes' in self.explogdict.get('oia_during_run','yes')
            self.exp_ecc_enabled = 'yes' in self.explogdict.get('ecc_enabled','no')
            self.exp_planned_run_short_id = self.explogdict.get('planned_run_short_id','')
            self.exp_chiptype = self.explogdict["chiptype"]

            self.exp_chipversion = self.explogdict.get('chipversion','')
            if not self.exp_chipversion or not self.exp_chipversion.startswith('P'):
                self.exp_chipversion = self.exp_chiptype

            logger.info("planned_run_short_id: '%s'" % self.exp_planned_run_short_id)
            logger.info("chiptype: '%s'" % self.exp_chiptype)
            logger.info("chipversion: '%s'" % self.exp_chipversion)


            if os.path.exists('/software/config/DataCollect.config'):
                f = open('/software/config/DataCollect.config')
                text = f.readlines()
                f.close()
                for line in text:
                    [key, value] = line.split(':')
                    key = key.strip()
                    value = value.strip()
                    logger.info('%s %s' % (key, value))
                    if key == 'TSUrl':
                        TSUrl = value
                    elif key == 'TSUserName':
                        TSUserName = value
                    elif key == 'TSPasswd':
                        TSPasswd = value
                    else:
                        continue
            else:
                raise

            logger.info('connect to:%s user:%s password:%s' % (TSUrl,TSUserName,TSPasswd))

            headers = {
                "Authorization": "Basic "+ string.strip(base64.encodestring(TSUserName + ':' + TSPasswd)),
                "Content-type": "application/json",
                "Accept": "application/json" }

            try:
                plan_found = False
                if self.exp_planned_run_short_id:
                    # get plan data
                    conn = httplib.HTTPConnection(TSUrl)
                    conn.request("GET", "/rundb/api/v1/plannedexperiment/?format=json&planShortID=%s&limit=20" % self.exp_planned_run_short_id, "", headers)
                    response = conn.getresponse()
                    logger.info('%s %s' % (response.status, response.reason))
                    data = response.read()
                    # Need to limit plans
                    plans = json.loads(data)
                    logger.debug("number of plans found on %s: %s" % (TSUrl,len(plans['objects'])))
                    logger.debug("search for plan with planShortID: '%s'" % self.exp_planned_run_short_id)

                    plan = [plan for plan in plans['objects'] if plan.get('planShortID','not_available') == self.exp_planned_run_short_id]
                    logger.debug("number of plans with planShortID '%s' found: %s" % (self.exp_planned_run_short_id,len(plan)))
                    if plan:
                        logger.debug("Plan: %s" % plan[0])
                        self.exp_beadfindArgs = plan[0].get('beadfindargs','justBeadFind')
                        self.exp_analysisArgs = plan[0].get('analysisargs','Analysis')
                        self.libraryKey       = plan[0].get('libraryKey','TCAG')
                        self.tfKey            = plan[0].get('tfKey','ATCG')
                        plan_found = True
                    else:
                        logger.debug("Problem with args in plan '%s'" % self.exp_planned_run_short_id)

                if not plan_found:
                    logger.debug("plan not available")
                    # TODO: user might starts run during TS upgrade
                    conn = httplib.HTTPConnection(TSUrl)
                    conn.request("GET", "/rundb/api/v1/analysisargs/?format=json", "", headers)
                    response = conn.getresponse()
                    data = response.read()
                    allanalysisargs = json.loads(data)

                    chip_found = False
                    logger.info('chipversion test: search for chipversion: [%s]' % self.exp_chipversion)

                    for analysisargs in allanalysisargs['objects']:
                        logger.info('chipversion test: compare with chipversion: [%s]' % analysisargs['name'])
                        if analysisargs['name']=="default_"+self.exp_chipversion:
                            logger.info('chipversion test: found analysis args for chipversion: [%s]' % analysisargs['name'])
                            logger.info(analysisargs)
                            self.exp_beadfindArgs = analysisargs['beadfindargs']
                            self.exp_analysisArgs = analysisargs['analysisargs']
                            self.libraryKey       = 'TCAG'
                            self.tfKey            = 'ATCG'
                            chip_found = True
                            break
                    if not chip_found:
                        raise Exception("Chiptype %s not found" % self.exp_chipversion)
            except:
                logger.error(traceback.format_exc())
                self.exp_beadfindArgs = 'justBeadFind --beadfind-minlivesnr 3 --region-size=216x224 --total-timeout 602'
                self.exp_analysisArgs = 'Analysis --from-beadfind --clonal-filter-bkgmodel on --region-size=216x224 --bkg-bfmask-update off --gpuWorkLoad 1 --total-timeout 602'
                self.libraryKey       = "TCAG"
                self.tfKey            = "ATCG"

            logger.info("beadfindArgs: '%s'" % self.exp_beadfindArgs)
            logger.info("analysisArgs: '%s'" % self.exp_analysisArgs)


        except:
            logger.error(traceback.format_exc())
            raise


        self.blocks = []
        try:
            self.block_to_process_start = config.getint('global','block_to_process_start')
            self.block_to_process_end = config.getint('global','block_to_process_end')
            self.discover_blocks()
        except:
            logger.error(traceback.format_exc())
            raise


    def aborted(self):

        aborted = False

        explogfinal = os.path.join(self.dat_path, "explog_final.txt")
        if os.path.exists(explogfinal):
            with open(explogfinal) as f:
                text = f.read()
                if "Critical: Aborted" in text:
                    self.status = 'aborted'
                    aborted = True

        oia_commands = os.path.join(self.dat_path, "oia_commands.txt")
        if os.path.exists(oia_commands):
            with open(oia_commands) as f:
                text = f.read()
                if "abort" in text:
                    self.status = 'aborted'
                    aborted = True

        if os.path.exists(os.path.join(self.dat_path, "aborted")):
            aborted = True

        return aborted


    def gettiming(self):
        # write header
        s = "TIMING run block chunk duration threadid start stop returncode\n"
        for block in self.blocks:
            s += block.gettiming()
        return s


    def killAnalysis(self, skipSeparator=False):
        for block in self.blocks:
            try:
                if 'justBeadFind' in block.command and skipSeparator:
                    continue
                if block.status == 'processing':
                    logger.debug("kill pid:%s, %s" % (block.process.pid, block))
                    block.process.kill()
                    #block.status = 'idle'
            except:
                logger.error(traceback.format_exc())

    def block_name_to_block_dir(self,block_name):
        block_dir = '%s_%s' % (self.explogdict[block_name].split(',')[0], self.explogdict[block_name].split(',')[1].strip())
        return block_dir

    def discover_blocks(self):

        for i in range(self.block_to_process_start,self.block_to_process_end+1):

            block_name = 'block_%03d' % i
            block_dir = self.block_name_to_block_dir(block_name)
            full_block_dat_path = os.path.join(self.dat_path, block_dir)
            full_block_sigproc_results_path = os.path.join(self.sigproc_results_path, "block_"+block_dir)

            if os.path.exists(os.path.join(self.sigproc_results_path, "block_"+block_dir, 'transfer_requested.txt')):
                block_status = "done"
            elif os.path.exists(full_block_sigproc_results_path):
                block_status = "ready_to_transfer"
            else:
                # default entry status
                block_status = "idle"

            # determine number of attempts
            nb_attempts = 0
            while os.path.exists(full_block_sigproc_results_path + "." + str(nb_attempts)):
                nb_attempts += 1

            newblock = Block(
                         self,
                         self.name,
                         block_dir,
                         block_status,
                         nb_attempts,
                         self.exp_flows,
                         full_block_dat_path,
                         full_block_sigproc_results_path)

            self.blocks.append(newblock)

    def update_status_file(self):
        try:
            with open(os.path.join(self.dat_path,'oia_status.txt'),'w') as f:
                f.write(str(self))
        except:
            logger.error(traceback.format_exc())

    def __str__(self):
        s  = "    run:" + self.name
        s += " status:" + self.status
        s += " use_synchdats:" + str(self.exp_usesynchdats)
        s += "\n"
        for block in self.blocks:
            s += str(block)
        return s

class Block:
    def __init__(self,run,run_name,name,status,nb_attempts,flows_total,dat_path,sigproc_results_path):
        self.name = name
        self.status = status
        self.process = None
        self.command = ""
        self.ret = 0
        self.run = run
        self.run_name = run_name
        self.beadfind_done = False
        self.successful_processed = 0
        self.nb_attempts = nb_attempts
        self.flows_total = flows_total
        self.flow_start = -1
        self.flow_end = -1
        self.dat_path = dat_path
        self.sigproc_results_path = sigproc_results_path
        self.sigproc_results_path_tmp = sigproc_results_path + "." + str(nb_attempts)

        self.jobtiming = []
        self.info = "no info"

    def __str__(self):
        s  = "  block:" + self.name
        s += "  status:" + str(self.status)
        s += "  flows_processed:" + str(self.flow_end+1)
        s += "  nb_attempts:" + str(self.nb_attempts)
        s += "\n"
        #s += self.gettiming()
        return s

    def gettiming(self):
        s = ""
        for x in self.jobtiming:
            starttime = x[0]
            stoptime  = x[1]
            block_info = x[2]
            block_return_code = x[3]
            thread_id = x[4]
            duration = int(time.strftime("%s",stoptime)) - int(time.strftime("%s",starttime))
            s += 'TIMING %s %s %s %s %s "%s" "%s" %s\n' % ( self.run_name, self.name, block_info, duration, thread_id, \
                                      str(time.strftime("%Y-%m-%d %H:%M:%S",starttime)), \
                                      str(time.strftime("%Y-%m-%d %H:%M:%S",stoptime)), \
                                      str(block_return_code) )

        return s



class App():
    def __init__(self):

        self.nb_max_jobs = config.getint('global','nb_max_jobs')
        logger.info('nb_max_jobs: %s' % self.nb_max_jobs)

        self.nb_max_analysis_jobs = get_number_max_analysis_jobs()
        logger.info('nb_max_analysis_jobs: %s' % self.nb_max_analysis_jobs)

        self.nb_max_beadfind_jobs = config.getint('global','nb_max_beadfind_jobs')
        logger.info('nb_max_beadfind_jobs: %s' % self.nb_max_beadfind_jobs)

        self.flowblocks = config.getint('global','flowblocks')

        # 1) Init a Thread pool with the desired number of threads
        self.pool = ThreadPool(self.nb_max_jobs,logger)

        self.blocks_to_process = []
        self.runs_in_process = []
        self.runs_processed = []

    def printStatus(self):
        logger.debug("\n***** STATUS: blocks to process: %d *****" % len(self.blocks_to_process))

    def update_oiastatus_file(self):
        try:
            with open("/software/config/OIAStatus",'w') as f:
                f.write("# this is a comment\n")
                f.write("Timestamp:%s\n" % time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
                f.write("# Processes can be Analysis or justBeadFind\n")
                f.write("Processes:%s\n" % str(self.pool.beadfind_counter + self.pool.analysis_counter))
        except:
            logger.error(traceback.format_exc())

    def get_run_dirs(self):

        run_dirs = []

        try:
            run_dirs = fnmatch.filter(os.listdir(config.get('global','results')), "R_*")
        except:
            pass

        return run_dirs

    def get_next_available_job(self,instr_busy,config):

        #logger.debug('get_next_available_job')
        # what kind of jobs are allowed to run?
        beadfind_request = True
        analysis_request = True

        if self.pool.beadfind_counter + self.pool.analysis_counter >= self.nb_max_jobs:
            logger.debug('max jobs limit reached (total)')
            beadfind_request = False
            analysis_request = False

        if self.pool.beadfind_counter >= self.nb_max_beadfind_jobs:
            logger.debug('max jobs limit reached (beadfind)')
            beadfind_request = False

        if self.pool.analysis_counter >= self.nb_max_analysis_jobs:
            logger.debug('max jobs limit reached (analysis)')
            analysis_request = False

        anablock = None

        if not beadfind_request and not analysis_request:
            # full queues
            return anablock

        for run in self.runs_in_process:

          runblocks = [block for block in self.blocks_to_process if block.run.name==run.name]

          rev=True
          # run in progress
          if run.last_flow == run.exp_flows-1:
              rev=False

          sorted_blocks = sorted(runblocks, key=lambda block: block.run.last_flow-block.successful_processed, reverse=rev)

          for block in sorted_blocks:

            if block.status != 'idle':
                #logger.debug('idle')
                continue

            # Analysis
            if analysis_request and block.beadfind_done:

                # how far can I go?

                # check for last flow
                if block.run.last_flow == block.flows_total-1:
                    new_flow_end = block.flows_total-1
                # run in progress and support odd number of flows (e.g. 396)
                else:
                    new_flow_end = ( (block.run.last_flow+1) / self.flowblocks ) * self.flowblocks - 1
                    if block.flows_total - 1 - new_flow_end < self.flowblocks or new_flow_end <= block.successful_processed-1:
                        logger.debug('new flowend for %s: (%s/%s/%s) filtered out' % (block.name, block.successful_processed-1,new_flow_end,block.flows_total-1))
                        continue

                logger.debug('new flowend for %s: (%s/%s/%s)' % (block.name, block.successful_processed-1,new_flow_end,block.flows_total-1))

                block.flow_start = block.successful_processed
                block.flow_end = new_flow_end
                block.command = getAnalysisCommand(config, block)
                anablock=block
                return anablock

            # Separator
            if beadfind_request and not block.beadfind_done:
                block.command = getSeparatorCommand(config, block)
                anablock=block
                return anablock


    def run(self):
        logger.info("Start OIA daemon")

        while True:

            if log_memory_usage:
                try:
                    logger.info("GC count: %s" % str(gc.get_count()))
                    h = hpy()
                    logger.debug(str(h.heap()))
                except:
                    logger.error(traceback.format_exc())

            all_run_dirs = self.get_run_dirs()
            logger.debug('RUNS DIRECTORIES: %s' % all_run_dirs)


            # check for runs in progress
            if instrument_busy():
                logger.info('run in progress, kill HD jobs')
                # abort all HD jobs if run in progress
                for run in self.runs_in_process:
                    logger.info('kill %s' % run.name)
                    #run.killAnalysis(skipSeparator=True)
                    #TODO set correct block status

            # update flow status
            logger.debug('runs in p: %s' % self.runs_in_process)
            for run in self.runs_in_process:
                # check for last flow
                logger.debug('runs in p: %s' % os.path.join(run.dat_path,'thumbnail','acq_*.dat'))
                files = glob.glob(os.path.join(run.dat_path,'thumbnail','acq_*.dat'))
                files.sort()
                #logger.debug('all files: %s' % files)
                if files:
                    s = int(re.match('acq_([0-9]*).dat',os.path.split(files[-1])[1]).group(1))
                    logger.debug('last file number: %d' % s)
                    run.last_flow = s

            # check for deleted runs
            # TODO

            # check for aborted runs
            for run in self.runs_in_process:
                if run.aborted():
                    logger.info('run aborted %s' % run.name)
                    run.killAnalysis(skipSeparator=False)
                    for block in run.blocks:
                        if block in self.blocks_to_process:
                            self.blocks_to_process.remove(block)
                    self.runs_in_process.remove(run)
                    self.runs_processed.append(run)

            # check for finished runs
            #TODO, add timeout per run?
            logger.debug('check for finished runs')
            for run in self.runs_in_process:
                # check whether all blocks have finished
                nb_blocks_finished = 0
                for block in run.blocks:
                    if block.status == 'done':
                        nb_blocks_finished += 1

                logger.info("Run %s: %s blocks ready" % (run.name, nb_blocks_finished))
                if nb_blocks_finished == len(run.blocks):

                    # update status
                    run.update_status_file()

                    # write timings
                    timing_file = os.path.join(run.sigproc_results_path,'timing.txt')
                    try:
                        with open(timing_file, 'a') as f:
                            f.write(run.gettiming())
                    except:
                        logger.error(traceback.format_exc())

                    #transfer timing.txt
                    try:
                        directory_to_transfer="onboard_results/sigproc_results"
                        file_to_transfer="timing.txt"
                        ret = Transfer(run.name, directory_to_transfer, file_to_transfer)
                        if ret == 0:
                            logger.debug("Transfer registered %s %s %s" % (run.name, directory_to_transfer, file_to_transfer))
                        else:
                            logger.error("Transfer failed %s %s %s, is datacollect running?" % (run.name, directory_to_transfer, file_to_transfer))
                    except:
                        logger.error(traceback.format_exc())

                    self.runs_in_process.remove(run)
                    self.runs_processed.append(run)
                else:
                    logger.debug(run.gettiming())


            processed_run_dirs = [run.name for run in self.runs_processed]
            logger.debug('RUNS PROCESSED: %s' % processed_run_dirs)

            in_process_run_dirs = [run.name for run in self.runs_in_process]
            logger.debug('RUNS IN PROCESS: %s' % in_process_run_dirs)

            new_run_dirs = list(set(all_run_dirs) - set(processed_run_dirs) - set(in_process_run_dirs))
            logger.debug('RUNS NEW: %s' % new_run_dirs)

            # update /software/config/OIAStatus
            self.update_oiastatus_file()

            for run in self.runs_in_process:
                run.update_status_file()

            # add new runs (blocks)
            for run_dir in new_run_dirs:
                logger.info('NEW RUN DETECTED: %s' % run_dir)

                try:
                    arun = Run(run_dir,config)
                    logger.info(arun)
                except:
                    logger.error(traceback.format_exc())
                    continue

                if arun.exp_oia_during_run:
                    wait_for_flow = 0
                else:
                    # wait for last flow
                    wait_for_flow = arun.exp_flows-1

                if arun.exp_flows < self.flowblocks:
                    logger.info('skip run: %s, not enough flows' % arun.name)
                    self.runs_processed.append(arun)
                    continue

                if arun.aborted():
                    logger.info('skip aborted run: %s' % arun.name)
                    self.runs_processed.append(arun)
                    continue

                if not os.path.exists(os.path.join(arun.dat_path , 'thumbnail', 'acq_%04d.dat' % wait_for_flow)):
                    logger.info('NEW RUN NOT READY, missing dat file: %s' % os.path.join(arun.dat_path , 'thumbnail', 'acq_%04d.dat' % wait_for_flow))
                    continue

                self.runs_in_process.append(arun)

                logger.info('arun.exp_oninstranalysis: %s' % arun.exp_oninstranalysis)
                if not arun.exp_oninstranalysis:
                    continue

                # ignore autoanalyze option in explog.txt
                logger.info('autoanalyze: %s' % arun.explogdict['autoanalyze'])
                #if not arun.explogdict['autoanalyze']: # contains True,False instead of yes, no
                #    continue


                full_sigproc_results_path = os.path.join(config.get('global','analysisresults'), run_dir, "onboard_results/sigproc_results")
                if os.path.exists(full_sigproc_results_path):
                    logger.info("partial on-instrument analysis detected %s" % run_dir)
                else:
                    try:
                        os.makedirs( full_sigproc_results_path )
                        os.symlink(os.path.join(config.get('global','analysisresults'),arun.name,'onboard_results'), os.path.join(config.get('global','results'),arun.name,'onboard_results'))
                    except:
                        logger.error('link failed %s' % arun)
                        logger.error(traceback.format_exc())
                logger.info("ADD %s blocks" % arun.name)
                for block in arun.blocks:
                    if block.status != 'done':
                        self.blocks_to_process.append(block)

            self.printStatus()

            # process runs
            timestamp = time.time()
            logger.info('time since last run check %s' % timestamp)
            while self.blocks_to_process:
                # wait a while before checking if queue is empty
                time.sleep(3)

                logger.debug('number blocks to process: %s %s/%s %s/%s' % ( len(self.blocks_to_process), self.pool.beadfind_counter, self.nb_max_beadfind_jobs, self.pool.analysis_counter, self.nb_max_analysis_jobs ) )

                # every 60 sec check for new run
                if time.time()-timestamp > 60:
                    logger.debug('check for new run')
                    break

                # check status of blocks
                for block in self.blocks_to_process:

                    # check for processed blocks
                    if block.status == 'processed':
                        block.status = "idle"
                        if block.ret == 0:
                            if block.flow_end == -1:
                                block.beadfind_done = True
                            else:
                                block.successful_processed = block.flow_end+1
                            if block.successful_processed == block.flows_total:
                                block.status = 'sigproc_done'
                        else:
                            logger.error('Block %s failed with return code %s' % (block.name, block.ret))
                            block.nb_attempts += 1
                            block.sigproc_results_path_tmp = block.sigproc_results_path + "." + str(block.nb_attempts)
                            block.beadfind_done = False
                            block.successful_processed = 0
                            block.flow_start = -1
                            block.flow_end = -1
                            block.status = "idle"

                    # processed blocks
                    if block.status == 'sigproc_done' or block.status == 'sigproc_failed':

                        # 1. rename block / last sigproc attempt
                        try:
                            if not os.path.exists(block.sigproc_results_path):
                                logger.info('rename block %s %s %s' % (block.name,block.sigproc_results_path_tmp,block.sigproc_results_path))
                                if block.nb_attempts >= config.getint('global','nb_retries'):
                                    shutil.move(block.sigproc_results_path + "." + str(block.nb_attempts-1), block.sigproc_results_path)
                                else:
                                    shutil.move(block.sigproc_results_path_tmp, block.sigproc_results_path)
                        except:
                            logger.error('renaming failed %s' % block.name)
                            logger.error(traceback.format_exc())
                            pass

                        # 2. remove *.step files
                        if block.status == 'sigproc_done':
                            try:
                                for filename in os.listdir(block.sigproc_results_path):
                                    if fnmatch.fnmatch(filename, 'step.*'):
                                        full_path_to_file = os.path.join(block.sigproc_results_path,filename)
                                        logger.info('remove step file: %s' % full_path_to_file)
                                        os.remove(full_path_to_file)
                            except:
                                logger.error('removing step file failed %s' % block.name)
                                logger.error(traceback.format_exc())
                                pass

                        # 3. generate MD5SUM for each output file
                        try:
                            md5sums = {}
                            for filename in os.listdir(block.sigproc_results_path):
                                full_filename = os.path.join(block.sigproc_results_path,filename)
                                if os.path.isdir(full_filename):
                                    continue
                                with open(full_filename,'rb') as f:
                                    binary_content = f.read()
                                    md5sums[filename] = hashlib.md5(binary_content).hexdigest()
                            with open(os.path.join(block.sigproc_results_path,'MD5SUMS'), 'w') as f:
                                for filename,hexdigest in md5sums.items():
                                    f.write("%s  %s\n" % (hexdigest,filename))
                        except:
                            logger.error(traceback.format_exc())

                        block.status = 'ready_to_transfer'

                    if block.status == 'ready_to_transfer':
                        directory_to_transfer="onboard_results/sigproc_results/"+"block_"+block.name
                        file_to_transfer=""
                        logger.info("Register transfer: %s %s %s" % (block.run_name, directory_to_transfer, file_to_transfer))
                        ret = Transfer(block.run_name, directory_to_transfer, file_to_transfer)
                        if ret == 0:
                            block.status = "transferred"
                            try:
                                open(os.path.join(block.sigproc_results_path,'transfer_requested.txt'), 'w').close()
                            except:
                                logger.error(traceback.format_exc())
                                pass
                        else:
                            logger.error("Transfer failed %s %s %s, is datacollect running?" % (block.run_name, directory_to_transfer, file_to_transfer))

                    if block.status == 'transferred':
                        logger.debug("DONE: %s" % (block.name))
                        block.status = 'done'
                        self.blocks_to_process.remove(block)

                instr_busy = instrument_busy()
                try:
                    ablock = self.get_next_available_job(instr_busy,config)
                except:
                    ablock = None
                    logger.error(traceback.format_exc())

                if ablock:
                    if ablock.nb_attempts >= config.getint('global','nb_retries'):
                        ablock.status = 'sigproc_failed'
                    else:
                        ablock.status = 'queued'
                        ablock.info = '%s-%s' % (ablock.flow_start, ablock.flow_end)
                        logger.debug('%s submitted (%s-%s)' % (ablock.name, ablock.flow_start, ablock.flow_end))
                        self.pool.add_task(ablock)

            #wait 10 sec if no blocks are available
            time.sleep(10)

if __name__ == '__main__':


    # parse on-instrument configuration file
    config_file = "/software/config/oia.config"
    config = ConfigParser.RawConfigParser()
    config.optionxform = str # don't convert to lowercase
    config.read(config_file)

    try:
        log_memory_usage = False
        if config.get('global','log_memory_usage') =='yes':
            from guppy import hpy
            import gc
            log_memory_usage = True
    except:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', dest='verbose', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        print "oiad:", args

        for run in get_run_list(config):
            print run


    logger = logging.getLogger("OIA")
    logger.setLevel(logging.DEBUG)
    rothandler = logging.handlers.RotatingFileHandler("/var/log/oia.log", maxBytes=1024*1024*10, backupCount=5)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    rothandler.setFormatter(formatter)
    cachehandler = logging.handlers.MemoryHandler(1024, logging.ERROR, rothandler)
    logger.addHandler(rothandler)
    logger.addHandler(cachehandler)

    print_config(config, logger)

    try:
        app = App()
        app.run()
    except:
        logger.error(traceback.format_exc())
