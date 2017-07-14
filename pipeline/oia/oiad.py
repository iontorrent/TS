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
import re
import json
import socket

from collections import deque

from Queue import Queue
from threading import Thread
import multiprocessing


#from ion.utils.explogparser import load_log
#from ion.utils.explogparser import parse_log
#from ion.utils import explogparser


import logging
import logging.handlers

try:
    from pynvml import *
    pynvml_available = True
except:
    pynvml_available = False

LowMemRetry=0

class Worker(Thread):
    """Thread executing tasks from a given tasks queue"""
    def __init__(self, id, tasks, pool, logger):
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
                block.status = "performJustBeadFind"
            if 'Analysis' in block.command:
                self.pool.analysis_counter += 1
                block.status = "performAnalysis"
            if 'BaseCaller' in block.command:
                self.pool.basecaller_counter += 1
                block.status = "performPhaseEstimation"

            logger.info("%s: T1: %s" % (self.id, block))
            starttime = time.localtime()

            try:
                if not os.path.exists(block.sigproc_results_path_tmp):
                    os.makedirs(block.sigproc_results_path_tmp)
                    logger.debug('mkdir %s' % block.sigproc_results_path_tmp)

                #try:
                #    if 'justBeadFind' in block.command and block.run.exp_ecc_enabled == "yes":
                #        outfile = open(os.path.join(block.sigproc_results_path_tmp, 'ecc.log'), "a")
                #        ecc_command = "/usr/share/ion/oia/ecc.py -i %s -o %s" % (
                #            block.dat_path, block.sigproc_results_path_tmp)
                #        args = shlex.split(ecc_command.encode('utf8'))
                #        logger.info("%s: run process: %s" % (self.id, args))
                #        p = subprocess.Popen(args, stdout=outfile, stderr=subprocess.STDOUT)
                #        # add popen process to block
                #        block.process = p
                #        ret = p.wait()
                #except:
                #    logger.error(traceback.format_exc())
                #    pass

                logger.info("%s: run process: %s" % (self.id, block))

                # don't use shell=True , otherwise child process cannot be killed
                outfile = open(os.path.join(block.sigproc_results_path_tmp, 'sigproc.log'), "a")
                args = shlex.split(block.command.encode('utf8'))
                my_environment = os.environ.copy()
                if not "/usr/local/bin" in my_environment['PATH']:
                    my_environment['PATH'] = my_environment['PATH']+":/usr/local/bin"
                # logger.info(my_environment)
                p = subprocess.Popen(args, stdout=outfile, stderr=subprocess.STDOUT, env=my_environment)
                # add popen process to block
                block.process = p
                ret = p.wait()

                # error generation
                # rand = random.randint(0,99)
                # if rand < 2:
                #    ret = 777
                # else:
                #    ret = 0

                if 0 and ret == 0 and block.flow_start < 180 and block.flow_end > 180:
                    try:
                        phase_estimation_cmd = "nice -n 1 %s --just-phase-estimation --num-threads %s --input-dir=%s --output-dir=%s" % (
                            block.run.exp_prebasecallerArgs_block, cpu_cores, block.sigproc_results_path_tmp, block.sigproc_results_path_tmp)
                        args = shlex.split(phase_estimation_cmd.encode('utf8'))
                        logger.info("%s: run process: %s" % (self.id, args))
                        p = subprocess.Popen(
                            args, stdout=outfile, stderr=subprocess.STDOUT, env=my_environment)
                        block.process = p
                        ret_phase_estimation = p.wait()
                    except:
                        logger.error('%s failed to run' % phase_estimation_cmd)
                        pass

                block.ret = ret
            except:
                logger.error(traceback.format_exc())
                block.ret = 666

            try:
                f = open(os.path.join(block.sigproc_results_path_tmp, 'analysis_return_code.txt'), 'w')
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
            if 'BaseCaller' in block.command:
                self.pool.basecaller_counter -= 1
                block.basecaller_done=True

            stoptime = time.localtime()
            block.jobtiming.append((starttime, stoptime, block.info, block.ret, self.id))
            logger.info("%s: T2: %s" % (self.id, block))


class ThreadPool:
    """Pool of threads consuming tasks from a queue"""
    def __init__(self, num_threads, logger):
        self.tasks = Queue(num_threads)  # limit the queue to the number of threads
        self.beadfind_counter = 0
        self.analysis_counter = 0
        self.basecaller_counter = 0

        for threadid in range(num_threads): Worker(threadid, self.tasks, self, logger)

    def add_task(self, block):
        """Add a task to the queue"""
        # print 'add task', block
        self.tasks.put(block)

    def wait_completion(self):
        """Wait for completion of all the tasks in the queue"""
        self.tasks.join()


def run_a_shell_process(command):
    # command = "echo "+command
    # print command
    # time.sleep(30)
    ret = subprocess.call(command, shell=True)
    return ret


def getSeparatorCommand(config, block):
    # command = "strace -o %s/strace.log %s" %
    # (block.sigproc_results_path_tmp,block.run.exp_beadfindArgs_block)
    if block.name == "thumbnail":
        command = "nice -n 3 %s" % block.run.exp_beadfindArgs_thumbnail
    else:
        command = "nice -n 3 %s" % block.run.exp_beadfindArgs_block
    command += " --local-wells-file false"
    command += " --beadfind-num-threads %s" % config.get('global', 'nb_beadfind_threads')
    command += " --no-subdir"
    command += " --output-dir=%s" % block.sigproc_results_path_tmp
    command += " --librarykey=%s" % block.run.libraryKey
    if block.run.tfKey != "":
        command += " --tfkey=%s" % block.run.tfKey
    # command += " --explog-path=%s" % os.path.join(block.run.analysis_path, 'explog.txt')
    command += " %s" % block.dat_path
    logger.debug('beadfindArgs(PR):"%s"' % command)
    return command




def getAnalysisCommand(config, block):
    if block.name == "thumbnail":
        command = "nice -n 1 %s" % block.run.exp_analysisArgs_thumbnail
    else:
        command = "nice -n 1 %s" % block.run.exp_analysisArgs_block
    command += " --numcputhreads %s" % config.get('global', 'nb_analysis_threads')
    command += " --local-wells-file false"
    if block.flow_start != 0:
        command += " --restart-from step.%s" % (block.flow_start-1)
    if block.flow_end != block.flows_total-1:
        command += " --restart-next step.%s" % (block.flow_end)
    command += " --flowlimit %s" % block.flows_total
    command += " --start-flow-plus-interval %s,%s" % (block.flow_start, block.flow_end-block.flow_start+1)
    command += " --no-subdir"
    command += " --output-dir=%s" % block.sigproc_results_path_tmp
    command += " --librarykey=%s" % block.run.libraryKey
    if block.run.tfKey != "":
        command += " --tfkey=%s" % block.run.tfKey
    # command += " --explog-path=%s" % os.path.join(block.run.analysis_path, 'explog.txt')
    command += " %s" % block.dat_path
    logger.debug('analysisArgs(PR):"%s"' % command)
    return command

def getBaseCallerCommand(config, block):
    # command = "strace -o %s/strace.log %s" %
    # (block.sigproc_results_path_tmp,block.run.exp_beadfindArgs_block)
    command =  "nice -n 3 %s --just-phase-estimation" % (block.run.exp_prebasecallerArgs_block)
    command += " --num-threads %s" % (cpu_cores) #config.get('global', 'nb_beadfind_threads'))
    command += " --input-dir=%s --output-dir=%s" % (
                       block.sigproc_results_path_tmp, block.sigproc_results_path_tmp)
    logger.debug('BaseCallerArgs(PR):"%s"' % command)
    return command

def Transfer(run_name, directory_to_transfer, file_to_transfer):
    if directory_to_transfer and file_to_transfer:
        args = ['/software/cmdControl', 'datamngt', 'WellsTrans',
                run_name, directory_to_transfer, file_to_transfer]
    elif directory_to_transfer and not file_to_transfer:
        args = ['/software/cmdControl', 'datamngt', 'WellsTrans', run_name, directory_to_transfer]
    else:
        logger.error("TransferBlock: no directory or file specified")
        return 1

    logger.info(args)
    p = subprocess.Popen(args, stdout=subprocess.PIPE)
    out, err = p.communicate()
    logger.debug("out:%s,err:%s" % (out, err))
    if 'done' in out:
        return 0
    else:
        return 1


def instrument_busy():
    return os.path.exists("/software/config/CurExperiment")


def get_run_list(config):

    tmp_list = []

    tmp_run_list = fnmatch.filter(os.listdir(config.get('global', 'results')), "R_*")
    for run_name in tmp_run_list:
        run = Run(run_name, config)
        if run:
            tmp_list.append(run)

    return tmp_list


class Run:
    def __init__(self, name, config):

        self.name = name
        self.dat_path = os.path.join(config.get('global', 'results'), name)
        self.analysis_path = os.path.join(config.get('global', 'analysisresults'), name)
	self.sigproc_results_path = os.path.join(
       		config.get('global', 'analysisresults'), name, "onboard_results", "sigproc_results")
        self.status = 'new'
        self.last_flow = -1

        try:
            if not os.path.exists(self.sigproc_results_path):
                os.makedirs(self.sigproc_results_path)
        except:
            logger.error(traceback.format_exc())

        explog_file = os.path.join(self.analysis_path, "explog.json")
        if not os.path.exists(explog_file):
            shutil.copyfile(os.path.join(self.dat_path, "explog.json"),
                            os.path.join(self.analysis_path, "explog.json"))
        if not os.path.exists(explog_file):
            raise Exception("%s doesn't exist" % explog_file)

        try:
            if not os.path.exists(os.path.join(self.dat_path, 'onboard_results')):
                os.symlink(os.path.join(self.analysis_path, 'onboard_results'),
                           os.path.join(self.dat_path, 'onboard_results'))
        except:
            logger.error(traceback.format_exc())

        try:
            self.explog_file = explog_file

            # parse explog.txt
            #(head, tail) = os.path.split(explog_file)
            #explogtext = load_log(head, tail)

            f=open(explog_file,'r')
            contents=f.read()
            f.close()
            contents = contents.replace('\\','\\\\')
            self.explogdict = json.loads(contents, strict=False)
            #parse_log(explogtext)
            self.exp_flows = int(self.explogdict["Flows"])
            self.exp_oninstranalysis = self.explogdict['OnInstrAnalysis']
            self.exp_oia_during_run = self.explogdict['OIA_During_Run']
            self.exp_ecc_enabled = self.explogdict['ECC Enabled']
            self.exp_planned_run_guid = self.explogdict['Planned Run GUID']
            self.exp_chiptype = self.explogdict["ChipType"]
            self.exp_seqkitplanname = self.explogdict['SeqKitPlanName']

            self.exp_chipversion = self.explogdict['ChipVersion']
            if not self.exp_chipversion:
                self.exp_chipversion = self.exp_chiptype

            logger.info("planned_run_guid: '%s'" % self.exp_planned_run_guid)
            logger.info("chiptype: '%s'" % self.exp_chiptype)
            logger.info("chipversion: '%s'" % self.exp_chipversion)

            try:
                analysisargs = {}
                if os.path.exists(os.path.join(self.dat_path, "reanalysis_args.json")):
                    f = open(os.path.join(self.dat_path, "reanalysis_args.json"), 'r')
                    analysisargs = json.load(f)
                    f.close()
                    logger.info("use analysisargs from %s" %
                                os.path.join(self.dat_path, "reanalysis_args.json"))
                elif os.path.exists(os.path.join(self.dat_path, "planned_run.json")):
                    f = open(os.path.join(self.dat_path, "planned_run.json"), 'r')
                    analysisargs = json.load(f)['objects'][0]
                    f.close()
                    logger.info("use analysisargs from %s" % os.path.join(self.dat_path, "planned_run.json"))
                else:
                    logger.info("trying to fetch planned run for %s" %
                                os.path.join(self.dat_path, "planned_run.json"))
                    http_command = "python /software/testing/get_reanalysis_args.py -e %s" % (name)
                    args = shlex.split(http_command.encode('utf8'))
                    logger.info("run process: %s" % (args))
                    outfile = open(os.path.join(self.dat_path, 'get_reanalysis_args_oia.log'), "w")
                    p = subprocess.Popen(args, stdout=outfile, stderr=subprocess.STDOUT)
                    # add popen process to block
                    ret = p.wait()
                    outfile.close()
                    if os.path.exists(os.path.join(self.dat_path, "planned_run.json")):
                        with open(os.path.join(self.dat_path, "planned_run.json"), 'r') as f:
                            try:
                                analysisargs = json.load(f)['objects'][0]
                                logger.info("Using Plan information from %s" %
                                            os.path.join(self.dat_path, "planned_run.json"))
                            except IndexError:
                                raise Exception("File %s is empty",
                                                os.path.join(self.dat_path, "planned_run.json"))

                if len(analysisargs) > 0:
                    self.exp_beadfindArgs_thumbnail = analysisargs['thumbnailbeadfindargs']
                    self.exp_analysisArgs_thumbnail = analysisargs['thumbnailanalysisargs']
                    self.exp_beadfindArgs_block = analysisargs['beadfindargs']
                    self.exp_analysisArgs_block = analysisargs['analysisargs']
                    self.exp_prebasecallerArgs_block = analysisargs['prebasecallerargs']
                    self.libraryKey = analysisargs['libraryKey']
                    self.tfKey = analysisargs['tfKey']
                else:
                    raise Exception("There are no Analysis args available")
            except:
                raise

            logger.info("thumbnail beadfindArgs: '%s'" % self.exp_beadfindArgs_thumbnail)
            logger.info("thumbnail analysisArgs: '%s'" % self.exp_analysisArgs_thumbnail)
            logger.info("block beadfindArgs: '%s'" % self.exp_beadfindArgs_block)
            logger.info("block analysisArgs: '%s'" % self.exp_analysisArgs_block)
            logger.info("block prebasecallerArgs: '%s'" % self.exp_prebasecallerArgs_block)

        except:
            logger.error(traceback.format_exc())
            raise

        self.blocks = []
        try:
            self.block_to_process_start = config.getint('global', 'block_to_process_start')
            self.block_to_process_end = config.getint('global', 'block_to_process_end')
            self.discover_blocks()
        except:
            logger.error(traceback.format_exc())
            raise

    def aborted(self):

        aborted = False

        if not os.path.exists(self.dat_path):
            logger.info("directory missing for '%s'" % self.dat_path)
            aborted = True

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

    def killAnalysis(self):
        for block in self.blocks:
            try:
                if block.status == 'performJustBeadFind' or block.status == 'performAnalysis' or block.status == 'performPhaseEstimation':
                    logger.debug("kill pid:%s, %s" % (block.process.pid, block))
                    block.process.kill()
                    # block.status = 'idle'
            except:
                logger.error(traceback.format_exc())

    def block_name_to_block_dir(self, block_name):
        block_dir = 'X%s_Y%s' % (self.explogdict[block_name]['X'],self.explogdict[block_name]['Y'])
        return block_dir

    def discover_blocks(self):

        range_of_blocks = [('block_%03d' % i)
                           for i in range(self.block_to_process_start, self.block_to_process_end+1)]
        block_dirs = [self.block_name_to_block_dir(x) for x in range_of_blocks if x in self.explogdict]
#        for block_dir in ['thumbnail'] + block_dirs:
        for block_dir in block_dirs:

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
            with open(os.path.join(self.dat_path, 'oia_status.txt'), 'w') as f:
                f.write(str(self))
        except:
            logger.error(traceback.format_exc())

    def __str__(self):
        s = "    run:" + self.name
        s += " status:" + self.status
        s += " use_synchdats:0"
        s += "\n"
        for block in self.blocks:
            s += str(block)
        return s


class Block:
    def __init__(self, run, run_name, name, status, nb_attempts, flows_total, dat_path, sigproc_results_path):
        self.name = name
        self.status = status
        self.process = None
        self.command = ""
        self.ret = 0
        self.run = run
        self.run_name = run_name
        self.beadfind_done = False
        self.analysis_done = False
        self.basecaller_done = False
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
        s = "  block:" + self.name
        s += "  status:" + str(self.status)
        s += "  flows_processed:" + str(self.flow_end+1)
        s += "  nb_attempts:" + str(self.nb_attempts)
        s += "\n"
        # s += self.gettiming()
        return s

    def gettiming(self):
        s = ""
        for x in self.jobtiming:
            starttime = x[0]
            stoptime = x[1]
            block_info = x[2]
            block_return_code = x[3]
            thread_id = x[4]
            duration = int(time.strftime("%s", stoptime)) - int(time.strftime("%s", starttime))
            s += 'TIMING %s %s %s %s %s "%s" "%s" %s\n' % (self.run_name, self.name, block_info, duration, thread_id, \
                                                           str(time.strftime("%Y-%m-%d %H:%M:%S", starttime)), \
                                                           str(time.strftime("%Y-%m-%d %H:%M:%S", stoptime)), \
                                                           str(block_return_code))

        return s


class App():
    def __init__(self):

        self.nb_max_jobs = config.getint('global', 'nb_max_jobs')
        logger.info('nb_max_jobs: %s' % self.nb_max_jobs)

        self.nb_max_analysis_jobs = config.getint('global', 'nb_max_analysis_jobs')
        logger.info('nb_max_analysis_jobs: %s' % self.nb_max_analysis_jobs)

        self.nb_max_beadfind_jobs = config.getint('global', 'nb_max_beadfind_jobs')
        logger.info('nb_max_beadfind_jobs: %s' % self.nb_max_beadfind_jobs)
        
	self.nb_max_basecaller_jobs = config.getint('global', 'nb_max_basecaller_jobs')
        logger.info('nb_max_basecaller_jobs: %s' % self.nb_max_basecaller_jobs)


        self.flowblocks = config.getint('global', 'flowblocks')

        # 1) Init a Thread pool with the desired number of threads
        self.pool = ThreadPool(self.nb_max_jobs, logger)

        self.blocks_to_process = []
        self.runs_in_process = []
        self.runs_processed = []

    def printStatus(self):
        logger.debug("\n***** STATUS: blocks to process: %d *****" % len(self.blocks_to_process))

    def update_oiastatus_file(self):
        try:
            with open("/software/config/OIAStatus", 'w') as f:
                f.write("# this is a comment\n")
                f.write("Timestamp:%s\n" % time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
                f.write("# Processes can be Analysis, justBeadFind or BaseCaller\n")
                f.write("Processes:%s\n" % str(self.pool.beadfind_counter + self.pool.analysis_counter + self.pool.basecaller_counter))
        except:
            logger.error(traceback.format_exc())

    def get_run_dirs(self):

        run_dirs = []

        try:
            run_dirs = fnmatch.filter(os.listdir(config.get('global', 'results')), "R_*")
        except:
            pass

        return run_dirs

    def get_next_available_job(self, config):

        # logger.debug('get_next_available_job')
        # what kind of jobs are allowed to run?
        beadfind_request = True
        analysis_request = True
        basecaller_request = True

        if self.pool.beadfind_counter + self.pool.analysis_counter + self.pool.basecaller_counter >= self.nb_max_jobs:
            logger.debug('max jobs limit reached (total)')
            beadfind_request = False
            analysis_request = False
            basecaller_request = False

        if self.pool.beadfind_counter >= self.nb_max_beadfind_jobs:
            logger.debug('max jobs limit reached (beadfind)')
            beadfind_request = False
            basecaller_request = False # don't run basecaller and justBeadFind at the same time..

        if self.pool.analysis_counter >= self.nb_max_analysis_jobs:
            logger.debug('max jobs limit reached (analysis)')
            analysis_request = False

        if (self.pool.beadfind_counter + self.pool.basecaller_counter) >= self.nb_max_basecaller_jobs:
            logger.debug('max jobs limit reached (basecaller)')
            basecaller_request = False
        # 6 analysis jobs are too much for some GPU's # find . -name sigproc.log | xargs grep -i StreamResources
        # sigproc.log:CUDA 0 StreamManager: No StreamResources could be aquired!
        # retry pending. jobs will he handled by CPU for now!
        if self.pool.analysis_counter >= total_GPU_memory/1000:
            logger.debug('GPU memory reached (analysis)')
            analysis_request = False

        # check if enough ressources are available
        # GPU limit
#       if current_HOST_memory + block.HOST_memory_requirement_beadfind >= 128G
#            beadfind_request = False

#       if current_HOST_memory + block.HOST_memory_requirement_analysis >= 128G
#            analysis_request = False
#        total_GPU_memory = 4000

#       if current_GPU_memory + block.GPU_memory_requirement_analysis >= 6G
#            analysis_request = False

# nb_beadfind_threads = 4
# nb_analysis_threads = 6
# nb_max_jobs = 5
# nb_max_beadfind_jobs = 5
# nb_max_analysis_jobs = 4
# HOST_memory_requirement_beadfind = 7G
# HOST_memory_requirement_analysis = 20G
# GPU_memory_requirement_analysis = 1G

        # for development: beadfind first
        # if self.pool.beadfind_counter > 0:
        #    analysis_request = False

        anablock = None

        if not beadfind_request and not analysis_request and not basecaller_request:
            # full queues
            return anablock

        for run in self.runs_in_process:

            runblocks = [block for block in self.blocks_to_process if block.run.name == run.name]

            rev = True
            # run in progress
            if run.last_flow == run.exp_flows-1:
                rev = False

            sorted_blocks = sorted(
                runblocks, key=lambda block: block.run.last_flow-block.successful_processed, reverse=rev)

            try:
                nb_max_jobs = config.getint(run.exp_chipversion, 'nb_max_jobs')
                nb_max_beadfind_jobs = config.getint(run.exp_chipversion, 'nb_max_beadfind_jobs')
                nb_max_analysis_jobs = config.getint(run.exp_chipversion, 'nb_max_analysis_jobs')
                nb_max_basecaller_jobs = config.getint(run.exp_chipversion, 'nb_max_basecaller_jobs')
            except:
                nb_max_jobs = config.getint('DefaultChip', 'nb_max_jobs')
                nb_max_beadfind_jobs = config.getint('DefaultChip', 'nb_max_beadfind_jobs')
                nb_max_analysis_jobs = config.getint('DefaultChip', 'nb_max_analysis_jobs')
                nb_max_basecaller_jobs = config.getint('DefaultChip', 'nb_max_basecaller_jobs')

            if self.pool.beadfind_counter >= nb_max_beadfind_jobs:
                logger.debug('max beadfind jobs limit reached (%s)' % run.exp_chipversion)
                beadfind_request = False
                basecaller_request = False  # don't run beadfind and basecaller at the same time

            if self.pool.analysis_counter >= nb_max_analysis_jobs:
                logger.debug('max analysis jobs limit reached (%s)' % run.exp_chipversion)
                analysis_request = False

            if self.pool.basecaller_counter >= nb_max_basecaller_jobs:
                logger.debug('max basecaller jobs limit reached (%s)' % run.exp_chipversion)
                basecaller_request = False
            
            if (self.pool.analysis_counter + self.pool.beadfind_counter + self.pool.basecaller_counter) >= nb_max_jobs:
                logger.debug('max jobs limit reached (%s)' % run.exp_chipversion)
                analysis_request = False
                beadfind_request = False
                basecaller_request = False

            for block in sorted_blocks:

                if block.status != 'idle':
                # logger.debug('idle')
                    continue

                # Analysis
                if basecaller_request and block.beadfind_done and not block.basecaller_done and block.flow_end > 170:
                    block.command = getBaseCallerCommand(config, block)
                    anablock = block
                    return anablock

                if analysis_request and block.beadfind_done and not block.analysis_done:

                    # how far can I go?

                    # check for last flow
                    if block.run.last_flow == block.flows_total-1:
                        new_flow_end = block.flows_total-1
                    # run in progress and support odd number of flows (e.g. 396)
                    else:
                        new_flow_end = ((block.run.last_flow+1) / self.flowblocks) * self.flowblocks - 1
                        if block.flows_total - 1 - new_flow_end < self.flowblocks or new_flow_end <= block.successful_processed-1:
                            logger.debug('new flowend for %s: (%s/%s/%s) filtered out' % (
                                block.name, block.successful_processed-1, new_flow_end, block.flows_total-1))
                            continue

                    logger.debug('new flowend for %s: (%s/%s/%s)' %
                                 (block.name, block.successful_processed-1, new_flow_end, block.flows_total-1))

                    block.flow_start = block.successful_processed
                    block.flow_end = new_flow_end
                    block.command = getAnalysisCommand(config, block)
                    anablock = block
                    return anablock

                # Separator
                if beadfind_request and not block.beadfind_done:
                    block.command = getSeparatorCommand(config, block)
                    anablock = block
                    return anablock

    def updateUsage(self):
        os.system("nvidia-smi --query-gpu=timestamp,utilization.gpu --format=csv | grep -v timestamp >> /var/log/gpu_util.log&")
        os.system("date >> /var/log/cpu_util.log")
        os.system("top -b -d 2 -n 2  | grep 'Cpu(s)' | tail -n+2 >> /var/log/cpu_util.log&")
        os.system("date >> /var/log/disk_util.log; vmstat -d | grep sd >> /var/log/disk_util.log")
        
    def getCurrentRunInformation(self):
        CurExp = ""
        CurFlow = 0

        # check for run in progress
        try:
            if os.path.isfile("/software/config/CurExperiment"):
                CurF = open("/software/config/CurExperiment", 'r')
                words = CurF.readline().split()
                CurF.close()
                CurExp = words[0]
                CurFlow = int(words[1])
                logger.info('run in progress %s' % CurExp)
                logger.info('run in progress flows %d', CurFlow)
        except:
            logger.info('failed to read /software/config/CurExperiment')

        return [CurExp, CurFlow]

    def run(self):
        global LowMemRetry
        logger.info("Start OIA daemon")

        while not os.path.isdir(config.get('global', 'analysisresults')):
            logger.debug('waiting for %s' % config.get('global', 'analysisresults'))
            time.sleep(10)

        while True:

 
            if os.path.isfile("/software/config/OIALargeChip"):
                # increase the retry count because our jobs were killed
                LowMemRetry = LowMemRetry + 1
                while os.path.isfile("/software/config/OIALargeChip"):
                    self.update_oiastatus_file()
                    logger.info("/software/config/OIALargeChip exists. LowMemRetry=%d sleeping..." % LowMemRetry)
                    time.sleep(15)
            elif (self.pool.beadfind_counter + self.pool.analysis_counter) == 0:
                # logger.info("resetting LowMemRetry counter")
                LowMemRetry = 0 # reset the counter after all the analysis are complete

            if log_memory_usage:
                try:
                    logger.info("GC count: %s" % str(gc.get_count()))
                    h = hpy()
                    logger.debug(str(h.heap()))
                except:
                    logger.error(traceback.format_exc())

            [CurExp, CurFlow] = self.getCurrentRunInformation()


            all_run_dirs = self.get_run_dirs()
            logger.debug('RUNS DIRECTORIES: %s' % all_run_dirs)

            # update flow status
            logger.debug('runs in p: %s' % self.runs_in_process)
            for run in self.runs_in_process:
                # check for last flow
                logger.debug('runs in p: %s' % run.dat_path)
                run.last_flow = run.exp_flows-1
                if run.name == CurExp:
                    run.last_flow = CurFlow
                    logger.debug('flows = %d' % run.last_flow)

            # check for deleted runs
            # TODO

            # check for aborted runs
            for run in self.runs_in_process:
                if run.aborted():
                    logger.info('run aborted %s' % run.name)
                    run.killAnalysis()
                    for block in run.blocks:
                        if block in self.blocks_to_process:
                            self.blocks_to_process.remove(block)
                    self.runs_in_process.remove(run)
                    self.runs_processed.append(run)

            # check for finished runs
            # TODO, add timeout per run?
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
                    timing_file = os.path.join(run.sigproc_results_path, 'timing.txt')
                    try:
                        with open(timing_file, 'a') as f:
                            f.write(run.gettiming())
                    except:
                        logger.error(traceback.format_exc())

                    # transfer timing.txt
                    try:
                        directory_to_transfer = "onboard_results/sigproc_results"
                        file_to_transfer = "timing.txt"
                        ret = Transfer(run.name, directory_to_transfer, file_to_transfer)
                        if ret == 0:
                            logger.debug("Transfer registered %s %s %s" % (
                                run.name, directory_to_transfer, file_to_transfer))
                        else:
                            logger.error("Transfer failed %s %s %s, is datacollect running?" %
                                         (run.name, directory_to_transfer, file_to_transfer))
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
                    arun = Run(run_dir, config)
                    logger.info(arun)
                except:
                    logger.error(traceback.format_exc())
                    continue

                logger.info('exp_oia_during_run = %s' % arun.exp_oia_during_run)
                if CurExp and arun.exp_oia_during_run == "no":
                    continue

                if CurExp == arun.name and CurFlow < 1:
                    continue

                if arun.exp_flows < self.flowblocks:
                    logger.info('skip run: %s, not enough flows' % arun.name)
                    self.runs_processed.append(arun)
                    continue

                if arun.aborted():
                    logger.info('skip aborted run: %s' % arun.name)
                    self.runs_processed.append(arun)
                    continue

                #don't let the utilization logs get too big.
                os.system("mv /var/log/cpu_util.log /var/log/cpu_util.bak")
                os.system("mv /var/log/gpu_util.log /var/log/gpu_util.bak")
                os.system("mv /var/log/disk_util.log /var/log/disk_util.bak")
                self.runs_in_process.append(arun)

                logger.info('arun.exp_oninstranalysis: %s' % arun.exp_oninstranalysis)
                if arun.exp_oninstranalysis == "no":
                    continue

                # ignore autoanalyze option in explog.txt
                logger.info('autoanalyze: %s' % arun.explogdict['AutoAnalyze'])
                # if not arun.explogdict['autoanalyze']: # contains True,False instead of yes, no
                #    continue

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
                self.updateUsage()
                time.sleep(5)

                if os.path.isfile("/software/config/OIALargeChip"):
                    break

                logger.info('Status:        Blocks: {0:3d}  Beadfind: {1:2d}/{2:2d}  Analysis: {3:2d}/{4:2d}  BaseCaller: {5:2d}/{6:2d} Total: {7:2d}/{8:2d}'.format(
                    len(self.blocks_to_process), self.pool.beadfind_counter, self.nb_max_beadfind_jobs, self.pool.analysis_counter, self.nb_max_analysis_jobs, 
                    self.pool.basecaller_counter,self.nb_max_basecaller_jobs,
                    self.pool.beadfind_counter+self.pool.analysis_counter+self.pool.basecaller_counter, self.nb_max_jobs))

                predicted_total_HOST_memory = 0
                predicted_total_GPU_memory = 0

                # get list of all different Runs
                for run in self.runs_in_process:
                    blocks_per_run = [i for i in self.blocks_to_process if i.run == run]
                    bf = len(
                        [i for i in self.blocks_to_process if i.run == run and i.status == 'performJustBeadFind'])
                    an = len(
                        [i for i in self.blocks_to_process if i.run == run and i.status == 'performAnalysis'])
                    bc = len(
                        [i for i in self.blocks_to_process if i.run == run and i.status == 'performPhaseEstimation'])
                    try:
                        nb_max_jobs = config.getint(run.exp_chipversion, 'nb_max_jobs')
                        nb_max_beadfind_jobs = config.getint(run.exp_chipversion, 'nb_max_beadfind_jobs')
                        nb_max_analysis_jobs = config.getint(run.exp_chipversion, 'nb_max_analysis_jobs')
                        nb_max_basecaller_jobs = config.getint(run.exp_chipversion, 'nb_max_basecaller_jobs')
                    except:
                        nb_max_jobs = config.getint('DefaultChip', 'nb_max_jobs')
                        nb_max_beadfind_jobs = config.getint('DefaultChip', 'nb_max_beadfind_jobs')
                        nb_max_analysis_jobs = config.getint('DefaultChip', 'nb_max_analysis_jobs')
                        nb_max_basecaller_jobs = config.getint('DefaultChip', 'nb_max_basecaller_jobs')
                    if len(blocks_per_run):
                        logger.info('Chip: {0:8} Blocks: {1:3d}  Beadfind: {2:2d}/{3:2d}  Analysis: {4:2d}/{5:2d} BaseCaller: {6:2d}/{7:2d} Total: {8:2d}/{9:2d}  ({10})'.format(
                            run.exp_chipversion, len(blocks_per_run), bf, nb_max_beadfind_jobs, an, nb_max_analysis_jobs, bc, nb_max_basecaller_jobs, an+bf,nb_max_jobs, run.name))
                    try:
                        HOST_memory_requirement_beadfind = config.getint(
                            run.exp_chipversion, 'HOST_memory_requirement_beadfind')
                        HOST_memory_requirement_analysis = config.getint(
                            run.exp_chipversion, 'HOST_memory_requirement_analysis')
                        GPU_memory_requirement_analysis = config.getint(
                            run.exp_chipversion, 'GPU_memory_requirement_analysis')
                    except:
                        HOST_memory_requirement_beadfind = config.getint(
                            'DefaultChip', 'HOST_memory_requirement_beadfind')
                        HOST_memory_requirement_analysis = config.getint(
                            'DefaultChip', 'HOST_memory_requirement_analysis')
                        GPU_memory_requirement_analysis = config.getint(
                            'DefaultChip', 'GPU_memory_requirement_analysis')
                    predicted_total_HOST_memory += int(HOST_memory_requirement_beadfind)*bf
                    predicted_total_HOST_memory += int(HOST_memory_requirement_analysis)*an
                    predicted_total_GPU_memory += int(GPU_memory_requirement_analysis)*an

                # every 60 sec
                if time.time()-timestamp > 60:
                    logger.info('HOST: {0} G   GPU: {1} G'.format(
                        predicted_total_HOST_memory/1073741824, predicted_total_GPU_memory/1073741824))

                # TODO: run.exp_oia_during_run
                # TODO: check for new runs only if no data acquisition
                # [CurExp,CurFlow] = self.getCurrentRunInformation()
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
				block.analysis_done = True
                                if block.basecaller_done == True or block.flows_total < 170:
	                            block.status = 'sigproc_done'
                        else:
                            logger.error('Block %s failed with return code %s' % (block.name, block.ret))
                            block.nb_attempts += 1
                            block.sigproc_results_path_tmp = block.sigproc_results_path + \
                                "." + str(block.nb_attempts)
                            block.beadfind_done = False
                            block.analysis_done = False
                            block.basecaller_done = False
                            block.successful_processed = 0
                            block.flow_start = -1
                            block.flow_end = -1
                            block.status = "idle"

                    # processed blocks
                    if block.status == 'sigproc_done' or block.status == 'sigproc_failed':

                        # 1. rename block / last sigproc attempt
                        try:
                            if not os.path.exists(block.sigproc_results_path):
                                logger.info('rename block LowMemRetry=%d %s %s %s' % (
                                    (config.getint('global', 'nb_retries') + LowMemRetry),
                                    block.name, block.sigproc_results_path_tmp, block.sigproc_results_path))
                                 
                                if block.nb_attempts >= (config.getint('global', 'nb_retries') + LowMemRetry):
                                    shutil.move(block.sigproc_results_path + "." + str(
                                        block.nb_attempts-1), block.sigproc_results_path)
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
                                        full_path_to_file = os.path.join(block.sigproc_results_path, filename)
                                        logger.info('remove step file: %s' % full_path_to_file)
                                        os.remove(full_path_to_file)
                            except:
                                logger.error('removing step file failed %s' % block.name)
                                logger.error(traceback.format_exc())
                                pass

                        '''
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
                        '''

                        block.status = 'ready_to_transfer'

                    if block.status == 'ready_to_transfer':
                        directory_to_transfer = "onboard_results/sigproc_results/"+"block_"+block.name
                        file_to_transfer = ""
                        logger.info("Register transfer: %s %s %s" %
                                    (block.run_name, directory_to_transfer, file_to_transfer))
                        ret = Transfer(block.run_name, directory_to_transfer, file_to_transfer)
                        if ret == 0:
                            block.status = "transferred"
                            try:
                                open(
                                    os.path.join(block.sigproc_results_path, 'transfer_requested.txt'), 'w').close()
                            except:
                                logger.error(traceback.format_exc())
                                pass
                        else:
                            logger.error("Transfer failed %s %s %s, is datacollect running?" %
                                         (block.run_name, directory_to_transfer, file_to_transfer))

                    if block.status == 'transferred':
                        logger.debug("DONE: %s" % (block.name))
                        block.status = 'done'
                        self.blocks_to_process.remove(block)

                try:
                    ablock = self.get_next_available_job(config)
                except:
                    ablock = None
                    logger.error(traceback.format_exc())

                if ablock:
                    if ablock.nb_attempts >= (config.getint('global', 'nb_retries') + LowMemRetry):
                        ablock.status = 'sigproc_failed'
                    else:
                        ablock.status = 'queued'
                        ablock.info = '%s-%s' % (ablock.flow_start, ablock.flow_end)
                        logger.debug('%s submitted (%s-%s)' %
                                     (ablock.name, ablock.flow_start, ablock.flow_end))
                        self.pool.add_task(ablock)

            # wait 10 sec if no blocks are available
            time.sleep(10)


def print_oiaconfig(config, logger):
    DEBUG = True
    # DEBUG = False
    if DEBUG:
        sections = config.sections()
        for section in sections:
            logger.info('section: %s' % section)
            options = config.options(section)
            for option in options:
                logger.info(option + ": " + config.get(section, option))


if __name__ == '__main__':

    os.system("nvidia-smi -pm 1; nvidia-smi -e 0; if [ \"`nvidia-smi | grep 'Tesla K40c'`\" != \"\" ]; then nvidia-smi -ac 3004,875; fi");
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', dest='verbose', action='store_true')
    args = parser.parse_args()

    # setup logger
    logger = logging.getLogger("OIA")
    logger.setLevel(logging.DEBUG)
    rothandler = logging.handlers.RotatingFileHandler(
        "/var/log/oia.log", maxBytes=1024*1024*10, backupCount=5)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    rothandler.setFormatter(formatter)
    cachehandler = logging.handlers.MemoryHandler(1024, logging.ERROR, rothandler)
    logger.addHandler(rothandler)
    logger.addHandler(cachehandler)

    # parse on-instrument configuration file
    config_file = "/software/config/oia.config"
    config = ConfigParser.RawConfigParser()
    config.optionxform = str  # don't convert to lowercase
    config.read(config_file)

    print_oiaconfig(config, logger)

    if args.verbose:
        print "oiad:", args

        for run in get_run_list(config):
            print run

    try:
        log_memory_usage = False
        if config.get('global', 'log_memory_usage') == 'yes':
            from guppy import hpy
            import gc
            log_memory_usage = True
    except:
        pass

    # number of cores for phase estimation
    try:
        cpu_cores = multiprocessing.cpu_count() / 2
    except:
        logger.error(traceback.format_exc())
        cpu_cores = 4
        pass

    # retrieve GPU information
    if pynvml_available:
        try:
            nvmlInit()
            logger.info("Driver Version:", nvmlSystemGetDriverVersion())
            deviceCount = nvmlDeviceGetCount()
            total_GPU_memory = 0
            for i in range(deviceCount):
                handle = nvmlDeviceGetHandleByIndex(i)
                logger.info("Device %s: %s" % (i, nvmlDeviceGetName(handle)))
                memory_info = nvmlDeviceGetMemoryInfo(handle)
                logger.info("Device %s: Total memory: %s" % (i, memory_info.total/1024/1024))
                logger.info("Device %s: Free memory: %s" % (i, memory_info.free/1024/1024))
                logger.info("Device %s: Used memory: %s" % (i, memory_info.used/1024/1024))
                total_GPU_memory += memory_info.total/1024/1024
            nvmlShutdown()
        except NVMLError as error:
            logger.error(error)
        except:
            logger.error(traceback.format_exc())
    else:
        total_GPU_memory = 4000

    try:
        app = App()
        app.run()
    except:
        logger.error(traceback.format_exc())
