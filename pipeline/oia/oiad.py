#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

# python 3.0 will have something similar: http://www.python.org/dev/peps/pep-3143/

# LIMITATION, can process only multiple 0f 20 flows

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
                    logger.info('mkdir %s' % block.sigproc_results_path_tmp)

                logger.info("%s: run process: %s" % (self.id, block) )

                # don't use shell=True , otherwise child process cannot be killed
                outfile = open(os.path.join(block.sigproc_results_path_tmp,'sigproc.log'), "a")
                args = shlex.split(block.command.encode('utf8'))
                p = subprocess.Popen(args, stdout=outfile, stderr=subprocess.STDOUT)
                # add popen process to block
                block.process = p
                ret = p.wait()

                # error generation
                #rand = random.randint(0,100)
                #if rand > 5: ret = 4
                #else: ret = 0
                block.ret = ret
            except:
                logger.error(traceback.format_exc())
                block.ret = 666


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
    command = "%s" % config.get('global','separatorArgs')
    command += " --beadfind-num-threads %s" % config.get('global','nb_beadfind_threads')
    command += " --no-subdir"
    command += " --output-dir=%s" % block.sigproc_results_path_tmp
    command += " --librarykey=%s" % config.get('global','libraryKey')
    command += " --tfkey=%s" % config.get('global','tfKey')
    command += " %s" % block.dat_path
    return command


def getAnalysisCommand(config,block,reentrant):
    command = "%s" % config.get('global','analysisArgs')
    command += " --numcputhreads %s" % config.get('global','nb_analysis_threads')
    if reentrant:
        command += " --from-beadfind"
        command += " --local-wells-file off"
        if block.flow_start != 0:
            command += " --restart-from step.%s" % (block.flow_start-1)
        if block.flow_end != block.flows_total-1:
            command += " --restart-next step.%s" % (block.flow_end)
        command += " --flowlimit %s" % block.flows_total
        command += " --start-flow-plus-interval %s,%s" % (block.flow_start, block.flow_end-block.flow_start+1)
    command += " --no-subdir"
    command += " --output-dir=%s" % block.sigproc_results_path_tmp
    command += " --librarykey=%s" % config.get('global','libraryKey')
    command += " --tfkey=%s" % config.get('global','tfKey')
    command += " %s" % block.dat_path
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
            self.exp_oninstranalysis = 'yes' in self.explogdict['oninstranalysis'] #yes, no
            self.exp_usesynchdats = 'yes' in self.explogdict['use_synchdats'] #yes, no
        except:
            logger.error(traceback.format_exc())
            raise


        self.blocks = []
        try:
            self.block_to_process_start = int(config.get('global','block_to_process_start'))
            self.block_to_process_end = int(config.get('global','block_to_process_end'))
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


    def killAnalysis(self, skip_nb_flows=-1, skipSSD=False, skipSeparator=False):
        for block in self.blocks:
            try:
                if block.storage == "SSD" and skipSSD:
                    continue
                if 'justBeadFind' in block.command and skipSeparator:
                    continue
                # ignore first flowchunk stored on SSD
                if block.flow_end == skip_nb_flows-1:
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
            storage = config.get('blocks',block_name).split(',')[0]
            full_block_dat_path = os.path.join(self.dat_path, block_dir)
            full_block_sigproc_results_path = os.path.join(self.sigproc_results_path, "block_"+block_dir)

            if os.path.exists(full_block_sigproc_results_path):
                # TODO transferred?
                block_status = "done"
            else:
                # default entry status
                block_status = "idle"

            # determine number of attempts
            nb_attempts = 0
            while os.path.exists(full_block_sigproc_results_path + "." + str(nb_attempts)):
                nb_attempts += 1
             
            newblock = Block(
                         self.name,
                         block_dir,
                         block_status,
                         storage,
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
    def __init__(self,run_name,name,status,storage,nb_attempts,flows_total,dat_path,sigproc_results_path):
        self.name = name
        self.status = status
        self.process = None
        self.reentrant = False
        self.command = ""
        self.ret = 0
        self.run_name = run_name
        self.successful_processed = -1
        self.storage = storage
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
        s += "  storage:" + str(self.storage)
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

        self.nb_max_jobs = int(config.get('global','nb_max_jobs'))
        logger.info('nb_max_jobs %s' % self.nb_max_jobs)

        self.nb_max_analysis_jobs = int(config.get('global','nb_max_analysis_jobs'))
        logger.info('nb_max_analysis_jobs %s' % self.nb_max_analysis_jobs)

        self.nb_max_beadfind_jobs = int(config.get('global','nb_max_beadfind_jobs'))
        logger.info('nb_max_beadfind_jobs %s' % self.nb_max_beadfind_jobs)

        self.flowblocks =  int(config.get('global','flowblocks'))

        # 1) Init a Thread pool with the desired number of threads
        self.pool = ThreadPool(self.nb_max_jobs,logger)

        self.blocks_to_process = []
        self.runs_in_process = []
        self.runs_processed = []

    def printStatus(self):
        logger.debug("\n***** STATUS: blocks to process: %d *****" % len(self.blocks_to_process))


    def get_run_dirs(self):

        run_dirs = []

        try:
            run_dirs = fnmatch.filter(os.listdir(config.get('global','results')), "R_*")
        except:
            pass

        return run_dirs

    def get_next_available_job(self,instr_busy,config):

        logger.debug('get_next_available_job')
        # what kind of jobs are allowed to run?
        beadfind_request = True
        analysis_request = True

        if self.pool.beadfind_counter + self.pool.analysis_counter >= self.nb_max_jobs:
            logger.debug('max jobs limit reached')
            beadfind_request = False
            analysis_request = False

        if self.pool.beadfind_counter >= self.nb_max_beadfind_jobs:
            logger.info('max beadfind jobs limit reached')
            beadfind_request = False

        if self.pool.analysis_counter >= self.nb_max_analysis_jobs:
            logger.debug('max analysis jobs limit reached')
            analysis_request = False



        anablock = None

        # Separator
        for block in self.blocks_to_process:

            if block.status != 'idle':
                continue

            if block.successful_processed == -1:
                if beadfind_request:
                    block.command = getSeparatorCommand(config,block)
                    anablock=block
                    break

        # Analysis
        if not anablock:
            for block in self.blocks_to_process:

                if block.status != 'idle':
                    continue

                if block.successful_processed != -1:
                    if analysis_request:
                        if instr_busy and block.storage == 'HD' and block.successful_processed >= self.flowblocks:
                            continue
                        new_flow_end = -1
                        # TODO test acquisition complete:        
                        # check for last flow
                        if os.path.exists(os.path.join(block.dat_path, 'acq_%04d.dat' % (block.flows_total-1) )):
                            new_flow_end = block.flows_total-1
                        # check for first flowchuck
                        elif os.path.exists(os.path.join(block.dat_path, 'acq_%04d.dat' % (self.flowblocks-1) )) and block.successful_processed == 0:
                            new_flow_end = self.flowblocks-1
                        else:
                            test_flow_end = block.successful_processed-1
                            while True:
                                test_flow_end += self.flowblocks
                                logger.debug('test new flowend ' + block.name + " " + str(test_flow_end))
                                if not os.path.exists(os.path.join(block.dat_path, 'acq_%04d.dat' % (test_flow_end) )):
                                    break
                                new_flow_end = test_flow_end
                            # not enough flows for chunk
                            if new_flow_end <= block.successful_processed-1:
                                continue
                        # don't process HD blocks greater than self.flowblocks
                        if instr_busy and block.storage == 'HD' and new_flow_end > self.flowblocks:
                            continue

                        # always reentrant because separator runs independently
                        block.flow_start = block.successful_processed
                        block.flow_end = new_flow_end
                        block.command = getAnalysisCommand(config, block, reentrant=True)
                        anablock=block
                        break

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
                    run.killAnalysis(skip_nb_flows=self.flowblocks,skipSSD=True,skipSeparator=True)
                    #TODO set correct block status


            # check for deleted runs
            # TODO

            # check for aborted runs
            for run in self.runs_in_process:
                if run.aborted():
                    logger.info('run aborted %s' % run.name)
                    run.killAnalysis(skip_nb_flows=-1,skipSSD=False,skipSeparator=False)
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

                if not arun.exp_usesynchdats:
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

                logger.info('autoanalyze: %s' % arun.explogdict['autoanalyze'])
                if not arun.explogdict['autoanalyze']: # contains True,False instead of yes, no
                    continue


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

                logger.debug('number blocks to process: %s %s %s' % ( len(self.blocks_to_process), self.pool.beadfind_counter, self.pool.analysis_counter ) )

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
                            block.successful_processed = block.flow_end+1
                            if block.successful_processed == block.flows_total:
                                block.status = 'sigproc_done'
                        else:
                            logger.error('Block %s failed with return code %s' % (block.name, block.ret))
                            block.nb_attempts += 1

                    # processed blocks
                    if block.status == 'sigproc_done' or block.status == 'sigproc_failed':
                        logger.info('rename block %s %s %s' % (block.name,block.sigproc_results_path_tmp,block.sigproc_results_path))
                        try:
                            # rename block / last sigproc attempt
                            shutil.move(block.sigproc_results_path_tmp, block.sigproc_results_path)
                        except:
                            logger.error('renaming failed %s' % block.name)
                            logger.error(traceback.format_exc())
                            pass

                        # 1. write return code into file
                        try:
                            f = open(os.path.join(block.sigproc_results_path,'analysis_return_code.txt'), 'w')
                            f.write(str(block.ret))
                            f.close()
                        except:
                            logger.error('%s failed to write return code' % block.name)
                            logger.error(traceback.format_exc())
                            pass

                        # 2. mark block as done
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
                        else:
                            logger.error("Transfer failed %s %s %s, is datacollect running?" % (block.run_name, directory_to_transfer, file_to_transfer))

                    if block.status == 'transferred':
                        logger.debug("DONE: %s" % (block.name) )
                        block.status = 'done'
                        self.blocks_to_process.remove(block)

                instr_busy = instrument_busy()
                try:
                    ablock = self.get_next_available_job(instr_busy,config)
                except:
                    ablock = None
                    logger.error(traceback.format_exc())

                if ablock:
                    if ablock.nb_attempts >= 5:#TODO, make it configurable
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
