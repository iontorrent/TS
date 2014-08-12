#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

# Flow of program 
# Load our tests
# Run each test

# Example: ./test_runner.py  -r __OUTPUT_ROOT__=out-check  -r __DATA__=/results3/csugnet/integration-data/ -r __GOLD_ROOT__=out-gold --environment PATH=/rhome/csugnet/code/TS_head/build/Analysis:$PATH --type all --jobs separator_tests.json --output topout-check
# We have notions of a source data directory, a gold results directory, and our test directory.
# __DATA_ROOT__
# __GOLD_OUPTUT_ROOT__
# __BIN_PATH_PREFIX__
# __TEST_OUTPUT_ROOT__

# Json format
#
# 
# { "job_id" : 
#   {
#    "job_name"  : "Name of job",
#    "job_command" : "Command line arguments of job", # "echo 'hello' > __JOB_OUTPUT_ROOT__/hello.txt
#    "job_check" : "Command line of program to check", # "grep 'hello' __JOB_OUTPUT_ROOT__/hello.txt
#    "job_logs" : [ "__JOB_OUTPUT_ROOT__/log.txt", "__JOB_OUTPUT_ROOT__/compare.log.txt"] ,
#    "description" : "What are we jobing here anywys.",
#    "dependencies" : [ "job_id1","job_id2", ..., "job_id3"],
#    "exp_ret_val" : "expected return value",
#    "tags" : [ "grep","echo","other injobing classes" ],
#    "stdout" : "output from stdout after running",
#    "stderr" : "output from stderr after running",
#    "results" : "Only present after processing."
#    }
# }

import sys
try:
    import argparse
except ImportError:
    sys.stderr.write( "Error: Can't import the python argparse package.\n" )
    sys.stderr.write( '       Perhaps you should do a "sudo apt-get install python-argparse".\n' )
    sys.exit( -1 )
import json
import subprocess
import logging
import re
import time
import os
import errno
import timeit
import datetime
try:
    from pygraph.classes.digraph import digraph
except ImportError:
    sys.stderr.write( "Error: Can't import the python digraph class from pygraph.\n" )
    sys.stderr.write( '       Perhaps you should do a "sudo apt-get install python-pygraph".\n' )
    sys.stderr.write( "       If that doesn't fix the problem, note that digraph isn't\n" )
    sys.stderr.write( "       available in Ubuntu 10.04.\n" )
    sys.exit( -1 )
from pygraph.algorithms.sorting import topological_sorting

class JobRunner :
    def __init__(self, args, report) :
        self.environment = args.environment
        self.report_file = report
        self.output = args.output
        splitter = re.compile("=");
        self.searches = [];
        self.replacements = [];
        for i in range(len(args.replace)) :
            s = splitter.split(args.replace[i])
            self.searches.append(re.compile(s[0]))
            self.replacements.append(s[1])
        self.run = 0
        self.passed = 0
        self.failed = 0
        
    def run_job(self,job_id,job) :
        out_file = os.path.join(self.output, '%s.stdout.txt' % job_id)
        err_file = os.path.join(self.output, '%s.stderr.txt' % job_id)
        job_stderr = open(err_file,mode="w")
        job_stdout = open(out_file ,mode="w") 
        command = job['job_command']
        
        for i in range(len(self.searches)) :
            command = self.searches[i].sub(self.replacements[i],command)

        # construct our arguments list noting that we don't want to break up something like myprogram --arg1 'this arg --has spaces'
        args_list=[]
        string_split = command.split("'");
        for i in range(len(string_split)) :
            if i % 2 == 0 :
                args_list = args_list + (string_split[i].split())
            else :
                args_list.append(string_split[i])
        start_time = time.time()
#        print "Call is: %s" % " ".join(["/usr/bin/env"] + self.environment.split(" ") + args_list)
        result = subprocess.call(["/usr/bin/env"] + self.environment.split(" ") + args_list, stdout=job_stdout,stderr=job_stderr);
        elapsed_time = (time.time() - start_time)
        job['elapsed_time'] = elapsed_time;
        job['job_command_run'] = command;
        self.run = self.run + 1
        job['ret_val'] = result
        if result == int(job['exp_ret_val']) :
            job['passed'] = True
            self.passed = self.passed + 1
        else :
            job['passed'] = False
            self.failed = self.failed + 1

        job_stderr.close()
        job_stdout.close()
        job['stdout'] = out_file
        job['stderr'] = err_file

def check_args(args):
    """See if we got our minimal set of valid arguments."""
    ok = True
    message = ""
    if not args.type :
        ok = False
        message += "Didn't get a type of jobs to run. "
    if not args.jobs :
        ok = False
        message += "Didn't get a jobs file to run. "
    if not args.output  :
        ok = False
        message += "Didn't get directory for output. "
    if args.update_root != None and args.update_root == args.output :
        ok = False
        message += "Can't specify output root same as update root."

    return([ok,message])

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def print_head_tail(file_in, file_out, alt_out=None, prefix="\t", target_lines=10) :
    lines = file_in.readlines();
    numlines = min(10,len(lines))
    for i in range(numlines) :
        file_out.write("\t" + lines[i]);
        if alt_out != None :
            alt_out.write("\t" + lines[i])
    if len(lines) > 10 :
        file_out.write("\t...\n");
        last_bit = lines[-1 * numlines:]
        for l in last_bit :
            file_out.write("\t" + l);
            if alt_out != None :
                alt_out.write("\t" + l);

def main():
    """Everybody's favorite starting function"""
    parser = argparse.ArgumentParser(description="Run a series of json specified jobs to exercise pipeline and test results.")
    parser.add_argument("-r", "--replace", help="Replace macro strings in commands with value in CURRENT=REPLACE format.", action='append', default=[])
    parser.add_argument("--environment", help="Generic additions to the environment variables (path, etc.)", default="DUMMY=DUMMY")
    parser.add_argument("--type", help="Types of jobs to run.", default="all")
    parser.add_argument("--not_type", help="Types of jobs not to run.", default="all")
    parser.add_argument("--jobs", help="json file with jobs to run")
    parser.add_argument("--output", help="Directory for results and intermediate files.")
    parser.add_argument("--jobnames", help="Name of particular jobs to run (e.g. 'job1|job2|job3').", default=None);
    parser.add_argument("--update_file", help="Name of particular job to run.", default=None);
    parser.add_argument("--update_decision", help="Name of particular job to run.", default=2);
    parser.add_argument("--update_root", help="Path to gold data to update.", default=None);
    parser.add_argument("--ts_base", help="Base of torrent suite checkout for auto setting of environment", default=None);
    parser.add_argument("--gold", help="Path to base of gold data root", default=None);
    parser.add_argument("--data", help="Path to base of data to be used", default=None);
    parser.add_argument("--failures", help="Failures log", default=None);
    args = parser.parse_args();
    args_ok = check_args(args);
    # Put the output root in our regular expressions to be replaced
    for i in range(len(args.replace)) :
        if re.match(args.replace[i],"__OUTPUT_ROOT__") != None :
            print "__OUTPUT_ROOT_ is specified from --output not manually."
            raise
    args.replace.append("__OUTPUT_ROOT__=%s" % args.output);

    # Put the gold root in our regular expressions to be replaced
    if args.gold != None :
        args.replace.append("__GOLD_ROOT__=%s" % args.gold)
        args.update_root = args.gold

    # Put the data root if specified in the regular expressions to be replaced
    if args.data != None :
        args.replace.append("__DATA__=%s" % args.data)

    # If specified, set up the paths based on specified torrent server code root
    if args.ts_base != None :
        args.environment = args.environment + " PATH=%s/build/Analysis:%s/build/Analysis/TMAP:%s/pipeline/bin:%s ION_CONFIG=%s/Analysis/config" % (args.ts_base, args.ts_base, args.ts_base, os.getenv("PATH"), args.ts_base)
    
    if args_ok[0] == False :
        sys.exit(args_ok[1])
    log_file = args.output + "/log.txt"
    json_results = args.output + "/results.json"
    mkdir_p(args.output);

    # Main logger at the debug level
    logging.basicConfig(filename=log_file, filemode="w", level=logging.DEBUG, format='%(asctime)s\t%(levelname)s\t%(message)s')
    
    # Add a logger for the console at the info level
    console_logger = logging.StreamHandler(sys.stdout);
    console_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console_logger.setFormatter(formatter);
    logging.getLogger('').addHandler(console_logger)

    if args.failures != None :
        fail_log = open(args.failures,'w');

    # Read our json
    logging.info("Reading json file: " + args.jobs);
    jobs_file = open(args.jobs);
    jobs = json.loads(jobs_file.read());
    jobs_file.close();
    logging.info('Got %d jobs from file' % len(jobs));

    report_file = os.path.join(args.output, "report.txt");
    runner = JobRunner(args, report_file);
    job_ids = jobs.keys()
    job_values = jobs.values()

    # Create our graph for dependency checking
    gr = digraph()
    # Add all of the nodes
    for i in range(len(job_ids)) :
        logging.debug( "Adding: %s" % job_values[i]['job_name'])
        if gr.has_node(job_values[i]['job_name']) :
            logging.info("Already here?? %s" % job_values[i]['job_name'])
        gr.add_node(job_values[i]['job_name']);

    # Add all the edges
    for i in range(len(job_ids)) :
        logging.debug( "Adding edges for: %s" % job_values[i]['job_name'])
        if "dependencies" in job_values[i] :
            for j in range(len(job_values[i]['dependencies'])) :
                logging.debug("Adding edge: %s %s" % (job_values[i]['dependencies'][j],job_values[i]['job_name']))
                gr.add_edge((job_values[i]['dependencies'][j],job_values[i]['job_name']))

    order = topological_sorting(gr);
    logging.debug("Order is: %s" % order)
    selector = re.compile(args.type);
    neg_selector = re.compile(args.not_type);
    jobnames = None
    if args.jobnames != None :
        jobnames = re.compile(args.jobnames);
    dorun = True
    start_time = time.time()
    for i in range(len(order)) :
        # See if we're running this class of jobs
        dorun = args.type == "all" 
        job = jobs[order[i]]
        if dorun == False :
            for tag_ix in range(len(job['tags'])) :
                if selector.match(job['tags'][tag_ix]) :
                    dorun = True
                    break
            for tag_ix in range(len(job['tags'])) :
                if neg_selector.match(job['tags'][tag_ix]) :
                    dorun = False
                break

        # Check to see if we're running a particular job which trumps tags
        if jobnames != None and jobnames.match(job['job_name']) :
            dorun = True
        elif jobnames != None :
            dorun = False

        # Run our job if necessary
        if dorun :
            logging.info("Running job %d %s" % (i, jobs[order[i]]['job_name']))
            runner.run_job(job['job_name'], job)
            if job["ret_val"] == 0 :
                logging.info("Passed in %.2f seconds" % job["elapsed_time"])
            else :
                logging.info("Failed in %.2f seconds" % job["elapsed_time"])
                if args.failures != None :
                    fail_log.write("Job %s failed\n" % job['job_name'])
                    fail_log.write("stdout file: %s\n" % job['stdout'])
                    sys.stdout.write("stdout file: %s\n" % job['stdout'])
                    out = open(job['stdout'])
                    print_head_tail(out, fail_log, alt_out=sys.stdout)
                    out.close();
                    fail_log.write("stderr file: %s\n" % job['stderr'])
                    sys.stdout.write("stderr file: %s\n" % job['stderr'])
                    out = open(job['stderr'])
                    print_head_tail(out, fail_log, alt_out=sys.stdout)
                    out.close();
        else :
            logging.info("Skipping job %d %s" % (i, job['job_name']))
    elapsed_time = (time.time() - start_time)
    logging.info("All tests took: %.2f seconds" % elapsed_time);
    if args.failures != None :
        fail_log.write('Ran %d jobs, %d passed and %d failed\n' % (runner.run, runner.passed, runner.failed))
        fail_log.close();
    json_out = open(json_results, mode="w");
    json_out.write(json.dumps(jobs,sort_keys=True, indent=4, separators=(',', ': ')))
    logging.info('Ran %d jobs, %d passed and %d failed' % (runner.run, runner.passed, runner.failed))
    logging.info('Full run took: %.3f seconds' % (elapsed_time))

    # Look to see if we need to update the gold data
    if args.update_file != None and args.update_root != None :
        ufile = open(os.path.join(args.output,args.update_file))
        ustats = json.loads(ufile.read())
        if float(ustats["quantile_min"]) > float(args.update_decision) and runner.failed == 0:
            gold_backup = "%s_%s" % (args.update_root, datetime.datetime.now().isoformat())
            print "Moving %s to %s and %s to new gold" % (args.update_root, gold_backup, args.output)
            os.rename(args.update_root, gold_backup)
            os.symlink(args.output, args.update_root)
        else :
            print "Skipping update as metric is: %.2f and threshold is %.2f or numfailed > 0 (%d)" % (ustats["quantile_min"],float(args.update_decision), runner.failed)
    else :
        print "Skipping update as file not specified"

    logging.info("Done.");
    if runner.failed > 0 :
        print "Failed."
        sys.exit(1)
    else :
        print "Success."
main()




