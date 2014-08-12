#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import argparse
import json
import subprocess
import logging
import time
import os
import errno
import sys
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def run_step(command, output, logprefix) :
    out_file = '%s/%s.stdout' % (output,logprefix)
    err_file = '%s/%s.stderr' % (output,logprefix)
    job_stderr = open(err_file,mode="w")
    job_stdout = open(out_file ,mode="w") 
    print "command is: %s" % command
    result = subprocess.call(command.split(), cwd=output, stdout=job_stdout,stderr=job_stderr)
    if result != 0 :
        raise error("Step failed with command: %s" % command)
    job_stderr.close()
    job_stdout.close()

def main():
  """Everybody's favorite function"""
  parser = argparse.ArgumentParser(description="Compare sequencing results from two runs.")
  parser.add_argument("-g","--gold", help="ionstats_alignment.json file from gold version", default="");
  parser.add_argument("-q","--query", help="ionstats_alignment.json file from gold version", default="");
  parser.add_argument("-m","--metric", help="metric to compare with minimum change (e.g. AQ20:1.00 for at least same AQ20", action='append');
  parser.add_argument("-c","--comparison", help="compare metric to be exact or just better than value supplied", default='exact', choices=['exact','better']);

  args = parser.parse_args();
  if args.gold == "" or args.query == "" :
      raise error("Must specify -g and -q")

  gold_stats_file = open(args.gold)
  gold_stats = json.loads(gold_stats_file.read());
  gold_stats_file.close();
  query_stats_file = open(args.query)
  query_stats = json.loads(query_stats_file.read());
  query_stats_file.close();
  
  allok = 1;
  comparison = {}
  for  i in range(len(args.metric)) : 
      metrics = args.metric[i].split(':')
      mult = float(metrics[1])
      gval = gold_stats[metrics[0]]["num_bases"]
      qval = query_stats[metrics[0]]["num_bases"]
      if gval != 0 :
          comparison[metrics[0]] = qval / gval;
      else :
          print "Warning - got 0 value from %s for metric %s" % (metrics[0],args.gold)
          continue
      if args.comparison == 'exact' and qval != gval * mult :
          allok = 0;
          print "Failed exact for gold %s %s being %d with query %d" % (metrics[0], metrics[1], gval, qval)
      elif args.comparison == 'better' and qval < gval * mult :
          print "Failed better for gold %s %s being %d with query %d" % (metrics[0], metrics[1], gval, qval)
          allok = 0;
      elif args.comparison == 'better' and qval >= gval * mult :
          print "Success better for gold %s %s being %d with query %d" % (metrics[0], metrics[1], gval, qval)
  comparison["pass"] = allok;
  end = args.query.rfind('/');
  out_root = args.query[0:end];
  
  comp_out = open('%s/comparison.json' % out_root, 'wb')
  json.dump(comparison, comp_out)
  comp_out.close()
  if allok != 1 :
      print "Not equivalent"
      sys.exit(1)
main()  
  


  
