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
from math import floor

def main() :
  """Everybody's favorite function"""
  parser = argparse.ArgumentParser(description="Summarize comparisons from a number of different tests.")
  parser.add_argument("-c","--comparison",help="json file from comparison analysis.", action="append");
  parser.add_argument("-o","--output", help="file to output with summary statistics");
  parser.add_argument("-m","--metric", help="name of metric to consider",action="append");
  parser.add_argument("-q","--quantile", help="quantile above 1 for passing",default=.9);
  args = parser.parse_args();
  
  allok = 1;
  results = dict()
  for m in range(len(args.metric)) :
      results[args.metric[m]] = list()

  for i in range(len(args.comparison)) :
      try:
          cfile = open(args.comparison[i])
          cstats = json.loads(cfile.read());
          cfile.close();
          for m in range(len(args.metric)) :
              print "Adding %.2f for %s from %s" % (cstats[args.metric[m]], args.metric[m], args.comparison[i])
              results[args.metric[m]].append(cstats[args.metric[m]])
      except:
          print "Error reading %s", args.comparison[i]
          allok = 0;
  
  q_min = 10000;
  for m in range(len(args.metric)) :
      q = sorted(results[args.metric[m]])
      qIx = int(floor(1.0 * float(args.quantile) * len(q)))
      print "QIx is: %d for %.2f and length %d" % (qIx, args.quantile, len(q))
      q_min = min(q[qIx], q_min)

  # If parsing had issues then set our qmin to zero
  if allok == 0 :
      q_min = 0

  print "Minimum quantile %.2f is %.2f\n" % (args.quantile, q_min)
  print "Writing to file: %s" % args.output
  sum_out = open(args.output, 'wb')        
  towrite = dict()
  towrite["quantile_min"] = q_min;
  json.dump(towrite, sum_out)
  sum_out.close()

main()
