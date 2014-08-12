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
        print ("Step failed with command: %s" % command)
        sys.exit(1)
    job_stderr.close()
    job_stdout.close()

def main():
  """Everybody's favorite function"""
  parser = argparse.ArgumentParser(description="Basic run of analysis pipeline.")
  parser.add_argument("-r","--raw", help="Directory of raw data");
  parser.add_argument("-o","--output", help="Output root directory");
  parser.add_argument("-a","--analysisx", help="Extra commands for analysis", default="");
  parser.add_argument("-b","--basecallerx", help="Extra commands for basecaller", default="");
  parser.add_argument("-s","--serialize", help="Test serialization of Analysis", action="store_true")
  parser.add_argument("--recalibrate", help="Do recalibration.", default=False, action="store_true");
  parser.add_argument("--calibratex", help="Cryptic calibrate arguments.", default="");
  parser.add_argument("--just_analysis", help="Just run to 1.wells file.", default=False)
  parser.add_argument("--bamfile_prefix", help="Prefix of bamfiles to align.", default="rawlib");
  args = parser.parse_args();
  if args.recalibrate == True and args.calibratex == "" :
      print("Error: Must specify appropriate arguments for --calibratex like '--xMin 0 --xMax 1199 --xCuts 2 --yMin 0 --yMax 799 --yCuts 2 --numFlows 280 --flowCuts 2'");
      sys.exit(1);

  mkdir_p(args.output);

  # Get the genome type from the explog
  f = open("%s/explog_final.txt" % args.raw)
  library = ""
  numflows = 0;
  for l in f.readlines():
      words = l.strip().split(":")
      if words[0] == "Library" :
          library = words[1];
          library = library.strip()
      elif words[0] == "Flows" :
          numflows = words[1].strip()
          numflows = int(numflows)
  f.close()

  if args.serialize :
      command = "justBeadFind %s --no-subdir --output-dir analysis-out %s" % (args.analysisx, args.raw)
      run_step(command, args.output, "justBeadFind");
      command = "Analysis --local-wells-file off --from-beadfind --restart-next step.19 --flowlimit %d --start-flow-plus-interval 0,20  %s --no-subdir --output-dir analysis-out %s" % (numflows,args.analysisx,args.raw)
      run_step(command, args.output, "analysis_20");
      command = "Analysis --local-wells-file off --from-beadfind --restart-from step.19 --flowlimit %d --start-flow-plus-interval 20,%d %s --no-subdir --output-dir analysis-out %s" % ( numflows, numflows-20, args.analysisx, args.raw)
      run_step(command, args.output, "analysis_rest");
  else :
      command = "Analysis %s --no-subdir --output-dir analysis-out %s" % (args.analysisx, args.raw)
      run_step(command, args.output, "analysis");
  alignment_file = "basecaller-out/%s.basecaller.bam" % args.bamfile_prefix
  datasets_arg = ""
  if os.path.isfile("%s/datasets_pipeline.json" % args.raw) :
      datasets_arg = "--datasets=%s/datasets_pipeline.json" % args.raw

  if args.recalibrate == True :
      print "Doing recalibration\n"
      mkdir_p("%s/basecaller-out/recalibration" % args.output)
      command = "BaseCaller --disable-all-filters on --phasing-residual-filter=2.0 --num-unfiltered 100000 --calibration-training=100000 --flow-signals-type scaled-residual --input-dir=analysis-out --librarykey=TCAG --tfkey=ATCG --run-id=029DH --output-dir=basecaller-out/recalibration --block-col-offset 0 --block-row-offset 0 %s --trim-adapter ATCACCGACTGCCCATAGAGAGGCTGAGAC --barcode-filter 0.01 --barcode-filter-minreads 0" % (datasets_arg)
      run_step(command, args.output, "basecaller_recal");

      command = "alignmentQC.py --logfile basecaller-out/recalibration/rawlib/alignmentQC_out.txt --output-dir basecaller-out/recalibration/rawlib --input basecaller-out/recalibration/rawlib.basecaller.bam --genome %s --out-base-name samplelib --skip-sorting" % library
      run_step(command, args.output, "recal_alignmentqc");

      command = "calibrate --skipDroop %s -i basecaller-out/recalibration/rawlib.basecaller.bam -o basecaller-out/recalibration/rawlib" % args.calibratex
      run_step(command, args.output, "calibrate")

      command = "calibrate --performMerge -o basecaller-out/recalibration --mergeParentDir basecaller-out/recalibration"
      run_step(command, args.output, "calibrate_merge");

      command = "BaseCaller --phasing-residual-filter=2.0 --num-unfiltered 100000 --calibration-file basecaller-out/recalibration/flowQVtable.txt --phase-estimation-file basecaller-out/recalibration/BaseCaller.json --model-file basecaller-out/recalibration/hpModel.txt --input-dir=analysis-out --librarykey=TCAG --tfkey=ATCG --run-id=029DH --output-dir=basecaller-recalibrated --block-col-offset 0 --block-row-offset 0 %s --trim-adapter ATCACCGACTGCCCATAGAGAGGCTGAGAC --barcode-filter 0.01 --barcode-filter-minreads 0 " % (datasets_arg)
      run_step(command, args.output, "basecaller_rescaled");  
      alignment_file = "basecaller-recalibrated/%s.basecaller.bam" % args.bamfile_prefix
  else :
      command = "BaseCaller --input-dir analysis-out --output-dir basecaller-out %s %s" % ( args.basecallerx, datasets_arg)
      run_step(command, args.output, "basecaller");  

  if args.just_analysis == False :
      mkdir_p("%s/align-out" % args.output)
      command = "alignmentQC.py -p --logfile alignmentQC_out.txt --output-dir align-out --input %s --genome %s --out-base-name samplelib --skip-sorting" % (alignment_file, library)
      run_step(command, args.output, "align");

      command = "ionstats alignment -i align-out/samplelib.bam -o ionstats_alignment.json"
      run_step(command, args.output, "stats");

      stats_file = open("%s/ionstats_alignment.json" % args.output)
      stats = json.loads(stats_file.read());
      stats_file.close();
      print "AQ20: %d AQ47: %d" % (stats["AQ20"]["num_bases"],stats["AQ47"]["num_bases"])

main()  
