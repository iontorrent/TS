#!/usr/bin/env python
# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved

# cmd-line wrapper to RNASeq_2step.sh (essentially original pipeline but per BAM) followed by
# collection of stats and additional analysis of the files produced for (plugin html) report

import os
import sys
import time

curdir = os.path.dirname(os.path.realpath(__file__))

def printlog(msg):
  sys.stderr.write(msg)
  sys.stderr.write('\n')
  sys.stderr.flush()

def printtime(msg):
  printlog( '(%s) %s'%(time.strftime('%X'),msg) )

def fileName(filepath):
  filepath = os.path.basename(filepath)
  return os.path.splitext(filepath)[0]

def runcmd( cmd, log, fatal=True ):
  retval = os.system(cmd)
  if retval != 0:
    sys.stderr.write( "ERROR: Failed running command (status = %d):\n" % retval )
    sys.stderr.write( "$ "+cmd+"\n" )
    if fatal:
      sys.stderr.write( "Check Plugin Log and '%s' for details.\n" % os.path.basename(log) )
      sys.exit(1)


if __name__ == '__main__':
  nargs = len(sys.argv)-1
  if nargs != 11:
    sys.stderr.write( "run_rnaseqanalysis.py: Incorrect number of arguments (%d).\n" % nargs )
    sys.stderr.write( "Usage: run_rnaseqanalysis.py <genome name> <reference.fasta> <reads.bam> <sampleid> <adapter> <frac_reads> <fpkm_thres> <output dir> <output file prefix> <run log> <ref index folder>\n" )
    sys.exit(1)

  genome = sys.argv[1]
  reference = sys.argv[2]
  bamfile = sys.argv[3]
  sample = sys.argv[4]
  adapter = sys.argv[5]
  fracreads = sys.argv[6]
  fpkm_thres = sys.argv[7]
  output_dir = sys.argv[8]
  output_prefix = sys.argv[9]
  run_log = sys.argv[10]
  ref_idxfolder = sys.argv[11]
  
  stem = os.path.join(output_dir,output_prefix)
  runlog = os.path.join(output_dir,run_log)

  # check to see if script will have to generate indexing files for warning message in arch log
  indexfolder = os.path.join(ref_idxfolder,genome)
  misindex = ""
  misindex = "" if os.path.exists( os.path.join(indexfolder,"STAR") ) else "STAR"
  if not os.path.exists( os.path.join(indexfolder,"bowtie2") ):
    if misindex != "": misindex += " and "
    misindex += "bowtie2"
  if misindex != "":
    printlog("Warning: %s index files for reference %s were not found and will be created."%(misindex,genome))
    printlog("- This may take several hours but only needs to be done once (per reference).")
    printtime( "Indexing reference and aligning reads..." )
  else:
    printtime( "Aligning reads..." )
  runcmd( "%s/RNASeq_2step.sh -A '%s' -D '%s' -F '%s' -L '%s' -P '%s' -S '%s' '%s' '%s' >> '%s' 2>&1" % (
    curdir,adapter,output_dir,output_prefix,genome,fracreads,ref_idxfolder,reference,bamfile,runlog), runlog )

  printtime( "Creating summary plots and collecting statistics..." )
  statsfile = stem + ".stats.txt"
  with open(statsfile,'w') as outstats:
    outstats.write("RNASeqAnalysis Summary Report\n\n");
    outstats.write("Sample Name: %s\n"%sample);
    outstats.write("Reference Genome: %s\n"%genome);
    outstats.write("Adapter Sequence: %s\n"%adapter);
    outstats.write("Reads Sampled: %s%%\n"%str(100*float(fracreads)));
    outstats.write("Alignments: %s\n\n"%fileName(bamfile));

  alnStem = stem+'.STARBowtie2'
  runcmd(
    "%s/scripts/createSummaryStats.pl -r '%s' '%s.alignmentSummary.txt' '%s.gene.count' '%s.RNAmetrics.txt' '%s.rnareads.xls' >> '%s' 2>> '%s'" % (
    curdir, os.path.join(output_dir,"xrRNA.basereads"), alnStem, alnStem, alnStem, stem, statsfile, runlog), runlog )

  # simple reformating to remove extra, but may add annotations later
  runcmd( "%s/scripts/createGeneCounts.pl '%s.gene.count' >> '%s' 2>> '%s'" % (
    curdir, alnStem, stem+".genereads.xls", runlog), runlog )

  # create a png version of the RNA metrics normalized target coverage plot
  runcmd( "R --no-save --slave --vanilla --args '%s.RNAmetrics.png' '%s.RNAmetrics.txt' < %s/scripts/plot_distcov.R 2>> '%s'" % (
    alnStem, alnStem, curdir, runlog), runlog, False )

  # create a pie chart of the typesof aligned reads
  runcmd( "R --no-save --slave --vanilla --args '%s' '%s' 'Alignment Distribution' < %s/scripts/plot_pie.R 2>> '%s'" % (
    stem+".rnareads.xls", stem+".rnareads.png", curdir, runlog), runlog, False )

  # create basic representation plot for raw gene counts
  runcmd( "R --no-save --slave --vanilla --args '%s' '%s' < %s/scripts/plot_rna_rep.R 2>> '%s'" % (
    stem+".genereads.xls", stem+".generep.png", curdir, runlog), runlog, False )

  # extraction of represented isoforms from cuflinks output, add basic isoform stats and create boxplot(s)
  runcmd( "%s/scripts/isoformExpressed.pl -F %s '%s.isoforms.fpkm_tracking' >> '%s' 2>> '%s'" % (
    curdir, fpkm_thres, stem, stem+".geneisoexp.xls", runlog), runlog )

  awkcmd="awk 'NR>1 {n+=$2;d+=$3} END {print \"\\nIsoforms Annotated: \"n\"\\nIsoforms Detected:  \"d}'"
  runcmd( "%s '%s' >> '%s' 2>> '%s'" % (
    awkcmd, stem+".geneisoexp.xls", statsfile, runlog), runlog )

  runcmd( "R --no-save --slave --vanilla --args '%s' '%s' 'Gene Isoform Expression' 25 < %s/scripts/plot_gene_isoexp.R 2>> '%s'" % (
    stem+".geneisoexp.xls", stem+".geneisoexp.png", curdir, runlog), runlog, False )
  runcmd( "R --no-save --slave --vanilla --args '%s' '%s' 'Gene Isoform Expression' 0 < %s/scripts/plot_gene_isoexp.R 2>> '%s'" % (
    stem+".geneisoexp.xls", stem+".geneisoexp_all.png", curdir, runlog), runlog, False )

