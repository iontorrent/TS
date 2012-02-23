#!/usr/bin/python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
import os.path
import getopt
import subprocess
from collections import defaultdict

def usage():
    global progName
    sys.stderr.write("Usage: %s [options] <outputdir> <reference> <bamfile>\n" % progName)
    sys.stderr.write("Options:\n")
    sys.stderr.write("  -b, --bedfile    =>  BED file specifing regions over which variant calls will be limited or filtered to\n")
    sys.stderr.write("  -f, --fsbam      =>  Flowspace BAM if required in addition to non-flowspace BAM file\n")
    sys.stderr.write("  -o, --floworder  =>  PGM base flow Order used (per cycle). Defaults to SAMBA\n")
    sys.stderr.write("  -p, --paramfile  =>  Parameters file for all variant programs used (simple json)\n")
    sys.stderr.write("  -r, --rundir     =>  Directory path to location of variant caller programs. Defaults to the directory this script is located\n")
    sys.stderr.write("  -l, --log        =>  Send additional Log messages to STDERR\n")
    sys.stderr.write("  -L, --logfile    =>  Send additional Log messages to specified log file\n")
    sys.stderr.write("  -h, --help       =>  Ignore command and output this Help message to STDERR\n")
    
paradict = defaultdict(lambda: defaultdict(defaultdict))

def ReadParamsFile( paramfile ):
    # defaults for expected dictionary elements
    paradict['dibayes'] = ""
    paradict['ion-variant-hunter'] = ""
    paradict['filter-indels'] = ""
    if paramfile == "":
        return
    # very simple json format parser: 2 levels only, no arrays
    newdict = 1
    dictkey = ""
    inf = open(paramfile,'r')
    for line in inf:
        line = line.strip()
        if line == '' or line[0] == '#':
            continue
        fields = line.split(':')
        name = fields[0].strip().replace('"','')
        if name == '}':
            if newdict == 1:
                sys.stderr.write("ERROR: paramfile format; unexpected key '}'\n")
                sys.exit(1)
            newdict = 1
            continue
        if len(fields) != 2:
            sys.stderr.write("ERROR: paramfile format; unexpected key '%s'\n"%fields[1])
            sys.exit(1)
        val = fields[1].strip().rstrip(',').replace('"','')
        if val == '{':
            if newdict == 0:
                sys.stderr.write("ERROR: paramfile format; unexpected value '{'\n")
                sys.exit(1)
            newdict = 0
            dictkey = name
            paradict[dictkey] = ""
            continue
        if dictkey == "":
            sys.stderr.write("ERROR: paramfile format; no progrma node before first key '%s'\n"%fields[1])
            sys.exit(1)
        if newdict == 0:
            paradict[dictkey] += "--%s=%s " % (name,val)
    inf.close()

def SNPsCallerCommandLine( binDir, paramstr, bamfile, reference, bedfile, flowseq, outDir ):
    # flowseq not currently employed
    fixedprms = "--AllSeq=1 -b 1 --platform=2 -d 1 -C 0 -W 0 -n diBayes_run -g %s/log/ -w %s/temp/ -o %s" % (outDir,outDir,outDir)
    if paramstr == "":
        paramstr  = "--call-stringency medium --het-skip-high-coverage 0 --reads-min-mapping-qv 2 --het-min-lca-start-pos 0 --het-min-lca-base-qv 14 " + \
            "--het-lca-both-strands 0 --het-min-allele-ratio 0.15 --het-max-coverage-bayesian 60 --het-min-nonref-base-qv 14 --snps-min-base-qv 14 " + \
            "--snps-min-nonref-base-qv 14 --reads-with-indel-exclude 0 --het-min-coverage 2 --het-min-start-pos 1 --hom-min-coverage 1 " + \
            "--hom-min-nonref-allele-count 3 --snps-min-filteredreads-rawreads-ratio 0.15 --het-min-validreads-totalreads-ratio 0.65 " + \
            "--reads-min-alignlength-readlength-ratio 0.2 --hom-min-nonref-base-qv 14 --hom-min-nonref-start-pos 0"
    if bedfile != "":
        bedfile = '-R ' + bedfile
    commandLineDiBayes = "%s/diBayes %s %s %s -f %s %s" % (
        binDir, fixedprms, paramstr, bedfile, reference, bamfile )
    return "export LOG4CXX_CONFIGURATION;export LD_LIBRARY_PATH=%s/diBayes_lib;%s" % (binDir,commandLineDiBayes)

def IndelCallerCommandLine( binDir, paramstr, bamfile, reference, bedfile, flowseq, outDir ):
    # bedfile and flowseq not currently employed
    fixedprms = "--dynamic-space-size 9216 --function write-alignment-n-deviation-lists-from-sam --fs-align-jar-args 'VALIDATION_STRINGENCY=SILENT' --java-bin-args _Xmx4G"
    if paramstr == "":
        paramstr = "--min-variant-freq ,0.15 --min-num-reads ,3 --strand-prob ,0 --min-mapq ,2"
    else:
        paramstr = paramstr.replace('=',' ,')
    return "%s/ion-variant-hunter-core %s %s --base-output-filename %s/%s --bam-file %s --reference-file %s --java-bin `which java`" % (
        binDir, fixedprms, paramstr, outDir, "variantCalls", bamfile, reference )

def IndelCallerBayesianRescoreCommandLine( binDir, reference, flowseq, outDir ):
    return "%s/bayesian-vh-rescorer %s %s/%s %s %s/%s %s/%s -log %s/%s" % (binDir, reference, outDir, "variantCalls.merged.dev", flowseq, outDir, "bayesian_scorer.vcf", outDir, "variantCalls.vcf", outDir, "bayesian_scorer.log")

def ScoreFilterIndelsCommandLine( binDir, paramstr, outDir ):
    return "%s/filter_indels.py %s %s/bayesian_scorer.vcf %s/variantCalls.filtered.vcf" % ( binDir, paramstr, outDir, outDir )

def BedFilterIndelsCommandLine( binDir, bedfile, outDir ):
    if bedfile != "":
        bedfile = '--bed ' + bedfile
    return "%s/vcftools --vcf %s/variantCalls.filtered.vcf %s --out %s/indels --recode --keep-INFO-all > /dev/null" % ( binDir, outDir, bedfile, outDir )
    
def RunCommand(command):
    command = command.strip()
    WriteLog( " $ %s\n" % command )
    stat = os.system( command )
    if( stat ) != 0:
        sys.stderr.write( "ERROR: variantCaller resource failed with status %d:\n" % stat )
        sys.stderr.write( "$ %s\n" % command )
        sys.exit(1)

def WriteLog(msg,force=0):
    global logout, logfile, logerr
    if logfile != "":
        if logout == 0:
            logout = open(logfile,'w')
        logout.write(msg)
    if logerr == 1 or force != 0:
        sys.stderr.write(msg)

def main(argv):
    # arg processing
    global progName, logout, logfile, logerr
    rundir = os.path.realpath(__file__)
    logerr = 0
    logout = 0
    logfile=""
    paramfile=""
    bedfile=""
    fsbam = ""
    floworder = "TACGTACGTCTGAGCATCGATCGATGTACAGC"
    try:
        opts, args = getopt.getopt( argv, "hlp:r:b:f:o:L:", ["help", "log", "paramfile=", "rundir=", "bedfile=", "fsbam=", "floworder=", "logfile="] )
    except getopt.GetoptError, msg:
        sys.stderr.write(msg)
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-p", "--paramfile"):
            paramfile = arg
            if not os.path.exists(paramfile):
                sys.stderr.write("No variant calling parameters file found at: %s\n" % paramfile)
                sys.exit(1)
        elif opt in ("-b", "--bedfile"):
            bedfile = arg
            if not os.path.exists(bedfile):
                sys.stderr.write("No bed file found at: %s\n" % bedfile)
                sys.exit(1)
        elif opt in ("-r", "--rundir"):
            rundir = arg
            if not os.path.isdir(rundir):
                sys.stderr.write("No run directory found at: %s\n" % rundir)
                sys.exit(1)
        elif opt in ("-f", "--fsbam"):
            fsbam = arg
        elif opt in ("-o", "--floworder"):
            floworder = arg
        elif opt in ("-l", "--log"):
            logerr = 1
        elif opt in ("-L", "--logfile"):
            logfile = arg
    if(len(args) != 3):
        sys.stderr.write("Error: Invalid number of arguments\n")
        usage()
        sys.exit(1)

    outdir = args[0]
    reference = args[1]
    bamfile = args[2]
    if( fsbam == "" ):
        fsbam = bamfile

    WriteLog("Running %s...\n" % progName)

    if not os.path.isdir(outdir):
        sys.stderr.write("No output directory found at: %s\n" % outdir)
        sys.exit(1)
    if not os.path.exists(reference):
        sys.stderr.write("No reference file found at: %s\n" % reference)
        sys.exit(1)
    if not os.path.exists(bamfile):
        sys.stderr.write("No bam file found at: %s\n" % bamfile)
        sys.exit(1)
    if not os.path.exists(fsbam):
        sys.stderr.write("No flowspace bam file found at: %s\n" % fsbam)
        sys.exit(1)

    # read calling parameters (germline/somatic)
    ReadParamsFile(paramfile)

    # create output directories for diBayes, if not already present
    snp_dir = outdir + "/dibayes_out"
    indel_dir = outdir
    if not os.path.exists(snp_dir):
        os.makedirs(snp_dir)
    if not os.path.exists(indel_dir):
        os.makedirs(indel_dir)
        
    # create command for SNP caller and run
    WriteLog(" Finding SNPs using diBayes...\n",1)
    cmdoptions = paradict['dibayes']
    RunCommand( SNPsCallerCommandLine( rundir, cmdoptions, bamfile, reference, bedfile, floworder, snp_dir ) )
    WriteLog(" > %s/diBayes_lib\n" % snp_dir)

    # create command for indel caller and run
    WriteLog(" Finding INDELs using variant-hunter...\n",1)
    cmdoptions = paradict['ion-variant-hunter']
    RunCommand( IndelCallerCommandLine( rundir, cmdoptions, fsbam, reference, bedfile, floworder, indel_dir ) )
    WriteLog(" > %s/variantCalls.merged.dev\n" % indel_dir)

    # apply bayesian score to vcf file produced by executing IndelCallerCommandLine()
    WriteLog(" Filtering variant-hunter calls using Bayesian scores...\n",1)
    RunCommand( IndelCallerBayesianRescoreCommandLine( rundir, reference, floworder, indel_dir ) )
    WriteLog(" > %s/bayesian_scorer.vcf\n" % indel_dir)

    # create commands for indel filtering and run
    cmdoptions = paradict['filter-indels']
    RunCommand( ScoreFilterIndelsCommandLine( rundir, cmdoptions, indel_dir ) )
    WriteLog(" > %s/variantCalls.filtered.vcf\n" % indel_dir)

    RunCommand( BedFilterIndelsCommandLine( rundir, bedfile, indel_dir ) )
    WriteLog(" > %s/indels.recode.vcf\n" % indel_dir)

    if logfile != "":
        logout.close()

if __name__ == '__main__':
    global progName
    progName = sys.argv[0]
    main(sys.argv[1:])

