#!/usr/bin/python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
import os.path
import getopt
import subprocess
import json
import datetime
from collections import defaultdict

def usage():
    global progName
    sys.stderr.write("Usage: %s [options] <outputdir> <reference> <bamfile>\n" % progName)
    sys.stderr.write("Options:\n")
    sys.stderr.write("  -b, --bedfile    =>  BED file specifing regions over which variant calls will be limited or filtered to\n")
    sys.stderr.write("  -s, --hotspotsfile    =>  BED file specifying exact hotspot positions\n")
    sys.stderr.write("  -f, --fsbam      =>  Flowspace BAM if required in addition to non-flowspace BAM file\n")
    sys.stderr.write("  -g, --genome     =>  Genome file. Defaults to <reference>.fai\n")
    sys.stderr.write("  -p, --paramfile  =>  Parameters file for all variant programs used (simple json)\n")
    sys.stderr.write("  -r, --rundir     =>  Directory path to location of variant caller programs. Defaults to the directory this script is located\n")
    sys.stderr.write("  -k, --keepdir    =>  Keep variant mapping to individual contigs (dibayes). (No genome file required.)\n")
    sys.stderr.write("  -n, --nosnp      =>  Call indels only.\n")
    sys.stderr.write("  -l, --log        =>  Send additional Log messages to STDERR\n")
    sys.stderr.write("  -L, --logfile    =>  Send additional Log messages to specified log file\n")
    sys.stderr.write("  -W, --warnto     =>  Exit normal status after an error occurs with valid partial results and send additional warning message to specified log file\n")
    sys.stderr.write("  -h, --help       =>  Ignore command and output this Help message to STDERR\n")
    
paradict = defaultdict(lambda: defaultdict(defaultdict))

def ReadParamsFromUI( paramfile):
    paradict['dibayes'] = ""
    paradict['torrent-variant-caller'] = ""
    paradict['long-indel-assembler'] = ""    
    paradict['filter-indels'] = ""
    if paramfile == "":
        return
    JSON_INPUT = json.load( open(paramfile, "r") )
    params = JSON_INPUT['pluginconfig']
    for line in params:
	value=params[line]
	line = line.strip()
	line = line.replace('___', '-')
	tokens = line.split('__')
	if(tokens[0].strip().replace('"', '') == "param"):
		group = tokens[1].strip().replace('"', '')
		name = tokens[2].strip().replace('"', '')
		if name == "C":
			paradict[group] += "-C %s " % (value)
		else:
			paradict[group] += "--%s=%s " % (name,value)

def ReadParamsFromPlan( paramfile):
    paradict['dibayes'] = ""
    paradict['torrent-variant-caller'] = ""
    paradict['long-indel-assembler'] = ""    
    paradict['filter-indels'] = ""
    if paramfile == "":
        return
    JSON_INPUT = json.load( open(paramfile, "r") )
    params = JSON_INPUT['runinfo']['plugin']['userInput']
    for line in params:
	value=params[line]
	line = line.strip()
	line = line.replace('___', '-')
	tokens = line.split('__')
	if(tokens[0].strip().replace('"', '') == "param"):
		group = tokens[1].strip().replace('"', '')
		name = tokens[2].strip().replace('"', '')
		if name == "C":
			paradict[group] += "-C %s " % (value)
		else:
			paradict[group] += "--%s=%s " % (name,value)
            	

def ReadParamsFile( paramfile ):
    # defaults for expected dictionary elements
    paradict['dibayes'] = ""
    paradict['torrent-variant-caller'] = ""
    paradict['long-indel-assembler'] = ""    
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
            if name == "C":
                paradict[dictkey] += "-C %s " % (val)
            else:
                paradict[dictkey] += "--%s=%s " % (name,val)   
    inf.close()

def RunCommand( command ):
    global errorExit, progName, haveBed, noerrWarn, noerrWarnFile
    command = command.strip()
    WriteLog( " $ %s\n" % command )
    stat = os.system( command )
    if( stat ) != 0:
        sys.stderr.write( "ERROR: resource failed with status %d:\n" % stat )
        sys.stderr.write( "$ %s\n" % command )
        if errorExit == 0 and noerrWarnFile != "":
            warnout = open(noerrWarnFile,'w')
            if noerrWarn != "":
                warnout.write("%s - see Log File.\n" % noerrWarn)
            if haveBed:
                warnout.write("SNP calls are not filtered to target regions.\n")
            warnout.close()
        sys.exit(errorExit)

def SNPsCallerCommandLine( binDir, paramstr, bamfile, reference, bedfile, hotspotsfile, outDir ):
    fixedprms = "--AllSeq=1 -b 1 --platform=2 -d 1 -W 0 -S 1 -n diBayes_run -g %s/log/ -w %s/temp/ -o %s" % (outDir,outDir,outDir)
    if paramstr == "":
        paramstr  = "--call-stringency medium --het-skip-high-coverage 0 --reads-min-mapping-qv 2 --het-min-lca-start-pos 0 --het-min-lca-base-qv 14 " + \
            "--het-lca-both-strands 0 --het-min-allele-ratio 0.15 --het-max-coverage-bayesian 60 --het-min-nonref-base-qv 14 --snps-min-base-qv 14 " + \
            "--snps-min-nonref-base-qv 14 --reads-with-indel-exclude 0 --het-min-coverage 2 --het-min-start-pos 1 --hom-min-coverage 1 " + \
            "--hom-min-nonref-allele-count 3 --snps-min-filteredreads-rawreads-ratio 0.15 --het-min-validreads-totalreads-ratio 0.65 " + \
            "--reads-min-alignlength-readlength-ratio 0.2 --hom-min-nonref-base-qv 14 --hom-min-nonref-start-pos 0"
    if bedfile != "":
        bedfile = '-R ' + bedfile
    if hotspotsfile != "":
	hotspotsfile = '-H ' + hotspotsfile
    commandLineDiBayes = "%s/diBayes %s %s %s %s -f %s %s" % (
        binDir, fixedprms, paramstr, bedfile, hotspotsfile, reference, bamfile )
    return "export LD_LIBRARY_PATH=%s/diBayes_lib;mkdir -p %s/temp;mkdir -p %s/log;export LOG4CXX_CONFIGURATION;%s" % (binDir,outDir,outDir,commandLineDiBayes)

def IndelCallerCommandLine( binDir, paramstr, bamfile, reference, bedfile, outDir ):
    if bedfile != "":
        bedfile = '-L ' + bedfile
    fixedprms = "java -Xmx8G -Djava.library.path=%s/TVC/lib -cp %s/TVC/jar/ -jar %s/TVC/jar/GenomeAnalysisTK.jar -T UnifiedGenotyper -R %s -I %s %s -o %s/%s -glm INDEL -S SILENT -U ALL -filterMBQ --selectMostFreqAllele --excludeHMM -combineSample ion-sample -nt 8 --max_alternate_alleles 2 --max_deletion_fraction 2 --min_base_quality_score 5 " % (
        binDir, binDir, binDir, reference, bamfile, bedfile, outDir, "small_indels.vcf")
    if paramstr == "":
       paramstr = " -minIndelCnt 5 -dcov 2000 "
    else:
       paramstr = paramstr.replace('=',' ')
    annotation = " -A IndelType -A AlleleBalance -A BaseCounts -A ReadDepthAndAllelicFractionBySample -A AlleleBalanceBySample -A DepthPerAlleleBySample -A MappingQualityZeroBySample "   
    return "%s %s %s > %s/%s;" % (fixedprms, annotation, paramstr, outDir, "indel_caller.log")

def IndelAssemblyCommandLine( binDir, paramstr, reference, bamfile, bedfile, outDir ):
    if bedfile != "":
       bedfile = '-L ' + bedfile
    fixedprms = "java -Xmx8G -cp %s/TVC/jar/ -jar %s/TVC/jar/GenomeAnalysisTK.jar -T IndelAssembly --bypassFlowAlign -R %s -I %s %s -o %s/%s -S SILENT -U ALL -filterMBQ " % (
        binDir, binDir, reference, bamfile, bedfile, outDir, "indel_assembly.vcf")    
    if paramstr == "":
       paramstr = " -nt 1 "
    else:
       paramstr = paramstr.replace('=',' ')
    return "%s %s  > %s/%s" % (fixedprms, paramstr, outDir, "indel_assembly.log")
    

def ScoreFilterIndelsCommandLine( binDir, paramstr, outDir ):
    return "%s/filter_indels.py %s %s/bayesian_scorer.vcf %s/variantCalls.filtered.vcf" % ( binDir, paramstr, outDir, outDir )

def BedFilterIndelsCommandLine( binDir, bedfile, outDir ):
    # add extra first line to bed file to avoid bug in vcftools
    rmbed = ""
    if bedfile != "":
        bedtmp = bedfile + "tmp.bed"
        RunCommand( 'awk \'{++c;if(c==1&&$1!~"^#"){print "#header line required by vcftools";print}else{print}}\' %s > %s' % (bedfile,bedtmp) )
        bedfile = '--bed ' + bedtmp
        rmbed = '; rm -f ' + bedtmp
    return "%s/vcftools --vcf %s/indels.merged.vcf %s --out %s/indels --recode --keep-INFO-all > /dev/null %s" % ( binDir, outDir, bedfile, outDir, rmbed )

def VCFSortFilterCommandLine( binDir, outDir ):
    return "java -Xmx4G -cp  %s/TVC/jar/VcfUtils.jar:%s/TVC/jar/VcfModel.jar:%s/TVC/jar/log4j-1.2.15.jar com.lifetech.ngs.vcfutils.FixQUALRun %s/variantCalls.filtered.vcf %s/indels.gatk-qual-rescored.vcf;java -Xmx4G -cp  %s/TVC/jar/VcfUtils.jar:%s/TVC/jar/VcfModel.jar:%s/TVC/jar/log4j-1.2.15.jar com.lifetech.ngs.vcfutils.SortVcfRun %s/indels.gatk-qual-rescored.vcf %s/indels.merged.vcf" % ( binDir, binDir, binDir, outDir, outDir, binDir, binDir, binDir, outDir, outDir )


def MergeSNPandIndelVCF( binDir, outDir ):
    return "java -Xmx4G -cp  %s/TVC/jar/VcfUtils.jar:%s/TVC/jar/VcfModel.jar:%s/TVC/jar/log4j-1.2.15.jar com.lifetech.ngs.vcfutils.MergeVcfRun %s/SNP_variants.vcf %s/indel_variants.vcf %s/TSVC_variants.vcf" % ( binDir, binDir, binDir, outDir, outDir, outDir )

    
def OutputSnpVCF( binDir, inDir, faifile, outDir ):
    global keepdir
    snpsout = "%s/SNP_variants.vcf" % outDir
    consensout = "%s/consensus_calls.txt" % outDir
    os.system('rm -f "%s"' % snpsout)
    os.system('rm -f "%s"' % consensout)
    contigs = open(faifile,'r')
    countout = 0
    conscountout = 0
    for lines in contigs:
        if len(lines) == 0:
            continue
        chrom = lines.split()
        fname = "%s/diBayes_run_%s_SNP.vcf" % (inDir,chrom[0])
        consfname = "%s/diBayes_run_%s_Consensus_Calls.txt" % (inDir,chrom[0])
        if os.path.exists(fname):
            countout += 1
            if countout > 1:
                os.system('sed -i -e "/^#/d" "%s"' % fname)
            os.system('cat "%s" >> "%s"' % (fname,snpsout))
        if os.path.exists(consfname):
            conscountout += 1
            if conscountout > 1:
                os.system('sed -i -e "/^#/d" "%s"' % consfname)
            os.system('cat "%s" >> "%s"' % (consfname,consensout))
    contigs.close()
    if countout == 0:
        CreateEmptyVcf(snpsout)
    keepdir = 0
    if keepdir == 0:
        RunCommand('rm -rf "%s"' % inDir)
        WriteLog(" (%s removed)\n" % inDir)
    WriteLog(" > %s\n" % snpsout)
    ZindexVcf(binDir,snpsout)
    if os.path.exists(consensout):
	ZindexVcf(binDir,consensout)
	os.system('rm -f "%s"' % consensout)

def OutputIndelVCF( binDir, outDir ):
    indelsout = "%s/indel_variants.vcf" % outDir
    os.system('rm -f "%s"' % indelsout)
    if os.path.exists("%s/indels.recode.vcf" % outDir):
        RunCommand( 'cat "%s/indels.recode.vcf" > "%s"' % (outDir,indelsout) )
    else:
        RunCommand( 'cat "%s/indels.merged.vcf" > "%s"' % (outDir, indelsout) )
    if not os.path.exists(indelsout):
        CreateEmptyVcf(indelsout)
    WriteLog(" > %s\n" % indelsout)
    ZindexVcf(binDir,indelsout)

def OutputMergedVCF( binDir, outDir ):
    vcfout = "%s/TSVC_variants.vcf" % outDir
    if not os.path.exists(vcfout):
        CreateEmptyVcf(vcfout)
    WriteLog(" > %s\n" % vcfout)
    RunCommand( '%s/bgzip -c "%s" > "%s.gz"' % (binDir,vcfout,vcfout) )
    WriteLog(" > %s.gz\n" % vcfout)

def CreateEmptyVcf( fileName ):
    try:
        fout = open(fileName,'w')
        fout.write("##fileformat=VCFv4.1\n");
        fout.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample\n");
        fout.close()
    except:
        sys.stderr.write("ERROR: Cannot open output file '%s'" % fileName)
        sys.exit(1)

def CreateParamsFile( fileName ):
    try:
	fout = open(fileName, 'w')
	now = datetime.datetime.now()
	fout.write("VariantCaller : ")
	fout.write(now.strftime("%Y-%m-%d %H:%M"))
	fout.write("\n\n")
	fout.write("SNP Caller Parameter Settings:")
	fout.write("\n")
	cmdStr = paradict['dibayes']
	parameters = cmdStr.split(' ')
	for p in parameters:
		p = p.replace('--', '')
		fout.write(p)
		fout.write("\n")	
	fout.write("\n")
	fout.write("Indel Caller Parameter Settings:")
	fout.write("\n")
	cmdStr = paradict['torrent-variant-caller']
	parameters = cmdStr.split(' ')
	for p in parameters:
		p = p.replace('--', '')
		fout.write(p)
		fout.write("\n")	
	fout.write("\n")
	fout.write("Long Indel Assembly Parameter Settings:")
	fout.write("\n")
	cmdStr = paradict['long-indel-assembler']
	parameters = cmdStr.split(' ')
	for p in parameters:
		p = p.replace('--', '')
		fout.write(p)
		fout.write("\n")	
	fout.write("\n")
	fout.write("Filter-Indels Parameter Settings:")
	fout.write("\n")
	cmdStr = paradict['filter-indels']
	parameters = cmdStr.split(' ')
	for p in parameters:
		p = p.replace('--', '')
		fout.write(p)
		fout.write("\n")	
	fout.write("\n")
	fout.close()
    except:
	sys.stderr.write("WARNING: Cannot open output file '%s'" % fileName)
	#sys.exit(1)

def ZindexVcf( binDir, fileName ):
    RunCommand( '%s/bgzip -c "%s" > "%s.gz"' % (binDir,fileName,fileName) )
    WriteLog(" > %s.gz\n" % fileName)
    RunCommand( '%s/tabix -p vcf "%s.gz"' % (binDir,fileName) )
    WriteLog(" > %s.gz.tbi\n" % fileName)

def WriteLog( msg, force=0 ):
    global logout, logfile, logerr
    if logfile != "":
        if logout == 0:
            logout = open(logfile,'w')
        logout.write(msg)
    if logerr == 1 or force != 0:
        sys.stderr.write(msg)
        
def main(argv):
    # arg processing
    global progName, logout, logfile, logerr, keepdir, errorExit, haveBed, noerrWarn, noerrWarnFile, callSNPs
    rundir = os.path.realpath(__file__)
    logerr = 0
    logout = 0
    keepdir = 0
    callSNPs = 1
    noerror = 0
    logfile=""
    paramfile=""
    bedfile=""
    hotspotsfile=""
    faifile=""
    fsbam = ""
    noerrWarnFile = ""
    try:
        opts, args = getopt.getopt( argv, "hlknp:r:b:s:f:g:o:W:L:",
            ["help", "log", "keepdir", "nosnp", "paramfile=", "rundir=", "bedfile=", "hotspotsfile=", "fsbam=", "genome=", "warnto=", "logfile="] )
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
        elif opt in ("-s", "--hotspotsfile"):
            hotspotsfile = arg
            if not os.path.exists(hotspotsfile):
                sys.stderr.write("No hotspots file found at: %s\n" % hotspotsfile)
                sys.exit(1)
        elif opt in ("-r", "--rundir"):
            rundir = arg
            if not os.path.isdir(rundir):
                sys.stderr.write("No run directory found at: %s\n" % rundir)
                sys.exit(1)
        elif opt in ("-f", "--fsbam"):
            fsbam = arg
        elif opt in ("-g", "--genome"):
            faifile = arg
        elif opt in ("-k", "--keepdir"):
            keepdir = 1
        elif opt in ("-n", "--nosnp"):
            callSNPs = 0    
        elif opt in ("-W", "--warnto"):
            noerrWarnFile = arg
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
    if( faifile == "" ):
        faifile = reference + ".fai"

    WriteLog("Running %s...\n" % progName)

    if not os.path.isdir(outdir):
        sys.stderr.write("No output directory found at: %s\n" % outdir)
        sys.exit(1)
    if not os.path.exists(reference):
        sys.stderr.write("No reference file found at: %s\n" % reference)
        sys.exit(1)
    if keepdir == 0:
        if not os.path.exists(faifile):
            sys.stderr.write("No genome/fasta index file found at: %s\n" % faifile)
            sys.exit(1)
    if not os.path.exists(bamfile):
        sys.stderr.write("No bam file found at: %s\n" % bamfile)
        sys.exit(1)
    if not os.path.exists(fsbam):
        sys.stderr.write("No flowspace bam file found at: %s\n" % fsbam)
        sys.exit(1)

    # used to track no fatal errors that need to be tracked outside log.
    haveBed = 0
    errorExit = 1
    noerrWarn = ""

    try:
       ReadParamsFromUI(os.environ['START_PLUGIN_JSON']);

       if(paradict['dibayes'] == ""):
          # read calling parameters (germline/somatic)
          ReadParamsFromPlan(os.environ['START_PLUGIN_JSON'])
       if(paradict['dibayes'] == ""):
          # read calling parameters (germline/somatic)
          ReadParamsFile(paramfile)
          
    except:
       ReadParamsFile(paramfile)

    CreateParamsFile(outdir + "/variantCaller_Params.xls");
    # create output directories for diBayes, if not already present
    snp_dir = outdir + "/dibayes_out"
    indel_dir = outdir
    if not os.path.exists(snp_dir):
        os.makedirs(snp_dir)
    if not os.path.exists(indel_dir):
        os.makedirs(indel_dir)
   
    if callSNPs == 1:
    # create command for SNP caller and run
       WriteLog(" Finding SNPs using diBayes...\n",1)
       cmdoptions = paradict['dibayes']
       RunCommand( SNPsCallerCommandLine( rundir, cmdoptions, bamfile, reference, bedfile, hotspotsfile, snp_dir ) )
       WriteLog(" > %s/\n" % snp_dir)

    # disable fatal errors from this point if option given
       if noerrWarnFile != "": errorExit = 0

    # merge diBayes outputs (per contig) into one file in reference index order
       noerrWarn = "No SNP call indexing performed"
       WriteLog(" Merging SNP calls and indexing VCF...\n",1)
       OutputSnpVCF( rundir, snp_dir, faifile, outdir )

    # ensure no filtering is tracked as issue if no error exit status returned
       if bedfile != "": haveBed = 1
       noerrWarn = "No INDEL calls made"
    else: 
       snpsout = "%s/SNP_variants.vcf" % outdir
       CreateEmptyVcf(snpsout)

    # create command for INDEL calling and run
    WriteLog("Calling small INDELs ...\n",1)
    cmdoptions = paradict['torrent-variant-caller']
    RunCommand( IndelCallerCommandLine( rundir, cmdoptions, fsbam, reference, bedfile, indel_dir ) )
    WriteLog(" > %s/small_indels.vcf\n" % indel_dir)
    if( os.path.exists( "%s/small_indels.vcf" % indel_dir ) ):
       RunCommand( "cat %s/small_indels.vcf > %s/bayesian_scorer.vcf" % (indel_dir , indel_dir) )
    WriteLog(" > %s/bayesian_scorer.vcf\n" % indel_dir)
       
    # create command for long indel assembly and run
    noerrWarn = "No long INDEL calls made"
    WriteLog(" Assembling Long INDELs using LiM ...\n",1)
    cmdoptions = paradict['long-indel-assembler']
    RunCommand( IndelAssemblyCommandLine( rundir, cmdoptions, reference, fsbam, bedfile, indel_dir ) )
    if( os.path.exists( "%s/indel_assembly.vcf" % indel_dir ) ):
        RunCommand( "cat %s/indel_assembly.vcf >> %s/bayesian_scorer.vcf" % (indel_dir , indel_dir) )
    WriteLog(" >> %s/bayesian_scorer.vcf\n" % indel_dir)

    # create commands for indel filtering and run
    noerrWarn = "No INDEL call filtering performed"
    WriteLog(" Score Filtering of INDEL vcf ...\n",1)
    cmdoptions = paradict['filter-indels']
    RunCommand( ScoreFilterIndelsCommandLine( rundir, cmdoptions, indel_dir ) )
    WriteLog(" > %s/variantCalls.filtered.vcf\n" % indel_dir)

    # sort and merge variants in combined indel file
    WriteLog(" Sorting and Merging INDEL vcf ...\n",1)
    RunCommand(VCFSortFilterCommandLine( rundir, indel_dir ))
    WriteLog(" > %s/indels.gatk-qual-rescored.vcf\n" % indel_dir)
    WriteLog(" > %s/indels.merged.vcf\n" % indel_dir)

    # BED filtering
    WriteLog(" BED Filtering of INDEL vcf ...\n",1)    
    RunCommand( BedFilterIndelsCommandLine( rundir, bedfile, indel_dir ) )
    WriteLog(" > %s/indels.recode.vcf\n" % indel_dir)

    # re-order indels to final output file
    noerrWarn = "No INDEL call indexing performed"
    WriteLog(" Sorting INDEL calls and indexing VCF...\n",1)
    OutputIndelVCF( rundir, indel_dir )
    
    # generating an unified SNPs and Indels file
    RunCommand( MergeSNPandIndelVCF( rundir, indel_dir ))
    WriteLog(" > %s/TSVC_variants.vcf\n" % indel_dir)
    OutputMergedVCF( rundir, indel_dir)

    if logfile != "":
        logout.close()

if __name__ == '__main__':
    global progName
    progName = sys.argv[0]
    main(sys.argv[1:])

