#!/usr/bin/python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
import subprocess
import time
try:
    import json
except:
    import simplejson as json
from optparse import OptionParser


def printtime(message, *args):
    if args:
        message = message % args
    print "[ " + time.strftime('%X') + " ] " + message
    sys.stdout.flush()
    sys.stderr.flush()


def RunCommand(command,description):
    printtime(' ')
    printtime('Task    : ' + description)
    printtime('Command : ' + command)
    printtime(' ')
    stat = subprocess.call(command,shell=True)
    if stat != 0:
        printtime('ERROR: command failed with status %d' % stat)
        sys.exit(1)


def VCFSortFilterCommandLine( binDir, outDir ):
    return "java -Xmx4G -cp  %s/TVC/jar/VcfUtils.jar:%s/TVC/jar/VcfModel.jar:%s/TVC/jar/log4j-1.2.15.jar com.lifetech.ngs.vcfutils.FixQUALRun %s/variantCalls.filtered.vcf %s/indels.gatk-qual-rescored.vcf;java -Xmx4G -cp  %s/TVC/jar/VcfUtils.jar:%s/TVC/jar/VcfModel.jar:%s/TVC/jar/log4j-1.2.15.jar com.lifetech.ngs.vcfutils.SortVcfRun %s/indels.gatk-qual-rescored.vcf %s/indels.merged.vcf" % ( binDir, binDir, binDir, outDir, outDir, binDir, binDir, binDir, outDir, outDir )


def MergeSNPandIndelVCF( binDir, outDir ):
    return "java -Xmx4G -cp  %s/TVC/jar/VcfUtils.jar:%s/TVC/jar/VcfModel.jar:%s/TVC/jar/log4j-1.2.15.jar com.lifetech.ngs.vcfutils.MergeVcfRun %s/SNP_variants.vcf %s/indel_variants.vcf %s/TSVC_variants.vcf" % ( binDir, binDir, binDir, outDir, outDir, outDir )


def SplitVcf(input_vcf,output_snp_vcf,output_indel_vcf):
    
    input = open(input_vcf,'r')
    output_snp = open(output_snp_vcf,'w')
    output_indel = open(output_indel_vcf,'w')
    
    for line in input:
        if not line or line[0]=='#':
            output_snp.write(line)
            output_indel.write(line)
            continue
        
        fields = line.split('\t')
        ref = fields[3]
        alt = fields[4].split(',')[0]
        
        if len(alt) == len(ref):
            output_snp.write(line)
        else:
            output_indel.write(line)
            
    input.close()
    output_snp.close()
    output_indel.close()



def main():
    
    parser = OptionParser()
    parser.add_option('-b', '--region-bed',     help='Limit variant calling to regions in this BED file (optional)', dest='bedfile') 
    parser.add_option('-s', '--hotspot-vcf',    help='VCF.gz (+.tbi) file specifying exact hotspot positions (optional)', dest='hotspot_vcf') 
    parser.add_option('-i', '--input-bam',      help='BAM file containing aligned reads (required)', dest='bamfile') 
    parser.add_option('-r', '--reference-fasta',help='FASTA file containing reference genome (requires)', dest='reference') 
    parser.add_option('-o', '--output-dir',     help='Output directory (default: current)', dest='outdir', default='.')
    parser.add_option('-p', '--parameters-file',help='JSON file containing variant calling parameters (recommended)', dest='paramfile')
    parser.add_option('-B', '--bin-dir',        help='Directory path to location of variant caller programs. Defaults to the directory this script is located', dest='rundir', default=os.path.realpath(__file__)) 
    parser.add_option('-n', '--num-threads',    help='Set TVC number of threads (default: 12)', dest='numthreads',default='12') 
    parser.add_option(      '--primer-trim-bam',help='Perform primer trimming, storing the results in provided BAM file name (optional)', dest='ptrim_bam')
    parser.add_option(      '--primer-trim-bed',help='BED file used for primer trimming. Must be provided with --primer-trim-bam', dest='ptrim_bed')
    
    (options, args) = parser.parse_args()
    
    if not options.bamfile or not options.reference:
        parser.print_help()
        exit(1)
    
	if not os.path.isdir(options.outdir):
		printtime('ERROR: No output directory found at: ' + options.outdir)
		sys.exit(1)
	if not os.path.exists(options.reference):
		printtime('ERROR: No reference file found at: ' + options.reference)
		sys.exit(1)
    if not os.path.exists(options.reference+'.fai'):
        printtime('ERROR: No reference index file found at: ' + options.reference + '.fai')
        sys.exit(1)
	if not os.path.exists(options.bamfile):
		printtime('ERROR: No bam file found at: ' + options.bamfile)
		sys.exit(1)
    if not os.path.exists(options.bamfile+'.bai'):
        printtime('ERROR: No bam index file found at: ' + options.bamfile + '.bai')
        sys.exit(1)

    if options.ptrim_bam:
        if not options.ptrim_bed:
            printtime('ERROR: --primer-trim-bam must be accompanied by --primer-trim-bed')
            exit(1)
        if not os.path.exists(options.ptrim_bed):
            printtime('ERROR: No bed file found at: ' + options.ptrim_bed)
            sys.exit(1)
            
    if options.hotspot_vcf:
        if not os.path.exists(options.hotspot_vcf):
            printtime('ERROR: No hotspots vcf.gz file found at: ' + options.hotspot_vcf)
            sys.exit(1)
        if not os.path.exists(options.hotspot_vcf+'.tbi'):
            printtime('ERROR: No hotspots index vcf.gz.tbi file found at: ' + options.hotspot_vcf + '.tbi')
            sys.exit(1)
    

    parameters = {}
    if options.paramfile:
        try:
            json_file = open(options.paramfile, 'r')
            parameters = json.load(json_file)
            json_file.close()
            if parameters.has_key('pluginconfig'):
                parameters = parameters['pluginconfig']
        except:
            printtime('ERROR: No parameter file found at: ' + options.paramfile)
            sys.exit(1)
    
    if not os.path.exists(options.outdir):
        os.makedirs(options.outdir)
   
   
    # Perform primer trimming, if requested
    if options.ptrim_bam:
        subprocess.call('rm -f "%s"' % options.ptrim_bam,shell=True)
        trimp_command = 'java -Xmx8G -cp %s/TRIMP_lib -jar %s/TRIMP.jar' % (options.rundir,options.rundir)
        trimp_command += ' ' + options.bamfile
        trimp_command += ' ' + options.ptrim_bam
        trimp_command += ' ' + options.reference
        trimp_command += ' ' + options.ptrim_bed
        RunCommand(trimp_command,'Trimming reads to target regions')
        RunCommand('samtools index "%s"' % options.ptrim_bam,'Generate index for trimmed BAM')
   
    
    # This logic might go to variant_caller_plugin.py
    meta_tvc_args = parameters.get('meta',{}).get('tvcargs','tvc')
    if meta_tvc_args == 'tvc' and os.path.exists(options.rundir + '/tvc'):   # try local binary first, then go to global one
        tvc_command =           '%s/tvc' % options.rundir
    else:
        tvc_command =           meta_tvc_args
    tvc_command +=              '   --output-dir %s' % options.outdir
    tvc_command +=              '   --output-vcf small_variants.vcf'
    tvc_command +=              '   --reference %s' % options.reference
    if options.ptrim_bam:
        tvc_command +=          '   --input-bam %s' % options.ptrim_bam
    else:
        tvc_command +=          '   --input-bam %s' % options.bamfile
    if options.bedfile:
        tvc_command +=          '   --target-file %s' % options.bedfile
    #if options.hotspot_vcf:
    #    tvc_command +=          '   --input-vcf %s' % options.hotspot_vcf
    if options.paramfile:
        tvc_command +=          '   --parameters-file %s' % options.paramfile
    tvc_command +=              '   --num-threads %s' % options.numthreads
    tvc_command +=              '   --error-motifs %s' % os.path.join(options.rundir,'TVC/sse/motifset.txt')
    RunCommand(tvc_command,'Call small indels and SNPs')

    vcfsort_command =           '%s/scripts/sort_vcf.py' % options.rundir
    vcfsort_command +=          '   --input-vcf %s/small_variants.vcf' % options.outdir
    vcfsort_command +=          '   --output-vcf %s/small_variants.sorted.vcf' % options.outdir
    vcfsort_command +=          '   --index-fai %s.fai' % options.reference
    RunCommand(vcfsort_command, 'Sort small variant vcf')

    left_align_command =        'java -Xmx8G -jar %s/TVC/jar/GenomeAnalysisTK.jar' % options.rundir
    left_align_command +=       '   -T LeftAlignVariants'
    left_align_command +=       '   -R %s' % options.reference
    left_align_command +=       '   --variant %s/small_variants.sorted.vcf' % options.outdir
    left_align_command +=       '   -o %s/small_variants.left.vcf' % options.outdir
    RunCommand(left_align_command, 'Ensure left-alignment of indels')


    if options.hotspot_vcf:
        if meta_tvc_args == 'tvc' and os.path.exists(options.rundir + '/tvc'):   # try local binary first, then go to global one
            tvc2_command =      '%s/tvc' % options.rundir
        else:
            tvc2_command =      meta_tvc_args
        tvc2_command +=         '   --output-dir %s' % options.outdir
        tvc2_command +=         '   --output-vcf hotspot_calls.vcf'
        tvc2_command +=         '   --reference %s' % options.reference
        if options.ptrim_bam:
            tvc2_command +=     '   --input-bam %s' % options.ptrim_bam
        else:
            tvc2_command +=     '   --input-bam %s' % options.bamfile
        if options.bedfile:
            tvc2_command +=     '   --target-file %s' % options.bedfile
        if options.paramfile:
            tvc2_command +=     '   --parameters-file %s' % options.paramfile
        tvc2_command +=         '   --input-vcf %s' % options.hotspot_vcf
        tvc2_command +=         '   --process-input-positions-only on'
        tvc2_command +=         '   --use-input-allele-only on'
        tvc2_command +=         '   --num-threads %s' % options.numthreads
        tvc2_command +=         '   --error-motifs %s' % os.path.join(options.rundir,'TVC/sse/motifset.txt')
        RunCommand(tvc2_command,'Call Hotspots')

    
    # create command for long indel assembly and run
    long_indel_command =        'java -Xmx8G -cp %s/TVC/jar/ -jar %s/TVC/jar/GenomeAnalysisTK.jar' % (options.rundir,options.rundir)
    long_indel_command +=       '   -T IndelAssembly --bypassFlowAlign'
    long_indel_command +=       '   -R %s' % options.reference
    if options.ptrim_bam:
        long_indel_command +=   '   -I %s' % options.ptrim_bam
    else:
        long_indel_command +=   '   -I %s' % options.bamfile
    if options.bedfile:
        long_indel_command +=   '   -L %s' % options.bedfile
    long_indel_command +=       '   -o %s/indel_assembly.vcf' % options.outdir
    long_indel_command +=       '   -S SILENT -U ALL -filterMBQ'
    cmdoptions = parameters.get('long_indel_assembler',{})
    for k,v in cmdoptions.iteritems():
        long_indel_command +=   '   --%s %s' % (k, str(v))
    if not cmdoptions:
        long_indel_command +=   '   -nt 1'
    #long_indel_command +=       ' > %s/indel_assembly.log' % options.outdir
    RunCommand(long_indel_command, 'Assemble long indels')


    # Perform variant unification step.

    unify_command =             '%s/scripts/unify_variants_and_annotations.py' % options.rundir
    unify_command +=            '   --novel-tvc-vcf %s/small_variants.left.vcf' % options.outdir
    if os.path.exists("%s/indel_assembly.vcf" % options.outdir):
        unify_command +=        '   --novel-assembly-vcf %s/indel_assembly.vcf' % options.outdir
    if options.hotspot_vcf:
        unify_command +=        '   --hotspot-tvc-vcf %s/hotspot_calls.vcf' % options.outdir
        unify_command +=        '   --hotspot-annotation-vcf %s' % options.hotspot_vcf[:-3] #remove .gz
    unify_command +=            '   --output-vcf %s/all.merged.vcf' % options.outdir
    unify_command +=            '   --index-fai %s.fai' % options.reference
    RunCommand(unify_command, 'Unify variants and annotations from all sources (tvc,IndelAssembly,hotpots)')
    
    # Scan through the merged vcf and count the number of lines. 
    num_variants_before_bed = 0
    input = open('%s/all.merged.vcf' % options.outdir,'r')
    for line in input:
        if line and line[0] != '#':
            num_variants_before_bed += 1
    input.close()


    # BED filtering
    if options.bedfile and num_variants_before_bed > 0:
        # This command merely prepends a fake header line to the bed file if it doesn't have one
        bedtmp = options.outdir + '/' + os.path.basename(options.bedfile) + "tmp.bed"
        RunCommand('awk \'{++c;if(c==1&&$1!~"^#"){print "track name=header";print}else{print}}\' %s > %s' % (options.bedfile,bedtmp),
                   'Append header line to bed file')
        
        bedfilter_command  =    'vcftools'
        bedfilter_command +=    '   --vcf %s/all.merged.vcf' %  options.outdir
        bedfilter_command +=    '   --bed %s' % bedtmp
        bedfilter_command +=    '   --out %s/all' % options.outdir
        bedfilter_command +=    '   --recode  --keep-INFO-all'
        #bedfilter_command +=    ' > /dev/null'
        RunCommand(bedfilter_command, 'Filter merged VCF using region BED')
        RunCommand("cp   %s/all.recode.vcf   %s/TSVC_variants.vcf" % (options.outdir,options.outdir),
                   'Move final VCF into place')
    else:
        RunCommand("cp   %s/all.merged.vcf   %s/TSVC_variants.vcf" % (options.outdir,options.outdir),
                   'Move final VCF into place')


    # Generate .gz and .tbi
    vcfout = '%s/TSVC_variants.vcf' % options.outdir
    RunCommand('bgzip   -c "%s"   > "%s.gz"' % (vcfout,vcfout), 'Generate compressed vcf')
    RunCommand('tabix   -p vcf   "%s.gz"' % (vcfout), 'Generate index for compressed vcf')


    SplitVcf("%s/TSVC_variants.vcf" % options.outdir , "%s/SNP_variants.vcf" % options.outdir , "%s/indel_variants.vcf" % options.outdir)
    

if __name__ == '__main__':
    main()

