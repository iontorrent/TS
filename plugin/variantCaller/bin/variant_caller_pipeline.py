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
    parser.add_option('-b', '--region-bed',       help='Limit variant calling to regions in this BED file (optional)', dest='bedfile')
    parser.add_option('-s', '--hotspot-vcf',      help='VCF file specifying exact hotspot positions (optional)', dest='hotspot_vcf')
    parser.add_option('-i', '--input-bam',        help='BAM file containing aligned reads (required)', dest='bamfile')
    parser.add_option('-r', '--reference-fasta',  help='FASTA file containing reference genome (required)', dest='reference')
    parser.add_option('-o', '--output-dir',       help='Output directory (default: current)', dest='outdir', default='.')
    parser.add_option('-p', '--parameters-file',  help='JSON file containing variant calling parameters (recommended)', dest='paramfile')
    parser.add_option('-B', '--bin-dir',          help='Directory path to location of variant caller programs. Defaults to the directory this script is located', dest='bindir')
    parser.add_option('-t', '--tvc-root-dir',     help='Directory path to TVC root directory', dest='tvcrootdir')
    parser.add_option('-n', '--num-threads',      help='Set TVC number of threads (default: 12)', dest='numthreads',default='12')
    parser.add_option(      '--primer-trim-bed',  help='Perform primer trimming using provided BED file. (optional)', dest='ptrim_bed')
    parser.add_option(      '--postprocessed-bam',help='Perform primer trimming, storing the results in provided BAM file name (optional)', dest='postprocessed_bam')
    parser.add_option(      '--error-motifs',     help='Error motifs file', dest='errormotifsfile')
    (options, args) = parser.parse_args()

    if options.tvcrootdir:
        tvcrootdir = options.tvcrootdir
    elif options.bindir:
        tvcrootdir = os.path.join( options.bindir, "..")
    else:
        tvcrootdir = os.path.join( os.path.dirname(os.path.realpath(__file__)), "..")
    tvcrootdir = os.path.normpath( tvcrootdir )

    #if options.errormotifsfile is None:
        # currently a required argument
     #   options.errormotifsfile = os.path.join(tvcrootdir,'share/TVC/sse/motifset.txt')

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


    if options.hotspot_vcf:
        if not os.path.exists(options.hotspot_vcf):
            printtime('ERROR: No hotspots vcf file found at: ' + options.hotspot_vcf)
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



    # New way of handling hotspots: single call to tvc

    # This logic might go to variant_caller_plugin.py
    meta_tvc_args = parameters.get('meta',{}).get('tvcargs','tvc')
    # try local binary first, then go to global one
    if meta_tvc_args == 'tvc' and os.path.exists(tvcrootdir + '/bin/tvc'):
        tvc_command =           '%s/bin/tvc' % tvcrootdir
    else:
        tvc_command =           meta_tvc_args
    tvc_command +=              '   --output-dir %s' % options.outdir
    tvc_command +=              '   --output-vcf small_variants.vcf'
    tvc_command +=              '   --reference %s' % options.reference
    tvc_command +=              '   --input-bam %s' % options.bamfile
    if options.ptrim_bed:
        tvc_command +=          '   --target-file %s' % options.ptrim_bed
        tvc_command +=          '   --trim-ampliseq-primers on'
    elif options.bedfile:
        tvc_command +=          '   --target-file %s' % options.bedfile
    if options.postprocessed_bam:
        postprocessed_bam_tmp = options.postprocessed_bam + '.tmp.bam'
        tvc_command +=          '   --postprocessed-bam %s' % postprocessed_bam_tmp
    if options.hotspot_vcf:
        tvc_command +=          '   --input-vcf %s' % options.hotspot_vcf
    if options.paramfile:
        tvc_command +=          '   --parameters-file %s' % options.paramfile
    tvc_command +=              '   --num-threads %s' % options.numthreads
    if options.errormotifsfile:
        tvc_command +=              '   --error-motifs %s' % options.errormotifsfile

    RunCommand(tvc_command,'Call small indels and SNPs')


    if options.postprocessed_bam:
        bamsort_command = 'samtools sort %s %s' % (postprocessed_bam_tmp, options.postprocessed_bam[:-4])
        RunCommand(bamsort_command,'Sort postprocessed bam')
        bamindex_command = 'samtools index %s' % options.postprocessed_bam
        RunCommand(bamindex_command,'Index postprocessed bam')
        RunCommand('rm -f ' + postprocessed_bam_tmp, 'Remove unsorted postprocessed bam')

    vcfsort_command =           '%s/share/TVC/scripts/sort_vcf.py' % tvcrootdir
    vcfsort_command +=          '   --input-vcf %s/small_variants.vcf' % options.outdir
    vcfsort_command +=          '   --output-vcf %s/small_variants.sorted.vcf' % options.outdir
    vcfsort_command +=          '   --index-fai %s.fai' % options.reference
    RunCommand(vcfsort_command, 'Sort small variant vcf')

    left_align_command =        'java -Xmx8G -jar %s/share/TVC/jar/GenomeAnalysisTK.jar' % tvcrootdir
    left_align_command +=       '   -T LeftAlignVariants'
    left_align_command +=       '   -R %s' % options.reference
    left_align_command +=       '   --variant %s/small_variants.sorted.vcf' % options.outdir
    left_align_command +=       '   -o %s/small_variants.left.vcf' % options.outdir
    RunCommand(left_align_command, 'Ensure left-alignment of indels')

    # write effective bed file
    if options.ptrim_bed and options.bedfile:
        tvcutils_command = "tvcutils validate_bed"
        tvcutils_command += ' --reference "%s"' % options.reference
        tvcutils_command += ' --target-regions-bed "%s"' % options.ptrim_bed
        tvcutils_command += ' --effective-bed "%s"' % os.path.join(options.outdir,'effective_regions.bed')
        RunCommand(tvcutils_command,'Write effective bed')

    # create command for long indel assembly and run
    long_indel_command =        'java -Xmx8G -cp %s/share/TVC/jar/ -jar %s/share/TVC/jar/GenomeAnalysisTK.jar' % (tvcrootdir,tvcrootdir)
    long_indel_command +=       '   -T IndelAssembly --bypassFlowAlign'
    long_indel_command +=       '   -R %s' % options.reference
    long_indel_command +=       '   -I %s' % options.bamfile
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

    unify_command =             '%s/share/TVC/scripts/unify_variants_and_annotations.py' % tvcrootdir
    #unify_command +=            '   --novel-tvc-vcf %s/small_variants.vcf' % options.outdir
    unify_command +=            '   --novel-tvc-vcf %s/small_variants.left.vcf' % options.outdir
    if os.path.exists("%s/indel_assembly.vcf" % options.outdir):
        unify_command +=        '   --novel-assembly-vcf %s/indel_assembly.vcf' % options.outdir
    if options.hotspot_vcf:
        unify_command +=        '   --hotspot-annotation-vcf %s' % options.hotspot_vcf
    unify_command +=            '   --output-vcf %s/all.merged.vcf' % options.outdir
    unify_command +=            '   --index-fai %s.fai' % options.reference
    if os.path.exists(options.outdir + '/tvc_metrics.json'):
        unify_command +=        '   --tvc-metrics %s/tvc_metrics.json' % options.outdir

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
        effectiveBedfile = "%s/effective_regions.bed" % options.outdir
        if os.path.exists(effectiveBedfile):
            bedtmp = effectiveBedfile
        else:
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

        if os.path.exists(options.outdir+'/all.recode.vcf'):
            RunCommand("cp   %s/all.recode.vcf   %s/TSVC_variants.vcf" % (options.outdir,options.outdir),
                       'Move final VCF into place')
        else:
            # Temporary workaround for cases where there are no variants left after filtering.
            # Just physically copy the header
            input = open('%s/all.merged.vcf' % options.outdir,'r')
            output = open('%s/TSVC_variants.vcf' % options.outdir,'w')
            for line in input:
                if line and line[0] == '#':
                    output.write(line)
            input.close()
            output.close()

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

