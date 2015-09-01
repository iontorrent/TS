#!/usr/bin/python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import re
import os
import subprocess
import time
import json
from optparse import OptionParser

def printtime(message, *args):
    if args:
        message = message % args
    print "[ " + time.strftime('%a %Y-%m-%d %X %Z') + " ] " + message
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




def main():
    parser = OptionParser()
    parser.add_option('-b', '--region-bed',       help='BED file specifing regions over which variant calls will be limited or filtered to', dest='bedfile') 
    parser.add_option('-s', '--hotspot-vcf',      help='VCF file specifying exact hotspot positions. TVC will force an evaluation for the alleles specified at the hotspot positions in this VCF file. For details, please visit Hotspot Variants (optional)', dest='hotspot_vcf')
    parser.add_option('-i', '--input-bam',        help='Input BAM file(s) containing aligned reads. Multiple file names must be concatenated with commas (required)', dest='bamfile')
    parser.add_option('-n', '--normal-bam',       help='BAM file(s) containing aligned reads for normal (reference) sample (IR only, optional)', dest='normbamfile')
    parser.add_option('-g', '--sample-name',      help='Sample Name for Test Sample (IR only, optional)', dest='testsamplename')
    parser.add_option('-r', '--reference-fasta',  help='FASTA file containing reference genome (required)', dest='reference')
    parser.add_option('-o', '--output-dir',       help='Output directory (default: current)', dest='outdir', default='.')
    parser.add_option('-p', '--parameters-file',  help='JSON file containing variant calling parameters. This file can be obtained from https://ampliseq.com for your panel. If not provided, default params will be used. for more information about parameters, please visit TVC 4.x Parameters Description (optional, recommended)', dest='paramfile')
    parser.add_option('-m', '--error-motifs',     help='System dependent motifs file helps improve variant calling accuracy. For Hi-Q chemistry use $TVC_ROOT_DIR/share/TVC/sse/ampliseqexome_germline_p1_hiq_motifset.txt else use $TVC_ROOT_DIR/share/TVC/sse/motifset.txt (optional)', dest='errormotifsfile')
    parser.add_option('-N', '--num-threads',      help='Set TVC number of threads (default: 12)', dest='numthreads',default='12')
    parser.add_option('-B', '--bin-dir',          help='Directory path to location of variant caller programs. Defaults to the directory this script is located', dest='bindir')
    parser.add_option('-t', '--tvc-root-dir',     help='Directory path to TVC root directory', dest='tvcrootdir')
    parser.add_option('-G', '--generate-gvcf',    help='Request generation of gvcf file in addition to vcf (on/off, default off)', dest='generate_gvcf', default='off')
    parser.add_option(      '--primer-trim-bed',  help='Perform primer trimming using provided unmerged BED file. (optional, recommended for ampliseq)', dest='ptrim_bed')
    parser.add_option(      '--postprocessed-bam',help='If provided, a primer trimmed BAM file will be produced for IGV viewing. This file does not contain the flow space data and should not be used as an input to TVC. Use of this option may increase TVC run time (optional, not recommended)', dest='postprocessed_bam')
    (options, args) = parser.parse_args()
    

    #parser.add_option('-B', '--build-dir',      help='Directory path to location of IR Build. Defaults to the directory this script is located', dest='builddir', default=os.path.realpath(__file__)) 
    #parser.add_option('-m', '--motifs-file', help='File containing motif set', dest='motifsfile')
    #parser.add_option('-s', '--hotspot-file',    help='VCF or BED file specifying exact hotspot positions', dest='hotspot_vcf')
    #parser.add_option('-c', '--contig-name',    help='Name of the Contig over which variant calls will be limited', dest='contigname')



    # Sort out directories and locations of the binaries

    if not os.path.exists(options.outdir):
        os.makedirs(options.outdir)
    
    if options.tvcrootdir:
        bin_dir = os.path.join(options.tvcrootdir, 'bin')
    elif options.bindir:
        bin_dir = options.bindir
    else:
        bin_dir = os.path.dirname(os.path.realpath(__file__))
    bin_dir = os.path.normpath(bin_dir)
    
    path_to_tvc = os.path.join(bin_dir,'tvc')
    if not os.path.exists(path_to_tvc):
        path_to_tvc = 'tvc'
    path_to_tvcassembly = os.path.join(bin_dir,'tvcassembly')
    if not os.path.exists(path_to_tvcassembly):
        path_to_tvcassembly = 'tvcassembly'
    path_to_tvcutils = os.path.join(bin_dir,'tvcutils')
    if not os.path.exists(path_to_tvcutils):
        path_to_tvcutils = 'tvcutils'


    # Verify that all pre-conditions are met
    
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
    for bam_filename in options.bamfile.split(','):
        if not os.path.exists(bam_filename):
            printtime('ERROR: No bam file found at: ' + bam_filename)
            sys.exit(1)
        if not os.path.exists(bam_filename+'.bai'):
            printtime('ERROR: No bam index file found at: ' + bam_filename + '.bai')
            sys.exit(1)
    if options.hotspot_vcf:
        if not os.path.exists(options.hotspot_vcf):
            printtime('ERROR: No hotspots vcf file found at: ' + options.hotspot_vcf)
            sys.exit(1)
    if options.ptrim_bed:
        if not os.path.exists(options.ptrim_bed):
            printtime('ERROR: No primer trim bed file found at: ' + options.ptrim_bed)
            sys.exit(1)
    if options.bedfile:
        if not os.path.exists(options.bedfile):
            printtime('ERROR: No target regions bed file found at: ' + options.bedfile)
            sys.exit(1)


    # write effective bed file
    if options.ptrim_bed and not options.bedfile:
        options.bedfile = options.ptrim_bed
        
    if options.ptrim_bed:
        tvcutils_command = path_to_tvcutils + " validate_bed"
        tvcutils_command += ' --reference "%s"' % options.reference
        tvcutils_command += ' --target-regions-bed "%s"' % options.ptrim_bed
        tvcutils_command += ' --effective-bed "%s"' % os.path.join(options.outdir,'effective_regions.bed')
        RunCommand(tvcutils_command,'Write effective bed')

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
    
   
    # TVC
    printtime('Calling small INDELs and SNPs using tvc ...')
    
    meta_tvc_args = parameters.get('meta',{}).get('tvcargs','tvc')
    if meta_tvc_args == 'tvc':
        tvc_command =   path_to_tvc
    else:
        tvc_command =   meta_tvc_args
    tvc_command +=      '   --output-dir %s' % options.outdir
    tvc_command +=      '   --reference %s' % options.reference
    if options.numthreads:
        tvc_command +=  '   --num-threads %s' % options.numthreads
    if options.ptrim_bed:
        tvc_command +=  '   --target-file %s' % options.ptrim_bed
        tvc_command +=  '   --trim-ampliseq-primers on'
    elif options.bedfile:
        tvc_command +=  '   --target-file %s' % options.bedfile
    if options.postprocessed_bam:
        #postprocessed_bam_tmp = options.postprocessed_bam + '.tmp.bam'
        #tvc_command +=  '   --postprocessed-bam %s' % postprocessed_bam_tmp
        tvc_command +=  '   --postprocessed-bam %s' % options.postprocessed_bam
    if options.paramfile:
        tvc_command +=  ' --parameters-file %s' % options.paramfile
    if options.errormotifsfile:
        tvc_command +=  '  --error-motifs %s' % options.errormotifsfile
    if options.normbamfile:
         tvc_command += '   --input-bam %s,%s' % ((options.bamfile, options.normbamfile))
         tvc_command += '   --sample-name "%s"' % (options.testsamplename)    
    else:
         tvc_command += '   --input-bam %s' % (options.bamfile)
    tvc_command +=      '   --output-vcf small_variants.vcf'
    if options.hotspot_vcf:
        tvc_command +=  '   --input-vcf %s' % options.hotspot_vcf
    
    if parameters and parameters['torrent_variant_caller'].get('process_input_positions_only','0') == '1' and parameters['freebayes'].get('use_input_allele_only','0') == '1':
        
        tvc_command +=  '   --process-input-positions-only on'
        tvc_command +=  '   --use-input-allele-only on'
        RunCommand(tvc_command, 'Call Hotspots Only')

    else:
        RunCommand(tvc_command,'Call small indels and SNPs')

        long_indel_command      =   path_to_tvcassembly
        long_indel_command     +=   '   --reference %s' % options.reference
        if options.normbamfile:
            long_indel_command +=   '   --input-bam %s,%s' % ((options.bamfile, options.normbamfile))
            long_indel_command +=   '   --sample-name "%s"' % (options.testsamplename)    
        else:
            long_indel_command +=   '   --input-bam %s' % (options.bamfile)
        if options.bedfile:
            long_indel_command +=   '   --target-file %s' % options.bedfile
        long_indel_command +=       '   --output-vcf %s/indel_assembly.vcf' % options.outdir
        if options.paramfile:
            long_indel_command +=   '   --parameters-file %s' % options.paramfile
        RunCommand(long_indel_command, "Assemble long indels")
        
    
    if options.postprocessed_bam:
        #bamsort_command = 'samtools sort -m 2G -l1 -@6 %s %s' % (postprocessed_bam_tmp, options.postprocessed_bam[:-4])
        #RunCommand(bamsort_command,'Sort postprocessed bam')
        bamindex_command = 'samtools index %s' % options.postprocessed_bam
        RunCommand(bamindex_command,'Index postprocessed bam')
        #RunCommand('rm -f ' + postprocessed_bam_tmp, 'Remove unsorted postprocessed bam')

    if options.generate_gvcf == "on":
        unify_command  =    'samtools depth'
        if options.bedfile != None:
            unify_command +=    '   -b ' + options.bedfile
        if options.postprocessed_bam:
            unify_command +=    '   ' + options.postprocessed_bam
        else:
            unify_command +=    '   ' + ' '.join(options.bamfile.split(','))
        unify_command +=    ' | '
    else:
        unify_command = ''

    unify_command     += path_to_tvcutils + ' unify_vcf'
    unify_command     +=    '   --novel-tvc-vcf %s/small_variants.vcf' % options.outdir
    if os.path.exists("%s/indel_assembly.vcf" % options.outdir):
        unify_command +=    '   --novel-assembly-vcf %s/indel_assembly.vcf' % options.outdir
    if options.hotspot_vcf:
        unify_command +=    '   --hotspot-annotation-vcf "%s"' % options.hotspot_vcf
    unify_command     +=    '   --output-vcf %s/TSVC_variants.vcf.gz' % options.outdir
    unify_command     +=    '   --reference-fasta %s' % options.reference
    if os.path.exists(options.outdir + '/tvc_metrics.json'):
        unify_command +=    '   --tvc-metrics %s/tvc_metrics.json' % options.outdir
    if options.bedfile:
        unify_command +=    '   --target-file "%s"' % options.bedfile
    if options.generate_gvcf == "on":
        unify_command +=    '    --input-depth stdin'
        if parameters and 'gen_min_coverage' in parameters.get('freebayes', {}):
            unify_command +='    --min-depth ' + str(parameters['freebayes']['gen_min_coverage']) 
            

    RunCommand(unify_command, 'Unify variants and annotations from all sources (tvc,IndelAssembly,hotpots)')

    # Generate uncompressed vcf file
    RunCommand('gzip -dcf "%s/TSVC_variants.vcf.gz" > "%s/TSVC_variants.vcf"' % (options.outdir,options.outdir), 'Generate uncompressed vcf')
    
    if options.generate_gvcf == "on" and os.path.exists(options.outdir+'/TSVC_variants.genome.vcf.gz'):
        RunCommand('gzip -dcf "%s/TSVC_variants.genome.vcf.gz" > "%s/TSVC_variants.genome.vcf"' % (options.outdir,options.outdir), 'Generate uncompressed gvcf')



if __name__ == '__main__':
    main()
