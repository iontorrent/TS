#!/usr/bin/python
# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved

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
    parser.add_option('-g', '--sample-name',      help='Sample Name for Test Sample (IR only, optional)', dest='testsamplename')
    parser.add_option('-r', '--reference-fasta',  help='FASTA file containing reference genome (required)', dest='reference')
    parser.add_option('-o', '--output-dir',       help='Output directory (default: current)', dest='outdir', default='.')
    parser.add_option('-p', '--parameters-file',  help='JSON file containing variant calling parameters. This file can be obtained from https://ampliseq.com for your panel. If not provided, default params will be used. for more information about parameters, please visit TVC 4.x Parameters Description (optional, recommended)', dest='paramfile')
    parser.add_option('-m', '--error-motifs',     help='System dependent motifs file helps improve variant calling accuracy. For Hi-Q chemistry use $TVC_ROOT_DIR/share/TVC/sse/ampliseqexome_germline_p1_hiq_motifset.txt else use $TVC_ROOT_DIR/share/TVC/sse/motifset.txt (optional)', dest='errormotifsfile')
    parser.add_option('-N', '--num-threads',      help='Set TVC number of threads (default: 12)', dest='numthreads',default='12')
    parser.add_option('-B', '--bin-dir',          help='Directory path to location of variant caller programs. Defaults to the directory this script is located', dest='bindir')
    (options, args) = parser.parse_args()
    

    # Sort out directories and locations of the binaries

    if not os.path.exists(options.outdir):
        os.makedirs(options.outdir)
    
    if options.bindir:
        bin_dir = options.bindir
    else:
        bin_dir = os.path.dirname(os.path.realpath(__file__))
    bin_dir = os.path.normpath(bin_dir)
    
    path_to_tvc = os.path.join(bin_dir,'tmol')
    if not os.path.exists(path_to_tvc):
        path_to_tvc = 'tmol'
    path_to_sort_vcf = os.path.join(bin_dir,'sort_vcf.py')
    if not os.path.exists(path_to_sort_vcf):
        path_to_sort_vcf = 'sort_vcf.py'
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
    else:
        printtime('ERROR: Hotspot file is not provided. Analysis requires hotspot file.')
        sys.exit(1)

    if options.bedfile:
        if not os.path.exists(options.bedfile):
            printtime('ERROR: No target regions bed file found at: ' + options.bedfile)
            sys.exit(1)
    else:
        printtime('ERROR: Target regions file is not provided. Analysis requires target regions file.')
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
    
   
    # TVC
    printtime('Calling small INDELs and SNPs using tmol ...')
    
    meta_tvc_args = parameters.get('meta',{}).get('tvcargs','tmol')
    if meta_tvc_args == 'tmol':
        tvc_command =   path_to_tvc
    else:
        tvc_command =   meta_tvc_args

    tvc_command = 'export LD_LIBRARY_PATH=%s/../lib:${LD_LIBRARY_PATH};%s' % (bin_dir,tvc_command)
    tvc_command +=      '   --fasta %s' % options.reference
    tvc_command += 		'   --bam %s' % (options.bamfile)
    tvc_command += 		'   --target %s' % options.bedfile
    tvc_command += 		'   --hotspot %s' % options.hotspot_vcf
    if int(parameters['torrent_variant_caller'].get('hotspots_only',1) or 0) == 0:
        tvc_command +=  '   --hs-mask-only'
    tvc_command +=  '   --oALT %s/tmol.stats.txt' % options.outdir
    tvc_command +=  '   --ofamily %s/tmol.family.txt' % options.outdir
    tvc_command +=  '   --oglobal %s/tmol.consensus.txt' % options.outdir
    tvc_command +=  '   --ostats %s/tmol.stats.txt' % options.outdir
    tvc_command +=  '   --ocalls %s/tmol_variants.vcf' % options.outdir
    tvc_command +=  '   --sample-name \"%s\"' % options.testsamplename
    tvc_command +=  '   --fam-size %s' % parameters['torrent_variant_caller'].get('min_fam_size','3')
    tvc_command +=  '   --min-num-fam %s' % parameters['torrent_variant_caller'].get('min_var_fam','2')
    tvc_command +=  '   --func-cov %s' % parameters['torrent_variant_caller'].get('min_func_cov','1500')
    tvc_command +=  '   --func-maf %s' % parameters['torrent_variant_caller'].get('min_func_maf','0.0005')	
	
    RunCommand(tvc_command,'Call small indels and SNPs')
    
    #Sort VCF
    sort_vcf_command = path_to_sort_vcf
    sort_vcf_command += ' --input-vcf %s/tmol_variants.vcf' % options.outdir
    sort_vcf_command += ' --output-vcf %s/small_variants.vcf' % options.outdir
    sort_vcf_command += ' --index-fai %s.fai' % options.reference
 
    RunCommand(sort_vcf_command,'Sort vcf file')
 
    #Annotate hotspots
    unify_command     = path_to_tvcutils + ' unify_vcf'
    unify_command     +=    '   --novel-tvc-vcf %s/small_variants.vcf' % options.outdir
    if options.hotspot_vcf:
        unify_command +=    '   --hotspot-annotation-vcf "%s"' % options.hotspot_vcf
    unify_command     +=    '   --output-vcf %s/TSVC_variants.vcf' % options.outdir
    unify_command     +=    '   --reference-fasta %s' % options.reference
    if options.bedfile:
        unify_command +=    '   --target-file "%s"' % options.bedfile

    RunCommand(unify_command, 'Unify variants and annotations from all sources (tmol,hotpots)')

    # Generate uncompressed vcf file
    #RunCommand('gzip -dcf "%s/TSVC_variants.vcf.gz" > "%s/TSVC_variants.vcf"' % (options.outdir,options.outdir), 'Generate uncompressed vcf')
    
  
if __name__ == '__main__':
    main()
