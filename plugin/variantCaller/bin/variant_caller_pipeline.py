#!/usr/bin/python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
import subprocess
import time
import json
import pandas as pd
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


# -------------------------------------------------------------------------
# splits argument strings in the torrent suite format into (arg_name, arg_value) tuples

def ion_argstr_to_tuples(arg_str):
    
    tuple_list = []
    try:
        arg_words = arg_str.split()
        index = 0
    
        while index < len(arg_words):
            # stand alone command like mapall, or stage1
            if not arg_words[index].startswith('-'):
                tuple_list.append((arg_words[index], ''))
            # argument name, value pair - could be split by space or '='
            else:
                argval = arg_words[index].split('=')
                if len(argval)>1:
                    tuple_list.append((argval[0], '='.join(argval[1])))
                elif index<len(arg_words)-1 and not arg_words[index+1].startswith('-'):
                    tuple_list.append((arg_words[index], arg_words[index+1]))
                    index+=1
                else:
                    tuple_list.append((arg_words[index], ''))
            index += 1
    except:
        pass
    return tuple_list


# -------------------------------------------------------------------------
# tmap call has the structure:
# [path to executable] [mapping algorithm] [global options] [flowspace options] [stage[0-9]+ [stage options] [algorithm [algorithm options]]+]+

def get_consensus_tmap_command(options, parameters, input_bam):
    
    # 1) Get path to tmap executable
    path_to_tmap = 'tmap'
    if options.path_to_tmap:
        path_to_tmap = options.path_to_tmap
    else:
        json_tmap_args = parameters.get('meta',{}).get('tmapargs','tmap').split()
        if len(json_tmap_args)>0:
            path_to_tmap = json_tmap_args[0]
    
    # 2) Get command line arguments from consensus.json file (does not include executable call)
    with open(os.path.join(options.outdir,'consensus.json'), 'r') as cj_in:
        consensus_json = json.load(cj_in)
    
    tmap_arg_tuples = ion_argstr_to_tuples(consensus_json.get('tmap',{}).get('command_line',{}))
    if len(tmap_arg_tuples) < 4:
        printtime('WARNING: Not able to read valid tmap command line from consensus.json: ' + os.path.join(options.outdir,'consensus.json'))
        sys.exit(1)
    
    # executable path plus mapping algorithm plus global options input bam, numthreads
    tmap_command  = path_to_tmap + ' ' + tmap_arg_tuples[0][0]
    tmap_command += ' -f "' + options.reference + '"'
    tmap_command += ' -r "' + input_bam + '" -n ' + str(options.numthreads)
    
    # Strip and update tmap arguments regarding input files, etc.
    for tp in tmap_arg_tuples[1:]:
        # Remove some input gloabl options from command string
        if tp[0] in ['-r','--fn-reads',    '-s','--fn-sam',
                     '-n','--num-threads', '-f','--fn-fasta', 
                     '-k','--shared-memory-key']:
            continue
        # Change path to bed file in place, if applicable
        elif tp[0] in ['--bed-file']:
            if options.ptrim_bed:
                tmap_command += ' --bed-file "' + options.ptrim_bed + '"'
            elif options.bedfile:
                tmap_command += ' --bed-file "' + options.bedfile + '"'
        
        # And other other options in their original order
        tmap_command += ' ' + ' '.join(tp).rstrip()
    
    return tmap_command
    

# -------------------------------------------------------------------------
# The call to the consensus exectable creates two unsorted bam files, one in need for realignment
# name extensions are hardcoded inside executable 
# 1) "consensus_bam_name".aln_not_needed.bam
# 2) "consensus_bam_name".aln_needed.bam

def consensus_alignment_pipeline(options, parameters, consensus_bam_name, remove_tmp_files):
    
    # 1) Sort "consensus_bam_name".aln_not_needed.bam
    command  = 'samtools sort -m 1000M -l1 -@' + options.numthreads + ' -T "' + consensus_bam_name + '.sort.tmp"'
    command += ' -o "' + consensus_bam_name + '.aln_not_needed.sorted.bam" '
    command += '"' + consensus_bam_name + '.aln_not_needed.bam"'
    RunCommand(command,"Sorting first partial consensus BAM.")
    if remove_tmp_files:
        try:
            os.remove(consensus_bam_name + '.aln_not_needed.bam')
        except:
            print('WARNING: Unable to delete file %s' % (consensus_bam_name + '.aln_not_needed.bam'))
    
    # 2) Align and sort "consensus_bam_name".aln_needed.bam
    command  = get_consensus_tmap_command(options, parameters, consensus_bam_name + '.aln_needed.bam')
    command += ' | samtools sort -m 1000M -l1 -@' + options.numthreads + ' -T "' + consensus_bam_name + '.sort.tmp"'
    command += ' -o "' + consensus_bam_name + '.aligned.sorted.bam" '
    RunCommand(command,"Aligning and sorting second partial consensus BAM.")
    if remove_tmp_files:
        try:
            os.remove(consensus_bam_name + '.aln_needed.bam')
        except:
            print('WARNING: Unable to delete file %s' % (consensus_bam_name + '.aln_needed.bam'))
    
    # 3) Merging the partial BAM files into one
    final_consensus_bam = consensus_bam_name + '.bam'
    command  = 'samtools merge -l1 -@' + options.numthreads + ' -c -p -f "' + final_consensus_bam + '"'
    command += ' "' +  consensus_bam_name + '.aln_not_needed.sorted.bam" "' +  consensus_bam_name +'.aligned.sorted.bam"'
    RunCommand(command,"Merging aligning partial consensus BAM files.")
    # And finally indexing consensus bam
    RunCommand('samtools index "'+final_consensus_bam+'"','Indexing merged consensus bam')
    
    if remove_tmp_files:
        try:
            os.remove(consensus_bam_name + ".aln_not_needed.sorted.bam")
            os.remove(consensus_bam_name + ".aligned.sorted.bam")
        except:
            print('WARNING: Unable to delete files %s' % (consensus_bam_name + "*.sorted.bam"))


# -------------------------------------------------------------------------
# write tvc consensus coverage metrics into text files
# TODO: use json files instead of text files

def create_consensus_metrics(options, parameters):
    
    targets_depth_file = os.path.join(options.outdir, 'targets_depth.txt') 
    df = pd.read_csv(targets_depth_file, sep = '\t')
    consensus_metrics = os.path.join(options.outdir, 'consensus_metrics.txt') 
    outFileFW = open(consensus_metrics, 'w')
    outFileFW.write("Median read coverage:%s\n" % df['read_depth'].median())
    outFileFW.write("Median molecular coverage:%s\n" %df['family_depth'].median())
    outFileFW.write("20th percentile read coverage:%s\n" % df['read_depth'].quantile(0.2))
    outFileFW.write("20th percentile molecular coverage:%s\n" % df['family_depth'].quantile(0.2))
    if 'torrent_variant_caller' in parameters:
        min_var_coverage = int(parameters['torrent_variant_caller'].get('snp_min_var_coverage', 3))
        if int(parameters['torrent_variant_caller'].get('use_fd_param', 0)):
            min_var_coverage = int(parameters['torrent_variant_caller'].get('fd_10', min_var_coverage).get('min_var_coverage', min_var_coverage))
        min_var_coverage = max(min_var_coverage, 1)
        try:
            lod = round(100.0 * (min_var_coverage - 0.5) / df['family_depth'].median(), 4)
        except ZeroDivisionError:
            lod = 100.0
        #CZB: The lod formula is totally heuristic and can not handle some edge cases. I have to fix it manually.
        lod = min(lod, 100.0)
        outFileFW.write("Median LOD percent:% 2.4f \n" % lod)
        try:
            lod = round(100.0 * (min_var_coverage - 0.5) / df['family_depth'].quantile(0.2), 4)
        except ZeroDivisionError:
            lod = 100.0
        lod = min(lod, 100.0)
        outFileFW.write("80th percentile LOD percent:% 2.4f \n" % lod)
    outFileFW.close()


# -------------------------------------------------------------------------
# Have the pipeline create a samtools depth file in case tvc does not do it

def create_depth_txt(options, out_file):
    cmdstr = 'samtools depth'
    if options.bedfile != None:
        cmdstr +=    '   -b "' + options.bedfile + '"'
    if options.postprocessed_bam:
        cmdstr +=    '   "' + options.postprocessed_bam + '"'
    else:
        cmdstr +=    '   ' + ' '.join(options.bamfile.split(','))
    cmdstr += ' > "' + out_file + '"'
    RunCommand(cmdstr,'Generating samtools depth file')
    

# -------------------------------------------------------------------------
# Run "tcvutils unify_vcf" to merge and post process vcf records

def run_tvcutils_unify(options, parameters):
    
    # Get json specified args and replace executable with command line specified executable path
    json_unify_args = parameters.get('meta',{}).get('unifyargs', '').split()
    if len(json_unify_args)<2 or json_unify_args[1] != 'unify_vcf':
        json_unify_args = ['tvcutils', 'unify_vcf']
    if options.path_to_tvcutils:
        json_unify_args[0] = options.path_to_tvcutils
    unify_command = ' '.join(json_unify_args)
    
    unify_command     +=    '   --novel-tvc-vcf "%s/small_variants.vcf"' % options.outdir
    unify_command     +=    '   --output-vcf "%s/TSVC_variants.vcf"' % options.outdir
    unify_command     +=    '   --reference-fasta "%s"' % options.reference
        
    if os.path.isfile("%s/indel_assembly.vcf" % options.outdir):
        unify_command +=    '   --novel-assembly-vcf "%s/indel_assembly.vcf"' % options.outdir
    if options.hotspot_vcf:
        unify_command +=    '   --hotspot-annotation-vcf "%s"' % options.hotspot_vcf
    if os.path.isfile(options.outdir + '/tvc_metrics.json'):
        unify_command +=    '   --tvc-metrics "%s/tvc_metrics.json"' % options.outdir
    if options.bedfile:
        unify_command +=    '   --target-file "%s"' % options.bedfile
    if options.generate_gvcf == "on":
        unify_command +=    '    --input-depth "%s/depth.txt"' % options.outdir
        if parameters and 'gen_min_coverage' in parameters.get('freebayes', {}):
            unify_command +='    --min-depth ' + str(parameters['freebayes']['gen_min_coverage']) 
            
    RunCommand(unify_command, 'Unify variants and annotations from all sources (tvc,IndelAssembly,hotpots)')


# -------------------------------------------------------------------------
# Straighten out which executables to use. The order of precedence is 
# 1) path explicity supplied through script options (regardless whether executable exists)*
# 2) non-standard executable^ supplied through parameter json (regardless whether executable exists)*
# 3) executable residing in "bin-dir" (if it exists)
# 4) system installed executable
# *) We want to hard error out if an explicitly supplied executable does mnot exists.
# ^) A non standard executable is, e.g., '/path/to/my_tmap', as opposed to system 'tmap'

def get_path_to_executable(parameters, system_exec, json_args, bin_dir):
    
    json_cmd = parameters.get('meta',{}).get(json_args,'').split()
    if json_cmd and json_cmd[0] != system_exec: # 2)
        return json_cmd[0] 
    elif bin_dir and os.path.isfile(os.path.join(bin_dir, system_exec)): #3)
        return os.path.join(bin_dir,system_exec)
    else:
        return system_exec #4)

# ===============================================================================

def main():
    
    # Parse and set options 
    parser = OptionParser()
    
    parser.add_option('-b', '--region-bed',       help='BED file specifing regions over which variant calls will be limited or filtered to', dest='bedfile') 
    parser.add_option('-s', '--hotspot-vcf',      help='VCF file specifying exact hotspot positions. TVC will force an evaluation for the alleles specified at the hotspot positions in this VCF file. For details, please visit Hotspot Variants (optional)', dest='hotspot_vcf')
    parser.add_option('-i', '--input-bam',        help='Input BAM file(s) containing aligned reads. Multiple file names must be concatenated with commas (required)', dest='bamfile')
    parser.add_option('-n', '--normal-bam',       help='BAM file(s) containing aligned reads for normal (reference) sample (IR only, optional)', dest='normbamfile')
    parser.add_option('-g', '--sample-name',      help='Sample Name for Test Sample (IR only, optional)', dest='testsamplename')
    parser.add_option('-r', '--reference-fasta',  help='FASTA file containing reference genome (required)', dest='reference')
    parser.add_option('-o', '--output-dir',       help='Output directory (default: current)', dest='outdir', default='.')
    parser.add_option('-p', '--parameters-file',  help='JSON file containing variant calling parameters. This file can be obtained from https://ampliseq.com for your panel. If not provided, default params will be used. for more information about parameters, please visit TVC 4.x Parameters Description (optional, recommended)', dest='paramfile')
    parser.add_option(      '--error-motifs-dir', help='Directory for error-motifs files', dest='errormotifsdir')
    parser.add_option('-m', '--error-motifs',     help='System dependent motifs file helps improve variant calling accuracy. For Hi-Q chemistry use $TVC_ROOT_DIR/share/TVC/sse/ampliseqexome_germline_p1_hiq_motifset.txt else use $TVC_ROOT_DIR/share/TVC/sse/motifset.txt (optional)', dest='errormotifsfile')
    parser.add_option('-e', '--sse-vcf',     help='strand-specific systematic error vcf (optional)', dest='sse_vcf')
    parser.add_option('-N', '--num-threads',      help='Set TVC number of threads (default: 12)', dest='numthreads',default='12')
    parser.add_option('-G', '--generate-gvcf',    help='Request generation of gvcf file in addition to vcf (on/off, default off)', dest='generate_gvcf', default='off')
    parser.add_option(      '--primer-trim-bed',  help='Perform primer trimming using provided unmerged BED file. (optional, recommended for ampliseq)', dest='ptrim_bed')
    parser.add_option(      '--postprocessed-bam',help='If provided, a primer trimmed BAM file will be produced for IGV viewing. This file does not contain the flow space data and should not be used as an input to TVC. Use of this option may increase TVC run time (optional, not recommended)', dest='postprocessed_bam')
    parser.add_option(      '--run-consensus',    help='Run consensus to compress the bam file for molecular tagging (on/off, default off).', dest='run_consensus', default='off')
    
    # Paths to Executables - new options in TS 5.4
    parser.add_option(      '--bin-tvc',         help='Path to tvc executable. Defaults to the system tvc.', dest='path_to_tvc')
    parser.add_option(      '--bin-tvcutils',    help='Path to tvcutils executable. Defaults to the system tvcutils.', dest='path_to_tvcutils')
    parser.add_option(      '--bin-tmap',        help='Path to tmap executable. Defaults to the system tmap.', dest='path_to_tmap')
    
    # We seem to still need this in TS 5.4 because i can't detangle the plugin code
    parser.add_option('-B', '--bin-dir',          help='DEPRECATED: Directory path to location of variant caller programs. Defaults to the directory this script is located', dest='bindir')
    parser.add_option('-t', '--tvc-root-dir',     help='DEPRECATED: Directory path to TVC root directory', dest='tvcrootdir')

    (options, args) = parser.parse_args()
    
    
    # -----------------------------------------------------------------------------------------
    # Load parameters json
    parameters = {}
    if options.paramfile:
        try:
            with open(options.paramfile, 'r') as json_file:
                parameters = json.load(json_file)
            if 'pluginconfig' in parameters:
                parameters = parameters['pluginconfig']
        except:
            printtime('ERROR: No parameter file found at: ' + options.paramfile)
            sys.exit(1)
            
    
    # -----------------------------------------------------------------------------------------
    # Straighten out which executables to use. The order of precedence is 
    # 1) path explicity supplied through script options (regardless whether executable exists)*
    # 2) non-standard executable^ supplied through parameter json (regardless whether executable exists)*
    # 3) executable residing in "bin-dir" (if it exists)
    # 4) system installed executable
    # *) We want to hard error out if an explicitly supplied executable does mnot exists.
    # ^) A non standard executable is, e.g., '/path/to/my_tmap', as opposed to system 'tmap'
    
    # Get executable directory
    bin_dir = ''
    if options.tvcrootdir:
        #printtime('WARNING: Option --tvc-root-dir is DEPRECATED and will be removed in a future release.')
        bin_dir = os.path.normpath(os.path.join(options.tvcrootdir, 'bin'))
    elif options.bindir:
        #printtime('WARNING: Option --bin-dir is DEPRECATED and will be removed in a future release.')
        bin_dir = os.path.normpath(options.bindir)
        
    # Get path to executables
    if not options.path_to_tvc: # 1)
        options.path_to_tvc =  get_path_to_executable(parameters, 'tvc', 'tvcargs', bin_dir)
    if not options.path_to_tvcutils:
        options.path_to_tvcutils =  get_path_to_executable(parameters, 'tvcutils', 'unifyargs', bin_dir)
    if not options.path_to_tmap:
        options.path_to_tmap =  get_path_to_executable(parameters, 'tmap', 'tmapargs', bin_dir)
    
    # And give some feedback about executables being used
    printtime('Using tvc      binary: ' + options.path_to_tvc)
    printtime('Using tvcutils binary: ' + options.path_to_tvcutils)
    printtime('Using tmap     binary: ' + options.path_to_tmap)
    
    
    # -----------------------------------------------------------------------------------------
    # Verify that all pre-conditions are met
    
    if not options.bamfile or not options.reference:
        parser.print_help()
        exit(1)
        
    multisample = (options.bamfile.find(",") != -1)
    if options.run_consensus.lower() == 'on' and multisample:
        printtime('ERROR: consensus currently does not support multisample runs.')
        sys.exit(1) # No need to waste time
    
    if not os.path.exists(options.outdir):
        os.makedirs(options.outdir)
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
        if (not os.path.exists(bam_filename)) or (not bam_filename.endswith('.bam')):
            printtime('ERROR: No bam file found at: ' + bam_filename)
            sys.exit(1)
        # If there is no index we try to simply index the existing bam before the workflow start
        if not os.path.exists(bam_filename+'.bai'):
            bamindex_command = 'samtools index "%s"' % bam_filename
            RunCommand(bamindex_command,('Index input bam '+bam_filename))
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


    # write effective_regions.bed file
    if options.ptrim_bed and not options.bedfile:
        options.bedfile = options.ptrim_bed
        
    if options.ptrim_bed:        
        tvcutils_command = options.path_to_tvcutils + " validate_bed"
        tvcutils_command += ' --reference "%s"' % options.reference
        tvcutils_command += ' --target-regions-bed "%s"' % options.ptrim_bed
        tvcutils_command += ' --effective-bed "%s"' % os.path.join(options.outdir,'effective_regions.bed')
        RunCommand(tvcutils_command,'Write effective bed')


    # -----------------------------------------------------------------------------------------
    # tvc consensus call and related pipeline operations
    tvc_input_bam = options.bamfile
    
    if options.run_consensus.lower() == 'on':
        
        printtime('Generating consensus bam file using consensus ...')
        
        bam_basename = os.path.basename(options.bamfile)
        consensus_bam = bam_basename[0:-4] + '_consensus'
        tvc_input_bam = os.path.join(options.outdir, consensus_bam+'.bam')
        
        # Get json specified args and replace executable with command line specified executable path
        json_consensus_args = parameters.get('meta',{}).get('consensusargs','').split()
        if len(json_consensus_args)<2 or json_consensus_args[1] != 'consensus':
            json_consensus_args = ['tvc', 'consensus']
        if options.path_to_tvc and json_consensus_args[0] == 'tvc':
            json_consensus_args[0] = options.path_to_tvc       
        consensus_command = ' '.join(json_consensus_args)
        
        consensus_command += '   --output-dir %s' % options.outdir
        consensus_command += '   --reference "%s"' % options.reference
        consensus_command += '   --num-threads 4'
        consensus_command += '   --input-bam "%s"' % (options.bamfile)
        consensus_command += '   --consensus-bam "%s"' % (consensus_bam)
        
        if options.ptrim_bed:
            consensus_command += '   --target-file "%s"' % options.ptrim_bed
        elif options.bedfile:
            consensus_command += '   --target-file "%s"' % options.bedfile
        if options.paramfile:
            consensus_command += ' --parameters-file "%s"' % options.paramfile

        RunCommand(consensus_command, 'Generate consensus bam file')
        
        # Alignment and merging of the two bam files produced by consensus
        consensus_alignment_pipeline(options, parameters, os.path.join(options.outdir, consensus_bam), True)
        
        #Generate Coverage Metrics for QC
        printtime('Generating coverage metrics') # TODO: Write json files rather than txt files
        create_consensus_metrics(options, parameters)
    
    # -----------------------------------------------------------------------------------------
    # TVC call and related pipeline operations
    
    printtime('Calling small INDELs and SNPs using tvc ...')
    
    # Get json specified args and replace executable with command line specified executable path
    json_tvc_args = parameters.get('meta',{}).get('tvcargs','').split()
    if len(json_tvc_args)<1:
        json_tvc_args = ['tvc']
    if options.path_to_tvc:
        json_tvc_args[0] = options.path_to_tvc      
    tvc_command = ' '.join(json_tvc_args)
    
    # Concatenate other command line args (json args take precedence)     
    tvc_command +=      '   --output-dir %s' % options.outdir
    tvc_command +=      '   --reference "%s"' % options.reference
    
    if options.normbamfile:
         tvc_command += '   --input-bam "%s","%s"' % ((options.bamfile, options.normbamfile))
         tvc_command += '   --sample-name "%s"' % (options.testsamplename)
    else:
        tvc_command +=  '   --input-bam "%s"' % tvc_input_bam
    
    if options.numthreads:
        tvc_command +=  '   --num-threads %s' % options.numthreads
    if options.ptrim_bed:
        tvc_command +=  '   --target-file "%s"' % options.ptrim_bed
        tvc_command +=  '   --trim-ampliseq-primers on'
    elif options.bedfile:
        tvc_command +=  '   --target-file "%s"' % options.bedfile
    if options.postprocessed_bam:
        tvc_command +=  '   --postprocessed-bam "%s"' % options.postprocessed_bam
    if options.paramfile:
        tvc_command +=  ' --parameters-file "%s"' % options.paramfile
    if options.errormotifsdir:
        tvc_command +=  '  --error-motifs-dir "%s"' % options.errormotifsdir
    if options.errormotifsfile:
        tvc_command +=  '  --error-motifs "%s"' % options.errormotifsfile
    tvc_command +=      '   --output-vcf "small_variants.vcf"'
    if options.hotspot_vcf:
        tvc_command +=  '   --input-vcf "%s"' % options.hotspot_vcf
    if options.sse_vcf:
        tvc_command +=  '   --sse-vcf "%s"' % options.sse_vcf
    if multisample:
        tvc_command += '   --heal-snps false'

    # --------------------------------------------------------
    # After creating the command line, we actually can run tvc now
    if parameters and parameters['torrent_variant_caller'].get('process_input_positions_only','0') == '1' and parameters['freebayes'].get('use_input_allele_only','0') == '1':       
        tvc_command +=  '   --process-input-positions-only on'
        tvc_command +=  '   --use-input-allele-only on'
        RunCommand(tvc_command, 'Call Hotspots Only')
    else:
        RunCommand(tvc_command,'Call small indels and SNPs')
    # Long indel assembly is done within tvc and needs not be called in a pipeline operation
    

    # -----------------------------------------------------------------------------------------
    # index a post processed bam file - no need to sort, we used an ordered BAM writer
    if options.postprocessed_bam:
        #bamsort_command = 'samtools sort -m 2G -l1 -@6 %s %s' % (postprocessed_bam_tmp, options.postprocessed_bam[:-4])
        #RunCommand(bamsort_command,'Sort postprocessed bam')
        bamindex_command = 'samtools index "%s"' % options.postprocessed_bam
        RunCommand(bamindex_command,'Index postprocessed bam')
        #RunCommand('rm -f ' + postprocessed_bam_tmp, 'Remove unsorted postprocessed bam')
        
    # -----------------------------------------------------------------------------------------
    # run tvcutils to merge and post process vcf records
    # tvcutils unify_vcf generates both, a compressed and a uncompressed vcf
    # Post processing settings can be adjusted through parameter json["meta"]["unifyargs"]
        
    if options.generate_gvcf == "on" and not os.path.isfile(options.outdir + '/depth.txt'):
        create_depth_txt(options, options.outdir + '/depth.txt')

    run_tvcutils_unify(options, parameters)

    # -----------------------------------------------------------------------------------------

    # run TS-14577 fixup
    swap_py = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'swap_duplicates.py')
    tvc_file = os.path.join(options.outdir, 'TSVC_variants.vcf')
    if options.bedfile and options.hotspot_vcf and os.path.isfile(tvc_file):

        tvc_file_tmp = os.path.join(options.outdir, 'TSVC_variants_tmp.vcf')
        RunCommand('mv ' + tvc_file + ' ' + tvc_file_tmp, 'Move')
        RunCommand('python ' + swap_py + ' ' + tvc_file_tmp + ' ' + options.bedfile + ' ' + tvc_file, 'Fix')
        if os.path.isfile(tvc_file):
            RunCommand('bgzip -c ' + tvc_file + ' > ' + tvc_file + '.gz', 'gzip')
            RunCommand('tabix -p vcf ' + tvc_file + '.gz', 'tabix')
        else:
            RunCommand('mv ' + tvc_file_tmp + ' ' + tvc_file, 'Move')

    tvc_file = os.path.join(options.outdir, 'TSVC_variants.genome.vcf')
    if options.bedfile and options.hotspot_vcf and os.path.isfile(tvc_file):

        tvc_file_tmp = os.path.join(options.outdir, 'TSVC_variants_tmp.genome.vcf')
        RunCommand('mv ' + tvc_file + ' ' + tvc_file_tmp, 'Move')
        RunCommand('python ' + swap_py + ' ' + tvc_file_tmp + ' ' + options.bedfile + ' ' + tvc_file, 'Fix')
        if os.path.isfile(tvc_file):
            RunCommand('bgzip -c '+ tvc_file + ' > ' + tvc_file + '.gz', 'gzip')
            RunCommand('tabix -p vcf ' + tvc_file + '.gz', 'tabix')
        else:
            RunCommand('mv ' + tvc_file_tmp + ' ' + tvc_file, 'Move')



    # -----------------------------------------------------------------------------------------

# =======================================================================================

if __name__ == '__main__':
    main()
