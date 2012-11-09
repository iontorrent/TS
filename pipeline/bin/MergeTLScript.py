#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

__version__ = filter(str.isdigit, "$Revision$")

import os
import sys
import subprocess
import argparse
import fnmatch
import traceback
import json

from ion.utils import blockprocessing
from ion.utils import explogparser
from ion.utils import sigproc
from ion.utils import basecaller
from ion.utils import alignment

from ion.utils.blockprocessing import printtime

from ion.utils.compress import make_zip

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('-s', '--do-sigproc', dest='do_sigproc', action='store_true', help='signal processing')
    parser.add_argument('-b', '--do-basecalling', dest='do_basecalling', action='store_true', help='base calling')
    parser.add_argument('-a', '--do-alignment', dest='do_alignment', action='store_true', help='alignment')
    parser.add_argument('-z', '--do-zipping', dest='do_zipping', action='store_true', help='zipping')

    args = parser.parse_args()

    if args.verbose:
        print "MergeTLScript:",args

    if not args.do_sigproc and not args.do_basecalling and not args.do_alignment and not args.do_zipping:
        parser.print_help()
        sys.exit(1)

    #ensure we permit read/write for owner and group output files.
    os.umask(0002)

    blockprocessing.printheader()
    env,warn = explogparser.getparameter()

    blockprocessing.write_version()
    sys.stdout.flush()
    sys.stderr.flush()

    blocks = explogparser.getBlocksFromExpLogJson(env['exp_json'], excludeThumbnail=True)
    dirs = ['block_%s' % block['id_str'] for block in blocks]


    if args.do_sigproc:

        merged_bead_mask_path = os.path.join(env['SIGPROC_RESULTS'], 'MaskBead.mask')

        sigproc.mergeSigProcResults(
            dirs,
            env['SIGPROC_RESULTS'],
            env['shortRunName'])


    if args.do_basecalling:
        # Only merge metrics and generate plots
        basecaller.merge_basecaller_stats(
            dirs,
            env['BASECALLER_RESULTS'],
            env['SIGPROC_RESULTS'],
            env['flows'],
            env['flowOrder'])
        ## Generate BaseCaller's composite "Big Data": unmapped.bam, sff, fastq. Totally optional
        if env.get('libraryName','') == 'none':        
            actually_merge_unmapped_bams = True
        else:
            actually_merge_unmapped_bams = False    
        actually_merge_bams_and_generate_sff_fastq = False
        if actually_merge_bams_and_generate_sff_fastq:
            basecaller.merge_basecaller_bigdata(
                dirs,
                env['BASECALLER_RESULTS'])
        elif actually_merge_unmapped_bams:
            basecaller.merge_basecaller_bam_only(
                dirs,
                env['BASECALLER_RESULTS'])

    if args.do_alignment and env['libraryName'] and env['libraryName']!='none':
        # Only merge metrics and generate plots
        alignment.merge_alignment_stats(
            dirs,
            env['BASECALLER_RESULTS'],
            env['ALIGNMENT_RESULTS'],
            env['flows'])
        ## Generate Alignment's composite "Big Data": mapped bam. Totally optional
        actually_merge_mapped_bams = True
        if actually_merge_mapped_bams:
            alignment.merge_alignment_bigdata(
                dirs,
                env['BASECALLER_RESULTS'],
                env['ALIGNMENT_RESULTS'],
                env['mark_duplicates'])

    if args.do_zipping:


        # This is a special procedure to create links with official names to all downloadable data files
        
        physical_file_prefix = 'rawlib'
        official_file_prefix = "%s_%s" % (env['expName'], env['resultsName'])
        
        link_src = [
            os.path.join(env['BASECALLER_RESULTS'], physical_file_prefix+'.basecaller.bam'),
            os.path.join(env['BASECALLER_RESULTS'], physical_file_prefix+'.sff'),
            os.path.join(env['BASECALLER_RESULTS'], physical_file_prefix+'.fastq'),
            os.path.join(env['ALIGNMENT_RESULTS'], physical_file_prefix+'.bam'),
            os.path.join(env['ALIGNMENT_RESULTS'], physical_file_prefix+'.bam.bai')]
        link_dst = [
            os.path.join(env['BASECALLER_RESULTS'], official_file_prefix+'.basecaller.bam'),
            os.path.join(env['BASECALLER_RESULTS'], official_file_prefix+'.sff'),
            os.path.join(env['BASECALLER_RESULTS'], official_file_prefix+'.fastq'),
            os.path.join(env['ALIGNMENT_RESULTS'], official_file_prefix+'.bam'),
            os.path.join(env['ALIGNMENT_RESULTS'], official_file_prefix+'.bam.bai')]
        
        for (src,dst) in zip(link_src,link_dst):
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.relpath(src,os.path.dirname(dst)),dst)
                except:
                    printtime("ERROR: Unable to symlink '%s' to '%s'" % (src, dst))
            

        libsff = "%s/%s_%s.sff" % (env['BASECALLER_RESULTS'], env['expName'], env['resultsName'])
        tfsff = "%s/%s_%s.tf.sff" % (env['BASECALLER_RESULTS'], env['expName'], env['resultsName'])
        fastqpath = "%s/%s_%s.fastq" % (env['BASECALLER_RESULTS'], env['expName'], env['resultsName'])
        libbam = "%s/%s_%s.bam" % (env['ALIGNMENT_RESULTS'], env['expName'], env['resultsName'])
        libbambai = "%s/%s_%s.bam.bai" % (env['ALIGNMENT_RESULTS'], env['expName'], env['resultsName'])
        tfbam = "%s/%s_%s.tf.bam" % (env['BASECALLER_RESULTS'], env['expName'], env['resultsName'])


        create_links = [
            ("rawtf.sff", tfsff),
            ("rawlib.bam", libbam),
            ("rawlib.bam.bai", libbambai),
            ("rawtf.bam", tfbam)
        ]
        for (src,dst) in create_links:
            if not os.path.exists(dst):
                try:
                    os.symlink(src,dst)
                except:
                    printtime("ERROR: Unable to symlink '%s' to '%s'" % (src, dst))
                    printtime(traceback.format_exc())

        ##################################################
        # Create zip of files
        ##################################################

        #sampled sff
        #make_zip(libsff.replace(".sff",".sampled.sff")+'.zip', libsff.replace(".sff",".sampled.sff"))

        #library sff
        make_zip(libsff + '.zip', libsff, arcname=libsff )

        #tf sff
        make_zip(tfsff + '.zip', tfsff, arcname=tfsff)

        #fastq zip
        make_zip(fastqpath + '.zip', fastqpath, arcname=fastqpath)

        #sampled fastq
        #make_zip(fastqpath.replace(".fastq",".sampled.fastq")+'.zip', fastqpath.replace(".fastq",".sampled.fastq"))

        ########################################################
        # barcode processing                                   #
        # Zip up and move sff, fastq, bam, bai files           #
        # Move zip files to results directory                  #
        ########################################################
        
        datasets_basecaller_path = os.path.join(env['BASECALLER_RESULTS'],"datasets_basecaller.json")
        datasets_basecaller = {}
    
        if os.path.exists(datasets_basecaller_path):
            try:
                f = open(datasets_basecaller_path,'r')
                datasets_basecaller = json.load(f);
                f.close()
            except:
                printtime("ERROR: problem parsing %s" % datasets_basecaller_path)
                traceback.print_exc()
        else:
            printtime("ERROR: %s not found" % datasets_basecaller_path)

        
        prefix_list = [dataset['file_prefix'] for dataset in datasets_basecaller.get("datasets",[])]
        
        if len(prefix_list) > 1:
            zip_task_list = [
                ('bam',             env['ALIGNMENT_RESULTS']),
                ('bam.bai',         env['ALIGNMENT_RESULTS']),
                ('basecaller.bam',  env['BASECALLER_RESULTS']),
                ('sff',             env['BASECALLER_RESULTS']),
                ('fastq',           env['BASECALLER_RESULTS']),]
            
            for extension,base_dir in zip_task_list:
                zipname = "%s_%s.barcode.%s.zip" % (env['expName'], env['resultsName'], extension)
                for prefix in prefix_list:
                    filename = os.path.join(base_dir, prefix+'.'+extension)
                    if os.path.exists(filename):
                        try:
                            make_zip(zipname, filename, arcname=filename)
                        except:
                            printtime("ERROR: target: %s" % filename)
                            traceback.print_exc()

        else:
            printtime("MergeTLScript: No barcode run")

    printtime("MergeTLScript exit")
    sys.exit(0)
