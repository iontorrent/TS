#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

__version__ = filter(str.isdigit, "$Revision$")

import os
import sys
import subprocess
import argparse
import fnmatch

from ion.utils import blockprocessing
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


    blockprocessing.printheader()
    env = blockprocessing.getparameter()

    blockprocessing.write_version()
    sys.stdout.flush()
    sys.stderr.flush()

    blocks = blockprocessing.getBlocksFromExpLog(env['exp_json'], excludeThumbnail=True)
    dirs = ['block_%s' % block['id_str'] for block in blocks]


    if args.do_sigproc:

        sigproc.mergeSigProcResults(
            dirs,
            env['pathToRaw'],
            env['skipchecksum'],
            env['SIGPROC_RESULTS'])



    if args.do_basecalling:

        QualityPath = os.path.join(env['BASECALLER_RESULTS'], 'quality.summary')
        libsff = "%s/%s_%s.sff" % (env['BASECALLER_RESULTS'], env['expName'], env['resultsName'])
        tfsff = "%s/%s_%s.tf.sff" % (env['BASECALLER_RESULTS'], env['expName'], env['resultsName'])
        merged_bead_mask_path = os.path.join(env['SIGPROC_RESULTS'], 'MaskBead.mask')

        basecaller.mergeBasecallerResults(
            dirs,
            QualityPath,
            merged_bead_mask_path,
            env['flowOrder'],
            libsff,
            tfsff,
            env['BASECALLER_RESULTS'])


    if args.do_alignment:

        alignment.mergeAlignmentResults(dirs, env, env['ALIGNMENT_RESULTS'])


    if args.do_zipping:

        libsff = "%s/%s_%s.sff" % (env['BASECALLER_RESULTS'], env['expName'], env['resultsName'])
        tfsff = "%s/%s_%s.tf.sff" % (env['BASECALLER_RESULTS'], env['expName'], env['resultsName'])
        fastqpath = "%s/%s_%s.fastq" % (env['BASECALLER_RESULTS'], env['expName'], env['resultsName'])
        libbam = "%s/%s_%s.bam" % (env['ALIGNMENT_RESULTS'], env['expName'], env['resultsName'])
        libbambai = "%s/%s_%s.bam.bai" % (env['ALIGNMENT_RESULTS'], env['expName'], env['resultsName'])
        tfbam = "%s/%s_%s.tf.bam" % (env['ALIGNMENT_RESULTS'], env['expName'], env['resultsName'])

        try:
            r = subprocess.call(["ln", "-s", "rawlib.sff", libsff])
        except:
            pass
        try:
            r = subprocess.call(["ln", "-s", "rawtf.sff", tfsff])
        except:
            pass
        try:
            r = subprocess.call(["ln", "-s", "rawlib.fastq", fastqpath])
        except:
            pass
        try:
            r = subprocess.call(["ln", "-s", "rawlib.bam", libbam])
        except:
            pass
        try:
            r = subprocess.call(["ln", "-s", "rawlib.bam.bai", libbambai])
        except:
            pass
        try:
            r = subprocess.call(["ln", "-s", "rawtf.bam", tfbam])
        except:
            pass

        ##################################################
        # Create zip of files
        ##################################################

        #sampled sff
        #make_zip(libsff.replace(".sff",".sampled.sff")+'.zip', libsff.replace(".sff",".sampled.sff"))

        #library sff
        make_zip(libsff + '.zip', libsff )

        #tf sff
        make_zip(tfsff + '.zip', tfsff)

        #fastq zip
        make_zip(fastqpath + '.zip', fastqpath)

        #sampled fastq
        #make_zip(fastqpath.replace(".fastq",".sampled.fastq")+'.zip', fastqpath.replace(".fastq",".sampled.fastq"))

        ########################################################
        # barcode processing                                   #
        # Zip up and move sff, fastq, bam, bai files           #
        # Move zip files to results directory                  #
        ########################################################
        if os.path.exists(env['DIR_BC_FILES']):
            filestem = "%s_%s" % (env['expName'], env['resultsName'])
            alist = os.listdir(env['DIR_BC_FILES'])
            extlist = ['sff','bam','bai','fastq']
            for ext in extlist:
                filelist = fnmatch.filter(alist, "*." + ext)
                zipname = filestem + '.barcode.' + ext + '.zip'
                for bfile in filelist:
                    afile = bfile.replace("rawlib",filestem)                
                    os.symlink(os.path.join(env['DIR_BC_FILES'],bfile),afile)
                    make_zip(zipname, afile)                
        else:
            printtime("No barcode run")

    printtime("MergeTLScript exit")
    sys.exit(0)
