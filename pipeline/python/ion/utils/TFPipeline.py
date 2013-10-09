#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import os, sys
import time
import argparse
import traceback
import json
import shutil
import subprocess
from ion.utils.blockprocessing import printtime
from ion.utils import ionstats
from ion.utils import ionstats_plots



class NoTFDataException(Exception):
    def __init__(self, msg):
        self.msg = msg
        

def buildTFReference(tfreffasta_filename,analysis_dir,tfkey):
    '''
    Build the DefaultTFs.fasta from DefaultTFs.conf
    '''

    DefaultTFconfPath = os.path.join(analysis_dir,'DefaultTFs.conf')
    if not os.path.exists(DefaultTFconfPath):
        if not os.path.exists('/opt/ion/config/DefaultTFs.conf'):
            printtime('ERROR: could not locate DefaultTFs.conf (tried %s and /opt/ion/config/DefaultTFs.conf)' % DefaultTFconfPath)
            raise IOError
        DefaultTFconfPath = '/opt/ion/config/DefaultTFs.conf'

    printtime('TFPipeline: Using TF sequences from %s' % DefaultTFconfPath)
    num_tfs = 0
    try:
        confFile = open(DefaultTFconfPath, 'r')
        fastaFile = open(tfreffasta_filename, 'w')

        for confLine in confFile.readlines():
            if len(confLine) == 0:
                continue
            if confLine[0] == '#':
                continue
            confEntries = confLine.split(',')
            if len(confEntries) != 3:
                continue
            if confEntries[1] != tfkey:
                continue

            fastaFile.write('>%s\n' % confEntries[0])
            fastaFile.write('%s\n' % str(confEntries[2]).strip())
            num_tfs += 1

        confFile.close()
        fastaFile.close()

    except Exception as e:
        printtime("ERROR: failed convert %s into %s" % (DefaultTFconfPath, tfreffasta_filename))
        raise e
    
    if num_tfs == 0:
        printtime("No suitable TFs with key %s found in %s" % (tfkey, DefaultTFconfPath))
        raise NoTFDataException('No TF reference sequences')
        



def alignTFs(basecaller_bam_filename,bam_filename,fasta_filename):

    # Step 1. Build tmap index for DefaultTFs.fasta in a temporary directory

    indexDir = 'tfref'  # Might instead use a folder in /tmp
    indexFile = os.path.join(indexDir,'DefaultTFs.fasta')

    printtime("TFPipeline: Building index '%s' and mapping '%s'" % (indexFile,basecaller_bam_filename))

    if not os.path.exists(indexDir):
        os.makedirs(indexDir)

    shutil.copyfile(fasta_filename, indexFile)

    subprocess.check_call("tmap index -f %s" % indexFile, shell=True)

    # Step 2. Perform mapping of the bam file

    com1 = "tmap mapall -n 12 -f %s -r %s -Y -v stage1 map4" % (indexFile, basecaller_bam_filename)
    com2 = "samtools view -Sb -o %s - 2>> /dev/null" % bam_filename
    p1 = subprocess.Popen(com1, stdout=subprocess.PIPE, shell=True)
    p2 = subprocess.Popen(com2, stdin=p1.stdout, shell=True)
    p2.communicate()
    p1.communicate()

    # Step 3. Delete index

    shutil.rmtree(indexDir, ignore_errors=True)

    if p1.returncode != 0:
        raise subprocess.CalledProcessError(p1.returncode, com1)
    if p2.returncode != 0:
        # Assumption: samtools view only fails when there are zero reads.
        printtime("Command %s failed, presumably because there are no TF reads" % (com2))
        raise NoTFDataException('No TF reads found')        
        #raise subprocess.CalledProcessError(p2.returncode, com2)


    # Step 4. Bonus: Make index for the fasta

    try:
        subprocess.check_call("samtools faidx %s" % fasta_filename, shell=True)
    except:
        printtime("WARNING: samtools faidx failed")





def processBlock(tf_basecaller_bam_filename, BASECALLER_RESULTS, tfkey, floworder, analysis_dir):

    try:

        # These files will be created
        tfstatsjson_path = os.path.join(BASECALLER_RESULTS,"TFStats.json")
        tfbam_filename = os.path.join(BASECALLER_RESULTS,"rawtf.bam")
        tfref_filename = os.path.join(BASECALLER_RESULTS,"DefaultTFs.fasta")
        ionstats_tf_filename = os.path.join(BASECALLER_RESULTS,"ionstats_tf.json")

        # TF analysis in 5 simple steps

        buildTFReference(tfref_filename,analysis_dir,tfkey)

        alignTFs(tf_basecaller_bam_filename, tfbam_filename, tfref_filename)

        ionstats.generate_ionstats_tf(tfbam_filename,tfref_filename,ionstats_tf_filename)
        
        ionstats_plots.tf_length_histograms(ionstats_tf_filename, '.')
        
        ionstats.generate_legacy_tf_files(ionstats_tf_filename,tfstatsjson_path)

    
    except NoTFDataException as e:
        printtime("No data to analyze Test Fragments (%s)" % e.msg)
        f = open(os.path.join(BASECALLER_RESULTS,'TFStats.json'),'w')
        f.write(json.dumps({}))
        f.close()

    except:
        traceback.print_exc()


def mergeBlocks(BASECALLER_RESULTS,dirs,floworder):

    ionstats_tf_filename = os.path.join(BASECALLER_RESULTS,"ionstats_tf.json")
    tfstatsjson_path = os.path.join(BASECALLER_RESULTS,"TFStats.json")
    composite_filename_list = [os.path.join(BASECALLER_RESULTS,dir,"ionstats_tf.json") for dir in dirs]
    composite_filename_list = [filename for filename in composite_filename_list if os.path.exists(filename)]

    ionstats.reduce_stats(composite_filename_list,ionstats_tf_filename)

    ionstats_plots.tf_length_histograms(ionstats_tf_filename, '.')
    
    ionstats.generate_legacy_tf_files(ionstats_tf_filename,tfstatsjson_path)


if __name__=="__main__":

    # Step 1. Parser command line arguments

    parser = argparse.ArgumentParser(description='Test Fragment evaluation pipeline.')
    parser.add_argument('-i','--input',dest='basecaller_bam', default='rawtf.basecaller.bam',
                        help='Input unmapped BAM file containing TF reads (Default: rawtf.basecaller.bam)')
    parser.add_argument('-b','--bam',  dest='bam', default='rawtf.bam',
                        help='Intermediate output BAM file for TF reads (Default: rawtf.bam)')
    parser.add_argument('-k','--key',  dest='key', default='ATCG',
                        help='TF key sequence (Default: ATCG)')
    parser.add_argument('-f','--ref',dest='ref', default=None,
                        help='FASTA file with TF sequences. If not specified, '
                             'the pipeline will generate one from DefaultTF.conf')
    parser.add_argument('-d','--dir', dest='analysis_dir', default='.',
                        help='Directory searched for DefaultTFs.conf (Default: current directory)')
    args = parser.parse_args()
    print "TFPipeline args :",args

    # Step 2. If reference fasta file not specified, build one

    try:
        if args.ref == None:
            args.ref = 'DefaultTFs.fasta'
            buildTFReference(args.ref,args.analysis_dir,args.key)

        # Step 3. Perform alignment and generate bam file

        alignTFs(args.basecaller_bam, args.bam, args.ref)

        # Step 4. Post-processing. Run alignStats and TFMapper


        ionstats.generate_ionstats_tf(args.bam,args.ref,'ionstats_tf.json')

        ionstats_plots.tf_length_histograms('ionstats_tf.json', '.')
        
        ionstats.generate_legacy_tf_files('ionstats_tf.json','TFStats.json')
        

    except:
        traceback.print_exc()




