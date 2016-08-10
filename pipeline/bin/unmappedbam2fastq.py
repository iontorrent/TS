#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

__version__ = filter(str.isdigit, "$Revision$")

import argparse
import pysam
import traceback
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('unmapped_bam', nargs='+')
    parser.add_argument('fastq', help='e.g. test.fastq')

    args = parser.parse_args()

    try:
        with open(args.fastq, 'w') as fastq_file:

            for bam_file in args.unmapped_bam:
                if os.path.exists(bam_file):
                    try:
                        samfile = pysam.Samfile(bam_file, mode="rb", check_header=False, check_sq=False)
                        for x in samfile.fetch(until_eof=True):
                            fastq_file.write("@%s\n%s\n+\n%s\n" % (x.qname, x.seq, x.qual))
                        samfile.close()
                    except:
                        traceback.print_exc()
    except:
        traceback.print_exc()
