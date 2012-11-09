# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
import os
import sys
import subprocess
import traceback

def filter_bam_file(inputBAMFile, outputBAMFile, filter_type, qscore, qlen):
    try:
        cmd = './BAMFilter %s %s %s %s > %s' % (qscore, qlen, filter_type, inputBAMFile, outputBAMFile)
        print 'DEBUG: Filtering BAM file: %s' % inputBAMFile
        subprocess.call(cmd,shell=True)
        print 'Generated filtered file: %s' % outputBAMFile
    except:
        print "BAMFilter failed on %s" % inputBAMFile
        traceback.print_exc()
        return 1

    return 0

def merge_bam_files(bamfilelist,composite_bam_filepath,composite_bai_filepath):

    try:
#        cmd = 'picard-tools MergeSamFiles'
        cmd = 'java -Xmx8g -jar /opt/picard/picard-tools-current/MergeSamFiles.jar'

        for bamfile in bamfilelist:
            cmd = cmd + ' I=%s' % bamfile
        cmd = cmd + ' O=%s' % (composite_bam_filepath)
        cmd = cmd + ' ASSUME_SORTED=true'
        cmd = cmd + ' CREATE_INDEX=true'
        cmd = cmd + ' USE_THREADING=true'
        cmd = cmd + ' VALIDATION_STRINGENCY=SILENT'
        print 'DEBUG: Calling %s' % cmd
        subprocess.call(cmd,shell=True)
    except:
        print 'bam file merge failed'
        traceback.print_exc()
        return 1

    try:
        # picard is using .bai , we want .bam.bai
        srcbaifilepath = composite_bam_filepath.replace(".bam",".bai")
        if os.path.exists(srcbaifilepath):
            os.rename(srcbaifilepath, composite_bai_filepath)
        else:
            print 'ERROR: %s doesnt exists' % srcbaifilepath
    except:
        traceback.print_exc()
        return 1

    return 0


def Usage():
    print 'Filter_and_Merge.py Usage:'
    print '-m\t\tmerge = true'
    print '-f\t\tfilter = true'
    print '-q x\t\tqscore thresh set to x'
    print '-l x\t\tqlen thresh set to x'
    print '-t x\t\tfilter type set to x (1=filter on aligned qscore, 2=filter on predicted qscore)'
    print '-i file\t\tInput BAM list file set to file'


if __name__ == '__main__':
    # establish defaults
    qscore = 17
    qlen = 50
    filter_type = 2
    wantMerge = False
    wantFilter = False
    BAMList = 'BAMList.txt'

    # parse cmd-line args
    argcc = 1
    argc = len(sys.argv)
    if argc < 2:
        Usage()
        sys.exit()

    while argcc < argc:
        if sys.argv[argcc] == '-m':
            wantMerge = True
        if sys.argv[argcc] == '-f':
            wantFilter = True
        if sys.argv[argcc] == '-q':
            argcc += 1
            qscore = int(sys.argv[argcc])
        if sys.argv[argcc] == '-l':
            argcc += 1
            qlen = int(sys.argv[argcc])
        if sys.argv[argcc] == '-t':
            argcc += 1
            filter_type = int(sys.argv[argcc])
        if sys.argv[argcc] == '-i':
            argcc += 1
            BAMList = sys.argv[argcc]
        if sys.argv[argcc] == '-h':
            Usage()
            sys.exit()
        argcc += 1


    # parse up the input file to get our bam file list
    bamFileList = []
    for line in open(BAMList,'rt').readlines():
        sline = line.strip()
        if '#' not in sline and len(sline) > 2:
            bamFileList.append(sline)
    numFiles = len(bamFileList)
    if numFiles == 0:
        print 'No BAM files in %s?' % BAMList
        sys.exit()

    mergedBAMFile = 'Merged.bam'
    mergedBAMIndex = 'Merged.bai'
    filteredBAMFile = 'Filtered.bam'

    # Two modes possible here:
    # 1. Merge BAM files, then optionally filter the result
    # 2. Loop through input file list, performing filtering on each file
    if wantMerge:
        merge_bam_files(bamFileList, mergedBAMFile, mergedBAMIndex)

        if wantFilter:
            filter_bam_file(mergedBAMFile, filteredBAMFile, filter_type, qscore, qlen)

    else:
        if wantFilter:
            for bamFile in bamFileList:
                filteredBamFile = 'Filtered_' + bamFile.split('/')[-1]
                filter_bam_file(bamFile, filteredBamFile, qscore, qlen)


