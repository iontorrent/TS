#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import os
import fnmatch
import zipfile
import socket
import re
import logging
import traceback
from glob import glob
import os.path

NATURAL_SORT_PATTERN = re.compile(r'(\d+|\D+)')


logger = logging.getLogger(__name__)


# From the net, a function to search for file pattern a la find command
# http://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python
def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory, followlinks=True):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def match_files(walk, pattern):
    for root, dirs, files in walk:
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def makeCSA(reportDir, rawDataDir, csa_file_name=None):
    '''Python replacement for the pipeline/web/db/writers/csa.php script
    reportDir Full path to Report directory, rawDataDir Full path to Raw Data directory
    Returns Full Path to file'''

    # Change cwd to the report directory
    os.chdir(reportDir)

    # Define output file name
    if not csa_file_name:
        csa_file_name = "%s.support.zip" % os.path.basename(reportDir)
    csaFullPath = os.path.join(reportDir, csa_file_name)

    # Files to find in the results directory hierarchy
    patterns = [
        'ion_params_00.json',
        'alignment.log',
        'report.pdf',
        'backupPDF.pdf',
        '*-full.pdf',
        'DefaultTFs.conf',
        'drmaa_stderr_block.txt',
        'drmaa_stdout.txt',
        'drmaa_stdout_block.txt',
        'ReportLog.html',
        'sysinfo.txt',
        'uploadStatus',
        'version.txt',
        # Signal processing files
        '*bfmask.stats',
        'avgNukeTrace_*.txt',
        'processParameters.txt',
        'separator.bftraces.txt',
        'separator.trace.txt',
        'BkgModelFilterData.h5',
        'pinsPerFlow.txt',
        # Basecaller files
        '*ionstats_alignment.json',
        '*ionstats_basecaller.json',
        'alignmentQC_out.txt',
        'alignmentQC_out_*.txt',
        'alignStats_err.json',
        'BaseCaller.json',
        'datasets_basecaller.json',
        'datasets_pipeline.json',
        'datasets_tf.json',
        '*ionstats_tf.json',
        'TFStats.json',
        '*.ionstats_error_summary.h5',
    ]

    tn = 1
    for root, dirs, files in os.walk(reportDir, followlinks=True):
        for dirName in dirs:
            if fnmatch.fnmatch(dirName, 'block_X*'):
                tn = 0
                break

    blockX = []
    blockY = []
    blockLeft = "blockLETT"
    blockRight = "blockRIGHT"
    blockMid = "blockMID"

    if tn == 0:
        combined_sigproc = os.path.join(reportDir, "combined_sigproc.log")
        combined_basecaller = os.path.join(reportDir, "combined_basecaller.log")
        combined_basecaller_recalibration = os.path.join(reportDir, "combined_basecaller_recalibration.log")
        fcs = open(combined_sigproc, "w")
        fcc = open(combined_basecaller, "w")
        fcr = open(combined_basecaller_recalibration, "w")
        for dirName in os.listdir(reportDir):
            if fnmatch.fnmatch(dirName, 'block_X*'):
                index = dirName.find("_Y")
                strX = dirName[7:index]
                strY = dirName[index + 2:]
                x = int(strX)
                y = int(strY)
                blockX.append(x)
                blockY.append(y)

                fcs.write('\n')
                fcs.write(dirName)
                fcs.write(':\n\n')

                fname = os.path.join(reportDir, dirName, "sigproc_results/sigproc.log")
                if os.path.exists(fname):
                    flagerr = 1
                    fn = open(fname, "r")
                    lines = fn.readlines()
                    for line in lines:
                        if "Command line =" in line:
                            fcs.write(line)
                            fcs.write('\n')

                        if "Beadfind Complete:" in line:
                            flagerr = 0
                            fcs.write(line)
                            fcs.write('\n')

                        if "Analysis (wells file only) Complete:" in line:
                            flagerr = 0
                            fcs.write(line)
                            fcs.write('\n')

                        if "PhaseEstimator::" in line:
                            fcs.write(line)
                            fcs.write('\n')

                        if "Basecalling Complete:" in line:
                            fcs.write(line)
                            fcs.write('\n')

                        if "Error" in line:
                            fcs.write(line)
                            fcs.write('\n')

                        if "ERROR" in line:
                            fcs.write(line)
                            fcs.write('\n')

                    fn.close()

                    if flagerr != 0:
                        fcs.write("Expected complete ending line is not found. There may be something wrong with this block.")
                        fcs.write('\n')

                else:
                    fcs.write("no sigproc_results/sigproc.log\n")

                fcc.write('\n')
                fcc.write(dirName)
                fcc.write(':\n\n')

                fname = os.path.join(reportDir, dirName, "basecaller_results/basecaller.log")
                if os.path.exists(fname):
                    flagerr = 1
                    fn = open(fname, "r")
                    lines = fn.readlines()
                    for line in lines:
                        if "Command line =" in line:
                            fcc.write(line)
                            fcc.write('\n')

                        if "PerBaseQual::" in line:
                            fcc.write(line)
                            fcc.write('\n')

                        if "MEM USAGE:" in line:
                            fcc.write(line)
                            fcc.write('\n')

                        if "Basecalling Complete:" in line:
                            flagerr = 0
                            fcc.write(line)
                            fcc.write('\n')

                        if "Error" in line:
                            fcc.write(line)
                            fcc.write('\n')

                        if "ERROR" in line:
                            fcc.write(line)
                            fcc.write('\n')

                    fn.close()

                    if flagerr != 0:
                        fcc.write("Expected complete ending line is not found. There may be something wrong with this block.")
                        fcc.write('\n')

                else:
                    fcc.write("no basecaller_results/basecaller.log\n")

                fcr.write('\n')
                fcr.write(dirName)
                fcr.write(':\n\n')

                fname = os.path.join(reportDir, dirName, "basecaller_results/recalibration/basecaller.log")
                if os.path.exists(fname):
                    flagerr = 1
                    fn = open(fname, "r")
                    lines = fn.readlines()
                    for line in lines:
                        if "Command line =" in line:
                            fcr.write(line)
                            fcr.write('\n')

                        if "PerBaseQual::" in line:
                            fcr.write(line)
                            fcr.write('\n')

                        if "MEM USAGE:" in line:
                            fcr.write(line)
                            fcr.write('\n')

                        if "Basecalling Complete:" in line:
                            flagerr = 0
                            fcr.write(line)
                            fcr.write('\n')

                        if "Error" in line:
                            fcr.write(line)
                            fcr.write('\n')

                        if "ERROR" in line:
                            fcr.write(line)
                            fcr.write('\n')

                    fn.close()

                    if flagerr != 0:
                        fcr.write("Expected complete ending line is not found. There may be something wrong with this block.")
                        fcr.write('\n')

                else:
                    fcr.write("no basecaller_results/recalibration/basecaller.log\n")

        fcs.close()
        patterns.append('combined_sigproc.log')

        fcc.close()
        patterns.append('combined_basecaller.log')

        fcr.close()
        patterns.append('combined_basecaller_recalibration.log')

        blockX.sort()
        blockY.sort()
        lenx = len(blockX)
        leny = len(blockY)

        leftx = blockX[blockX.count(blockX[0])+1]
        midx = blockX[lenx/2]
        rightx = blockX[-(blockX.count(blockX[-1])+1)]
        midy = blockY[leny/2]
        blockLeft = ("block_X" + str(leftx) + "_Y" + str(midy))
        blockRight = ("block_X" + str(rightx) + "_Y" + str(midy))
        blockMid = ("block_X" + str(midx) + "_Y" + str(midy))

        block_list = os.path.join(reportDir, "whole_block_list.txt")
        fbl = open(block_list, "w")
        fbl.write(blockLeft)
        fbl.write('\n')
        fbl.write(blockMid)
        fbl.write('\n')
        fbl.write(blockRight)
        fbl.close()
        patterns.append('whole_block_list.txt')

    else:
        patterns.append('dcOffset*')
        patterns.append('NucStep*')
        patterns.append('sigproc.log')
        patterns.append('basecaller.log')

    zipList = []
    walk = list(os.walk(reportDir, followlinks=True))
    skipDir = reportDir
    skipDir += "/block_X"
    for pattern in patterns:
        for file in match_files(walk, pattern):
            # Ignore all files from the plugin_out subdirectory
            if "plugin_out" not in file and skipDir not in file:
                file = str(file).replace(reportDir, "")
                if file[0] == '/': file = file[1:]
                zipList.append(file)

    if tn == 0:
        patterns3 = [
            'dcOffset*',
            'NucStep*',
            'sigproc.log',
            'basecaller.log',
            ]

        for pattern3 in patterns3:
            for file3 in match_files(walk, pattern3):
                if "plugin_out" not in file3 and skipDir not in file3 and (blockLeft in file3 or blockRight in file3 or blockMid in file3):
                    file3 = str(file3).replace(reportDir, "")
                    if file3[0] == '/': file3 = file3[1:]
                    zipList.append(file3)

    # Open a zip archive file (overwrite if it already exists)
    csa = zipfile.ZipFile(csaFullPath, mode='w',
                          compression=zipfile.ZIP_DEFLATED, allowZip64=True)

    # Compress/Add each file to the zip archive file
    for file in zipList:
        if os.path.exists(file):
            csa.write(file)

    # Add files from raw data directory, if they exist
    # See TS-9178, TS-12390
    zipList = [
            'explog_final.txt',
            'explog.txt',
            'InitLog.txt',
            'InitLog1.txt',
            'InitLog2.txt',
            'RawInit.txt',
            'RawInit.jpg',
            'InitValsW3.txt',
            'InitValsW2.txt',
            'Controller',
            'debug',
            'Controller_1',
            'debug_1',
            'chipCalImage.bmp.bz2',
            'InitRawTrace0.png',
        ]
    for item in zipList:
        srcfile = os.path.join(rawDataDir, item)
        if os.path.exists(srcfile):
            csa.write(srcfile, item)
    # Add contents of pgm_logs.zip
    # This file generated by TLScript.py (src in pipeline/python/ion/reports)
    # list of files included in zip should match list above
    if zipfile.is_zipfile("pgm_logs.zip"):
        # Open the pgm log zip file
        pgmlogzip = zipfile.ZipFile("pgm_logs.zip", mode="r")
        for file in pgmlogzip.namelist():
            contents = pgmlogzip.read(file)
            csa.writestr(file, contents)
        pgmlogzip.close()

    csa.close()
    return csaFullPath


def natsort_key(s):
    return [int(s) if s.isdigit() else s for s in NATURAL_SORT_PATTERN.findall(s)]


def make_ssa():
    globs = [
        "/etc/torrentserver/tsconf.conf",
        "/opt/sge/iontorrent/spool/master/messages",
        "/usr/share/ion-tsconfig/mint-config/*",
        "/var/log/apache2/access.log",
        "/var/log/apache2/access.log.1.gz",
        "/var/log/apache2/error.log",
        "/var/log/apache2/error.log.1.gz",
        "/var/log/ion/crawl.log",
        "/var/log/ion/crawl.log.1",
        "/var/log/ion/django.log",
        "/var/log/ion/django.log.1.gz",
        "/var/log/ion/celery_*.log",
        "/var/log/ion/celery_*.log.1.gz",
        "/var/log/ion/celerybeat.log",
        "/var/log/ion/celerybeat.log.1.gz",
        "/var/log/ion/ionPlugin.log",
        "/var/log/ion/ionPlugin.log.1",
        "/var/log/ion/jobserver.log",
        "/var/log/ion/jobserver.log.1",
        "/var/log/ion/tsconfig_*.log",
        "/var/log/ion/tsconfig_*.log.1",
        "/var/log/ion/tsconf.log",
        "/var/log/ion/RSMAgent.log",
        "/var/log/ion/data_management.log",
        "/var/log/ion/data_management.log.1.gz",
        "/var/log/kern.log",
        "/var/log/postgresql/postgresql-*-main.log",
        "/var/log/syslog",
        "/tmp/stats_sys.txt",
    ]
    try:
        with open("/etc/torrentserver/tsconf.conf") as conf:
            for l in conf:
                if l.startswith("serialnumber:"):
                    servicetag = l[len("serialnumber:"):].strip()
                    break
            else:
                servicetag = socket.gethostname()
    except:
        servicetag = "0"
    archive_name = "%s_systemStats.zip" % servicetag
    path = os.path.join("/tmp", archive_name)
    archive = zipfile.ZipFile(path, mode='w',
                              compression=zipfile.ZIP_DEFLATED, allowZip64=True)
    for pattern in globs:
        files = glob(pattern)
        files.sort(key=natsort_key)
        for filename in files:
            try:
                archive.write(filename, os.path.basename(filename))
            except:
                logger.warn(traceback.format_exc())
    archive.close()
    return path, archive_name

if __name__ == '__main__':
    # Command line to make CSA for debugging code purposes.
    import sys
    reportDir = sys.argv[1]
    rawDataDir = sys.argv[2]
    makeCSA(reportDir, rawDataDir)
