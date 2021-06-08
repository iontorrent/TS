
#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved
import fnmatch
import os
import shutil
import tarfile
import tempfile
import zipfile
import StringIO

result_patterns = [
    'ion_params_00.json',
    # 'alignment.log',
    # 'report.pdf',
    # 'backupPDF.pdf',
    # '*-full.pdf',
    'DefaultTFs.conf',
    # 'drmaa_stderr_block.txt',
    # 'drmaa_stdout.txt',
    # 'drmaa_stdout_block.txt',
    # 'ReportLog.html',
    # 'sysinfo.txt',
    # 'uploadStatus',
    'version.txt',
    "analysisSamples.json",
    "AnalysisData.json",
    "primaryAnalysis.log",
    "summary.log",
    #Signal processing files
    '*bfmask.stats',
    # 'avgNukeTrace_*.txt',
    # 'dcOffset*',
    # 'NucStep*',
    'processParameters.txt',
    'separator.bftraces.txt',
    'separator.trace.txt',
    'sigproc.log',
    # 'BkgModelFilterData.h5',
    # 'pinsPerFlow.txt',
    # Basecaller files
    '*ionstats_alignment.json',
    '*ionstats_basecaller.json',
    'alignmentQC_out.txt',
    'alignmentQC_out_*.txt',
    'alignStats_err.json',
    'BaseCaller.json',
    'basecaller.log',
    'datasets_basecaller.json',
    'datasets_pipeline.json',
    'datasets_tf.json',
    '*ionstats_tf.json',
    'TFStats.json',
    'readLenHisto*.png',
    'Bead_density_1000.png',
    # '*.sparkline.png'
    # tertiary analysis and  secondary analysis
    '*.ini',
    # VariantCaller
    'VariantCaller.json',
    'summary*.log',
    '*inline_control.png',
    'inline_control_stats.json',
]

dir_patterns = [
    'log/AnnotatorActor-00',
    'log/TmapExecutionActor-00',
    'log/VcMetricActor-00',
    'log/AppRunnerActor-00',
    'log/TmapMergeActor-00',
    'log/CnvActor-00',
    'log/VariantCallerActor-00',
    'log/AssayQCMetricActor-00',
    'log/BaseCallingActor-00',
    'log/BarcodeCrosstalkActor-00',
    'log/PlanLevelAppRunnerActor-00',
    'log/RNACountsActor-00'
]

rawDir_patterns = [
    'pipPres'
]

rawdata_patterns = [
    'explog_final.txt',
    'explog.txt',
    'explog.json',
    'explog_final.json',
    'planned_run.json',
    'InitLog.txt',
    'InitLog1.txt',
    'InitLog2.txt',
    'debug',
    'liveview',
    'Controller_1',
    'debug_1',
    'InitRawTrace0.png',
    'DataCollect.config',
    'Events.log',
    'impi_out.txt',
    'impi_out_bin.txt',
    'tslink.log',
    'kern.log',
    'T0Estimate_dbg*',
    'libPrep_log.csv',
    'ScriptStatus.csv',
    'vacuum_log.csv',
    # 'vacuum_data_lane*.csv',
    'pipetteUsage.json',
    'tubeVolHistory.csv',
    'TubeBottomLog.csv*',
    'DeckStatus.json',
    'LiquidClassCommonTbl.json',
    'ContainerGeometryCommonTbl.json',
    'LiquidClassWorkFlowTbl.json',
    'DeckGeometryTbl.json',
    'ContainerGeometryWorkflowTbl.json',
    'leakTestHeadSpace.csv',
    'initFill_pressureVerify.csv',
    'initFill_R*.csv',
    'blockedTipPressureCurve_*.csv',
]

duplicate_patterns = {
    "outputs/SigProcActor-00" : "sigproc.log",
    "outputs/BaseCallingActor-00" : "basecaller.log"
}

qcreport_patterns = {
    "Info.csv",
    "Bead_density_contour.png",
    "*amplicon.cov.xls",
    "*.summary.pdf",
    "*.stats.cov.txt",
    "wells_beadogram.png",
    "*.pdf"
}

def match_files(walk, pattern):
    for root, _, files in walk:
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

def match_dir(walk, pattern):
    for root, dirname, files in walk:
        if root.endswith(pattern):
            for file in files:
                filename = os.path.join(root, file)
                yield filename


def get_file_list(directory, patterns, block_list=[], check_list = []):
    print "getting files from " + directory
    mylist = []
    try:
        walk = list(os.walk(directory, followlinks=True))
    except:
        raise Exception('Unable to access files from %s' % directory)

    for pattern in patterns:
        for filename in match_files(walk, pattern):
            blockthis = False
            for blocker in block_list:
                if blocker in filename:
                    blockthis = True
            if not blockthis:
                mylist.append(filename)
    return mylist

def get_file_list2(directory, patterns):
    print "getting files from the log dir" + directory
    mylist = []
    try:
        walk = list(os.walk(directory, followlinks=True))
    except:
        raise Exception('Unable to access files from %s' % directory)

    for pattern in patterns:
        for filename in match_dir(walk, pattern):
            mylist.append(filename)
    return mylist

def get_list_from_results(directory, blockList):
    file_list = get_file_list(directory, result_patterns, block_list=["plugin", "thumbnail", "rawdata"] + blockList)
    file_list = remove_duplicate_from_outputs(file_list, duplicate_patterns, block_ele = "recalibration")
    return file_list

def get_list_from_rawdata(directory):
    return get_file_list(directory, rawdata_patterns)

def get_list_from_dir(directory):
    return get_file_list2(directory, dir_patterns)

def get_list_from_dir2(directory):
    return get_file_list2(directory, rawDir_patterns)

def get_list_from_report(directory):
    return get_file_list(directory, qcreport_patterns)

def remove_duplicate_from_outputs(file_list, duplicate_patterns, block_ele):
    print "removing duplicated files"
    # A copy of the file list
    file_list_copy = list(file_list)
    nInclude = 3 # number of blocks to include when no error on any block
    # find which file to include
    for block, logfile in duplicate_patterns.items():
        nSuccess = 0
        nBlock = 0
        for filename in file_list_copy:
            if block in filename and logfile in filename and block_ele not in filename:
                nBlock = nBlock + 1
                lastline = os.popen("tail -n 2 %s" % filename).read()
                if "Complete" in lastline:
                    nSuccess = nSuccess + 1
        # remove log files when all the block completed without error
        if nSuccess == nBlock:
            nRemove = nSuccess - nInclude
            for filename in file_list_copy:
                if block in filename and logfile in filename and block_ele not in filename and nRemove > 0:
                    # print "FOUND: " + filename
                    lastline = os.popen("tail -n 2 %s" % filename).read()
                    if "Complete" in lastline:
                        blockid = block + '/'  + filename.split(block + '/')[1].split('/')[0]
                        # print "pattern - " + blockid
                        for f in file_list:
                            if blockid in f:
                                file_list.remove(f)
                        nRemove = nRemove - 1

    return file_list



def writeZip(fullnamepath, filelist, dirtrim="", recursive = False, openmode="w"):
    '''Add files to zip archive.  dirtrim is a string which will be deleted
    from each file entry.  Used to strip leading directory from filename.'''
    # Open a zip archive file
    # csa = zipfile.ZipFile(fullnamepath, mode=openmode, compression=zipfile.ZIP_DEFLATED, allowZip64=True)
    csa = tarfile.TarFile.open(fullnamepath, mode=openmode)
    # Add each file to the zip archive file
    for item in filelist:
        if os.path.exists(item):
            if recursive:
                arcname = item.replace(dirtrim, 'CSA/'+ os.path.basename(os.path.dirname(item)) + '/')
            else:
                arcname = item.replace(dirtrim, 'CSA/')
            csa.add(item, arcname)

    # Close zip archive file
    csa.close()


def makeCSA(reportDir, rawDataDir, sampleResultsDirs = [] ,csa_file_name=None, block_list = []):
    """Create the CSA zip file"""

    # reportDir must exist
    if not os.path.exists(reportDir):
        raise IOError("%s: Does not exist" % reportDir)

    # Define output file name
    if not csa_file_name:
        csa_file_name = "%s.CSA.tar" % os.path.basename(reportDir)

    csaFullPath = os.path.join(reportDir, csa_file_name)

    # Generate a list of files from report dir to write to the archive
    zipList = get_list_from_results(reportDir, block_list)
    writeZip(csaFullPath, zipList, dirtrim=reportDir, openmode="w")

    # Generate a list of files from rawdata dir to append to the archive
    zipList = get_list_from_rawdata(rawDataDir)
    writeZip(csaFullPath, zipList, dirtrim=rawDataDir, openmode="a")

    # Generate a list of files from log dir to append to the archive
    zipList = get_list_from_dir(reportDir)
    writeZip(csaFullPath, zipList, dirtrim=reportDir, openmode="a")

    zipList = get_list_from_dir2(rawDataDir)
    writeZip(csaFullPath, zipList, dirtrim=rawDataDir, openmode="a")

    # Generate a list of files from report dir to append to the archive
    if len(sampleResultsDirs) > 0:
        for sampleResultsDir in sampleResultsDirs:
            try:
                zipList = get_list_from_report(sampleResultsDir)
                writeZip(csaFullPath, zipList, dirtrim=sampleResultsDir, recursive = True, openmode="a")
            except:
                pass

    version = '0.4.4'
    pluginInfo = tarfile.TarInfo('pluginInfo.txt')
    pluginInfo.size = len(version)
    tar = tarfile.open(csaFullPath, 'a')
    tar.addfile(pluginInfo, StringIO.StringIO(version))
    tar.close()

    return csaFullPath
