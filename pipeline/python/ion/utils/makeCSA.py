#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import os
import fnmatch
import zipfile

    
# From the net, a function to search for file pattern a la find command
#http://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python
def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory,followlinks=True):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename
                    
def makeCSA(reportDir,rawDataDir):
    '''Python replacement for the pipeline/web/db/writers/csa.php script
    reportDir Full path to Report directory, rawDataDir Full path to Raw Data directory
    Returns Full Path to file'''

    # Change cwd to the report directory
    os.chdir(reportDir)
    
    # Define output file name
    csaFileName = "%s.support.zip" % os.path.basename(reportDir)
    csaFullPath = os.path.join(reportDir,csaFileName)
    
    # Files to find in the results directory heirarchy
    patterns = [
                'alignment.log',
                'backupPDF.pdf',
                'DefaultTFs.conf',
                'drmaa_stderr_block.txt',
                'drmaa_stdout.txt',
                'drmaa_stdout_block.txt',
                'ReportLog.html',
                'sysinfo.txt',
                'uploadStatus',
                'version.txt',
                #Signal processing files
                '*bfmask.stats',
                'avgNukeTrace_*.txt',
                'dcOffset*',
                'NucStep*',
                'processParameters.txt',
                'separator.bftraces.txt',
                'separator.trace.txt',
                'sigproc.log',
                # Basecaller files
                '*alignment.summary',
                '*quality.summary',
                'alignmentQC_out.txt',
                'alignmentQC_out_*.txt',
                'alignStats_err.json'
                'BaseCaller.json',
                'basecaller.log',
                'datasets_basecaller.json',
                'datasets_pipeline.json',
                'datasets_tf.json',
                'TFStats.json',
               ]
    
    zipList = []
    for pattern in patterns:
        for file in find_files(reportDir,pattern):
            # Ignore all files from the plugin_out subdirectory
            if "plugin_out" not in file:
                file = str(file).replace(reportDir,"")
                if file[0] == '/': file = file[1:]
                zipList.append(file)
    
    # Open a zip archive file (overwrite if it already exists)
    csa = zipfile.ZipFile(csaFullPath, mode='w', 
        compression=zipfile.ZIP_DEFLATED, allowZip64=True)
    
    # Compress/Add each file to the zip archive file
    for file in zipList:
        if os.path.exists(file):
            csa.write(file)

    # Add contents of pgm_logs.zip
    if zipfile.is_zipfile("pgm_logs.zip"):
        # Open the pgm log zip file
        pgmlogzip = zipfile.ZipFile("pgm_logs.zip", mode="r")
        for file in pgmlogzip.namelist():
            contents = pgmlogzip.read(file)
            csa.writestr(file, contents)
        pgmlogzip.close()
    else:
        # If pgm_logs.zip not available, try the raw data directory
        
        if os.path.isdir(rawDataDir):
            os.chdir(rawDataDir)
            # Files to find in the Raw Data Directory
            zipList = [
                'explog_final.txt',
                'InitLog.txt',
                'RawInit.txt',
                'RawInit.jpg',
                'InitValsW3.txt',
                'InitValsW2.txt',
                'Controller',
            ]
            for file in zipList:
                if os.path.exists(file):
                    csa.write(file)
        else:
            pass
            #TODO: Write out a README file saying no PGM logs were found.

    csa.close()

    return csaFullPath
