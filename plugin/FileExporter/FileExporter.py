#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import json
import os
import subprocess
import tarfile
import re
import urllib2
import zipfile

from ion.plugin import *
from ion.utils import blockprocessing
from subprocess import *

class FileExporter(IonPlugin):
    version = '5.4.0.0'
    runtypes = [ RunType.FULLCHIP, RunType.THUMB, RunType.COMPOSITE ]
    runlevels = [ RunLevel.DEFAULT ]

    pluginDir = ''
    envDict = dict(os.environ)
    json_dat = {}
    json_barcodes = {}
    renameString = ""
    isBarcodedRun = False
    variantCallerPath = "variantCaller_out" # path to the most recently run variant caller plugin, TS 4.4 and newer enables multiple instances, so this path may change
    delim = '.'
    selections= ""
    variantExists = False
    downloadDir =''
    run_name = ''
    report_name = ''
    run_date = ''
    chiptype = ''
    instrument = ''
    analysis_dir = ''

    # set defaults for user options
    fastqCreate = False
    vcfCreate   = False
    xlsCreate   = False
    bamCreate   = False

    zipFASTQ = False
    zipBAM = False
    zipVCF = False
    zipXLS = False
    wantTar = True

    # imported from FastqCreator
    alignment_dir  = ''
    reference_path = ''
    basecaller_dir = ''
    datasets_basecaller = dict()
    barcodeId = ''

    def applyBarcodeName(self, barcodeName):
        """
        This method will convert the replace name with all the barcode specific items
        :param barcodeName:
        :return:
        """
        if not barcodeName or barcodeName == 'nonbarcoded':
            return self.renameString

        finalName = self.renameString.replace('@BARINFO@', barcodeName)
        sampleName = self.json_barcodes[barcodeName].get('sample', '').replace(' ', '_')

        #if we cannot find a sample for this barcode, then we remove the sample id
        if sampleName:
            return re.sub(r"@SAMPLEID@", sampleName, finalName)
        else:
            return re.sub(self.delim + r"?@SAMPLEID@" + self.delim + r"?", sampleName, finalName)


    def bamRename(self):
        """
        Method to rename and symlink the .bam and .bai files.
        :return:
        """
        for barcodeName in self.json_barcodes.keys():
            fileName = self.json_barcodes[barcodeName]['bam_filepath']
            finalName = self.applyBarcodeName(barcodeName)

            destName      = self.pluginDir   + '/' + finalName
            downloadsName = self.downloadDir + '/' + finalName

            if os.path.isfile(fileName):
                if not os.path.exists(destName + '.bam'):
                    os.symlink(fileName, destName + '.bam')

                if os.path.lexists(downloadsName + '.bam'):
                    os.remove(downloadsName + '.bam')
                os.symlink(fileName, downloadsName + '.bam')

            fileName = fileName.replace('.bam', '.bam.bai')

            if os.path.isfile(fileName):
                if not os.path.exists(destName + '.bam.bai'):
                    os.symlink(fileName, destName + '.bam.bai')

                if os.path.exists(downloadsName + '.bam.bai'):
                    os.remove(downloadsName + '.bam.bai')
                os.symlink(fileName, downloadsName + '.bam.bai')

    def vcfRename(self):
        """
        Method to rename and symlink the .vcf files.
        :return:
        """
        for barcode in self.json_barcodes.keys():
            # build our new filename and move this file
            finalName = self.applyBarcodeName(barcode)

            destName      = self.pluginDir   + '/' + finalName + '.vcf'
            downloadsName = self.downloadDir + '/' + finalName + '.vcf'

            if self.isBarcodedRun:
                srcName = '%s/plugin_out/%s/%s/TSVC_variants.vcf' % (self.analysis_dir, self.variantCallerPath, barcode)
            else:
                srcName = '%s/plugin_out/%s/TSVC_variants.vcf'    % (self.analysis_dir, self.variantCallerPath)

            if os.path.isfile(srcName):
                if not os.path.lexists(destName):
                    os.symlink(srcName, destName)
                if os.path.lexists(downloadsName):
                    os.remove(downloadsName)
                os.symlink(srcName, downloadsName)

    def xlsRename(self):
        """
        Method to rename and symlink the .xls files.
        :return:
        """
        for barcode in self.json_barcodes.keys():
            # build our new filename and move this file
            finalName = self.applyBarcodeName(barcode)

            if self.isBarcodedRun:
                srcNameAlleles  = '%s/plugin_out/%s/%s/alleles.xls'  % (self.analysis_dir, self.variantCallerPath, barcode)
                srcNameVariants = '%s/plugin_out/%s/%s/variants.xls' % (self.analysis_dir, self.variantCallerPath, barcode)
            else:
                srcNameAlleles  = '%s/plugin_out/%s/alleles.xls'  % (self.analysis_dir, self.variantCallerPath)
                srcNameVariants = '%s/plugin_out/%s/variants.xls' % (self.analysis_dir, self.variantCallerPath)

            destNameAlleles       = self.pluginDir   + '/' + finalName + self.delim + 'alleles.xls'
            downloadsNameAlleles  = self.downloadDir + '/' + finalName + self.delim + 'alleles.xls'
            destNameVariants      = self.pluginDir   + '/' + finalName + self.delim + 'variants.xls'
            downloadsNameVariants = self.downloadDir + '/' + finalName + self.delim + 'variants.xls'

            if os.path.isfile(srcNameAlleles):
                if not os.path.lexists(destNameAlleles):
                    os.symlink(srcNameAlleles, destNameAlleles)
                if os.path.lexists(downloadsNameAlleles):
                    os.remove(downloadsNameAlleles)
                os.symlink(srcNameAlleles, downloadsNameAlleles)

            if os.path.isfile(srcNameVariants):
                if not os.path.lexists(destNameVariants):
                    os.symlink(srcNameVariants, destNameVariants)
                if os.path.lexists(downloadsNameVariants):
                    os.remove(downloadsNameVariants)
                os.symlink(srcNameVariants, downloadsNameVariants)

    def createFastq(self):
        """
        Creates Fastq files
        :return:
        """
        for barcodeName in self.json_barcodes.keys():
            # build our new filename and move this file
            bam = self.json_barcodes[barcodeName]['bam_filepath']

            if not os.path.exists(bam):
                continue

            # generate final place for the fastq file
            fastqFileName = self.applyBarcodeName(barcodeName) + '.fastq'
            finalName = os.path.join(self.pluginDir, fastqFileName)

            # create and execute the subprocess command
            command = blockprocessing.bam2fastq_command(bam, finalName)
            subprocess.call(command,shell=True)

            # create a symlink for the fastq in the downloads folder
            downloadsName = os.path.join(self.downloadDir, fastqFileName)
            if os.path.exists(finalName):
                if os.path.lexists(downloadsName):
                    os.remove(downloadsName)
                os.symlink(finalName, downloadsName)

    def compressFiles(self, CompressType, zipSubdir, htmlOut):
        htmlFiles = '<div id="files">'
        for fileName in os.listdir(self.pluginDir):
            if (self.zipBAM and (fileName.endswith('.bam') or fileName.endswith('.bai') )) or (self.zipFASTQ and fileName.endswith('.fastq')) or (self.zipVCF and fileName.endswith('.vcf')) or (self.zipXLS and fileName.endswith('.xls')):
                fullPathAndFileName = self.pluginDir + '/' + fileName
                if os.path.isfile(fullPathAndFileName): # need to make sure sym links exist
                    if self.wantTar:
                        storedName = zipSubdir + '/' + fileName
                        CompressType.add(fullPathAndFileName, arcname=storedName)
                    else:

                        if os.path.getsize(fullPathAndFileName) > 1024*1024*1024*2: # store large files (>2G), don't try and compress
                            CompressType.write(fullPathAndFileName, zipSubdir + '/' + fileName, zipfile.ZIP_STORED)

                        else:
                            CompressType.write(fullPathAndFileName, zipSubdir + '/' + fileName, zipfile.ZIP_DEFLATED)
                    htmlFiles = htmlFiles + '<span><pre>' + fileName + '</pre></span>\n'

            elif fileName.endswith('.bam') or fileName.endswith('.bai') or fileName.endswith('.fastq') or fileName.endswith('.vcf') or fileName.endswith('.xls'):
                htmlOut.write('<a href="%s/%s" download>%s</a><br>\n'%(self.envDict['TSP_URLPATH_PLUGIN_DIR'],fileName, fileName))

        htmlFiles += "</div>\n"
        return htmlFiles

    def launch(self, data=None):

        # In this first section we are going to attempt to get all the required inputs from a variety of different sources
        try:
            with open('startplugin.json', 'r') as fh:
                self.json_dat = json.load(fh)

            with open('barcodes.json', 'r') as fh:
                self.json_barcodes = json.load(fh)

            self.basecaller_dir = self.json_dat['runinfo']['basecaller_dir']
            with open(os.path.join(self.basecaller_dir, "datasets_basecaller.json"), 'r') as f:
                self.datasets_basecaller = json.load(f)

            # sanity check to make sure we have all of the settings
            if not 'pluginconfig' in self.json_dat:
                print("ERROR: Failed to get plugin configuration.")
                return

            # Define output directory.
            self.pluginDir      = self.json_dat['runinfo']['results_dir']
            self.alignment_dir  = self.json_dat['runinfo']['alignment_dir']
            self.barcodeId      = self.json_dat['runinfo']['barcodeId']
            self.analysis_dir   = self.json_dat['runinfo']['analysis_dir']
            self.run_name       = self.json_dat['expmeta']['run_name']
            self.report_name    = self.json_dat['expmeta']['results_name']
            self.run_date       = self.json_dat['expmeta']['run_date']
            self.chiptype       = self.json_dat['expmeta']['chiptype']
            self.instrument     = self.json_dat['expmeta']['instrument']
            self.reference_path = os.getenv('TSP_FILEPATH_GENOME_FASTA','')
            self.downloadDir    = os.path.join(self.analysis_dir, 'plugin_out', 'downloads')

            if not os.path.isdir(self.downloadDir):
                os.makedirs(self.downloadDir)

            self.isBarcodedRun = not 'nonbarcoded' in self.json_barcodes

            pluginconfig = self.json_dat.get('pluginconfig', {})

            #get all the button information
            self.delim       = pluginconfig.get('delimiter_select', '.')
            self.selections  = pluginconfig.get('select_dialog', ['run_name'])

            # remove any empty entries and fall back to default value if there are no selections... selected...
            self.selections[:] = [entry for entry in self.selections if entry != '']
            if len(self.selections) == 0:
                self.selections = ['run_name']

            if self.isBarcodedRun:
                # if this is a barcoded run then we need to assert that be barcode name be selected
                if not 'barcodename' in self.selections:
                    self.selections.append('barcodename')
            else:
                # remove the barcode and sample selection are not present in a non-barcoded run
                if 'barcodename' in self.selections:
                    self.selections.remove('barcodename')
                if 'sample' in self.selections:
                    self.selections.remove('sample')

            # parse all of the creation attributes
            self.bamCreate   = pluginconfig.get('bamCreate'  , 'off') == 'on'
            self.fastqCreate = pluginconfig.get('fastqCreate', 'off') == 'on'
            self.vcfCreate   = pluginconfig.get('vcfCreate'  , 'off') == 'on'
            self.xlsCreate   = pluginconfig.get('xlsCreate'  , 'off') == 'on'
            self.zipFASTQ    = pluginconfig.get('zipFASTQ'   , 'off') == 'on'
            self.zipBAM      = pluginconfig.get('zipBAM'     , 'off') == 'on'
            self.zipVCF      = pluginconfig.get('zipVCF'     , 'off') == 'on'
            self.zipXLS      = pluginconfig.get('zipXLS'     , 'off') == 'on'

            # parse if you wish to tar the results
            self.wantTar     = pluginconfig.get('compressedType', '') == 'tar'

        except:
            print('ERROR: Failed read and parse the required information sources.')
            traceback.print_exc()
            return

        # get the actual path to the variant caller plugin
        if self.vcfCreate or self.xlsCreate:
            try:
                api_url = self.json_dat['runinfo']['api_url'] + '/v1/pluginresult/?format=json&plugin__name=variantCaller&result=' + str(self.json_dat['runinfo']['pk'])
                api_key = self.json_dat['runinfo']['api_key']
                pluginNumber = self.json_dat['runinfo'].get('pluginresult') or self.json_dat['runinfo'].get('plugin',{}).get('pluginresult')
                if pluginNumber is not None and api_key is not None:
                    api_url += '&pluginresult=%s&api_key=%s' % (pluginNumber, api_key)
                else:
                    print('No API key available')
                d = json.loads(urllib2.urlopen(api_url).read())
                self.variantCallerPath = d['objects'][0]['path']
                self.variantCallerPath = self.variantCallerPath.split('/')[-1]
                self.variantExists = True

            except:
                print('ERROR!  Failed to get variant caller path. Will not generate variant files.')

        # report settings
        print("INFO: *************************************")
        print("INFO: Plugin Dir    : " + str(self.pluginDir))
        print("INFO: Download Dir  : " + str(self.downloadDir))
        print("INFO: Is Barcoded   : " + str(self.isBarcodedRun))
        print("INFO: Delimiter     : " + str(self.delim))
        print("INFO: Selections    : " + str(self.selections))
        print("INFO: Bam Create    : " + str(self.bamCreate))
        print("INFO: FastQ Create  : " + str(self.fastqCreate))
        print("INFO: VCF Create    : " + str(self.vcfCreate))
        print("INFO: XLS Create    : " + str(self.xlsCreate))
        print("INFO: Zip FastQ     : " + str(self.zipFASTQ))
        print("INFO: Zip BAM       : " + str(self.zipBAM))
        print("INFO: Zip VCF       : " + str(self.zipVCF))
        print("INFO: Zip XLS       : " + str(self.zipXLS))
        print("INFO: Want Tar      : " + str(self.wantTar))
        print("INFO: Alignment Dir : " + str(self.alignment_dir))
        print("INFO: Reference Path: " + str(self.reference_path))
        print("INFO: Basecaller Dir: " + str(self.basecaller_dir))
        print("INFO: Barcode Id    : " + str(self.barcodeId))
        print("INFO: Run name      : " + str(self.run_name))
        print("INFO: Report name   : " + str(self.report_name))
        print("INFO: Run date      : " + str(self.run_date))
        print("INFO: Chip type     : " + str(self.chiptype))
        print("INFO: Instrument    : " + str(self.instrument))
        print("INFO: Variant Exists: " + str(self.variantExists))
        print("INFO: Variant Caller: " + str(self.variantCallerPath))
        print("INFO: Analysis Dir  : " + str(self.analysis_dir))
        print("INFO: *************************************")

        # replace all of the values in the selection with their appropriate values
        self.selections = [self.run_name    if selection=='run_name'    else selection for selection in self.selections]
        self.selections = [self.report_name if selection=='report_name' else selection for selection in self.selections]
        self.selections = [self.run_date    if selection=='run_date'    else selection for selection in self.selections]
        self.selections = [self.chiptype    if selection=='chiptype'    else selection for selection in self.selections]
        self.selections = [self.instrument  if selection=='instrument'  else selection for selection in self.selections]
        self.selections = ['@SAMPLEID@'     if selection=='samplename'  else selection for selection in self.selections]
        self.selections = ['@BARINFO@'      if selection=='barcodename' else selection for selection in self.selections]
        self.renameString = self.delim.join(self.selections)

        # Create fastq file(s) if requested.
        if self.fastqCreate:
            self.createFastq()

        # Link to TVC files if requested.
        if self.bamCreate:
            self.bamRename()

        # variant caller dependent items
        if self.variantExists:
            if self.vcfCreate:
                self.vcfRename()
            if self.xlsCreate:
                self.xlsRename()

        # write HTML output
        htmlOut = open('FileExporter_block.html', 'w')
        htmlOut.write('<!DOCTYPE html>\n<html lang="en">\n<head>\n')
        htmlOut.write('<script type="text/javascript" src="/site_media/resources/jquery/jquery-1.8.2.js">\n</script>\n')
        htmlOut.write('<script>\n$(function() {\n\t$("#files").hide();\n\t$("#params").hide();\n')

        htmlOut.write('\t$("#showParams\").click(function() {\n\t\t$("#params").slideToggle("toggle");\n')
        htmlOut.write('\t\tif ($("#showParams").html() == "Show Parameters"){\n\t\t\t$("#showParams").html("Hide Parameters");\n\t\t}\n')
        htmlOut.write('\t\telse{\n\t\t\t$("#showParams").html("Show Parameters");\n\t\t}\n\t});\n\n')

        htmlOut.write('\t$("#showFiles\").click(function() {\n\t\t$("#files").slideToggle("toggle");\n')
        htmlOut.write('\t\tif ($("#showFiles").html() == "Show Contents of Compressed File"){\n\t\t\t$("#showFiles").html("Hide Contents of Compressed File");\n\t\t}\n')
        htmlOut.write('\t\telse{\n\t\t\t$("#showFiles").html("Show Contents of Compressed File");\n\t\t}\n\t});\n});\n')
        htmlOut.write('\n</script>\n</head>\n<html>\n<body>\n')
        htmlOut.write('<b>Output Files:</b><br>\n')

        # Create compressed files (note that we create html links to them if we are not adding to the compressed file)
        if self.zipBAM or self.zipFASTQ or self.zipVCF or self.zipXLS:
            zipSubdir = self.renameString
            if self.isBarcodedRun:
                zipSubdir = self.renameString.replace('@BARINFO@'+ self.delim , '')
                zipSubdir = zipSubdir.replace(self.delim +'@BARINFO@' , '')
                zipSubdir = zipSubdir.replace('@BARINFO@' , '')
                zipSubdir = zipSubdir.replace('@SAMPLEID@'+ self.delim , '')
                zipSubdir = zipSubdir.replace(self.delim +'@SAMPLEID@', '')
                zipSubdir = zipSubdir.replace('@SAMPLEID@', '')

            if self.wantTar:
                compressedFileName = zipSubdir + '.tar.bz2'
                tarName = tarfile.open(compressedFileName, "w:bz2")
                tarName.dereference = True
                htmlFiles = self.compressFiles(tarName, zipSubdir, htmlOut)
                tarName.close()
            else:
                compressedFileName = zipSubdir + '.zip'
                zipName = zipfile.ZipFile(compressedFileName, "w", zipfile.ZIP_DEFLATED, True) # note we are enabling zip64 extensions here
                htmlFiles = self.compressFiles(zipName, zipSubdir, htmlOut)
                zipName.close()

            htmlOut.write('<a href="%s" download><b>Compressed Files</b></a><br>\n'%compressedFileName)
            htmlOut.write('<div style="height:30px"><button type="button" class="btn" id="showFiles">Show Contents of Compressed File</button></div>\n')
            htmlOut.write(htmlFiles)

        # Create file links.
        else:
            for datum in sorted(os.listdir(self.pluginDir)):
                if datum.endswith('.bam') or datum.endswith('.bai') or datum.endswith('.fastq') or datum.endswith('.vcf') or datum.endswith('.xls'):
                    #at this time the download attribute with the filename specification is only supported by a few browsers
                    # if at some point it is supported fully, the process will be much easier and the download link itsself can be used to rename the file
                    htmlOut.write('<a href="%s/%s" download>%s</a><br>\n' % (self.envDict['TSP_URLPATH_PLUGIN_DIR'], datum, datum))

        htmlOut.write('<div style="height:50px"><button type="button" class="btn" id="showParams">Show Parameters</button>\n<div id="params">\n')
        htmlOut.write("<h3>Selected Parameters:</h3>\n<pre>Include BAM: %s &nbsp;Archive: %s\n" % (self.bamCreate, self.zipBAM))
        htmlOut.write("Include VCF: %s &nbsp;Archive: %s\n" % (self.vcfCreate, self.zipVCF))
        htmlOut.write("Include XLS: %s &nbsp;Archive: %s\n" % (self.xlsCreate, self.zipXLS))
        htmlOut.write("Include FASTQ: %s Archive: %s\n</pre>\n" % (self.fastqCreate, self.zipFASTQ))

        htmlOut.write('<h3>Naming Parameters:</h3>\n<pre>Delimiter: "%s"</pre>\n'%self.delim)
        for sel in self.selections:
            if sel != '':
                htmlOut.write('<pre>"%s"</pre>\n'%sel)
        htmlOut.write('</div>')
        htmlOut.close()
        return True

if __name__ == "__main__":
    PluginCLI(FileExporter())

