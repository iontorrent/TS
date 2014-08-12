#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

"""Generate PDF files of the Report Page and associated plugin results pages."""

from __future__ import division
import os
import sys
import json
import urllib2
import logging
import shutil
import subprocess
import urllib
import glob
import Image
import math
import traceback
from django import shortcuts

from iondb.rundb import models

logger = logging.getLogger(__name__)

REPORT_PDF = "report.pdf"
PLUGIN_PDF = "plugins.pdf"

def write_report_pdf(_result_pk, report_dir=None):
    '''Writes pdf file of the Report Page'''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    result_obj = shortcuts.get_object_or_404(models.Results, pk=_result_pk)
    if not report_dir:
        report_dir = result_obj.get_report_dir()
    #first get all the major plugins
    major_plugins = {}
    pluginList = models.PluginResult.objects.filter(result__pk=_result_pk)
    for major_plugin in pluginList:
        if major_plugin.plugin.majorBlock:
            #list all of the _blocks for the major plugins, just use the first one
            try:
                majorPluginFiles = glob.glob(os.path.join(major_plugin.path(),"*_block.html"))[0]
            except IndexError:
                majorPluginFiles = False

            major_plugins[major_plugin.plugin.name] = majorPluginFiles

    #make sure the pdf dir is there
    try:
        os.makedirs(os.path.join(report_dir,"pdf"))
    except OSError:
        pass

    #if there are major plugins take screenshots of them and include them in the PDF
    for plugin, pluginFile in major_plugins.iteritems():
        #check that the plugin has a file to display
        if pluginFile:
            try:
                image_path = os.path.join(report_dir, "pdf", plugin + ".png")
                plugin_image =  [
                    "/opt/ion/iondb/bin/wkhtmltoimage-amd64",
                    "--width",
                    "1024",
                    "--crop-w",
                    "1024",
                    pluginFile,
                    image_path
                    ]
                proc = subprocess.Popen(plugin_image, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=report_dir)
                stdout, stderr = proc.communicate()
                if proc.returncode:
                    logger.warn("Error executing %s" % plugin_image[0])
                    logger.warn(" stdout: %s" % stdout)
                    logger.warn(" stderr: %s" % stderr)
                    return False
                #now the fancy part, split the image up
                long_slice(image_path, plugin, os.path.join(report_dir,"pdf"), 1200)
            except:
                logger.warning("ERROR creating PNG of plugin : %s " % plugin)

    page_url = "http://127.0.0.1/report/" + str(_result_pk) + "/?latex=1"
    urllib.urlretrieve(page_url , report_dir + "/report.tex")
    pdf = [
        "pdflatex",
        report_dir + "/report.tex",
        "-output-directory",
        report_dir,
        "-interaction",
        "batchmode"
        ]

    proc = subprocess.Popen(pdf, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=report_dir)
    stdout, stderr = proc.communicate()
    if proc.returncode > 1:     # TODO: Why does pdflatex return non-zero status despite creating the pdf file?
        logger.warn("Error executing %s" % pdf[0])
        logger.warn(" stdout: %s" % stdout)
        logger.warn(" stderr: %s" % stderr)
        return False

    cleanup_latex_files(report_dir)

    return os.path.join(report_dir, REPORT_PDF)


def write_plugin_pdf(_result_pk, directory = None):
    '''Writes pdf files of the plugin results pages'''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)

    def create_pdf(url, outputFile, title):
        '''Creates pdf of plugin's output page'''
        pdf_str = '/opt/ion/iondb/bin/wkhtmltopdf-amd64 -q'
        pdf_str += ' --javascript-delay 1200'
        pdf_str += ' --no-outline'
        pdf_str += ' --margin-top 15'
        pdf_str += ' --header-spacing 5'
        pdf_str += ' --footer-left "' + title + '"'
        pdf_str += ' --footer-right "Page [page] of [toPage]"'
        pdf_str += ' --header-left "[title]"'
        pdf_str += ' --footer-font-size 12'
        pdf_str += ' --header-font-size 12'
        pdf_str += ' --disable-internal-links'
        pdf_str += ' --disable-external-links'
        pdf_str += ' --outline'
        pdf_str += ' ' + url + " " + outputFile
        logger.debug("PDFSTR for " + url + " was " + pdf_str)
        try:
            retcode = subprocess.call(pdf_str, shell = True)
            if retcode < 0:
                logger.error("Child was terminated by signal %d" % (-retcode))
            else:
                logger.info("Child returned %d" % (retcode))
        except Exception as inst:
            logger.exception("create_pdf error")

    result_obj = shortcuts.get_object_or_404(models.Results, pk=_result_pk)
    report_dir = result_obj.get_report_dir()
    if directory:
        report_dir = directory

    #==========================================================================
    # Get list of plugins and their output html pages
    #==========================================================================
    host = "http://127.0.0.1"
    djangoURL =  "%s/rundb/api/v1/results/%s/pluginresults/?format=json" % (host, _result_pk)
    pageOpener = urllib2.build_opener()
    jsonPage = pageOpener.open(djangoURL)
    djangoJSON = jsonPage.read()
    decodedJSON = json.loads(djangoJSON)

    fileNames = []
    paths = []
    for JSON in decodedJSON:
        roughFN = JSON['Files']
        for filename in roughFN[:1]:    # TODO: Don't we want to grab all html??
        #for filename in roughFN:    # Grab every html file
            fileNames.append(filename)
            paths.append(JSON['URL'])

    #if there is no plugin output return false
    if not fileNames:
        return None

    #create the directory to store the pdf files
    try:
        os.makedirs(os.path.join(report_dir, "pdf"))
    except OSError:
        pass

    #=========================================================================
    # Create pdf for each html file
    #=========================================================================
    for i, entry in enumerate(fileNames):
        #create the url
        full_url = host + os.path.join(paths[i], entry)
        #check to see if it returns a 200 code
        if get_status_code(full_url) == 200:
            uniq = paths[i].split("/")[-2]
            outpath  = os.path.join(report_dir, "pdf" ,  uniq + entry +  ".pdf")
            create_pdf(full_url, outpath, entry)
        else:
            logger.debug("Did NOT get 200 response from " + full_url)

    #=========================================================================
    # Concatenate all the individual plugin pdf files into single pdf
    #=========================================================================
    cmd = "/usr/bin/pdftk " + os.path.join(report_dir, "pdf", "*.pdf") + " cat output " + os.path.join(report_dir, "plugins.pdf")
    logger.debug("Command String is:\"%s\"" % cmd)

    try:
        retcode = subprocess.call(cmd, shell = True)
        if retcode < 0:
            logger.error("Child was terminated by signal %d" % (-retcode))
            return None
        else:
            logger.info("Child returned %d" % (retcode))
            return os.path.join(report_dir, PLUGIN_PDF)
    except OSError as e:
        logger.error("Execution failed: %s" % e)
        if os.path.exists(os.path.join(report_dir, PLUGIN_PDF)):
            return os.path.join(report_dir, PLUGIN_PDF)
        else:
            return None


def write_summary_pdf(_result_pk, directory = None):
    '''Writes pdf file combining Report Page and Plugin Pages'''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)

    result_obj = shortcuts.get_object_or_404(models.Results, pk=_result_pk)
    report_dir = result_obj.get_report_dir()
    if directory:
        report_dir = directory
    pdf_reportfile = write_report_pdf(_result_pk)
    pdf_pluginfile = write_plugin_pdf(_result_pk)
    pdf_summaryfile = os.path.join(report_dir, os.path.basename(report_dir)+"-full.pdf")
    if pdf_reportfile and os.path.exists(pdf_reportfile):
        if pdf_pluginfile and os.path.exists(pdf_pluginfile):

            cmd = ['pdftk', pdf_reportfile, pdf_pluginfile, 'cat', 'output', pdf_summaryfile]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                logger.error(stdout)
                logger.error(stderr)

        if not os.path.exists(pdf_summaryfile):
            shutil.copyfile(pdf_reportfile, pdf_summaryfile)

    else:
        logger.exception("Error generating %s" % pdf_reportfile)

    try:
        cleanup_latex_files(report_dir)
    except:
        pass

    return os.path.join(pdf_summaryfile)


def get_summary_pdf(pkR):
    '''Report Page + Plugins Page PDF'''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    filename = write_summary_pdf(pkR)
    if filename:
        return open(filename)
    else:
        return False


def get_plugin_pdf(pkR):
    '''Plugins Page PDF'''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    filename = write_plugin_pdf(pkR)
    if filename:
        return open(filename)
    else:
        return False


def get_report_pdf(pkR):
    '''Report Page PDF'''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    filename = write_report_pdf(pkR)
    if filename:
        return open(filename)
    else:
        return False


def cleanup_latex_files(_report_dir):
    '''Cleanup intermediate files created by latex'''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    # Clean up intermediate files
    for file in ['report.tex', 'report.aux', 'report.log']:
        filepath = os.path.join(_report_dir, file)
        if os.path.exists(filepath):
            os.remove(filepath)


def enum(iterable, start = 1):
    """enumerate but with a starting position"""
    n = start
    for i in iterable:
        yield n, i
        n += 1


def long_slice(image_path, out_name, out_dir, slice_size):
    """slice an image into parts slice_size tall"""
    img = Image.open(image_path)
    width, height = img.size

    upper = 0

    slices = int(math.ceil(height/slice_size))

    for i, slice in enum(range(slices)):
        left = 0
        upper = upper
        if i == slices:
            lower = height
        else:
            lower = int(i * slice_size)
        bbox = (left, upper, width, lower)
        working_slice = img.crop(bbox)
        upper += slice_size
        working_slice.save(os.path.join(out_dir, "slice_" + out_name + "_" + str(i)+".png"))


def get_status_code(url):
    """get the HTTP status of a URL, returns an int"""
    try:
        connection = urllib2.urlopen(url)
        connection.close()
        return connection.getcode()
    except urllib2.HTTPError:
        return None


#NOTE: May not work as advertised
def get_pdf_for_report_directory(directory):
    '''
    Instead of using the database primary key of a report, use a report directory
    to generate a PDF.
    '''
    #TODO: Calculate the primary key of this report directory
    # Isolate the Report Directory from the fullpath
    reportnamedir = os.path.split(os.path.abspath(directory))[1]
    # Strip off the underscore and experiment PK to get the Report Name
    reportname = reportnamedir.replace("_"+reportnamedir.rsplit("_")[-1],"")
    # Lookup the Report Name in the database
    pkR = models.Results.objects.get(resultsName=reportname).id
    print "Got PK = %d" % (pkR)
    pdfpath = write_summary_pdf(pkR)
    print "Wrote file: %s" % (pdfpath)
    return

#NOTE: May not work as advertised
def generate_pdf_from_archived_report(source_dir):
    from iondb.rundb.data.dmactions import _copy_to_dir
    import shutil
    def get_reportPK(directory):
        # Isolate the Report Directory from the fullpath
        reportnamedir = os.path.split(os.path.abspath(directory))[1]
        # Strip off the underscore and experiment PK to get the Report Name
        reportname = reportnamedir.replace("_"+reportnamedir.rsplit("_")[-1],"")
        # Lookup the Report Name in the database
        reportPK = models.Results.objects.get(resultsName=reportname)
        return reportPK

    result = get_reportPK(source_dir)
    dmfilestat = result.get_filestat('Output Files')
    report_dir = result.get_report_dir()
    # set archivepath for get_report_dir to find files when generating pdf
    print "Current archivepath: %s" % (dmfilestat.archivepath)
    print "Changing to: %s" % (source_dir)
    dmfilestat.archivepath = source_dir
    dmfilestat.save()

    if False:
        pdfpath = write_summary_pdf(result.pk)
        print "Wrote file: %s" % (pdfpath)
        shutil.copyfile(pdfpath, os.path.join(os.path.abspath(os.path.split(pdfpath)[0]), 'backupPDF.pdf'))
    else:
        # create report pdf via latex
        latex_filepath = os.path.join('/tmp', os.path.basename(report_dir)+'-full.tex' )
        url = "http://127.0.0.1/report/" + str(result.pk) + "/?latex=1"
        urllib.urlretrieve(url , latex_filepath)
        pdf = ["pdflatex", "-output-directory", "/tmp", "-interaction", "batchmode", latex_filepath]
        proc = subprocess.Popen(pdf, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=source_dir)
        stdout, stderr = proc.communicate()
        if stderr:
            log.write('Error: '+ stderr)
        else:
            _copy_to_dir(os.path.join('/tmp', os.path.basename(report_dir)+'-full.pdf' ), '/tmp', report_dir)
            shutil.copyfile(os.path.join('/tmp', os.path.basename(report_dir)+'-full.pdf' ),
                            os.path.join(report_dir, 'backupPDF.pdf'))
    return


if __name__ == '__main__':
    #if(len(sys.argv) > 1):
    #    write_report_pdf(sys.argv[1], directory = "./")
    #else:
    #    print "Need to provide a Report's pk"
    if(len(sys.argv) > 1):
        generate_pdf_from_archived_report(sys.argv[1], "./")
    else:
        print "Need to provide a Report's pk"
