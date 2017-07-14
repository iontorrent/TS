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
from django import shortcuts

from iondb.rundb import models

logger = logging.getLogger(__name__)

LOCALHOST = "127.0.0.1"
REPORT_PDF = "report.pdf"
PLUGIN_PDF = "plugins.pdf"


def write_report_pdf(_result_pk, output_dir=None):
    '''Writes pdf file of the Report Page'''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    result_obj = shortcuts.get_object_or_404(models.Results, pk=_result_pk)
    report_dir = result_obj.get_report_dir()
    if not output_dir:
        output_dir = report_dir
    
    pdf_dir = os.path.join(report_dir, "pdf")
    # create pdf dir if does not exist
    try:
        os.makedirs(pdf_dir)
    except OSError:
        pass

    # if there are major plugins take screenshots of them to include in the PDF
    plugins = result_obj.pluginresult_set.filter(plugin__majorBlock=True)

    # for plugins using Kendo tables this javascript will show all rows on single page
    js_str = "$('.k-grid table').each(function(){var dataSource=$(this).data('kendoGrid').dataSource; dataSource.pageSize(dataSource.total());})"
    for major_plugin in plugins:
        try:
            # list all of the _blocks for the major plugins, just use the first one
            majorPluginFile = glob.glob(os.path.join(major_plugin.path(), "*_block.html"))[0]
            pluginPath, pluginFile = os.path.split(majorPluginFile)
            pluginName = major_plugin.plugin.name
        except IndexError:
            continue

        try:
            pluginOutLink = pluginPath.replace(report_dir, result_obj.reportLink)
            url = "http://" + LOCALHOST + os.path.normpath(os.path.join(pluginOutLink, pluginFile))
            image_path = os.path.join(pdf_dir, pluginName + ".png")
            # create png file
            _wkhtmltopdf_create_image(url, image_path, js_str)
            # now the fancy part, split the image up
            long_slice(image_path, pluginName, pdf_dir, 1200)
        except Exception as e:
            logger.warning("ERROR creating PNG of plugin : %s " % pluginName)
            logger.warning(e)

    # get latex version of the report page
    page_url = "http://%s/report/%s/?latex=1" % (LOCALHOST, _result_pk)
    urllib.urlretrieve(page_url, output_dir + "/report.tex")
    cmd = [
        "pdflatex",
        output_dir + "/report.tex",
        "-output-directory",
        output_dir,
        "-interaction",
        "batchmode"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=output_dir)
    stdout, stderr = proc.communicate()
    if proc.returncode > 1:     # Why does pdflatex return non-zero status despite creating the pdf file?
        logger.warn("Error executing %s" % cmd[0])
        logger.warn(" stdout: %s" % stdout)
        logger.warn(" stderr: %s" % stderr)
        return False

    cleanup_latex_files(output_dir)

    return os.path.join(output_dir, REPORT_PDF)


def write_plugin_pdf(_result_pk, directory=None):
    '''Writes pdf files of the plugin results pages'''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)

    result_obj = shortcuts.get_object_or_404(models.Results, pk=_result_pk)
    report_dir = result_obj.get_report_dir()
    if directory:
        report_dir = directory

    #==========================================================================
    # Get list of plugins and their output html pages
    #==========================================================================
    host = "http://%s" % LOCALHOST
    djangoURL = "%s/rundb/api/v1/pluginresult/?result=%s" % (host, _result_pk)
    pageOpener = urllib2.build_opener()
    jsonPage = pageOpener.open(djangoURL)
    djangoJSON = jsonPage.read()
    decodedJSON = json.loads(djangoJSON)['objects']

    plugins = []
    for JSON in decodedJSON:
        files = JSON['files']
        if files:
            filename = files[0]
            plugins.append({
                'name': JSON['pluginName'],
                'id': JSON['id'],
                'url': os.path.join(JSON['URL'], filename),
                'path': os.path.join(JSON['path'], filename),
                'filename': filename
            })

    # if there is no plugin output return false
    if not plugins:
        return None

    # create the directory to store the pdf files
    try:
        os.makedirs(os.path.join(report_dir, "pdf"))
    except OSError:
        pass

    #=========================================================================
    # Create pdf for each html file
    #=========================================================================
    plugin_pdf_files = []
    # for plugins using Kendo tables this javascript will show all rows on single page
    js_str = "$('.k-grid table').each(function(){var dataSource=$(this).data('kendoGrid').dataSource; dataSource.pageSize(dataSource.total());})"
    for plugin in plugins:
        # create the url
        full_url = host + plugin['url']
        # check to see if it returns a 200 code
        if get_status_code(full_url) == 200:
            outpath = os.path.join(report_dir, "pdf", "%s.%s.pdf" % (plugin['filename'], plugin['id']) )
            _wkhtmltopdf_create_pdf(full_url, outpath, plugin['filename'], plugin['path'], js_str)
            plugin_pdf_files.append(outpath)
        else:
            logger.debug("Did NOT get 200 response from " + full_url)

    #=========================================================================
    # Concatenate all the individual plugin pdf files into single pdf
    #=========================================================================
    cmd = "/usr/bin/pdftk " + ' '.join(plugin_pdf_files) + " cat output " + os.path.join(report_dir, "plugins.pdf")
    logger.debug("Command String is:\"%s\"" % cmd)

    try:
        retcode = subprocess.call(cmd, shell=True)
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


def write_summary_pdf(_result_pk, directory=None):
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
    for filename in ['report.tex', 'report.aux', 'report.log']:
        filepath = os.path.join(_report_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)


def enum(iterable, start=1):
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

    for i, _ in enum(range(slices)):
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


# NOTE: May not work as advertised
def get_pdf_for_report_directory(directory):
    '''
    Instead of using the database primary key of a report, use a report directory
    to generate a PDF.
    '''
    # Isolate the Report Directory from the fullpath
    reportnamedir = os.path.split(os.path.abspath(directory))[1]
    # Strip off the underscore and experiment PK to get the Report Name
    reportname = reportnamedir.replace("_"+reportnamedir.rsplit("_")[-1], "")
    # Lookup the Report Name in the database
    pkR = models.Results.objects.get(resultsName=reportname).id
    print "Got PK = %d" % (pkR)
    pdfpath = write_summary_pdf(pkR)
    print "Wrote file: %s" % (pdfpath)
    return

# NOTE: May not work as advertised


def generate_pdf_from_archived_report(source_dir):
    from iondb.rundb.data.dmactions import _copy_to_dir

    def get_reportPK(directory):
        # Isolate the Report Directory from the fullpath
        reportnamedir = os.path.split(os.path.abspath(directory))[1]
        # Strip off the underscore and experiment PK to get the Report Name
        reportname = reportnamedir.replace("_"+reportnamedir.rsplit("_")[-1], "")
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
        latex_filepath = os.path.join('/tmp', os.path.basename(report_dir)+'-full.tex')
        url = "http://127.0.0.1/report/" + str(result.pk) + "/?latex=1"
        urllib.urlretrieve(url, latex_filepath)
        pdf = ["pdflatex", "-output-directory", "/tmp", "-interaction", "batchmode", latex_filepath]
        proc = subprocess.Popen(pdf, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=source_dir)
        _, stderr = proc.communicate()
        if stderr:
            logger.error('Error: ' + stderr)
        else:
            _copy_to_dir(os.path.join('/tmp', os.path.basename(report_dir)+'-full.pdf'), '/tmp', report_dir)
            shutil.copyfile(os.path.join('/tmp', os.path.basename(report_dir)+'-full.pdf'),
                            os.path.join(report_dir, 'backupPDF.pdf'))
    return


def _wkhtmltopdf_create_image(url, outputFile, js_str=""):
    pdf_str = "/opt/ion/iondb/bin/wkhtmltoimage-amd64"
    if js_str:
        pdf_str += ' --run-script "%s"' % js_str.replace("$", "\$")
    pdf_str += ' --javascript-delay 1200'
    pdf_str += ' --width 1024 --crop-w 1024'
    pdf_str += ' ' + url
    pdf_str += ' ' + outputFile

    proc = subprocess.Popen(pdf_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = proc.communicate()
    if proc.returncode:
        logger.warn("Error executing %s" % pdf_str)
        logger.warn(" stdout: %s" % stdout)
        logger.warn(" stderr: %s" % stderr)
        return False


def _wkhtmltopdf_create_pdf(url, outputFile, name, filePath="", js_str=""):
    '''Creates pdf of plugin's output page'''
    pdf_str = '/opt/ion/iondb/bin/wkhtmltopdf-amd64 -q'
    if js_str:
        pdf_str += ' --run-script "%s"' % js_str.replace("$", "\$")

    pdf_str += ' --javascript-delay 1200'
    pdf_str += ' --margin-top 5 --margin-bottom 5 --margin-left 5 --margin-right 5'
    pdf_str += ' --footer-left "' + name + '"'
    pdf_str += ' --footer-right "Page [page] of [toPage]"'
    pdf_str += ' --header-left "[title]"'
    pdf_str += ' --footer-font-size 12'
    pdf_str += ' --header-font-size 12'
    pdf_str += ' --disable-internal-links'
    pdf_str += ' --disable-external-links'
    pdf_str += ' --outline'
    pdf_str += ' %s '
    pdf_str += outputFile
    # prevent stuck wkhtmltopdf jobs
    pdf_str = "timeout 5m " + pdf_str

    try:
        p = subprocess.Popen(pdf_str % url, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = p.communicate()
        # known issue in wkhtmltopdf: sometimes it returns exit code 2 but the pdf file is actually created
        if p.returncode and p.returncode != 2:
            logger.error("wkhtmltopdf_create_pdf from %s returned %d %s" % (url, p.returncode, stderr))
            # retry from html file path directly
            if filePath:
                retcode = subprocess.call(pdf_str % filePath, shell=True)
                logger.debug("wkhtmltopdf_create_pdf from %s returned %d" % (filePath,retcode))
    except Exception:
        logger.exception("create_pdf error for %s" % name)


if __name__ == '__main__':
    # if(len(sys.argv) > 1):
    #    write_report_pdf(sys.argv[1], directory = "./")
    # else:
    #    print "Need to provide a Report's pk"
    if len(sys.argv) > 1:
        generate_pdf_from_archived_report(sys.argv[1])
    else:
        print "Need to provide a Report's pk"
