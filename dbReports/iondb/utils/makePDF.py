#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

"""Generate PDF file of the Report Page"""

from __future__ import division
import os
import sys
import json
import urllib2
import logging
import shutil

import iondb.bin.djangoinit
from django import shortcuts
from pwd import getpwnam

from iondb.rundb import models
import logging
import subprocess
import urllib
import glob

import Image
import math

logger = logging.getLogger(__name__)

import urllib2

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

    for i,slice in enum(range(slices)):
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
    except urllib2.HTTPError, e:
        return None

def makePDFdir(pkR, savePath):
    """this function makes one combined PDF for all of the plugin output"""

    host = "http://127.0.0.1"
    ret = shortcuts.get_object_or_404(models.Results, pk=pkR)
    djangoURL =  "%s/rundb/api/v1/results/%s/pluginresults/?format=json" % (host, pkR)
    pageOpener = urllib2.build_opener()
    jsonPage = pageOpener.open(djangoURL)
    djangoJSON = jsonPage.read()
    decodedJSON = json.loads(djangoJSON)

    fileNames = []
    paths = []
    for JSON in decodedJSON:
        roughFN = JSON['Files']
        try:
            fn = roughFN[0]
        except IndexError:
            #some plugins don't produce files
            continue

        #if it doesn't end in '.html', remove the '/' that will be there.
        if fn.endswith("/"):
            fn = fn[:-1]
        #if there's no filename, the plugin didn't work properly; don't bother appending it to the list.
        if fn != '':
            fileNames.append(fn)
            paths.append(JSON['URL'])

    #if there is no plugin output return false
    if not fileNames:
        return False

    #create the directory to store the pdf files
    try:
        os.makedirs(os.path.join(savePath,"pdf"))
        os.chown(os.path.join(savePath,"pdf"),getpwnam('www-data').pw_uid,getpwnam('www-data').pw_gid)
    except OSError:
        pass

    def create_pdf(url, outputFile, title):
        pdf_str = 'sudo -u www-data '
        pdf_str += '/opt/ion/iondb/bin/wkhtmltopdf-amd64 -q'
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
            os.system(pdf_str)
        except Exception as inst:
            logger.exception("PDF generation error")

    for i, entry in enumerate(fileNames):
        #create the url
        entry_url = paths[i] + entry
        full_url = host + entry_url
        #check to see if it returns a 200 code
        if get_status_code(full_url) == 200:
            uniq = paths[i].split("/")[-2]
            outpath  = os.path.join(ret.get_report_dir(), "pdf" ,  uniq + entry +  ".pdf")
            create_pdf( full_url, outpath ,  entry)
        else:
            logger.debug("Did NOT get 200 response from " + full_url)

    pdfs = glob.glob( ret.get_report_dir() + "/pdf/*.pdf")
    pdf_string = " ".join(pdfs)

    try:
        os.system("sudo -u www-data pdftk " + pdf_string + " cat output " + savePath + "/plugins.pdf")
        return True
    except Exception as inst:
        logger.exception("PDF generation error")
        return False

def latex(pkR, directory):
    """get the latex, create the major plugin images and render the pdf"""

    #first get all the major plugins
    major_plugins = {}
    pluginList = models.PluginResult.objects.filter(result__pk=pkR)
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
        os.makedirs(os.path.join(directory,"pdf"))
    except OSError:
        pass

    #if there are major plugins take screenshots of them and include them in the PDF
    for plugin, pluginFile in major_plugins.iteritems():
        #check that the plugin has a file to display
        if pluginFile:
            try:
                image_path = os.path.join(directory,"pdf",plugin + ".png")
                plugin_image =  ['sudo','-u','www-data',"/opt/ion/iondb/bin/wkhtmltoimage-amd64", "--width", "1024", "--crop-w", "1024"
                , pluginFile, image_path]
                plugin_proc = subprocess.Popen(plugin_image, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=directory)
                plugin_stdout, plugin_stderr = plugin_proc.communicate()
                #now the fancy part, split the image up
                long_slice(image_path, plugin, os.path.join(directory,"pdf"), 1200)
            except:
                logger.warning("ERROR creating PNG of plugin : %s " % plugin)

    page_url = "http://127.0.0.1/report/" + str(pkR) + "/?latex=1"
    latex = urllib.urlretrieve(page_url , directory + "/report.tex")
    pdf =  ['sudo','-u','www-data',"pdflatex",  directory + "/report.tex", "-output-directory", directory
        ,"-interaction", "batchmode"]

    proc = subprocess.Popen(pdf, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=directory)
    stdout, stderr = proc.communicate()

    logger.info("LaTeX pdf created using this command, '%s'" % " ".join(pdf))
    if stderr:
        logger.warning("ERROR creating LaTeX PDF: %s" % stderr)


#if it gets called without a dirname argument, just use the report's directory.
def makePDF(pkR):
    ret = shortcuts.get_object_or_404(models.Results, pk=pkR)
    makePDFdir(pkR, ret.get_report_dir())
    latex(pkR, ret.get_report_dir())

    latexReport = os.path.join(ret.get_report_dir(),"report.pdf")
    pluginReport = os.path.join(ret.get_report_dir(),"plugins.pdf")
    fullReport = os.path.join(ret.get_report_dir(),"backupPDF.pdf")
    if os.path.exists(latexReport):
        if os.path.exists(pluginReport):

            #cmd = "pdftk %s %s cat output %s" % (latexReport,pluginReport,fullReport)
            #os.system(cmd)
            cmd = ['pdftk', latexReport, pluginReport, 'cat', 'output', fullReport]
            cmd = ['sudo','-u','www-data','pdftk', latexReport, pluginReport, 'cat', 'output', fullReport]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            plugin_stdout, plugin_stderr = proc.communicate()
            if proc.returncode != 0:
                logger.error(plugin_stderr)

        if not os.path.exists(fullReport):
            try:
                shutil.copyfile(latexReport,fullReport)
            except:
                logger.exception("Copy %s to %s error" % (latexReport,fullReport))

        # Clean up intermediate files
        for file in ['report.tex','report.aux','report.log']:
            filepath = os.path.join(ret.get_report_dir(),file)
            if os.path.exists(filepath):
                os.remove(filepath)

    else:
        logger.exception("Error generating %s" % latexReport)

def getPDF(pkR):
    ret = shortcuts.get_object_or_404(models.Results, pk=pkR)
    makePDF(pkR)
    return open(ret.get_report_dir() + "/backupPDF.pdf")

def getPlugins(pkR):
    ret = shortcuts.get_object_or_404(models.Results, pk=pkR)
    status = makePDFdir(pkR, ret.get_report_dir())
    if status:
        return open(ret.get_report_dir() + "/plugins.pdf")
    else:
        return False

def getlatex(pkR):
    ret = shortcuts.get_object_or_404(models.Results, pk=pkR)
    error = latex(pkR, ret.get_report_dir())
    return open(ret.get_report_dir() + "/report.pdf")

def getOldPDF(pkR):
    ret = shortcuts.get_object_or_404(models.Results, pk=pkR)
    makeOldPDFdir(pkR, ret.get_report_dir())
    return open(ret.get_report_dir() + "/backupPDF.pdf")


def makeOldPDFdir(pkR, dirname):
    savePath = dirname
    ret = shortcuts.get_object_or_404(models.Results, pk=pkR)
    djangoURL = "http://127.0.0.1/rundb/api/v1/results/%s/pluginresults/?format=json" % pkR
    pageOpener = urllib2.build_opener()
    jsonPage = pageOpener.open(djangoURL)
    djangoJSON = jsonPage.read()
    decodedJSON = json.loads(djangoJSON)

    fileNames = []
    paths = []
    for JSON in decodedJSON:
        roughFN = JSON['Files']
        fn = roughFN[0]

        #if it doesn't end in '.html', remove the '/' that will be there.
        if fn.endswith("/"):
            fn = fn[:-1]
        #if there's no filename, the plugin didn't work properly; don't bother appending it to the list.
        if fn != '':
            fileNames.append(fn)
            paths.append(JSON['Path'])

    link = ret.reportWebLink() + "/"

    page = ""
    page_url = "http://127.0.0.1" + link + "/?no_header=True "
    #get the links for the plugin pages. Save them in one string with spaces in between; wkhtmltopdf takes them as 'arg1 arg2 arg3 arg4...'
    for i, entry in enumerate(fileNames):
        newPage = "http://127.0.0.1" + os.path.join(link, 'plugin_out', paths[i], entry) + " "
        page += newPage

    #this string will be really long for reports with lots of plugins. One of the bigger ones caused wkhtmltopdf to segfault.
    #but, it did successfully produce a .pdf with the report and all of the plugins in one piece, so I'm not really sure what to make of that.
    pdf_str = "sudo -u www-data "
    pdf_str += '/opt/ion/iondb/bin/wkhtmltopdf-amd64 -q'
    pdf_str += ' --margin-top 15'
    pdf_str += ' --header-spacing 5'
    pdf_str += ' --header-left " ' + ret.resultsName + ' - [date]"'
    pdf_str += ' --header-right "Page [page] of [toPage]"'
    pdf_str += ' --header-font-size 9'
    pdf_str += ' --disable-internal-links'
    pdf_str += ' --disable-external-links'
    pdf_str += ' --enable-forms ' + page_url + page + os.path.join(savePath, 'backupPDF.pdf')

    try:
        os.system(pdf_str)
    except:
        pass

if __name__ == '__main__':
    if(len(sys.argv) > 1):
        makePDFdir(sys.argv[1], "./")
        #makeOldPDFdir(sys.argv[1],"./")
    else:
        print "Need to provide a Report's pk"
