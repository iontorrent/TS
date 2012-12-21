#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved


"""Generate PDF file of the Report Page"""

import os
import sys
import json
import urllib2
import logging

import iondb.bin.djangoinit
from django import shortcuts

from iondb.rundb import models
import logging
import subprocess
import urllib
import glob

logger = logging.getLogger(__name__)

import urllib2

def get_status_code(url):
    """get the HTTP status of a URL, returns an int"""
    try:
        connection = urllib2.urlopen(url)
        connection.close()
        return connection.getcode()
    except urllib2.HTTPError, e:
        return None

def makePDFdir(pkR, dirname):
    host = "http://127.0.0.1"
    savePath = dirname
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
            paths.append(JSON['Path'])

    #if there is no plugin output return false 
    if not fileNames:
        return False 

    #create the directory to store the pdf files
    try:
        os.makedirs(os.path.join(savePath,"pdf"))
    except OSError:
        pass

    def create_pdf(url, outputFile, title):
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
            os.system(pdf_str)
        except Exception as inst:
            logger.exception("PDF generation error")
            pass

    #get the links for the plugin pages. Save them in one string with spaces in between; wkhtmltopdf takes them as 'arg1 arg2 arg3 arg4...'
    for i, entry in enumerate(fileNames):
        #create the url
        entry_url = os.path.join(ret.reportWebLink(), 'plugin_out', paths[i], entry)
        full_url = host + entry_url
        #check to see if it returns a 200 code 
        if get_status_code(full_url) == 200:
            create_pdf( host + "/" + entry_url,
                        os.path.join(savePath,"pdf", paths[i] + "-" + entry + ".pdf"),
                        paths[i] + "-" + entry
                      )
        else:
            logger.debug("Did NOT get 200 response from " + full_url)

    pdfs = glob.glob(savePath + "/pdf/*.pdf")
    pdf_string = " ".join(pdfs)

    try:
        os.system("pdftk " + pdf_string + " cat output " + savePath + "/plugins.pdf")
        return True
    except Exception as inst:
        logger.exception("PDF generation error")
        return False

def latex(pkR, directory):
 
    page_url = "http://127.0.0.1/report/" + str(pkR) + "/?latex=1"
    latex = urllib.urlretrieve(page_url , directory + "/report.tex")
    pdf =  ["pdflatex",  directory + "/report.tex", "-output-directory", directory
    ,"-interaction", "batchmode"]

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

    for plugin, pluginFile in major_plugins.iteritems():
        #check that the plugin has a file to display
        if pluginFile:
            try:
                plugin_image =  ["/opt/ion/iondb/bin/wkhtmltoimage-amd64", "--crop-w", "1000"
                , pluginFile, os.path.join(directory,"pdf",plugin + ".png")]
                plugin_proc = subprocess.Popen(plugin_image, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=directory)
                plugin_stdout, plugin_stderr = plugin_proc.communicate()
                logger.warning("OKAY creating PNG of plugin : %s " % plugin_image)
            except:
                logger.warning("ERROR creating PNG of plugin : %s " % plugin)


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
    try:
        os.system("pdftk " + latexReport + " " + pluginReport + " cat output " + ret.get_report_dir() + "/backupPDF.pdf")
    except Exception as inst:
        logger.exception("Full PDF generation error")
        pass

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
    pdf_str = '/opt/ion/iondb/bin/wkhtmltopdf-amd64 -q'
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
