#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
import os
from os import path
import sys
import commands
import statvfs
import datetime

import iondb.bin.djangoinit
from iondb.rundb import models
from django import shortcuts

import json
from iondb.rundb import json_field

import urllib2
import urllib

'''
Args:
X: X = the pk of the report to make a .pdf of.
'''

def makePDFdir(pkR, dir):
    ret = shortcuts.get_object_or_404(models.Results, pk=pkR)
    djangoURL = "http://localhost/rundb/api/v1/results/%s/pluginresults/?format=json"%pkR
    pageOpener = urllib2.build_opener()
    jsonPage = pageOpener.open(djangoURL)
    djangoJSON = jsonPage.read()
    decodedJSON = json.loads(djangoJSON)
    
    fileNames = []
    names = []
    for JSON in decodedJSON:
        #print JSON
        roughFN = '%s'%JSON['Files']
        #get rid of the apostrophes to make a happy file name.
        fn = roughFN[roughFN.find("'")+1:]
        fn = fn[:fn.find("'")]
        roughName = '%s'%JSON['Name']
        #if it doesn't end in '.html', remove the '/' that will be there.
        if fn[len(fn)-1:] == '/':
            fn = fn[:len(fn)-1]
        #if there's no filename, the plugin didn't work properly; don't bother appending it to the list.
        #also, the name will still exist if the plugin failed, but should only be added if it succeeded.
        if fn != '':
            fileNames.append(fn)
            names.append(roughName)
    
    link = "/report/%s"%pkR
    
    page = ""
    page_url = "http://localhost"+link+"?no_header=True"
    #get the links for the plugin pages. Save them in one string with spaces in between; wkhtmltopdf takes them as 'arg1 arg2 arg3 arg4...'
    for i, entry in enumerate(fileNames):
        url =  ret.reportLink + 'plugin_out/' + names[i] + '_out/' + entry
        name = names[i]

        newPage = " http://localhost"
        newPage += url
        page += newPage

    #file.write("args created...")
    #note to self: probably could just use dir in the following line...
    savePath = dir
    #this string will be really long for reports with lots of plugins. One of the bigger ones caused wkhtmltopdf to segfault.
    #but, it did successfully produce a .pdf with the report and all of the plugins in one piece, so I'm not really sure what to make of that. 
    pdf_str = '/opt/ion/iondb/bin/wkhtmltopdf-amd64 -q'
    pdf_str += ' --javascript-delay 1000 --no-outline --margin-top 15 --header-spacing 5 --header-left "[title]" --header-right "Page [page] of [toPage]" --header-font-size 12 --disable-internal-links --disable-external-links '
    pdf_str += page_url + page + " " + savePath +'/backupPDF.pdf'

    #have to cook up a way to get all the plugin output on one page

    try:
        #file.write("%s..."%pdf_str)
        tmpstr = os.system(pdf_str)
        #file.write("done.")
    except Exception as inst:
        #file.write("error.")
        pass

#if it gets called without a dir argument, just use the report's directory.
def makePDF(pkR):
    ret = shortcuts.get_object_or_404(models.Results, pk=pkR)
    makePDFdir(pkR, ret.get_report_dir())
    
def getPDF(pkR):
    ret = shortcuts.get_object_or_404(models.Results, pk=pkR)
    makePDFdir(pkR, ret.get_report_dir())
    return open(ret.get_report_dir()+"/backupPDF.pdf")

def getOldPDF(pkR):
    ret = shortcuts.get_object_or_404(models.Results, pk=pkR)
    makeOldPDFdir(pkR, ret.get_report_dir())
    return open(ret.get_report_dir()+"/backupPDF.pdf")

def makeOldPDFdir(pkR, dir):
    #fileName = "/tmp/makePDFLog.txt"
    #file = open(fileName, 'w')
    #file.write("makePDFdir called; %s"%pkR+", %s..."%dir)
    ret = shortcuts.get_object_or_404(models.Results, pk=pkR)
    djangoURL = "http://localhost/rundb/api/v1/results/%s/pluginresults/?format=json"%pkR
    pageOpener = urllib2.build_opener()
    jsonPage = pageOpener.open(djangoURL)
    djangoJSON = jsonPage.read()
    decodedJSON = json.loads(djangoJSON)
    #file.write("JSON obained...")
    
    fileNames = []
    names = []
    for JSON in decodedJSON:
        #print JSON
        roughFN = '%s'%JSON['Files']
        #get rid of the apostrophes to make a happy file name.
        fn = roughFN[roughFN.find("'")+1:]
        fn = fn[:fn.find("'")]
        roughName = '%s'%JSON['Name']
        #if it doesn't end in '.html', remove the '/' that will be there.
        if fn[len(fn)-1:] != 'l':
            fn = fn[:len(fn)-1]
        #if there's no filename, the plugin didn't work properly; don't bother appending it to the list.
        #also, the name will still exist if the plugin failed, but should only be added if it succeeded.
        if fn != '':
            fileNames.append(fn)
            names.append(roughName)
    
    #file.write("filenames found...")
    
    link = ret.reportLink
    if link[len(link)-5:] == '.html':
        #if there's an error, it will point to 'log.html'...there'll still be some sort of report page, though. So, chop the end off if that's the case.
        link = link[:len(link)-8]
        #on second thought, I'm not entirely certain that this 'save' line is necessary...
        ret.reportLink = link
        ret.save()
    #since we'll be concatenating things starting with '/' and/or '?' to the link, get rid of a '/' at the end if there is one.
    if link[len(link)-1:] == '/':
        link = link[:len(link)-1]
    
    #file.write("link produced...")
    
    i = 0
    page = ""
    page_url = "http://localhost"+link+"?no_header=True "
    #get the links for the plugin pages. Save them in one string with spaces in between; wkhtmltopdf takes them as 'arg1 arg2 arg3 arg4...'
    for entry in fileNames:
        newPage = "http://localhost"+link
        newPage = newPage + '/plugin_out/' + names[i] + '_out/' + entry + " "
        page += newPage
        i+=1
    #file.write("args created...")
    #note to self: probably could just use dir in the following line...
    savePath = dir
    #this string will be really long for reports with lots of plugins. One of the bigger ones caused wkhtmltopdf to segfault.
    #but, it did successfully produce a .pdf with the report and all of the plugins in one piece, so I'm not really sure what to make of that. 
    pdf_str = '/opt/ion/iondb/bin/wkhtmltopdf-amd64 -q --margin-top 15 --header-spacing 5 --header-left " ' + ret.resultsName + ' - [date]" --header-right "Page [page] of [toPage]" --header-font-size 9 --disable-internal-links --disable-external-links --enable-forms '+ page_url + page + ' %s/backupPDF.pdf'%savePath
    #and, run it. Note to self: maybe add 'except Exception as inst' and print what kind of exception caused the error. Not that that works with any regularity anyways...
    try:
        #file.write("%s..."%pdf_str)
        os.system(pdf_str)
        #file.write("done.")
    except:
        #file.write("error.")
        pass
    
#if it's called from the command line, run it with whatever number comes after the command as the pk.
#It could be neater and maybe have an arg for a directory, but I don't see this being called from the command line beyond debugging.
if __name__ == '__main__':
    rpk = 1
    if(len(sys.argv) > 1):
        for arg in sys.argv:
            rpk = arg
    makePDF(rpk)