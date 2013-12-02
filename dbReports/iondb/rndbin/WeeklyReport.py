# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
import datetime
from os import path
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email.MIMEText import MIMEText
from email.Utils import COMMASPACE, formatdate
from email import Encoders

import re

from iondb.bin.djangoinit import *
from iondb.rundb import models

import math
import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import scipy.stats as stats
import locale
locale.setlocale(locale.LC_ALL, 'en_US.utf8')

Proton = {
    'Proton_East':{
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'iondb',
        'USER': 'ion',
        'PASSWORD': 'ion',
        'HOST': 'blackbird.ite'
    },
    'Proton_West':{
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'iondb',
        'USER': 'ion',
        'PASSWORD': 'ion',
        'HOST': 'blackbird.itw'
    },
    'Proton_Carlsbad':{
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'iondb',
        'USER': 'ion',
        'PASSWORD': 'ion',
        'HOST': 'blackbird.cbd'
    },
    'Proton_Beverly':{
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'iondb',
        'USER': 'ion',
        'PASSWORD': 'ion',
        'HOST': 'blackbird.bev'
    },
    'Proton_Pringle':{
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'iondb',
        'USER': 'ion',
        'PASSWORD': 'ion',
        'HOST': 'pringle.ite'
    },
    'Proton_Bluth':{
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'iondb',
        'USER': 'ion',
        'PASSWORD': 'ion',
        'HOST': 'bluth.itw'
    }
}
for site in Proton:
    settings.DATABASES[site] = Proton[site]

chipListLookup = models.Chip.objects.all().values_list('name', 'description')
chipDictLookup = dict(i for i in chipListLookup)

chipLookup = {'P0':['P1.0.19', 'P2.0.16'], 'PI':['900', '900v2', 'P1.1.17', 'P1.1.16', 'P2.1.16']}


def lookupSite(site):
    printName = ''
    if 'ite' in site:
        printName = 'IE'
    elif 'itw' in site:
        printName = 'IW'
    elif 'bev' in site:
        printName = 'Bev'
    elif 'cbd' in site:
        printName = 'SoCal'
    else:
        printName = 'other'
    return printName

rowToggle = 0
tableCounter = 0
itemsPerLine = 1
itemPercent = 1.0

#pretty number display
def ion_readable(value):
    try:
        charlist = []
        charlist.append("")
        charlist.append("K ")
        charlist.append("M ")
        charlist.append("G ")
        charlist.append("T ")

        charindex = 0
        val = float(value)
        while (val >= 1000):
            val = val / 1000
            charindex = charindex + 1

        converted_text = ""
        if (charindex > 0):
            val2 = math.floor(val*10)
            val2 = val2 / 10
            text = "%.1f" % val2
            if text[-1:] == '0':
                text = text.split('.')[0]
            textIntPart = text.split('.')[0]
            if len(textIntPart) > 2:
                text = textIntPart
            converted_text = str(text) + charlist[charindex]
        else:
            converted_text = str(value)

        return converted_text
    except:
        pass


def ion_pretty(value):
    return locale.format("%d", value, grouping=True)    


def ion_latex_safe(text):
    text = text.replace('_', '\\_')
    return text


def ion_table_number(text):
    return '\\multicolumn{1}{r}{%s}' % text


#define a function to make nice tables
def initdoc(f, titleTxt):
    f.write('\\documentclass{article}\n')
    f.write('\\usepackage{booktabs}\n')
    f.write('\\usepackage{colortbl}\n')
    f.write('\\usepackage{amsmath}\n')
    f.write('\\usepackage{xcolor}\n')
    f.write('\\usepackage{graphicx}\n')
    f.write('\\usepackage{hyperref}\n')
    f.write('\\usepackage{fancyhdr}\n')
    f.write('\\usepackage{array}\n')

    f.write('\\colorlet{tableheadcolor}{gray!25}\n')
    f.write('\\colorlet{tablerowcolor}{gray!10}\n')
    f.write('\\newcommand{\headcol}{\\rowcolor{tableheadcolor}}\n')
    f.write('\\newcommand{\\rowcol}{\\rowcolor{tablerowcolor}}\n')
    f.write('\\newcommand{\\topline}{\\arrayrulecolor{black}\\specialrule{0.1em}{\\abovetopsep}{0pt}\\arrayrulecolor{tableheadcolor}\\specialrule{\\belowrulesep}{0pt}{0pt}\\arrayrulecolor{black}}\n')
    f.write('\\newcommand{\\midline}{\\arrayrulecolor{tableheadcolor}\\specialrule{\\aboverulesep}{0pt}{0pt}\\arrayrulecolor{black}\\specialrule{\\lightrulewidth}{0pt}{0pt}\\arrayrulecolor{white}\\specialrule{\\belowrulesep}{0pt}{0pt}\\arrayrulecolor{black}}\n')
    f.write('\\newcommand{\\rowmidlinecw}{\\arrayrulecolor{tablerowcolor}\\specialrule{\\aboverulesep}{0pt}{0pt}\\arrayrulecolor{black}\\specialrule{\\lightrulewidth}{0pt}{0pt}\\arrayrulecolor{white}\\specialrule{\\belowrulesep}{0pt}{0pt}\\arrayrulecolor{black}}\n')
    f.write('\\newcommand{\\rowmidlinewc}{\\arrayrulecolor{white}\\specialrule{\\aboverulesep}{0pt}{0pt}\\arrayrulecolor{black}\\specialrule{\\lightrulewidth}{0pt}{0pt}\\arrayrulecolor{tablerowcolor}\\specialrule{\\belowrulesep}{0pt}{0pt}\\arrayrulecolor{black}}\n')
    f.write('\\newcommand{\\rowmidlinew}{\\arrayrulecolor{white}\\specialrule{\\aboverulesep}{0pt}{0pt}\\arrayrulecolor{black}}\n')
    f.write('\\newcommand{\\rowmidlinec}{\\arrayrulecolor{tablerowcolor}\\specialrule{\\aboverulesep}{0pt}{0pt}\\arrayrulecolor{black}}\n')
    f.write('\\newcommand{\\bottomline}{\\arrayrulecolor{white}\\specialrule{\\aboverulesep}{0pt}{0pt}\\arrayrulecolor{black}\\specialrule{\\heavyrulewidth}{0pt}{\\belowbottomsep}}\n')
    f.write('\\newcommand{\\bottomlinec}{\\arrayrulecolor{tablerowcolor}\\specialrule{\\aboverulesep}{0pt}{0pt}\\arrayrulecolor{black}\\specialrule{\\heavyrulewidth}{0pt}{\\belowbottomsep}}\n')
    f.write('\\newcolumntype{C}[1]{>{\\centering\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}\n')
    f.write('\\newcolumntype{L}[1]{>{\\raggedright\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}\n')
    f.write('\\newcolumntype{R}[1]{>{\\raggedleft\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}\n')

    f.write(''
        '\\makeatletter\n'
        '\\def\\maxwidth{%\n'
        '\\ifdim\\Gin@nat@width>\\linewidth\n'
        '\\linewidth\n'
        '\\else\n'
        '\\Gin@nat@width\n'
        '\\fi\n'
        '}\n'
        '\\makeatother\n')

    # f.write('\\textwidth 7.5in\n')
    # f.write('\\marginsize{0.5in}{0.5in}{1.0in}{1.0in}\n')
    f.write('\\usepackage[top=1.0in, bottom=1.0in, left=0.5in, right=0.5in]{geometry}\n')

    f.write(''
        '\\fancypagestyle{mystyle}{%\n'
        '\\fancyhead{}\n'
        '\\fancyhead[C]{\\large ')
    f.write(titleTxt);
    f.write('}\n'
        '\\fancyfoot{}\n'
        '\\fancyfoot[L]{\\thepage}\n'
        '\\fancyfoot[R]{\\includegraphics[width=20mm]{/opt/ion/iondb/media/IonLogo.png}}\n'
        '\\renewcommand{\\headrulewidth}{1pt}                             % the header rule\n'
        '\\renewcommand{\\footrulewidth}{0pt}\n'
        '}\n')

    f.write('\\begin{document}\n')
    f.write('\\pagestyle{mystyle}\n')

def table_multi_begin(f,items):
    global itemsPerLine
    global itemPercent
    itemsPerLine = items
    itemPercent = (1.0 - itemsPerLine*0.05)/itemsPerLine

    f.write('\\begin{table}[ht!]\n')

def table_begin(f,header,formatText):
    global rowToggle
    rowToggle = 0

    if itemsPerLine > 1:
        f.write('\\begin{minipage}[ht!]{%s\\linewidth}\n' % itemPercent)

    f.write(''
        '\\begin{tabular}{%s}\n'
        '\\topline\n'
        '\\headcol %s \\\\\n'
        '\\midline\n' % (formatText, header))

def table_addrow(f,s):
    global rowToggle
    outText = ''
    if rowToggle == 1:
       outText = '\\rowcol '
    outText += s
    outText += ' \\\\\n'

    f.write(outText)

    if rowToggle == 0:
        rowToggle = 1
    else:
        rowToggle = 0

def table_end(f):
    global itemsPerLine

    f.write('\\bottomlinec\n'
        '\\end{tabular}\n')

    if itemsPerLine > 1:
        f.write('\\end{minipage}\n')
        f.write('\\hspace{0.5cm}\n')

def table_multi_end(f):
    global itemsPerLine
    itemsPerLine = 1
    f.write('\\end{table}\n')

def figure_add(f,s, asFigure):
    # note: placement options are h:here, t:top, b:bottom, p:page_for_floats_only, !:override_latex_opinion_and_do_what_I_say
    # s in an array of figures, let's see how many
    howmanyfigures = len(s)

    options = 'width=\\maxwidth'

    if asFigure:
        f.write('\\begin{figure}[ht!]\n')
    # f.write('\\begin{center}\n')

    if howmanyfigures > 1:
        w = 6.0 / howmanyfigures
        options = 'width=%sin' % w
        #f.write('\\begin{center}$\n')
        f.write('$\n')
        placementOptions = ''
        for i in range(0,howmanyfigures):
            placementOptions = placementOptions + 'l'
        f.write('\\begin{array}{%s}\n' % placementOptions)

    count = 0
    for figure in s:
        if howmanyfigures > 1 and count > 0:
            f.write(' & ')
        f.write('\\includegraphics[%s]{%s}\n' % (options, figure))
        count = count + 1

    if howmanyfigures > 1:
        f.write('\\end{array}$\n')
        #f.write('\\end{center}\n')

    if asFigure:
        f.write('\\end{figure}\n')


def GetWebRoot(site):
    gc = models.GlobalConfig.objects.using(site).all()[0]
    webRoot = gc.web_root
    if len(webRoot) > 0:
        if webRoot[-1] == '/':
            webRoot = webRoot[:len(webRoot) - 1]
    return webRoot

class MetricRecord:
    def __init__(self, metricName, numBest, chip=None):
        self.metricName = metricName		# name of metric to track (from models.py in LibMetrics class)
        self.numBest = numBest			# numBest is an integer indicating how many top records of a given type to maintain in the list
        self.chip = chip			# chip is a string containing part of the chip name to match
        self.recordReportList = []		# the record report list will be maintained with the up to top N reports, sorted best to worst
        self.numInList = 0
        self.recordValue = 0
        self.track = False
        self.count = 0
        self.metricSum = 0
        self.i100Q17_reads = 0
        self.q17bases = 0
        self.sum314 = 0
        self.sum316 = 0
        self.sum318 = 0
        self.dateFilter = False
        self.dateMin = datetime.datetime(2012, 1, 1)
        self.dateMax = datetime.datetime(2012, 1, 1)
        self.site = ""
        self.siteFilter = False
        self.plugin = False
        self.reverse = False
        self.floatval = False
        self.ref = ''
        self.refMatchMode = 0

        # historical tracking fields
        self.history = False
        self.historyDaysPerPeriod = 0 
        self.historyPeriods = 0
        self.historyValue = []

        # plugin fields
        if (':' in metricName):
            self.plugin = True
            self.pluginStore = metricName.split(':')[0]
            self.pluginMetric = metricName.split(':')[1]
    def Reverse(self):
        self.reverse = True
        self.recordValue = 999999

    def History(self,periodLen,numPeriods):
        self.history = True
        self.historyDaysPerPeriod = periodLen
        self.historyPeriods = numPeriods
        self.historyValue = [0]*self.historyPeriods

def isChipMatch(chipName, exp):
    global chipLookup

    match = False

    # try our most recent naming scheme first
    chipVersion = ''
    try:
        chipVersion = exp.log['chipversion']
    except:
        if exp.chipType == '900': # legacy Proton support
            chipVersion = 'P1.1.17'

    if len(chipVersion) > 0 and len(chipVersion) < 7:
        if chipVersion[0] == 'P' or chipVersion[0] == '1': # intermediate style naming, assume P1 chip
            chipVersion = 'P1.1.17'

    # chip might have been a P0 chip run in advAvg mode 1 so it reports 164M sensors, treat as a P1
    try:
        if chipVersion == 'P1.0.19' and int(exp.log['columns']) > 15000:
            chipVersion = 'P1.1.17'
    except:
        pass

    if chipVersion in chipLookup[chipName]:
        match = True

    return match

def isChipMatch_old(chipName, exp):
    global chipDictLookup

    match = False

    chipVersion = exp.chipType

    if chipVersion == '900': # see if we can get name from stored explog file for older proton runs
        try:
            chipVersion = exp.log['chipversion']
        except:
            chipVersion = 'P1.1.17'

    if len(chipVersion) > 0 and len(chipVersion) < 7:
        if chipVersion[0] == 'P' or chipVersion[0] == '1': # intermediate style naming, assume P1 chip
            chipVersion = 'P1.1.17'

    # chip might have been a P0 chip run in advAvg mode 1 so it reports 164M sensors, treat as a P1
    try:
        if chipVersion == 'P1.0.19' and int(exp.log['columns']) > 15000:
            chipVersion = 'P1.1.17'
    except:
        pass

    try:
        chipDescription = chipDictLookup[chipVersion]
    except:
        chipDescription = ''

    if chipName == chipDescription:
        match = True

    return match


def BuildTrackingMetrics(metricRecordList, site):
    for metricRecord in metricRecordList:
        if not metricRecord.track:
            continue

        if metricRecord.site != site:
            continue

        print 'Processing tracking metric %s for site %s' % (metricRecord.metricName, site)

        # get the url base for the site
        web_root = GetWebRoot(site)

        expList = models.Experiment.objects.using(site).select_related().filter(date__range=(metricRecord.dateMin,metricRecord.dateMax))
        print 'Loading experiments in range...'
        theLen = len(expList)
 
        overallBest = 0

        print 'Tracking...'
        for exp in expList:
            repList = exp.results_set.all()
            # MGD - speedup here - would like to avoid the per-record query above, maybe re-use of hash table from normal metricRecord code?
            bestVal = 0
            bestResult = None
            bestLib = None

            # optional filters
            ok = True
            if ok and metricRecord.chip is not None:
                ok = isChipMatch(metricRecord.chip, exp)

            if ok:
                for rep in repList:
                    try:
                        libmetrics = rep.libmetrics_set.all()[0] # ok, there's really only one libmetrics set per result, but we still need to get at it
                        # MGD - speedup here - we can re-use the hash lookup we created per site below so avoid the database query per record here
                    except IndexError:
                        libmetrics = None
        
                    if libmetrics is not None:
                        if libmetrics.align_sample == 0:
                            valtext = getattr(libmetrics, metricRecord.metricName)
                            if metricRecord.floatval:
                                val = float(valtext)
                            else:
                                val = int(valtext)
                            if val > bestVal:
                                bestVal = val
                                bestResult = rep
                                bestLib = libmetrics
    
            if bestResult is not None:
                metricRecord.count = metricRecord.count + 1
                metricRecord.metricSum = metricRecord.metricSum + bestVal
                metricRecord.q17bases = metricRecord.q17bases + bestLib.q17_mapped_bases
                metricRecord.i100Q17_reads = metricRecord.i100Q17_reads + bestLib.i100Q17_reads
                if '314' in bestResult.experiment.chipType:
                    metricRecord.sum314 = metricRecord.sum314 + 1
                if '316' in bestResult.experiment.chipType:
                    metricRecord.sum316 = metricRecord.sum316 + 1
                if '318' in bestResult.experiment.chipType:
                    metricRecord.sum318 = metricRecord.sum318 + 1

                if bestVal > overallBest:
                    overallBest = bestVal
                    if metricRecord.numInList == 0:
                        metricRecord.numInList = 1
                        metricRecord.recordReportList.append((bestResult, bestVal, bestLib.i100Q17_reads, bestLib.q17_mapped_bases, web_root))
                    else:
                        metricRecord.recordReportList[0] = (bestResult, bestVal, bestLib.i100Q17_reads, bestLib.q17_mapped_bases, web_root)


def BuildMetrics(metricRecordList, site):
    # metricRecordList is an array of type MetricRecord
    # site is a string containing the site name - must match the database configuration name from settings.py

    print 'Processing site: %s' % site

    # get the url base for the site
    web_root = GetWebRoot(site)

    # some init

    # filters we can use to ignore/keep records of interest while looping over all of them
    # projectFilter = True
    projectFilter = False
    project = 'ava'
    dateFilter = False

    #today = datetime.date.today()
    #timeEnd = datetime.datetime(today.year, today.month, today.day)
    #timeStart = timeEnd - datetime.timedelta(days=14)
    #repList = models.Results.objects.using(site).filter(timeStamp__range=(timeStart, timeEnd)).select_related()

    # select all result records (maybe here would be a good place to just get them all into memory fast?)
    # and on performance, we will hit this site's report list many times, looking for a specific chip, metric, etc so would be good to keep this around
    repList = models.Results.objects.using(site).select_related()  # or we can filter by date for example: filter(timeStamp__range=(queryStart,queryEnd))

    if projectFilter:
        repList = repList.filter(projects__name__icontains=project)

    print "forcing repList loading"
    theLen = len(repList) # force load of database table across net

    libList = models.LibMetrics.objects.using(site).select_related()
    print "forcing libmetrics loading"
    theLen = len(libList) # force load of database table across net

    # make a lookup table of library metrics pk's
    print 'Building library metrics hash...'
    libpk_hash = []
    num = 0
    largest = 0
    for lib in libList:
        pk = lib.report.pk
        if pk > largest:
            delta = pk - largest
            largest = pk
            for i in range(0, delta+1): # now why can't python simply resize the list when I perform assignments?
                libpk_hash.append(-1)
        # if (libpk_hash[pk] == -1):
            # libpk_hash[pk] = num
        if libpk_hash[pk] == -1:
            libpk_hash[pk] = num
        else:
            # see if this new libmterics associated with a report is 'better' than the current one
            if lib.q17_mapped_bases > libList[libpk_hash[pk]].q17_mapped_bases:
                libpk_hash[pk] = num
        num = num + 1

    # pre-generate a few of the more common queries, so we don't need to re-generate these for each metric
    repList_hg19 = repList.filter(eas__reference='hg19')
    repList_other = repList.exclude(eas__reference='hg19')
    repList_hg19_default = repList_hg19.filter(resultsName__icontains='auto')
    repList_other_default = repList_other.filter(resultsName__icontains='auto')
    repList_hg19 = repList_hg19.exclude(resultsName__icontains='auto')
    repList_other = repList_other.exclude(resultsName__icontains='auto')

    # loop through all metrics, updating top 5 (numBest) list for each
    for metricRecord in metricRecordList:
        if metricRecord.track:
            continue

        print 'Processing metric: %s' % metricRecord.metricName

        # generate or point the the desired report list.  Note that some have been cached for speed
        repListFinal = repList
        if metricRecord.refMatchMode == 1 and metricRecord.ref == 'hg19':
            repListFinal = repList_hg19
        elif metricRecord.refMatchMode == 2 and metricRecord.ref == 'hg19':
            repListFinal = repList_other
        elif metricRecord.refMatchMode == 3 and metricRecord.ref == 'hg19':
            repListFinal = repList_hg19_default
        elif metricRecord.refMatchMode == 4 and metricRecord.ref == 'hg19':
            repListFinal = repList_other_default
        elif metricRecord.refMatchMode == 1:
            repListFinal = repList.filter(eas__reference=metricRecord.ref)
        elif metricRecord.refMatchMode == 2:
            repListFinal = repList.exclude(eas__reference=metricRecord.ref)
        else:
            repListFinal = repList

        if metricRecord.history:
            print 'updating history list for metric: %s' % metricRecord.metricName
            today = datetime.date.today()
            timeStart = datetime.datetime(today.year, today.month, today.day)
            for rep in repListFinal:

                # figure our what period this report is in
                expDate = datetime.datetime(rep.experiment.date.year, rep.experiment.date.month, rep.experiment.date.day)
                dateAgo = timeStart - expDate
                daysAgo = dateAgo.days
                period = daysAgo / metricRecord.historyDaysPerPeriod
                # anything too old gets lumped into the oldest period.  We could also just skip old records
                if (period >= metricRecord.historyPeriods):
                    period = metricRecord.historyPeriods - 1

                # get the metric value
                libmetrics = None
                # look up the library table entry from the report pk via our pre-generated hash lookup array
                try:
                    libIndex = libpk_hash[rep.pk]
                    libmetrics = libList[libIndex]
                    if libmetrics.align_sample != 0:
                        libmetrics = None # ignore any sampled & extrapolated runs
                except:
                    libmetrics = None

                val = 0

                # optional filters
                ok = True
                if ok and metricRecord.chip is not None:
                    ok = isChipMatch(metricRecord.chip, rep.experiment)

                if ok:
                    if metricRecord.plugin:
                        try:
                            pluginDict = rep.pluginStore[metricRecord.pluginStore]
                        except:
                            pluginDict = None
                        if pluginDict is not None:
                            try:
                                valtext = pluginDict[metricRecord.pluginMetric]
                                val = float(valtext.rstrip('%'))
                            except:
                                val = 0
                    else:
                        if libmetrics is not None:
                            valtext = getattr(libmetrics, metricRecord.metricName)
                            if metricRecord.floatval:
                                val = float(valtext)
                            else:
                                val = int(valtext)
    
                # if this value is better than the current stored value, replace it
                if val > 0:
                    if ((metricRecord.reverse == False and val > metricRecord.historyValue[period]) or (metricRecord.reverse == True and val < metricRecord.historyValue[period])):
                        metricRecord.historyValue[period] = val

            # and no need to process this metric further
            continue

        print 'updating report list...'
        # look at all reports
        for rep in repListFinal:
            libmetrics = None
            # look up the library table entry from the report pk via our pre-generated hash lookup array
            try:
                libIndex = libpk_hash[rep.pk]
                libmetrics = libList[libIndex]
                if libmetrics.align_sample != 0:
                    libmetrics = None # ignore any sampled & extrapolated runs
            except:
                libmetrics = None

            if libmetrics is not None:
                # optional filters
                ok = True
                if ok and metricRecord.chip is not None:
                    ok = isChipMatch(metricRecord.chip, rep.experiment)

                if ok and metricRecord.dateFilter:
                    ok = False
                    if metricRecord.dateMin <= rep.experiment.date and metricRecord.dateMax > rep.experiment.date:
                        ok = True

                if ok and metricRecord.siteFilter:
                    ok = False
                    if metricRecord.site == site:
                        ok = True

                if ok:
                    val = 0
                    if metricRecord.plugin:
                        try:
                            pluginDict = rep.pluginStore[metricRecord.pluginStore]
                        except:
                            pluginDict = None
                        if pluginDict is not None:
                            try:
                                valtext = pluginDict[metricRecord.pluginMetric]
                                val = float(valtext.rstrip('%'))
                            except:
                                val = 0
                    else:
                        valtext = getattr(libmetrics, metricRecord.metricName)
                        if metricRecord.floatval:
                            val = float(valtext)
                        else:
                            val = int(valtext)

                    if (val > 0):
                        if ((metricRecord.reverse == False and val > metricRecord.recordValue) or (metricRecord.reverse == True and val < metricRecord.recordValue)):
                            # if report's parent experiment already exists in the list, replace it, else insert it
                            repIndex = 0
                            repToReplace = -1
                            repFound = False
                            for rep_item, rep_value, i100Q17_reads, q17bases, webRoot in metricRecord.recordReportList:
                                # only replace it if its a better record
                                if (rep_item.experiment.pk == rep.experiment.pk):
                                    repFound = True # found it, but report might not be better than the one already added
                                    if ((metricRecord.reverse == False and val > rep_value) or (metricRecord.reverse == True and val < rep_value)):
                                        repToReplace = repIndex
                                repIndex = repIndex + 1
                            if repToReplace > -1: # found and its better than the one already added
                                metricRecord.recordReportList[repToReplace] = (rep, val, libmetrics.i100Q17_reads, libmetrics.q17_mapped_bases, web_root)
                                #print 'replaced experiment %s using newer report value %s' % (rep.experiment.expName, val)
                            else:
                                # only add if we didn't add this experiment already
                                if repFound == False:
                                    if (metricRecord.numInList < metricRecord.numBest):
                                        metricRecord.recordReportList.append((rep, val, libmetrics.i100Q17_reads, libmetrics.q17_mapped_bases, web_root))
                                        metricRecord.numInList = metricRecord.numInList + 1
                                    else:
                                        # replace the worst item in the list
                                        metricRecord.recordReportList[metricRecord.numInList-1] = (rep, val, libmetrics.i100Q17_reads, libmetrics.q17_mapped_bases, web_root)
    
                            #re-sort the list, and set the min recordValue to 0 if list not full (so we can continue to add to the list), or to the worst item in the list, so we can replace that item if/when new record is found
                            metricRecord.recordReportList = sorted(metricRecord.recordReportList, key=lambda val: val[1], reverse=(not metricRecord.reverse))
                            if metricRecord.numInList == metricRecord.numBest:
                                rep, metricRecord.recordValue, i100Q17_reads, q17bases, webRoot = metricRecord.recordReportList[metricRecord.numInList-1]
                            else:
                                if metricRecord.reverse == False:
                                    metricRecord.recordValue = 0
                                else:
                                    metricRecord.recordValue = 999999



def DumpMetric(metricRecord):

    if metricRecord.track:
        # display tracking summary metrics a bit different
        if metricRecord.dateFilter:
            print 'Site Tracking for %s %s to %s' % (metricRecord.site, metricRecord.dateMin, metricRecord.dateMax)
            html.write('Site Tracking for %s %s to %s<br>' % (metricRecord.site, metricRecord.dateMin, metricRecord.dateMax))
        else:
            print 'Site Tracking for %s' % metricRecord.site
            html.write('Site Tracking for %s<br>' % metricRecord.site)
        print 'Runs: %s  %s: %s  Q17 bases: %s  100AQ17 reads: %s  314/316/318: %s/%s/%s' % (metricRecord.count, metricRecord.metricName, metricRecord.metricSum, metricRecord.q17bases, metricRecord.i100Q17_reads, metricRecord.sum314, metricRecord.sum316, metricRecord.sum318)
        html.write('Runs: %s  100Q17 reads: %s  Q17 bases: %s  314/316/318: %s/%s/%s<br>' % (metricRecord.count, metricRecord.i100Q17_reads, metricRecord.q17bases, metricRecord.sum314, metricRecord.sum316, metricRecord.sum318))
        try:
            recordReport, recordValue, i100Q17_reads, q17bases, webRoot = metricRecord.recordReportList[0]
            print 'Top run 100Q17 reads: %s Q17 bases: %s Run date: %s  Analysis date: %s' % (i100Q17_reads, q17bases, recordReport.experiment.date, recordReport.timeStamp)
            # html.write('<a href="%s">Top Run 100Q17 reads: %s Q17 bases: %s Run date: %s Analysis date: %s</a><br>\n' % (webRoot + recordReport.reportLink, i100Q17_reads, q17bases, recordReport.experiment.date, recordReport.timeStamp))
            html.write('<a href="%s">Top Run 100Q17 reads: %s Q17 bases: %s Run date: %s Analysis date: %s</a><br>\n' % (webRoot + '/report/' + str(recordReport.pk), i100Q17_reads, q17bases, recordReport.experiment.date, recordReport.timeStamp))
        except:
            print 'No top run found?'
            html.write('No top run found?<br>')
    elif metricRecord.history:
        print 'History for %s:\n' % metricRecord.metricName
        for i in range(0, metricRecord.historyPeriods):
            print '%s ' % metricRecord.historyValue[i]
        print '\n'

    else:
        # display our top N
        if metricRecord.numBest > 1:
            html.write('Ion Chip: %s Top %s %s Runs<br>' % (metricRecord.chip, metricRecord.numBest, metricRecord.metricName))
        else:
            html.write('Ion Chip: %s Top %s Run<br>' % (metricRecord.chip, metricRecord.metricName))

        for recordReport, recordValue, i100Q17_reads, q17bases, webRoot in metricRecord.recordReportList:
            print '%s 100Q17 reads: %s Q17 bases: %s Run date: %s  Analysis date: %s' % (recordValue, i100Q17_reads, q17bases, recordReport.experiment.date, recordReport.timeStamp)
            # print 'URL: %s' % (webRoot + recordReport.reportLink)
            print 'URL: %s' % (webRoot + '/report/' + str(recordReport.pk))
            html.write('<a href="%s">%s 100Q17 reads: %s  Q17 bases: %s Run date: %s Analysis date: %s</a><br>\n' % (webRoot + '/report/' + str(recordReport.pk), recordValue, i100Q17_reads, q17bases, recordReport.experiment.date, recordReport.timeStamp))

 
def sendMail(attachmentName):
    emails = models.EmailAddress.objects.filter(selected=True)
    emailList = []
    if len(emails) > 0:
        emailList = [i.email for i in emails]

    # hack for testing!
    # emailList = ['Mel.Davey@Lifetech.com']

    msg = MIMEMultipart()
    sendFrom = 'donotreply@Lifetech.com'
    msg['From'] = sendFrom
    msg['To'] = COMMASPACE.join(emailList)
    msg['Date'] = str(datetime.date.today())
    msg['Subject'] = 'Proton Weekly Summary Report'
    msg.attach( MIMEText('Weekly Proton Summary Report is attached.') )

    part = MIMEBase('application', "octet-stream")
    part.set_payload( open(attachmentName,"rb").read() )
    Encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(attachmentName))
    msg.attach(part)

    server="localhost"
    smtp = smtplib.SMTP(server)
    smtp.sendmail(sendFrom, emailList, msg.as_string())
    smtp.close()



def table_make(f, header, formatText, metricRecord):
    table_begin(f, header, formatText)
    for recordReport, recordValue, i100Q17_reads, q17bases, webRoot in metricRecord.recordReportList:
        site = ion_latex_safe('\\href{' + webRoot + '/report/' + str(recordReport.pk) + '}{' + lookupSite(webRoot) + '}')
        dateStr = str(recordReport.experiment.date).split(' ')[0]
        rowText = ion_pretty(recordValue) + ' & ' + dateStr + ' & ' + site
        table_addrow(f, rowText)
    table_end(f)


def add_links(f, titleText, metricRecord):

    f.write('\\subsection*{%s}\n' % titleText)
    for recordReport, recordValue, i100Q17_reads, q17bases, webRoot in metricRecord.recordReportList:
        linkName = ion_pretty(recordValue) + ' ' + recordReport.resultsName
        linkURL = webRoot + '/report/' + str(recordReport.pk)
        text = ion_latex_safe('\\href{' + linkURL + '}{' + linkName + '}\\\\\n')
        f.write(text)


if __name__=="__main__":
    siteList = ['Proton_East', 'Proton_West', 'Proton_Carlsbad', 'Proton_Bluth']
    #siteList = ['Proton_East']
    #siteList = ['Proton_Pringle']
    #siteList = ['Proton_Carlsbad']
    #siteList = ['Proton_West']
    #siteList = ['Proton_Bluth']

    metricRecords = []

    #
    # Capture a bunch of company-wide records
    #

    metricRecords.append(MetricRecord('i50Q17_reads', 5))
    metricRecords.append(MetricRecord('i100Q17_reads', 5))
    IonStats_Q10 = MetricRecord('q10_longest_alignment', 5)
    metricRecords.append(IonStats_Q10)
    IonStats_Q17 = MetricRecord('q17_longest_alignment', 5)
    metricRecords.append(IonStats_Q17)
    IonStats_Q20 = MetricRecord('q20_longest_alignment', 5)
    metricRecords.append(IonStats_Q20)
    IonStats_Q47 = MetricRecord('q47_longest_alignment', 5)
    metricRecords.append(IonStats_Q47)
    metricRecords.append(MetricRecord('q7_mapped_bases', 10))
    metricRecords.append(MetricRecord('q10_mapped_bases', 10))
    metricRecords.append(MetricRecord('q17_mean_alignment_length', 10))
    snr_stats = MetricRecord('sysSNR', 10)
    snr_stats.floatval = True
    metricRecords.append(snr_stats)

    #
    # set up some useful vars for use later on
    #

    today = datetime.date.today()
    timeStart = datetime.datetime(today.year, today.month, today.day)
    daysFromMonday = timeStart.weekday() # Monday is 0, so if its Thursday (3), we need to go back 3 days
    lengthOfReport = 7 # report for 7 days, with Monday being the first day included in a report
    if daysFromMonday < lengthOfReport: # we want to go back to the start of the full week, if we are in the middle of a week, need to go back to the start of last week
        daysFromMonday = daysFromMonday + 7
    timeStart = timeStart - datetime.timedelta(days=daysFromMonday)
    timeEnd = timeStart + datetime.timedelta(days=lengthOfReport)

    #
    # Capture P0, P1 metrics
    # For each metric, we capture per chip, and have 4 variations: hg19-default, other-default, hg19-record, other-record
    #

    # chipList = ['P1.0.19', 'P1.1.17']
    #chipList = ['P0', 'PI']
    #chipListDesc = ["P0", "P1"]
    chipList = ['PI']
    chipListDesc = ["P1"]
    metricList = ['total_mapped_target_bases', 'q17_mapped_bases', 'q20_mapped_bases']
    metricListDesc = ['Mapped Bases', 'AQ17 Bases', 'AQ20 Bases']
    protonMetrics = {}
    for chip in chipList:
        protonMetrics[chip] = {}
        for metric in metricList:
            protonMetrics[chip][metric] = []
            for var in range(0,4):
                item = MetricRecord(metric, 5, chip)
                item.ref = 'hg19'
                item.refMatchMode = var + 1 # 1 = contains, 2 = does not contain, 3 = contains and default, 4 = does not contain and default
                metricRecords.append(item)
                protonMetrics[chip][metric].append(item)

        # chips each store some fun historical records also
        protonMetrics[chip]['hist'] = {}

        # a 6 month (26 week) weekly record for AQ17 bases
        protonMetrics[chip]['hist']['6m-q17_mapped_bases'] = MetricRecord('q17_mapped_bases', 1, chip)
        protonMetrics[chip]['hist']['6m-q17_mapped_bases'].History(7,26)
        metricRecords.append(protonMetrics[chip]['hist']['6m-q17_mapped_bases'])

        # a 6 month (26 week) weekly record for AQ20 bases
        protonMetrics[chip]['hist']['6m-q20_mapped_bases'] = MetricRecord('q20_mapped_bases', 1, chip)
        protonMetrics[chip]['hist']['6m-q20_mapped_bases'].History(7,26)
        metricRecords.append(protonMetrics[chip]['hist']['6m-q20_mapped_bases'])

        # a 30 day history for AQ17 bases
        protonMetrics[chip]['hist']['30d-q17_mapped_bases'] = MetricRecord('q17_mapped_bases', 1, chip)
        protonMetrics[chip]['hist']['30d-q17_mapped_bases'].History(1,30)
        metricRecords.append(protonMetrics[chip]['hist']['30d-q17_mapped_bases'])

        # a 30 day history for AQ20 bases
        protonMetrics[chip]['hist']['30d-q20_mapped_bases'] = MetricRecord('q20_mapped_bases', 1, chip)
        protonMetrics[chip]['hist']['30d-q20_mapped_bases'].History(1,30)
        metricRecords.append(protonMetrics[chip]['hist']['30d-q20_mapped_bases'])

        # Capture the last full week's runs to sumarize
        for site in siteList:
            # weeklySite = MetricRecord('i100Q17_reads', 1, '9')
            weeklySite = MetricRecord('q17_mapped_bases', 1, chip)
            weeklySite.track = True
            weeklySite.dateFilter = True
            weeklySite.dateMin = timeStart
            weeklySite.dateMax = timeEnd
            weeklySite.site = site
            weeklySite.siteFilter = True
            metricRecords.append(weeklySite)
            protonMetrics[chip][site] = weeklySite

    #
    # Do the heavy-lifting of data mining
    #

    for site in siteList:
        BuildMetrics(metricRecords, site)
        BuildTrackingMetrics(metricRecords, site)

    #
    # dump out all the raw metrics into an html file
    #

    html = open("top-" + str(datetime.date.today())+ ".html",'w')
    html.write('<html><body>\n')

    print 'Multi-site metrics'
    html.write('Multi-site metrics')

    for metricRecord in metricRecords:
        # quick cleanup of recordValue to make sure it now reflects the top record, not the barrier to entry for a record
        if metricRecord.numInList > 0:
            rep, metricRecord.recordValue, i100Q17_reads, q17bases, webRoot = metricRecord.recordReportList[0]
        DumpMetric(metricRecord)

    html.write('</body></html>\n')
    html.close()

    #
    # dump out a metrics file for use with PGM IonStats screensaver
    #

    IonStats = open("Proton_metrics.txt", 'w')
    IonStats.write('%s,%s,%s\n' % (IonStats_Q10.metricName, IonStats_Q10.recordValue, 0))
    IonStats.write('%s,%s,%s\n' % (IonStats_Q17.metricName, IonStats_Q17.recordValue, 0))
    IonStats.write('%s,%s,%s\n' % (IonStats_Q20.metricName, IonStats_Q20.recordValue, 0))
    IonStats.write('%s,%s,%s\n' % (IonStats_Q47.metricName, IonStats_Q47.recordValue, 0))
    IonStats.close()

    #
    # Generate trend chart png files
    #

    today = datetime.date.today()
    trendStart = datetime.datetime(today.year, today.month, today.day)

    for chip in chipList:
        for histRecName in protonMetrics[chip]['hist']:
            histRec = protonMetrics[chip]['hist'][histRecName]
            plotdataVals = []
            plotdataDate = []
            for i in range(0, histRec.historyPeriods):
                gig = float(histRec.historyValue[histRec.historyPeriods-i-1]) / (1000.0*1000.0*1000.0)
                plotdataVals.append(gig)
                daysago = (histRec.historyPeriods-i-1)*histRec.historyDaysPerPeriod
                plotDate = trendStart - datetime.timedelta(days=daysago)
                plotdataDate.append(plotDate)
                #plotdataDate.append(i)
            Qtxt = 'AQ17'
            if '20' in histRec.metricName:
                Qtxt = 'AQ20'
            plt.xlabel('Date')
            plt.ylabel('%s bases (G)' % Qtxt)
            plt.title('Historical Best %s Bases Run' % Qtxt)
            plt.plot(plotdataDate, plotdataVals)
            plt.legend([Qtxt + ' bases'],loc = 'best')
	    pngName = '%s_%s.png' % (chip.replace('.', '_'), histRecName)
            plt.savefig(pngName)
            plt.close()


    #
    # generate the LaTeX layout report
    #

    reportName = 'WeeklyReport-' + str(datetime.date.today()) + '.tex'
    f = open(reportName, 'w')
    titleTxt = 'Weekly Proton Report %s to %s' % (str(timeStart).split(' ')[0], str(timeEnd).split(' ')[0])
    initdoc(f, titleTxt)

    #
    # Generate basically the same report, but one per chip
    #

    for (chipID, chip) in enumerate(chipList):

        f.write('\\section*{Proton Summary for chip: %s}\n' % chipListDesc[chipID])

        # Section 1 - Summary Charts
        f.write('Weekly run summary from %s to %s\n\\\\' % (str(timeStart).split(' ')[0], str(timeEnd).split(' ')[0]))

        # add table of site tracking
        table_begin(f,'Site & Total \\newline 100AQ17 Reads & Total \\newline AQ17 Bases & Full \\newline Analyses & Best Run \\newline 100AQ17 Reads & Best Run \\newline AQ17 Bases & Best Run', 'L{2cm} R{2cm} R{2cm} R{2cm} R{2cm} R{2cm} L{3cm}')
        for site in siteList:
            metricRecord = protonMetrics[chip][site]
            rowText = '%s & %s & %s & %s & ' % (ion_latex_safe(metricRecord.site), ion_pretty(metricRecord.i100Q17_reads), ion_readable(metricRecord.q17bases), metricRecord.count)
            try:
                recordReport, recordValue, i100Q17_reads, q17bases, webRoot = metricRecord.recordReportList[0]
                rowText = rowText + '%s & %s & %s' % (ion_pretty(i100Q17_reads), ion_readable(q17bases), ion_latex_safe('\\href{' + webRoot + '/report/' + str(recordReport.pk) + '}{' + recordReport.resultsName.replace('Auto_user_', '') + '}'))
            except:
                rowText = rowText + 'none & none & none'
            table_addrow(f, rowText)

        table_end(f)
        f.write('\\\\\\\\\n')

        # add list of default analysis runs hg19 & other
        f.write('\\subsection*{Default Analysis Runs}\n')
        f.write('\\begin{minipage}[t]{0.5\\textwidth}\n\\centerline{\\textbf{HG19 runs}}\n\\end{minipage}\n')
        f.write('\\begin{minipage}[t]{0.5\\textwidth}\n\\centerline{\\textbf{Other runs}}\n\\end{minipage}\n')

        for (i, metric) in enumerate(metricList):
            table_multi_begin(f, 2)
            table_make(f, '%s & Date & Site' % metricListDesc[i], 'rrr', protonMetrics[chip][metric][2])
            table_make(f, '%s & Date & Site' % metricListDesc[i], 'rrr', protonMetrics[chip][metric][3])
            table_multi_end(f)

        # add list of custom analysis runs hg19 & other
        f.write('\\newpage\n')
        f.write('\\subsection*{Custom Analysis Runs}\n')
        f.write('\\begin{minipage}[t]{0.5\\textwidth}\n\\centerline{\\textbf{HG19 runs}}\n\\end{minipage}\n')
        f.write('\\begin{minipage}[t]{0.5\\textwidth}\n\\centerline{\\textbf{Other runs}}\n\\end{minipage}\n')

        for (i, metric) in enumerate(metricList):
            table_multi_begin(f, 2)
            table_make(f, '%s & Date & Site' % metricListDesc[i], 'rrr', protonMetrics[chip][metric][0])
            table_make(f, '%s & Date & Site' % metricListDesc[i], 'rrr', protonMetrics[chip][metric][1])
            table_multi_end(f)

        # add trend charts
        f.write('\\newpage\n')
        f.write('\\subsection*{Historical Performance Graphs}\n')

        # add plot of AQ17 & AQ20 26 week trend
        f.write('6 month view\n')
        figure_add(f, ['%s_6m-q17_mapped_bases.png' % chip.replace('.', '_'), '%s_6m-q20_mapped_bases.png' % chip.replace('.', '_')], False)
        f.write('\\\\\n')
        f.write('30 day view\n')
        figure_add(f, ['%s_30d-q17_mapped_bases.png' % chip.replace('.', '_'), '%s_30d-q20_mapped_bases.png' % chip.replace('.', '_')], False)
        f.write('\\clearpage\n')

        # add links to top runs
        #f.write('\\newpage\n')

        #f.write('\\newpage\n\\section*{Links to top runs}\n')
        #titleTextList = ['Top 5 aligned base runs', 'Top 5 AQ17 runs', 'Top 5 AQ20 runs']
        #i = 0
        #for metric in metricList:
            #add_links(f, titleTextList[i], protonMetrics[chip][metric][3])
            #i = i + 1

    f.write('\\end{document}\n')
    f.close()

    # generate the PDF report
    cmd = 'pdflatex %s' % reportName
    os.system(cmd)

    # email pdf report
    pdfName = reportName.replace('.tex', '.pdf')
    sendMail(pdfName)

