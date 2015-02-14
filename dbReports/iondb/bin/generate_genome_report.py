# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
import datetime
from os import path

import djangoinit
from django.db import models
from iondb.rundb import models

def getResults():
    # calculate the date range to generate the report
    # the intent is to auto-generate a weekly report for the 'prior' week, no matter when the script is run
    # week ends Friday at midnight.  So for example, if run on a Friday, will generate the report for the prior week,
    # but run on a Sat will generate the current report for the week just ended.
    #today = datetime.date.today()
    #timeStart = datetime.datetime(today.year, today.month, today.day)
    #daysFromMonday = timeStart.weekday() # Monday is 0, so if its Thursday (3), we need to go back 3 days
    #lengthOfReport = 7 # report for 7 days, with Monday being the first day included in a report
    #if daysFromMonday < lengthOfReport: # we want to go back to the start of the full week, if we are in the middle of a week, need to go back to the start of last week
        #daysFromMonday = daysFromMonday + 7
    #timeStart = timeStart - datetime.timedelta(days=daysFromMonday)
    #print 'TimeStart is %s' % timeStart
    #timeEnd = timeStart + datetime.timedelta(days=lengthOfReport) # technically this is one second too much but who really will notice
    #print 'TimeEnd is %s' % timeEnd

    # and now we have a date range to query on, grab all 'new' runs, sum their 100AQ17 values, and track the best weekly 100Q17 run also
    #exp = models.Experiment.objects.filter(date__range=(timeStart,timeEnd))
    library = "hg19"
    exp = models.Experiment.objects.filter(library=library)
    print 'Found %s experiments for library %s' % (len(exp), library)
    # get best result for each experiment, the 'best' is 100Q17 reads right now
    # we will build an array of the best results for each experiment and return that to the caller
    res = []
    for e in exp:
        rep = e.results_set.all()
        bestNumReads = 0
        bestResult = None
        for r in rep:
            try:
                libmetrics = r.libmetrics_set.all()[0] # ok, there's really only one libmetrics set per result, but we still need to get at it
            except IndexError:
                libmetrics = None

            if libmetrics is not None:
                if libmetrics.align_sample == 1:
                    numReads = libmetrics.extrapolated_100q17_reads
                else:
                    numReads = libmetrics.i100Q17_reads
                if numReads > bestNumReads:
                        bestNumReads = numReads
                        bestResult = r

        if bestResult is not None:
            res.append(bestResult)

    return res

class topRecord:
    def __init__(self):
        self.r = None
        self.reads = 0


def byReads(a, b):
    return cmp(b.reads, a.reads) # compare the integer reads values in each topRecord instance

def generateReport():
    # get the web root path for building an html link
    gc = models.GlobalConfig.get()
    web_root = gc.web_root
    if len(web_root) > 0:
        if web_root[-1] == '/':
            web_root = web_root[:len(web_root)-1]
    if gc.site_name:
        print 'Cumulative Report for %s' % gc.site_name
    else:
        print 'Cumulative Report for %s' % web_root

    # get all analysis results from the specified period
    res = getResults()

    # calculate some metrics for the results
    sum_100Q17Reads = 0
    sum_Q17Bases = 0
    sum_Runs = 0

    bestRun = None
    curBestReads = 0
    curBestBases = 0

    topN = []
    topNRecord = []

    for r in res:
        try:
            libmetrics = r.libmetrics_set.all()[0] # ok, there's really only one libmetrics set per result, but we still need to get at it
        except IndexError:
            libmetrics = None

        if libmetrics is not None:
            if libmetrics.align_sample == 1:
                # print 'Extrapolated run'
                reads = libmetrics.extrapolated_100q17_reads
                bases = libmetrics.extrapolated_mapped_bases_in_q17_alignments
            else:
                reads = libmetrics.i100Q17_reads
                bases = libmetrics.q17_mapped_bases

            sum_100Q17Reads = sum_100Q17Reads + reads
            sum_Q17Bases = sum_Q17Bases + bases
            sum_Runs = sum_Runs + 1
            if reads > curBestReads:
                curBestReads = reads
                curBestBases = bases
                bestRun = r

            topNRecord = topRecord()
            topNRecord.r = r
            topNRecord.reads = reads
            topN.append(topNRecord)

    print 'Totals  100Q17 reads: %s 100Q17 bases: %s in %s runs' % (sum_100Q17Reads, sum_Q17Bases, sum_Runs)
    if bestRun:
        print 'Best run: %s' % (web_root + bestRun.reportLink) # need to handle bestRun = None case?
        print 'Best run 100Q17 reads: %s 100Q17 bases: %s' % (curBestReads, curBestBases)
    else:
        print 'There were no best runs for this report.'

    # top 5
    top5 = sorted(topN, byReads)[:5]
    for i, top in enumerate(top5):
        print 'Run: %s 100Q17 reads: %s link: %s' % (i+1, top.reads, web_root + top.r.reportLink)

if __name__=="__main__":
    generateReport()

