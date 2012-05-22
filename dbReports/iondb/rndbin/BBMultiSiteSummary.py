# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os

# Settings is emotionally complicated, when you import it, nothing happens, but
# when you either read or write any of it's variables, it will configure and
# load Django's logger as defined in settings.
from django.conf import global_settings
# Settings inherits the properties of global_settings which is a much simpler
# beast.  By settings LOGGING_CONFIG to None here, settings will not act on
# it's logging configuration when we read other stuff from it.
global_settings.LOGGING_CONFIG=None

import datetime
from os import path
sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from django.db import models
from iondb.rundb import models
import re

from django.conf import settings
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
    'Proton_Bev':{
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'iondb',
        'USER': 'ion',
        'PASSWORD': 'ion',
        'HOST': 'blackbird.bev'
    }
}
for site in Proton:
    settings.DATABASES[site] = Proton[site]

def GetWebRoot(site):
    gc = models.GlobalConfig.objects.using(site).all()[0]
    webRoot = gc.web_root
    if len(webRoot) > 0:
        if webRoot[-1] == '/':
            webRoot = webRoot[:len(webRoot) - 1]
    return webRoot

class MetricRecord:
    def __init__(self, metricName, numBest, chip):
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
        if (':' in metricName):
            self.plugin = True
            self.pluginStore = metricName.split(':')[0]
            self.pluginMetric = metricName.split(':')[1]
    def Reverse(self):
        self.reverse = True
        self.recordValue = 999999


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
            for rep in repList:
                try:
                    libmetrics = rep.libmetrics_set.all()[0] # ok, there's really only one libmetrics set per result, but we still need to get at it
                    # MGD - speedup here - we can re-use the hash lookup we created per site below so avoid the database query per record here
                except IndexError:
                    libmetrics = None
    
                if libmetrics is not None:
                    if libmetrics.align_sample == 0:
                        valtext = getattr(libmetrics, metricRecord.metricName)
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
    chipFilter = True
    projectFilter = False
    project = ''
    dateFilter = False

    # select all result records (maybe here would be a good place to just get them all into memory fast?)
    # and on performance, we will hit this site's report list many times, looking for a specific chip, metric, etc so would be good to keep this around
    repList = models.Results.objects.using(site).select_related()  # or we can filter by date for example: filter(timeStamp__range=(queryStart,queryEnd))
    print "forcing repList loading"
    theLen = len(repList) # force load of database table across net

    libList = models.LibMetrics.objects.using(site).select_related()
    print "forcing libmetrics loading"
    theLen = len(libList) # force load of database table across net

    # make a lookup table of library metrics pcsRecord.chip, k
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
            # see if this new libmterics assosiated with a report is 'better' than the current one
            if lib.q17_mapped_bases > libList[libpk_hash[pk]].q17_mapped_bases:
                libpk_hash[pk] = num
        num = num + 1

    # loop through all metrics, updating top 5 (numBest) list for each
    for metricRecord in metricRecordList:
        if metricRecord.track:
            continue

        print 'Processing metric: %s' % metricRecord.metricName

        print 'updating report list...'
        # look at all reports
        for rep in repList:
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
                if ok and chipFilter:
                    ok = False
                    if (metricRecord.chip in rep.experiment.chipType):
                        ok = True

                if ok and projectFilter:
                    ok = False
                    if re.search(project, rep.experiment.project, re.IGNORECASE):
                        ok = True

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
            html.write('<a href="%s">Top Run 100Q17 reads: %s Q17 bases: %s Run date: %s Analysis date: %s</a><br>\n' % (webRoot + recordReport.reportLink, i100Q17_reads, q17bases, recordReport.experiment.date, recordReport.timeStamp))
        except:
            print 'No top run found?'
            html.write('No top run found?<br>')
    else:
        # display our top N
        if metricRecord.numBest > 1:
            html.write('Ion Chip: %s Top %s %s Runs<br>' % (metricRecord.chip, metricRecord.numBest, metricRecord.metricName))
        else:
            html.write('Ion Chip: %s Top %s Run<br>' % (metricRecord.chip, metricRecord.metricName))

        for recordReport, recordValue, i100Q17_reads, q17bases, webRoot in metricRecord.recordReportList:
            print '%s 100Q17 reads: %s Q17 bases: %s Run date: %s  Analysis date: %s' % (recordValue, i100Q17_reads, q17bases, recordReport.experiment.date, recordReport.timeStamp)
            print 'URL: %s' % (webRoot + recordReport.reportLink)
            html.write('<a href="%s">%s 100Q17 reads: %s  Q17 bases: %s Run date: %s Analysis date: %s</a><br>\n' % (webRoot + recordReport.reportLink, recordValue, i100Q17_reads, q17bases, recordReport.experiment.date, recordReport.timeStamp))

 


if __name__=="__main__":
    siteList = ['Proton_East', 'Proton_West', 'Proton_Bev']

    metricRecords = []
    metricRecords.append(MetricRecord('i50Q17_reads', 5, '9'))
    metricRecords.append(MetricRecord('i100Q17_reads', 5, '9'))
    IonStats_Q10 = MetricRecord('q10_longest_alignment', 5, '9')
    metricRecords.append(IonStats_Q10)
    IonStats_Q17 = MetricRecord('q17_longest_alignment', 5, '9')
    metricRecords.append(IonStats_Q17)
    IonStats_Q20 = MetricRecord('q20_longest_alignment', 5, '9')
    metricRecords.append(IonStats_Q20)
    IonStats_Q47 = MetricRecord('q47_longest_alignment', 5, '9')
    metricRecords.append(IonStats_Q47)
    metricRecords.append(MetricRecord('q17_mapped_bases', 5, '9'))
    metricRecords.append(MetricRecord('q17_mean_alignment_length', 10, '9'))

    today = datetime.date.today()
    timeStart = datetime.datetime(today.year, today.month, today.day)
    daysFromMonday = timeStart.weekday() # Monday is 0, so if its Thursday (3), we need to go back 3 days
    lengthOfReport = 7 # report for 7 days, with Monday being the first day included in a report
    if daysFromMonday < lengthOfReport: # we want to go back to the start of the full week, if we are in the middle of a week, need to go back to the start of last week
        daysFromMonday = daysFromMonday + 7
    timeStart = timeStart - datetime.timedelta(days=daysFromMonday)
    timeEnd = timeStart + datetime.timedelta(days=lengthOfReport)
    # timeEnd = timeEnd - datetime.timedelta(seconds=1)

    for site in siteList:
        # weeklySite = MetricRecord('i100Q17_reads', 1, '9')
        weeklySite = MetricRecord('q17_mapped_bases', 1, '9')
        weeklySite.track = True
        weeklySite.dateFilter = True
        weeklySite.dateMin = timeStart
        weeklySite.dateMax = timeEnd
        weeklySite.site = site
        weeklySite.siteFilter = True
        metricRecords.append(weeklySite)

    for site in siteList:
        BuildMetrics(metricRecords, site)
        BuildTrackingMetrics(metricRecords, site)

    html = open("top-" + str(datetime.date.today())+ ".html",'w')
    html.write('<html><body>\n')

    #print 'Weekly metrics captured from %s to %s' % (timeStart, timeEnd)
    #html.write('Weekly metrics captured from %s to %s<br>' % (timeStart, timeEnd))
    print 'Multi-site metrics'
    html.write('Multi-site metrics')

    for metricRecord in metricRecords:
        # quick cleanup of recordValue to make sure it now reflects the top record, not the barrier to entry for a record
        if metricRecord.numInList > 0:
            rep, metricRecord.recordValue, i100Q17_reads, q17bases, webRoot = metricRecord.recordReportList[0]
        DumpMetric(metricRecord)

    html.write('</body></html>\n')
    html.close()

    # dump out a metrics file for use with PGM IonStats screensaver
    IonStats = open("metrics.txt", 'w')
    IonStats.write('%s,%s,%s\n' % (IonStats_Q10.metricName, IonStats_Q10.recordValue, 0))
    IonStats.write('%s,%s,%s\n' % (IonStats_Q17.metricName, IonStats_Q17.recordValue, 0))
    IonStats.write('%s,%s,%s\n' % (IonStats_Q20.metricName, IonStats_Q20.recordValue, 0))
    IonStats.write('%s,%s,%s\n' % (IonStats_Q47.metricName, IonStats_Q47.recordValue, 0))

