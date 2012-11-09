# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
import datetime
from os import path
sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from django.db import models
from iondb.rundb import models
import re

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
        self.q17bases = 0
        self.q20bases = 0
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
        self.projectFilter = False
        self.project = ''
        if (':' in metricName):
            self.plugin = True
            self.pluginStore = metricName.split(':')[0]
            self.pluginMetric = metricName.split(':')[1]
    def Reverse(self):
        self.reverse = True
        self.recordValue = 999999
    def SetProject(self, project):
        self.projectFilter = True
        self.project = project


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
                    try:
                        isPairedEnd = (rep.metaData["paired"] == 1)
                    except:
                        isPairedEnd = False
                    if libmetrics.align_sample == 0 and not isPairedEnd:
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
                metricRecord.q20bases = metricRecord.q20bases + bestLib.q20_mapped_bases
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
                        metricRecord.recordReportList.append((bestResult, bestVal, bestLib.q17_mapped_bases, bestLib.q20_mapped_bases, web_root))
                    else:
                        metricRecord.recordReportList[0] = (bestResult, bestVal, bestLib.q17_mapped_bases, bestLib.q20_mapped_bases, web_root)


def BuildMetrics(metricRecordList, site):
    # metricRecordList is an array of type MetricRecord
    # site is a string containing the site name - must match the database configuration name from settings.py

    print 'Processing site: %s' % site

    # get the url base for the site
    web_root = GetWebRoot(site)

    # some init

    # filters we can use to ignore/keep records of interest while looping over all of them
    chipFilter = True
    dateFilter = False

    # select all result records (maybe here would be a good place to just get them all into memory fast?)
    # and on performance, we will hit this site's report list many times, looking for a specific chip, metric, etc so would be good to keep this around
    repList = models.Results.objects.using(site).select_related()  # or we can filter by date for example: filter(timeStamp__range=(queryStart,queryEnd))
    print "forcing repList loading"
    theLen = len(repList) # force load of database table across net

    # for any projects we have, make additional sub-filtered lists
    repList_avalanche = repList.filter(projects__name='avalanche')

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
        libpk_hash[pk] = num
        num = num + 1

    # loop through all metrics, updating top 5 (numBest) list for each
    for metricRecord in metricRecordList:
        if metricRecord.track:
            continue

        print 'Processing metric: %s' % metricRecord.metricName

        print 'updating report list...'
        # look at all reports
        useRepList = repList
        if metricRecord.projectFilter:
            useRepList = repList_avalanche
        for rep in useRepList:
            libmetrics = None
            # look up the library table entry from the report pk via our pre-generated hash lookup array
            try:
                libIndex = libpk_hash[rep.pk]
                libmetrics = libList[libIndex]
                if libmetrics.align_sample != 0:
                    libmetrics = None # ignore any sampled & extrapolated runs
            except:
                libmetrics = None

            try:
                isPairedEnd = (rep.metaData["paired"] == 1)
            except:
                isPairedEnd = False

            if libmetrics is not None and not isPairedEnd:
                # optional filters
                ok = True
                if ok and chipFilter:
                    ok = False
                    if (metricRecord.chip in rep.experiment.chipType):
                        ok = True

                #if ok and metricRecord.projectFilter:
                    #ok = False
                    #if re.search(metricRecord.project, rep.experiment.project, re.IGNORECASE):
                    #projectNames = rep.projectNames()
                    #for projectName in projectNames.split(','):
                        #if re.search(metricRecord.project, projectName, re.IGNORECASE):
                            #ok = True

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
                            for rep_item, rep_value, q17bases, q20bases, webRoot in metricRecord.recordReportList:
                                # only replace it if its a better record
                                if (rep_item.experiment.pk == rep.experiment.pk):
                                    repFound = True # found it, but report might not be better than the one already added
                                    if ((metricRecord.reverse == False and val > rep_value) or (metricRecord.reverse == True and val < rep_value)):
                                        repToReplace = repIndex
                                repIndex = repIndex + 1
                            if repToReplace > -1: # found and its better than the one already added
                                metricRecord.recordReportList[repToReplace] = (rep, val, libmetrics.q17_mapped_bases, libmetrics.q20_mapped_bases, web_root)
                                #print 'replaced experiment %s using newer report value %s' % (rep.experiment.expName, val)
                            else:
                                # only add if we didn't add this experiment already
                                if repFound == False:
                                    if (metricRecord.numInList < metricRecord.numBest):
                                        metricRecord.recordReportList.append((rep, val, libmetrics.q17_mapped_bases, libmetrics.q20_mapped_bases, web_root))
                                        metricRecord.numInList = metricRecord.numInList + 1
                                    else:
                                        # replace the worst item in the list
                                        metricRecord.recordReportList[metricRecord.numInList-1] = (rep, val, libmetrics.q17_mapped_bases, libmetrics.q20_mapped_bases, web_root)
    
                            #re-sort the list, and set the min recordValue to 0 if list not full (so we can continue to add to the list), or to the worst item in the list, so we can replace that item if/when new record is found
                            metricRecord.recordReportList = sorted(metricRecord.recordReportList, key=lambda val: val[1], reverse=(not metricRecord.reverse))
                            if metricRecord.numInList == metricRecord.numBest:
                                rep, metricRecord.recordValue, q17bases, q20bases, webRoot = metricRecord.recordReportList[metricRecord.numInList-1]
                            else:
                                if metricRecord.reverse == False:
                                    metricRecord.recordValue = 0
                                else:
                                    metricRecord.recordValue = 999999



def DumpMetric(metricRecord):

    if metricRecord.track:
        # display tracking summary metrics a bit different
        print 'Site Tracking for %s' % metricRecord.site
        html.write('Site Tracking for %s<br>' % metricRecord.site)
        print 'Runs: %s  %s: %s  AQ17 bases: %s  AQ20 bases: %s  314/316/318: %s/%s/%s' % (metricRecord.count, metricRecord.metricName, metricRecord.metricSum, metricRecord.q17bases, metricRecord.q20bases, metricRecord.sum314, metricRecord.sum316, metricRecord.sum318)
        html.write('Runs: %s  %s: %s  AQ17 bases: %s  AQ20 bases: %s  314/316/318: %s/%s/%s<br>' % (metricRecord.count, metricRecord.metricName, metricRecord.metricSum, metricRecord.q17bases, metricRecord.q20bases, metricRecord.sum314, metricRecord.sum316, metricRecord.sum318))
        try:
            recordReport, recordValue, q17bases, q20bases, webRoot = metricRecord.recordReportList[0]
            print 'Top run %s: %s AQ17 bases: %s AQ20 bases: %s Run date: %s  Analysis date: %s' % (metricRecord.metricName, recordValue, q17bases, q20bases, recordReport.experiment.date, recordReport.timeStamp)
            html.write('<a href="%s">Top Run %s: %s AQ17 bases: %s AQ20bases: %s Run date: %s Analysis date: %s</a><br>\n' % (webRoot + recordReport.reportLink, metricRecord.metricName, recordValue, q17bases, q20bases, recordReport.experiment.date, recordReport.timeStamp))
        except:
            print 'No top run found?'
            html.write('No top run found?<br>')
    else:
        # display our top N
        if metricRecord.numBest > 1:
            html.write('Ion Chip: %s Top %s %s Runs<br>' % (metricRecord.chip, metricRecord.numBest, metricRecord.metricName))
        else:
            html.write('Ion Chip: %s Top %s Run<br>' % (metricRecord.chip, metricRecord.metricName))

        for recordReport, recordValue, q17bases, q20bases, webRoot in metricRecord.recordReportList:
            print '%s: %s AQ17 bases: %s AQ20 bases: %s Run date: %s  Analysis date: %s' % (metricRecord.metricName, recordValue, q17bases, q20bases, recordReport.experiment.date, recordReport.timeStamp)
            print 'URL: %s' % (webRoot + recordReport.reportLink)
            html.write('<a href="%s">%s AQ17 bases: %s AQ20 bases: %s Run date: %s Analysis date: %s</a><br>\n' % (webRoot + recordReport.reportLink, recordValue, q17bases, q20bases, recordReport.experiment.date, recordReport.timeStamp))


def lookupSite(url):
    if 'cbd01' in url:
        return 'SoCal'
    if 'aruba' in url:
        return 'Bev'
    if 'ioneast' in url:
        return 'IE'
    if 'ionwest' in url:
        return 'IW'
    if 'pbox' in url:
        return 'Bev/pbox'
    return 'unknown'


if __name__=="__main__":
    #siteList = ['ioneast', 'ionwest', 'beverly', 'pbox', 'ioncarlsbad']
    siteList = ['ioneast', 'ionwest', 'beverly', 'ioncarlsbad']
    #siteList = ['ioncarlsbad']

    metricRecords = []

    m1 = MetricRecord('i100Q17_reads', 5, '314')
    m2 = MetricRecord('i200Q17_reads', 5, '314')
    m3 = MetricRecord('i300Q17_reads', 5, '314')
    m4 = MetricRecord('i400Q17_reads', 5, '314')
    m5 = MetricRecord('i100Q17_reads', 5, '316')
    m6 = MetricRecord('i200Q17_reads', 5, '316')
    m7 = MetricRecord('i300Q17_reads', 5, '316')
    m8 = MetricRecord('i400Q17_reads', 5, '316')
    m9 = MetricRecord('i100Q17_reads', 5, '318')
    m10 = MetricRecord('i200Q17_reads', 5, '318')
    m11 = MetricRecord('i300Q17_reads', 5, '318')
    m12 = MetricRecord('i400Q17_reads', 5, '318')

    m13 = MetricRecord('i100Q20_reads', 5, '314')
    m14 = MetricRecord('i200Q20_reads', 5, '314')
    m15 = MetricRecord('i300Q20_reads', 5, '314')
    m16 = MetricRecord('i400Q20_reads', 5, '314')
    m17 = MetricRecord('i100Q20_reads', 5, '316')
    m18 = MetricRecord('i200Q20_reads', 5, '316')
    m19 = MetricRecord('i300Q20_reads', 5, '316')
    m20 = MetricRecord('i400Q20_reads', 5, '316')
    m21 = MetricRecord('i100Q20_reads', 5, '318')
    m22 = MetricRecord('i200Q20_reads', 5, '318')
    m23 = MetricRecord('i300Q20_reads', 5, '318')
    m24 = MetricRecord('i400Q20_reads', 5, '318')

    metricRecords.append(m1)
    metricRecords.append(m2)
    metricRecords.append(m3)
    metricRecords.append(m4)
    metricRecords.append(m5)
    metricRecords.append(m6)
    metricRecords.append(m7)
    metricRecords.append(m8)
    metricRecords.append(m9)
    metricRecords.append(m10)
    metricRecords.append(m11)
    metricRecords.append(m12)
    metricRecords.append(m13)
    metricRecords.append(m14)
    metricRecords.append(m15)
    metricRecords.append(m16)
    metricRecords.append(m17)
    metricRecords.append(m18)
    metricRecords.append(m19)
    metricRecords.append(m20)
    metricRecords.append(m21)
    metricRecords.append(m22)
    metricRecords.append(m23)
    metricRecords.append(m24)

    IonStats_200Q20 = MetricRecord('i200Q20_reads', 1, '31')
    metricRecords.append(IonStats_200Q20)
    IonStats_Q7 = MetricRecord('q7_longest_alignment', 1, '31')
    metricRecords.append(IonStats_Q7)
    IonStats_Q10 = MetricRecord('q10_longest_alignment', 1, '31')
    metricRecords.append(IonStats_Q10)
    IonStats_Q17 = MetricRecord('q17_longest_alignment', 1, '31')
    metricRecords.append(IonStats_Q17)
    IonStats_Q20 = MetricRecord('q20_longest_alignment', 1, '31')
    metricRecords.append(IonStats_Q20)
    IonStats_Q47 = MetricRecord('q47_longest_alignment', 1, '31')
    metricRecords.append(IonStats_Q47)
    metricRecords.append(MetricRecord('q17_mean_alignment_length', 5, '31'))
    metricRecords.append(MetricRecord('q20_mean_alignment_length', 5, '31'))
    metricRecords.append(MetricRecord('q47_mean_alignment_length', 5, '31'))
    metricRecords.append(MetricRecord('q17_mapped_bases', 3, '314'))
    metricRecords.append(MetricRecord('q17_mapped_bases', 3, '316'))
    metricRecords.append(MetricRecord('q17_mapped_bases', 3, '318'))
    metricRecords.append(MetricRecord('q20_mapped_bases', 5, '314'))
    metricRecords.append(MetricRecord('q20_mapped_bases', 5, '316'))
    metricRecords.append(MetricRecord('q20_mapped_bases', 5, '318'))
    #pluginMetric = MetricRecord('ampliconGeneralAnalysis:coverage_needed_for_99_percentile_base_with_at_least_1x_coverage', 5, '31')
    #pluginMetric.Reverse()
    #metricRecords.append(pluginMetric)
    #pluginMetric = MetricRecord('ampliconGeneralAnalysis:coverage_needed_for_98_percentile_base_with_at_least_10x_coverage', 5, '31')
    #pluginMetric.Reverse()
    #metricRecords.append(pluginMetric)

    # Avalanche metrics tracking
    av_314_q17 = MetricRecord('q17_mapped_bases', 10, '314')
    av_314_q17.SetProject('avalanche')
    metricRecords.append(av_314_q17)
    av_316_q17 = MetricRecord('q17_mapped_bases', 10, '316')
    av_316_q17.SetProject('avalanche')
    metricRecords.append(av_316_q17)
    av_318_q17 = MetricRecord('q17_mapped_bases', 10, '318')
    av_318_q17.SetProject('avalanche')
    metricRecords.append(av_318_q17)

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
        weeklySite = MetricRecord('i100Q17_reads', 1, '31')
        weeklySite.track = True
        weeklySite.dateFilter = True
        weeklySite.dateMin = timeStart
        weeklySite.dateMax = timeEnd
        weeklySite.site = site
        weeklySite.siteFilter = True
        metricRecords.append(weeklySite)

        weeklySite = MetricRecord('i100Q20_reads', 1, '31')
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

    print 'Weekly metrics captured from %s to %s' % (timeStart, timeEnd)
    html.write('Weekly metrics captured from %s to %s<br>' % (timeStart, timeEnd))

    # write all non project-specific metrics first
    for metricRecord in metricRecords:
        # quick cleanup of recordValue to make sure it now reflects the top record, not the barrier to entry for a record
        if metricRecord.numInList > 0:
            rep, metricRecord.recordValue, q17bases, q20bases, webRoot = metricRecord.recordReportList[0]
        if not metricRecord.projectFilter:
            DumpMetric(metricRecord)

    # now write out any project-specific metrics
    project = 'avalanche'
    html.write('</br></br>Tracking for project: %s</br></br>' % project)
    for metricRecord in metricRecords:
        if metricRecord.projectFilter and project == metricRecord.project:
            DumpMetric(metricRecord)

    html.write('</body></html>\n')
    html.close()

    # dump out a metrics file for use with PGM IonStats screensaver
    IonStats = open("metrics.txt", 'w')
    IonStats.write('%s,%s,%s\n' % (IonStats_Q7.metricName, IonStats_Q7.recordValue, 0))
    IonStats.write('%s,%s,%s\n' % (IonStats_Q10.metricName, IonStats_Q10.recordValue, 0))
    IonStats.write('%s,%s,%s\n' % (IonStats_Q17.metricName, IonStats_Q17.recordValue, 0))
    IonStats.write('%s,%s,%s\n' % (IonStats_Q20.metricName, IonStats_Q20.recordValue, 0))
    IonStats.write('%s,%s,%s\n' % (IonStats_Q47.metricName, IonStats_Q47.recordValue, 0))
    IonStats.write('%s,%s,%s\n' % (IonStats_200Q20.metricName, IonStats_200Q20.recordValue, 0))
    IonStats.close()

    # dump out a csv file for copy/paste into excel tracker
    IonWeekly = open("weekly.csv", "w")
    IonWeekly.write('314 RECORDS\n')
    for i in range(5):
        recordReport1, recordValue1, q17bases1, q20bases1, webRoot1 = m1.recordReportList[i]
        recordReport2, recordValue2, q17bases2, q20bases2, webRoot2 = m13.recordReportList[i]
        siteText1 = lookupSite(webRoot1)
        siteText2 = lookupSite(webRoot2)
        IonWeekly.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (recordValue1, q17bases1, recordReport1.timeStamp, siteText1, webRoot1 + recordReport1.reportLink, recordValue2, q20bases2, recordReport2.timeStamp, siteText2, webRoot2 + recordReport2.reportLink))
    IonWeekly.write('\n\n\n\n');
    for i in range(5):
        recordReport1, recordValue1, q17bases1, q20bases1, webRoot1 = m2.recordReportList[i]
        recordReport2, recordValue2, q17bases2, q20bases2, webRoot2 = m14.recordReportList[i]
        siteText1 = lookupSite(webRoot1)
        siteText2 = lookupSite(webRoot2)
        IonWeekly.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (recordValue1, q17bases1, recordReport1.timeStamp, siteText1, webRoot1 + recordReport1.reportLink, recordValue2, q20bases2, recordReport2.timeStamp, siteText2, webRoot2 + recordReport2.reportLink))
    IonWeekly.write('\n\n\n\n');
    for i in range(5):
        recordReport1, recordValue1, q17bases1, q20bases1, webRoot1 = m3.recordReportList[i]
        recordReport2, recordValue2, q17bases2, q20bases2, webRoot2 = m15.recordReportList[i]
        siteText1 = lookupSite(webRoot1)
        siteText2 = lookupSite(webRoot2)
        IonWeekly.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (recordValue1, q17bases1, recordReport1.timeStamp, siteText1, webRoot1 + recordReport1.reportLink, recordValue2, q20bases2, recordReport2.timeStamp, siteText2, webRoot2 + recordReport2.reportLink))
    IonWeekly.write('\n\n\n\n');
    for i in range(5):
        recordReport1, recordValue1, q17bases1, q20bases1, webRoot1 = m4.recordReportList[i]
        recordReport2, recordValue2, q17bases2, q20bases2, webRoot2 = m16.recordReportList[i]
        siteText1 = lookupSite(webRoot1)
        siteText2 = lookupSite(webRoot2)
        IonWeekly.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (recordValue1, q17bases1, recordReport1.timeStamp, siteText1, webRoot1 + recordReport1.reportLink, recordValue2, q20bases2, recordReport2.timeStamp, siteText2, webRoot2 + recordReport2.reportLink))
    IonWeekly.write('\n\n\n\n');

    IonWeekly.write('316 RECORDS\n')
    for i in range(5):
        recordReport1, recordValue1, q17bases1, q20bases1, webRoot1 = m5.recordReportList[i]
        recordReport2, recordValue2, q17bases2, q20bases2, webRoot2 = m17.recordReportList[i]
        siteText1 = lookupSite(webRoot1)
        siteText2 = lookupSite(webRoot2)
        IonWeekly.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (recordValue1, q17bases1, recordReport1.timeStamp, siteText1, webRoot1 + recordReport1.reportLink, recordValue2, q20bases2, recordReport2.timeStamp, siteText2, webRoot2 + recordReport2.reportLink))
    IonWeekly.write('\n\n\n\n');
    for i in range(5):
        recordReport1, recordValue1, q17bases1, q20bases1, webRoot1 = m6.recordReportList[i]
        recordReport2, recordValue2, q17bases2, q20bases2, webRoot2 = m18.recordReportList[i]
        siteText1 = lookupSite(webRoot1)
        siteText2 = lookupSite(webRoot2)
        IonWeekly.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (recordValue1, q17bases1, recordReport1.timeStamp, siteText1, webRoot1 + recordReport1.reportLink, recordValue2, q20bases2, recordReport2.timeStamp, siteText2, webRoot2 + recordReport2.reportLink))
    IonWeekly.write('\n\n\n\n');
    for i in range(5):
        recordReport1, recordValue1, q17bases1, q20bases1, webRoot1 = m7.recordReportList[i]
        recordReport2, recordValue2, q17bases2, q20bases2, webRoot2 = m19.recordReportList[i]
        siteText1 = lookupSite(webRoot1)
        siteText2 = lookupSite(webRoot2)
        IonWeekly.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (recordValue1, q17bases1, recordReport1.timeStamp, siteText1, webRoot1 + recordReport1.reportLink, recordValue2, q20bases2, recordReport2.timeStamp, siteText2, webRoot2 + recordReport2.reportLink))
    IonWeekly.write('\n\n\n\n');
    for i in range(5):
        recordReport1, recordValue1, q17bases1, q20bases1, webRoot1 = m8.recordReportList[i]
        recordReport2, recordValue2, q17bases2, q20bases2, webRoot2 = m20.recordReportList[i]
        siteText1 = lookupSite(webRoot1)
        siteText2 = lookupSite(webRoot2)
        IonWeekly.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (recordValue1, q17bases1, recordReport1.timeStamp, siteText1, webRoot1 + recordReport1.reportLink, recordValue2, q20bases2, recordReport2.timeStamp, siteText2, webRoot2 + recordReport2.reportLink))
    IonWeekly.write('\n\n\n\n');

    IonWeekly.write('318 RECORDS\n')
    for i in range(5):
        recordReport1, recordValue1, q17bases1, q20bases1, webRoot1 = m9.recordReportList[i]
        recordReport2, recordValue2, q17bases2, q20bases2, webRoot2 = m21.recordReportList[i]
        siteText1 = lookupSite(webRoot1)
        siteText2 = lookupSite(webRoot2)
        IonWeekly.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (recordValue1, q17bases1, recordReport1.timeStamp, siteText1, webRoot1 + recordReport1.reportLink, recordValue2, q20bases2, recordReport2.timeStamp, siteText2, webRoot2 + recordReport2.reportLink))
    IonWeekly.write('\n\n\n\n');
    for i in range(5):
        recordReport1, recordValue1, q17bases1, q20bases1, webRoot1 = m10.recordReportList[i]
        recordReport2, recordValue2, q17bases2, q20bases2, webRoot2 = m22.recordReportList[i]
        siteText1 = lookupSite(webRoot1)
        siteText2 = lookupSite(webRoot2)
        IonWeekly.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (recordValue1, q17bases1, recordReport1.timeStamp, siteText1, webRoot1 + recordReport1.reportLink, recordValue2, q20bases2, recordReport2.timeStamp, siteText2, webRoot2 + recordReport2.reportLink))
    IonWeekly.write('\n\n\n\n');
    for i in range(5):
        recordReport1, recordValue1, q17bases1, q20bases1, webRoot1 = m11.recordReportList[i]
        recordReport2, recordValue2, q17bases2, q20bases2, webRoot2 = m23.recordReportList[i]
        siteText1 = lookupSite(webRoot1)
        siteText2 = lookupSite(webRoot2)
        IonWeekly.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (recordValue1, q17bases1, recordReport1.timeStamp, siteText1, webRoot1 + recordReport1.reportLink, recordValue2, q20bases2, recordReport2.timeStamp, siteText2, webRoot2 + recordReport2.reportLink))
    IonWeekly.write('\n\n\n\n');
    for i in range(5):
        recordReport1, recordValue1, q17bases1, q20bases1, webRoot1 = m12.recordReportList[i]
        recordReport2, recordValue2, q17bases2, q20bases2, webRoot2 = m24.recordReportList[i]
        siteText1 = lookupSite(webRoot1)
        siteText2 = lookupSite(webRoot2)
        IonWeekly.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (recordValue1, q17bases1, recordReport1.timeStamp, siteText1, webRoot1 + recordReport1.reportLink, recordValue2, q20bases2, recordReport2.timeStamp, siteText2, webRoot2 + recordReport2.reportLink))

    IonWeekly.close()

