# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
import datetime
from os import path
sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from django.db import models
from iondb.rundb import models
import re
import math

RECIPS = "Mel.Davey@Lifetech.com"
SENDER = "donotreply@iontorrent.com"

def send_html(sender,recips,subject,html):
    msg = mail.EmailMessage(subject,html,sender,recips)
    msg.content_subtype = "html"
    msg.send()

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
            self.reverse = True
            self.recordValue = 99999

def ion_readable(value):
    charlist = []
    charlist.append("")
    charlist.append("K")
    charlist.append("M")
    charlist.append("G")

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
        converted_text = str(text) + charlist[charindex]
    else:
        converted_text = str(value)

    return converted_text

def BuildTrackingMetrics(metricRecordList, site):
    print 'Tracking site: %s' % site

    # get the url base for the site
    web_root = GetWebRoot(site)

    repList = models.Results.objects.using(site).filter(timeStamp__range=(timeStart, timeEnd))

    for rep in repList:
        if (rep.status == 'Completed'):
            val = 0
            libmetrics = rep.libmetrics_set.all()[0]
            exp = rep.experiment
            rset = exp.results_set.all().order_by('timeStamp')
            for metricRecord in metricRecordList:
                if metricRecord.track:
                    val = 0
                    pluginDict = None
                    if metricRecord.plugin:
                        try:
                            pluginDict = rep.getPluginStore()[metricRecord.pluginStore]
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
        
                    if val > 0:
                        if rset[0].pk == rep.pk:
                            metricRecord.recordReportList.append((rep, libmetrics.q7_alignments, libmetrics.q17_mapped_bases, pluginDict, libmetrics.sysSNR, web_root))
                            metricRecord.numInList = metricRecord.numInList + 1
                        # else its a re-analysis



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

    pluginResults = models.PluginResult.objects.using(site).select_related() 
    print "forcing pluginResults loading"
    theLen = len(pluginResults) # force load of database table across net

    plugins = models.Plugin.objects.using(site).select_related()
    print "forcing plugins loading"
    theLen = len(plugins) # force load of database table across net

    # make a lookup table of library metrics pk
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

    # make a lookup table of plugin results pk
    print 'Building plugin results hash...'
    pluginHashDict = {}
    for plugin in plugins:
        pluginHashDict[plugin.name] = []
        whichPluginHash = pluginHashDict[plugin.name]
        for i in range(0, largest+1):
            whichPluginHash.append(-1)
    num = 0
    for pluginResult in pluginResults:
        pk = pluginResult.result.pk
        whichPluginHash = pluginHashDict[pluginResult.plugin.name]
        whichPluginHash[pk] = num
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
                    pluginDict = None
                    if metricRecord.plugin:
                        try:
                            # pluginDict = rep.getPluginStore()[metricRecord.pluginStore]

                            whichPluginHash = pluginHashDict[metricRecord.pluginStore]
                            pluginDictIndex = whichPluginHash[rep.pk]
                            pluginDict = pluginResults[pluginDictIndex].store
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

                    if val > 0:
                        if (((metricRecord.reverse ==  False) and (val > metricRecord.recordValue)) or ((metricRecord.reverse ==  True) and (val < metricRecord.recordValue))):
                            # if report's parent experiment already exists in the list, replace it, else insert it
                            repIndex = 0
                            repToReplace = -1
                            repFound = False
                            for rep_item, rep_value, q7reads, q17bases, pdict, snr, webRoot in metricRecord.recordReportList:
                                # only replace it if its a better record
                                if (rep_item.experiment.pk == rep.experiment.pk):
                                    repFound = True # found it, but report might not be better than the one already added
                                    if (((metricRecord.reverse ==  False) and (val > rep_value)) or ((metricRecord.reverse ==  True) and (val < rep_value))):
                                        repToReplace = repIndex
                                repIndex = repIndex + 1
                            if repToReplace > -1: # found and its better than the one already added
                                metricRecord.recordReportList[repToReplace] = (rep, val, libmetrics.q7_alignments, libmetrics.q17_mapped_bases, pluginDict, libmetrics.sysSNR, web_root)
                                #print 'replaced experiment %s using newer report value %s' % (rep.experiment.expName, val)
                            else:
                                # only add if we didn't add this experiment already
                                if repFound == False:
                                    if (metricRecord.numInList < metricRecord.numBest):
                                        metricRecord.recordReportList.append((rep, val, libmetrics.q7_alignments, libmetrics.q17_mapped_bases, pluginDict, libmetrics.sysSNR, web_root))
                                        metricRecord.numInList = metricRecord.numInList + 1
                                    else:
                                        # replace the worst item in the list
                                        metricRecord.recordReportList[metricRecord.numInList-1] = (rep, val, libmetrics.q7_alignments, libmetrics.q17_mapped_bases, pluginDict, libmetrics.sysSNR, web_root)
    
                            #re-sort the list, and set the min recordValue to 0 if list not full (so we can continue to add to the list), or to the worst item in the list, so we can replace that item if/when new record is found
                            metricRecord.recordReportList = sorted(metricRecord.recordReportList, key=lambda val: val[1], reverse=(not metricRecord.reverse))
                            if metricRecord.numInList == metricRecord.numBest:
                                rep, metricRecord.recordValue, q7reads, q17bases, pdict, snr, webRoot = metricRecord.recordReportList[metricRecord.numInList-1]
                            else:
                                if metricRecord.reverse:
                                    metricRecord.recordValue = 99999
                                else:
                                    metricRecord.recordValue = 0




if __name__=="__main__":
    siteList = ['ioneast', 'ionwest', 'beverly', 'pbox', 'ioncarlsbad']

    # get the list of all results generated over a 24hr period, ending at 6am
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    timeEnd = datetime.datetime(today.year, today.month, today.day)
    timeEnd = timeEnd + datetime.timedelta(hours=6)
    timeStart = timeEnd - datetime.timedelta(days=1)

    metricRecords = []
    top5_m1 = MetricRecord('ampliconGeneralAnalysis:coverage_needed_for_99_percentile_base_with_at_least_1x_coverage', 5, '31')
    top5_m2 = MetricRecord('ampliconGeneralAnalysis:coverage_needed_for_98_percentile_base_with_at_least_10x_coverage', 5, '31')
    summary = MetricRecord('ampliconGeneralAnalysis:coverage_needed_for_98_percentile_base_with_at_least_10x_coverage', 1, '31')
    metricRecords.append(top5_m1)
    metricRecords.append(top5_m2)
    metricRecords.append(summary)

    # mark the summary metric as a tracking metric
    summary.track = True

    for site in siteList:
        BuildMetrics(metricRecords, site)
        BuildTrackingMetrics(metricRecords, site)

    html = open("/results/custom_reports/ampl-" + str(datetime.date.today())+ ".html",'w')
    html.write('<html><body>\n')

    html.write(\
        '<h3>Amplicon Report Summary</h3>'
        '<h3>%s</h3><br>'\
        % (today.strftime('%B %d %Y'))\
    )

    html.write(\
        '<div style="width:100%%; background-color:#DDD;'
              'margin-top:16px; margin-left:auto;'
              'margin-right:auto; padding:8px;">'
              '<p><h2><center>New Run Results from %s</center></h2></p></div>\n' % yesterday.strftime('%B %d %Y'))

    # New Run Results Table
    html.write('<table style="width: 100%; border-collapse: collapse;">')
    html.write('<thead style="text-align:left;">')
    html.write(\
        '<tr>'
          '<th ALIGN="left" rowspan="2">Name</th>'
          '<th ALIGN="center" rowspan="2">Chip Type</th>'
          '<th ALIGN="center" rowspan="2">Project</th>'
          '<th ALIGN="center" rowspan="2">Sample</th>'
          '<th ALIGN="center" rowspan="2">Reference</th>'
          '<th ALIGN="right" rowspan="2">AQ7 Reads</th>'
          '<th ALIGN="center" colspan="2">Required depth for</th>'
          '<th ALIGN="right" rowspan="2">AQ17 bases</th>'
          '<th ALIGN="right" rowspan="2">SNR</th>'
        '</tr>'
        '<tr>'
          '<th ALIGN="right">99% bases >1x</th>'
          '<th ALIGN="right">98% bases >10x</th>'
        '</tr>'
      '</thead>'
      '<tbody>\n')

    cycle = 0
    backgroundColor = []
    backgroundColor.append('inherit')
    backgroundColor.append('#AAA')
    for recordReport, q7reads, q17bases, pluginDict, snr, webRoot in summary.recordReportList:
        html.write('<tr style="background-color:%s;">' % (backgroundColor[cycle%2]))
        html.write('<td><a href="%s">%s</a></td>' % (webRoot + recordReport.reportLink, recordReport.resultsName))
        html.write('<td ALIGN="center">%s</td>' % recordReport.experiment.chipType)
        html.write('<td ALIGN="center">%s</td>' % recordReport.experiment.project)
        html.write('<td ALIGN="center">%s</td>' % recordReport.experiment.sample)
        html.write('<td ALIGN="center">%s</td>' % recordReport.experiment.library)
        html.write('<td ALIGN="right">%s</td>' % ion_readable(q7reads))
        html.write('<td ALIGN="right" bgcolor=#88CC88>%s</td>' % pluginDict['coverage_needed_for_99_percentile_base_with_at_least_1x_coverage'])
        html.write('<td ALIGN="right" bgcolor=#88CC88>%s</td>' % pluginDict['coverage_needed_for_98_percentile_base_with_at_least_10x_coverage'])
        html.write('<td ALIGN="right">%s</td>' % ion_readable(q17bases))
        html.write('<td ALIGN="right">%s</td>' % snr)
        html.write('</tr>\n')

        html.write('<tr colspan="11"style="background-color:%s;">' % (backgroundColor[cycle%2]))
        html.write('<td colspan="11">Notes: %s</td>' % recordReport.experiment.notes)
        html.write('</tr>\n')

        cycle = cycle + 1

    html.write(\
        '</tbody>'
        '</table>\n')


    # Top 5 high qquality runs
    html.write(\
        '<br><br><div style="width:100%%; background-color:#DDD;'
              'margin-top:16px; margin-left:auto;'
              'margin-right:auto; padding:8px;">'
              '<p><h2><center>Top 5 high quality runs</center></h2></p></div>\n')

    html.write('<table style="width: 100%; border-collapse: collapse;">')
    html.write('<thead style="text-align:left;">')
    html.write(\
        '<tr>'
          '<th ALIGN="left">Name</th>'
          '<th ALIGN="center">Chip Type</th>'
          '<th ALIGN="center">Project</th>'
          '<th ALIGN="center">Sample</th>'
          '<th ALIGN="center">Reference</th>'
          '<th ALIGN="right">AQ7 Reads</th>'
          '<th ALIGN="center">Required depth for 98% bases >10x</th>'
          '<th ALIGN="right">AQ17 bases</th>'
          '<th ALIGN="right">SNR</th>'
        '</tr>'
      '</thead>'
      '<tbody>\n')

    cycle = 0
    for recordReport, recordVal, q7reads, q17bases, pluginDict, snr, webRoot in top5_m2.recordReportList:
        html.write('<tr style="background-color:%s;">' % (backgroundColor[cycle%2]))
        html.write('<td><a href="%s">%s</a></td>' % (webRoot + recordReport.reportLink, recordReport.resultsName))
        html.write('<td ALIGN="center">%s</td>' % recordReport.experiment.chipType)
        html.write('<td ALIGN="center">%s</td>' % recordReport.experiment.project)
        html.write('<td ALIGN="center">%s</td>' % recordReport.experiment.sample)
        html.write('<td ALIGN="center">%s</td>' % recordReport.experiment.library)
        html.write('<td ALIGN="right">%s</td>' % ion_readable(q7reads))
        html.write('<td ALIGN="right" bgcolor=#88CC88>%s</td>' % recordVal)
        html.write('<td ALIGN="right">%s</td>' % ion_readable(q17bases))
        html.write('<td ALIGN="right">%s</td>' % snr)
        html.write('</tr>\n')

        html.write('<tr colspan="11"style="background-color:%s;">' % (backgroundColor[cycle%2]))
        html.write('<td colspan="11">Notes: %s</td>' % recordReport.experiment.notes)
        html.write('</tr>\n')

        cycle = cycle + 1

    html.write(\
        '</tbody>'
        '</table>\n')

    # close up shop
    html.write('</body></html>\n')
    html.close()

    print 'Wrote %d entries' % summary.numInList

    #print 'Generating and sending email...'
    #htmlFile = open("/results/custom_reports/ampl-" + str(datetime.date.today())+ ".html",'r')
    #htmlText = htmlFile.read()
    #send_html(SENDER, RECIPS, subject, htmlText)
    #htmlFile.close()

    print 'Finished.'

