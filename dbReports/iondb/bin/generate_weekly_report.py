# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
import datetime
from os import path
import urllib, urllib2

import djangoinit
from django.db import models
from iondb.rundb import models

siteName = "collab"


def getResults(htmlFile):
    # calculate the date range to generate the report
    # the intent is to auto-generate a weekly report for the 'prior' week, no matter when the script is run
    # week ends Friday at midnight.  So for example, if run on a Friday, will generate the report for the prior week,
    # but run on a Sat will generate the current report for the week just ended.
    today = datetime.date.today()
    timeStart = datetime.datetime(today.year, today.month, today.day)
    daysFromMonday = (
        timeStart.weekday()
    )  # Monday is 0, so if its Thursday (3), we need to go back 3 days
    lengthOfReport = (
        7
    )  # report for 7 days, with Monday being the first day included in a report
    if (
        daysFromMonday < lengthOfReport
    ):  # we want to go back to the start of the full week, if we are in the middle of a week, need to go back to the start of last week
        daysFromMonday = daysFromMonday + 7
    timeStart = timeStart - datetime.timedelta(days=daysFromMonday)
    timeEnd = timeStart + datetime.timedelta(days=lengthOfReport)
    timeEnd = timeEnd - datetime.timedelta(seconds=1)

    # and now we have a date range to query on, grab all 'new' runs, sum their
    # 100AQ17 values, and track the best weekly 100Q17 run also
    exp = models.Experiment.objects.filter(date__range=(timeStart, timeEnd))
    htmlFile.write(
        "Found %s experiments between %s and %s\n<br>\n"
        % (len(exp), timeStart, timeEnd)
    )

    xml_string = ""
    xml_string = xml_string + "      <TimeStart>" + str(timeStart) + "</TimeStart>\n"
    xml_string = xml_string + "      <TimeEnd>" + str(timeEnd) + "</TimeEnd>\n"

    # get best result for each experiment, the 'best' is 100Q17 reads right now
    # we will build an array of the best results for each experiment and return that to the caller
    res = []
    for e in exp:
        rep = e.results_set.all()
        bestNumReads = 0
        bestResult = None
        for r in rep:
            try:
                libmetrics = r.libmetrics_set.all()[
                    0
                ]  # ok, there's really only one libmetrics set per result, but we still need to get at it
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

    return res, xml_string


def installAuth():
    password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(
        None, "http://updates.iontorrent.com/metrics", "metrics", "ionmetrics"
    )
    handler = urllib2.HTTPBasicAuthHandler(password_mgr)
    opener = urllib2.build_opener(handler)
    urllib2.install_opener(opener)


def sendXML(xml_string):
    installAuth()
    query_args = {"xml": xml_string}
    encoded_args = urllib.urlencode(query_args)
    url = "http://updates.iontorrent.com/metrics/recvxml.php"
    response = urllib2.urlopen(url, encoded_args).read()


def generateReport():
    # get the web root path for building an html link
    gc = models.GlobalConfig.get()
    web_root = gc.web_root
    if len(web_root) > 0:
        if web_root[-1] == "/":
            web_root = web_root[: len(web_root) - 1]

    if gc.site_name:
        siteName = gc.site_name
    if siteName.find("Set") > 0:
        siteName = "default"

    heading = "Weekly Report for %s\n" % siteName

    # generate the report name using today's date
    today = datetime.date.today()
    reportName = "Weekly_Summary_%s_%s_%s.html" % (today.month, today.day, today.year)
    location = models.Location.objects.all()[0]
    # Always use the last, latest ReportStorage location
    reportStorages = models.ReportStorage.objects.all()
    reportStorage = reportStorages.filter(default=True)[0]
    path = "/var/www%s/%s/reports" % (reportStorage.webServerPath, location.name)
    if not os.path.isdir(path):
        os.mkdir(path, 0o0775)
        os.chmod(path, 0o0775)
    report_path = os.path.join(path, reportName)
    htmlFile = open(report_path, "w")
    htmlFile.write("<html><body>\n")
    htmlFile.write(heading)
    htmlFile.write("<br>\n")

    # we generate an xml string so we can optionally send the results to a remote Ion monitor site
    xml_string = '<?xml version="1.0"?>\n<Metrics>\n'
    xml_string = xml_string + "   <Site>\n"
    xml_string = xml_string + "      <Name>" + siteName + "</Name>\n"

    # get all analysis results from last week
    res, local_xml_data = getResults(htmlFile)
    xml_string = xml_string + local_xml_data

    # calculate some metrics for the results
    sum_100Q17Reads = 0
    sum_Q17Bases = 0
    sum_Runs = 0
    sum_Runs_314 = 0
    sum_Runs_316 = 0
    sum_Runs_318 = 0

    bestRun = None
    curBestReads = 0
    curBestBases = 0

    for r in res:
        try:
            libmetrics = r.libmetrics_set.all()[
                0
            ]  # ok, there's really only one libmetrics set per result, but we still need to get at it
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

            if "314" in r.experiment.chipType:
                sum_Runs_314 = sum_Runs_314 + 1
            if "316" in r.experiment.chipType:
                sum_Runs_316 = sum_Runs_316 + 1
            if "318" in r.experiment.chipType:
                sum_Runs_318 = sum_Runs_318 + 1

            if reads > curBestReads:
                curBestReads = reads
                curBestBases = bases
                bestRun = r

    htmlFile.write(
        "Totals  100Q17 reads: %s 100Q17 bases: %s in %s reports  314/316/318: %s/%s/%s\n"
        % (
            sum_100Q17Reads,
            sum_Q17Bases,
            sum_Runs,
            sum_Runs_314,
            sum_Runs_316,
            sum_Runs_318,
        )
    )
    htmlFile.write("<br>\n")
    if bestRun:
        htmlFile.write(
            'Best run: <a href="%s">%s</a>\n'
            % (web_root + bestRun.reportLink, bestRun.reportLink)
        )  # need to handle bestRun = None case?
        htmlFile.write("<br>\n")
        htmlFile.write(
            "Best run 100Q17 reads: %s 100Q17 bases: %s\n"
            % (curBestReads, curBestBases)
        )
        htmlFile.write("<br>\n")
    else:
        htmlFile.write("There were no best runs for this report.\n")
        htmlFile.write("<br>\n")

    htmlFile.write("</body></html>\n")
    htmlFile.close()

    xml_string = (
        xml_string
        + "      <Total100AQ17reads>"
        + str(sum_100Q17Reads)
        + "</Total100AQ17reads>\n"
    )
    xml_string = (
        xml_string
        + "      <TotalAQ17bases>"
        + str(sum_Q17Bases)
        + "</TotalAQ17bases>\n"
    )
    xml_string = (
        xml_string + "      <TotalReports>" + str(sum_Runs) + "</TotalReports>\n"
    )
    xml_string = (
        xml_string
        + "      <TotalReports314>"
        + str(sum_Runs_314)
        + "</TotalReports314>\n"
    )
    xml_string = (
        xml_string
        + "      <TotalReports316>"
        + str(sum_Runs_316)
        + "</TotalReports316>\n"
    )
    xml_string = (
        xml_string
        + "      <TotalReports318>"
        + str(sum_Runs_318)
        + "</TotalReports318>\n"
    )
    if bestRun:
        xml_string = (
            xml_string
            + "      <Best100AQ17reads>"
            + str(curBestReads)
            + "</Best100AQ17reads>\n"
        )
        xml_string = (
            xml_string
            + "      <BestAQ17bases>"
            + str(curBestBases)
            + "</BestAQ17bases>\n"
        )
        xml_string = (
            xml_string
            + "      <BestRunURL>"
            + web_root
            + bestRun.reportLink
            + "</BestRunURL>\n"
        )
    xml_string = xml_string + "   </Site>\n</Metrics>\n"
    sendXML(xml_string)


if __name__ == "__main__":
    generateReport()
