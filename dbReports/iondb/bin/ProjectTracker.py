#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Find the top micole reports

This script can extract data from multiple Torrent Servers
it requires Django 1.2 or greater.

All of the databases must be defined in settings.py similar to the following

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
        "NAME": "iondb",
        "USER": "ion",
        "PASSWORD": "ion",
        "HOST": "localhost"
    },
    "ioneast": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
        "NAME": "iondb",
        "USER": "ion",
        "PASSWORD": "ion",
        "HOST": "ioneast.ite"
    },
    "beverly": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
        "NAME": "iondb",
        "USER": "ion",
        "PASSWORD": "ion",
        "HOST": "aruba.bev"
    },
    "ionwest": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
        "NAME": "iondb",
        "USER": "ion",
        "PASSWORD": "ion",
        "HOST": "ionwest.itw"
    }
}

"""

from djangoinit import *

import datetime

from os import path
from iondb.rundb import models

import locale

# format the numbers is a more readable way
try:
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
except locale.Error:
    locale.setlocale(locale.LC_ALL, "")


def get_project2(db, project_name, daterange, timeStart, timeEnd):
    """get exps with that look that they are for project micole"""
    results = models.Experiment.objects.using(db).filter(library="hg19")
    results = (
        results.exclude(project__icontains="test")
        .exclude(project__icontains="library")
        .exclude(project__icontains="enrichment")
        .exclude(project__icontains="emulsionstability")
    )

    report_storage = models.ReportStorage.objects.using(db).all().order_by("id")[0]
    if daterange:
        results = results.filter(date__range=(timeStart, timeEnd))

    gc = models.GlobalConfig.objects.using(db).all()[0]
    web_root = gc.web_root
    if len(web_root) > 0:
        if web_root[-1] == "/":
            web_root = web_root[: len(web_root) - 1]

    return (db, results, report_storage, web_root)


def get_project(db, project_name, daterange, timeStart, timeEnd):
    """get exps with that look that they are for project micole"""
    # results = models.Experiment.objects.using(db).filter(project__iexact=project_name)
    results = models.Experiment.objects.using(db).filter(
        project__istartswith=project_name
    )
    report_storage = models.ReportStorage.objects.using(db).all().order_by("id")[0]
    if daterange:
        results = results.filter(date__range=(timeStart, timeEnd))

    gc = models.GlobalConfig.objects.using(db).all()[0]
    web_root = gc.web_root
    if len(web_root) > 0:
        if web_root[-1] == "/":
            web_root = web_root[: len(web_root) - 1]

    return (db, results, report_storage, web_root)


def data_find(project_name, big_list):
    """
    give this a big_list and it will return the sum of the 100bpaq17 values
    and the aq17 sum
    """

    # 100bp q17
    i100bpAQ17 = 0
    i100bpLIST = []

    # aq17
    iAQ17 = 0

    # now we will iterate through all of the reports returned to us by get_project.
    # while going through the loop we will pull out the values we are interested in.

    csv = open(project_name + "_All-" + str(datetime.date.today()) + ".csv", "w")
    csv.write("reportname , id , chip , path , site , timestamp\n")

    count = 0
    for place in big_list:
        for exp in place[1]:
            count = count + 1
            # get the best report
            r = exp.best_aq17()

            try:
                r = r[0]
                # if not r.libmetrics_set.all()[0].align_sample:
                if True:
                    iAQ17 = iAQ17 + r.libmetrics_set.all()[0].q17_mapped_bases
                    i100bpAQ17 = (
                        i100bpAQ17
                        + r.libmetrics_set.order_by("i100Q17_reads")[0].i100Q17_reads
                    )
                    i100bpLIST.append(
                        (place[0], r, r.libmetrics_set.all()[0].i100Q17_reads, place[3])
                    )
                    """
                else:
                    iAQ17 = iAQ17 + r.libmetrics_set.all()[0].extrapolated_mapped_bases_in_q17_alignments
                    i100bpAQ17 = i100bpAQ17 + r.libmetrics_set.all()[0].extrapolated_100q17_reads
                    i100bpLIST.append( (place[0], r, r.libmetrics_set.all()[0].extrapolated_100q17_reads) )
                    """

                    # write out our csv file entry for this result
                    result_dir = place[2].dirPath
                    innerpath = r.reportLink.replace("output/", "")
                    # east
                    innerpath = innerpath.replace("outputB/", "")
                    # west
                    innerpath = innerpath.replace(
                        "http://ionwest.iontorrent.com/IonWest", "/IonWest"
                    )
                    innerpath = "/".join(innerpath.split("/")[:-1])
                    runid = str(exp.pk)
                    chipType = exp.chipType
                    csv.write(
                        '"'
                        + r.resultsName
                        + '",'
                        + runid
                        + ","
                        + chipType
                        + ',"'
                        + result_dir
                        + innerpath
                        + '","'
                        + place[0]
                        + '","'
                        + str(r.timeStamp)
                        + '"\n'
                    )

            except Exception:
                pass
    csv.close()

    results = []
    # sort on second value of the tuple which is the 100bpq17 value
    top5 = sorted(i100bpLIST, key=lambda bp: bp[2])[-5:]
    top5_100AQ17Reads = 0
    top5_AQ17Bases = 0
    for top in reversed(top5):
        r = top[1]
        one = "100AQ17 reads: "
        two = locale.format("%d", top[2], True)
        three = " -- From Report: " + str(top[1])
        four = " -- URL: " + top[3] + r.reportLink
        tu = (one, two, three, four)
        results.append(tu)
        if not r.libmetrics_set.all()[0].align_sample:
            top5_AQ17Bases = top5_AQ17Bases + r.libmetrics_set.all()[0].q17_mapped_bases
            top5_100AQ17Reads = (
                top5_100AQ17Reads
                + r.libmetrics_set.order_by("i100Q17_reads")[0].i100Q17_reads
            )
        else:
            top5_AQ17Bases = (
                top5_AQ17Bases
                + r.libmetrics_set.all()[0].extrapolated_mapped_bases_in_q17_alignments
            )
            top5_100AQ17Reads = (
                top5_100AQ17Reads + r.libmetrics_set.all()[0].extrapolated_100q17_reads
            )

    return i100bpAQ17, iAQ17, count, results, top5_AQ17Bases, top5_100AQ17Reads


if __name__ == "__main__":

    # get the project to track
    try:
        project_name = sys.argv[1]
    except Exception:
        quit()

    # get the start and end data (thanks Mel)
    today = datetime.date.today()
    timeStart = datetime.datetime(today.year, today.month, today.day)
    daysFromMonday = timeStart.weekday()
    lengthOfReport = 7
    if daysFromMonday < lengthOfReport:
        daysFromMonday = daysFromMonday + 7
    timeStart = timeStart - datetime.timedelta(days=daysFromMonday)
    timeEnd = timeStart + datetime.timedelta(days=lengthOfReport)
    timeEnd = timeEnd - datetime.timedelta(seconds=1)
    DoWeekly = False

    html = open(project_name + "_Report-" + str(datetime.date.today()) + ".html", "w")

    html.write("<html><head><title>Micol Report</title></head><body>")
    if DoWeekly:
        html.write("<h1>Experiments between %s and %s </h1>" % (timeStart, timeEnd))

    # build the list of all project best results
    beverly = get_project2("beverly", project_name, DoWeekly, timeStart, timeEnd)
    west = get_project2("ionwest", project_name, DoWeekly, timeStart, timeEnd)
    east = get_project2("ioneast", project_name, DoWeekly, timeStart, timeEnd)
    big_list = [beverly, west, east]

    html.write("</br> <h2>Summary Micol Report for site(s)")
    for i, site in enumerate(big_list):
        html.write(site[0])
        if i + 1 != len(big_list):
            html.write(",")
    html.write("</h2>")

    # get the data for all the sites in big_list
    i100bpAQ17, iAQ17, count, results, top5_AQ17Bases, top5_100AQ17Reads = data_find(
        project_name, big_list
    )

    # print the summary totals for all runs
    html.write("<ul><li>Runs: " + str(count))
    html.write("</li><li>Total 100AQ17 reads: ")
    html.write(locale.format("%d", i100bpAQ17, True))
    html.write("</li><li>Total AQ17 bases: ")
    html.write(locale.format("%d", iAQ17, True))
    html.write("</li><ul>")

    # print top 5 results
    html.write("<h3>Top 5 Reports</h3>")
    html.write("<ol>")
    for result in results:
        html.write("<li>")
        html.write(result[0])
        html.write(result[1])
        # html.write(result[2])
        html.write(result[3])
        html.write("</li>")
    html.write("</ol>")

    html.write("<h3>Totals for top 5 runs</h3><ul>")
    html.write("<li>Total 100AQ17 reads: ")
    html.write(locale.format("%d", top5_100AQ17Reads, True))
    html.write("</li><li>Total AQ17 bases: ")
    html.write(locale.format("%d", top5_AQ17Bases, True))
    html.write("</li></ul>")
