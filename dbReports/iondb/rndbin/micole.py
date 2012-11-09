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

#format the numbers is a more readable way 
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    locale.setlocale(locale.LC_ALL, '')

def micole(db, daterange, timeStart, timeEnd):
    """get exps with that look that they are for project micole"""
    #results = models.Experiment.objects.using(db).filter(library='e_coli_dh10b')
    #results = models.Results.objects.using(db).filter(projects__name__istartswith='mi')
    #results = models.Experiment.objects.using(db).filter(library='hg19')
    #results = models.Results.objects.using(db).filter(projects__name__iexact='micol') | results = models.Results.objects.using(db).filter(projects__name__iexact='micole') | results = models.Results.objects.using(db).filter(projects__name__iexact='mikol')
    results = models.Experiment.objects.using(db).filter(library='hg19')
    results = results.exclude(expName__contains='exome')
    if daterange:
        results = results.filter(date__range=(timeStart,timeEnd))

    return (db,results)

def data_find(big_list):
    """
    give this a big_list and it will return the sum of the 100bpaq17 values
    and the aq17 sum
    """

    #100bp q17
    i100bpAQ17 = 0
    i100bpLIST = []

    #aq17
    iAQ17 = 0

    #now we will iterate through all of the reports returned to us by micole.
    #while going through the loop we will pull out the values we are interested in.

    count = 0
    for place in big_list:
        for exp in place[1]:
            count = count + 1
            #get the best report
            r = exp.best_aq17()

            try:
                r = r[0]
                if not r.libmetrics_set.all()[0].align_sample:
                    iAQ17 = iAQ17 + r.libmetrics_set.all()[0].q17_mapped_bases
                    i100bpAQ17 = i100bpAQ17 + r.libmetrics_set.order_by('i100Q17_reads')[0].i100Q17_reads
                    i100bpLIST.append( (place[0], r ,r.libmetrics_set.all()[0].i100Q17_reads) )
                else:
                    iAQ17 = iAQ17 + r.libmetrics_set.all()[0].extrapolated_mapped_bases_in_q17_alignments
                    i100bpAQ17 = i100bpAQ17 + r.libmetrics_set.all()[0].extrapolated_100q17_reads
                    i100bpLIST.append( (place[0], r, r.libmetrics_set.all()[0].extrapolated_100q17_reads) )
            except:
                pass
    
    results = []
    #sort on second value of the tuple which is the 100bpq17 value
    top5 =  sorted(i100bpLIST, key=lambda bp: bp[2])[-5:]
    for top in reversed(top5):
        one = "100AQ17 reads: "
        two = locale.format('%d', top[2], True)
        three = " -- From Report:" + str(top[1])
        tu = (one,two,three)
        results.append(tu)

    return i100bpAQ17, iAQ17, count, results

if __name__ == '__main__':

    #get the start and end data (thanks Mel)
    today = datetime.date.today()
    timeStart = datetime.datetime(today.year, today.month, today.day)
    daysFromMonday = timeStart.weekday() 
    lengthOfReport = 7
    if daysFromMonday < lengthOfReport: 
        daysFromMonday = daysFromMonday + 7
    timeStart = timeStart - datetime.timedelta(days=daysFromMonday)
    timeEnd = timeStart + datetime.timedelta(days=lengthOfReport)
    timeEnd = timeEnd - datetime.timedelta(seconds=1)

    html = open("top-" + str(datetime.date.today())+ ".html",'w')

    html.write("<html><head><title>top runs</title></head><body>")
    html.write('<h1>Experiments between %s and %s </h1>' % ( timeStart, timeEnd))

    #build the list of all micole exps
    beverly = micole("beverly", True, timeStart,timeEnd)
    west = micole("ionwest", True, timeStart,timeEnd)
    east = micole("ioneast", True, timeStart, timeEnd)
    big_list = [beverly,west,east]
    
    html.write("</br> <h2>Summary Micole Report for site(s)")
    for i, site in enumerate(big_list):
        html.write(site[0])
        if i+1 != len(big_list): html.write(",")
    html.write("</h2>")

    #get the data for all the sites in big_list
    i100bpAQ17, iAQ17, count, results = data_find(big_list)

    #print top results
    html.write("<h3>Top 5 Reports</h3>")
    html.write("<ol>")
    for result in results:
        html.write("<li>")
        html.write(result[0])
        html.write(result[1])
        html.write(result[2])
        html.write("</li>")
    html.write("</ol>") 

    html.write("<h3>Totals for top 5 runs</h3><ul>")
    html.write("<li>Total 100AQ17 reads: " )
    html.write (locale.format('%d', i100bpAQ17, True))
    html.write("</li><li>Total AQ17 bases: ")
    html.write(locale.format('%d', iAQ17, True))
    html.write("</li></ul>")

    #build the list of all micole exps
    beverly = micole("beverly",False, timeStart, timeEnd)
    west = micole("ionwest",False, timeStart, timeEnd)
    east = micole("ioneast",False, timeStart, timeEnd)
    big_list = [beverly,west,east]

    html.write("</br><h2>Cumulative Data</h2>")

    i100bpAQ17, iAQ17, count, results = data_find(big_list)
    
    html.write ("<ul><li>Runs: " + str(count))
    html.write("</li><li>Total 100AQ17 reads: " )
    html.write (locale.format('%d', i100bpAQ17, True))
    html.write("</li><li>Total AQ17 bases: ")
    html.write(locale.format('%d', iAQ17, True))
    html.write("</li><ul>")

