#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Write out all the micole runs

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

    report_storage = models.ReportStorage.objects.using(db).all().order_by('id')[0]

    results = models.Experiment.objects.using(db).filter(project__istartswith='mi')
    if daterange:
        results = results.filter(date__range=(timeStart,timeEnd))

    return (db,results,report_storage)

def data_find(big_list):
    """
    print the report and its path
    """
    results = []
    csv = open("all-" + str(datetime.date.today())+ ".csv",'w')
    csv.write("reportname , path , site \n")
    for place in big_list:
        for exp in place[1]:
            for result in exp.results_set.all():
                #result_dir = place[2].dirPath + result.reportLink.replace( place[2].webServerPath, "")
                #the manual disk change causes problems
                result_dir = place[2].dirPath
                innerpath = result.reportLink.replace( "output/", "")
                #east
                innerpath = innerpath.replace( "outputB/", "")
                #west
                innerpath = innerpath.replace( "http://ionwest.iontorrent.com/IonWest", "/IonWest")
                innerpath = "/".join(innerpath.split("/")[:-1])
                csv.write ('"' + result.resultsName + '",' + '"' + result_dir + innerpath + '"' + ',"' + place[0] + '",' + '"' + str(result.timeStamp) + '"\n')
                results.append( (result.resultsName ,  result_dir + innerpath , place[0], str(result.timeStamp) ) )

    csv.close()
    return results


if __name__ == '__main__':

    #get the start and end data (thanks Mel)
    today = datetime.date.today()
    timeStart = datetime.datetime(today.year, today.month, today.day)
    daysFromMonday = timeStart.weekday() 
    lengthOfReport = 14 
    if daysFromMonday < lengthOfReport: 
        daysFromMonday = daysFromMonday + 7
    timeStart = timeStart - datetime.timedelta(days=daysFromMonday)
    timeEnd = timeStart + datetime.timedelta(days=lengthOfReport)
    timeEnd = timeEnd - datetime.timedelta(seconds=1)
    
    #build the list of all micole exps
    time_range = False
    beverly = micole("beverly", time_range , timeStart,timeEnd)
    west = micole("ionwest", time_range, timeStart,timeEnd)
    east = micole("ioneast", time_range, timeStart, timeEnd)
    big_list = [beverly,west,east]
    
    results = data_find(big_list)

    #print to html results
    html = open("all-" + str(datetime.date.today())+ ".html",'w')
    html.write("<html><head><title>" + str(datetime.date.today()) + "</title></head> <body>")
    html.write("<table border=2><tr><th>result name</th> <th> path </th> <th> server </th><th>time stamp</th>")
    
     
    for result in results:
        html.write("<tr>")
        html.write("<td>" + result[0] + "</td>")
        html.write("<td>" + result[1] + "</td>")
        html.write("<td>" + result[2] + "</td>")
        html.write("<td>" + result[3] + "</td>")
        html.write("</tr>\n")

    html.write("</table>")

    html.write('</br> <a href="all-' + str(datetime.date.today()) + '.csv">csv file</a>'  )

    html.write("</body></html>")

    html.close()

