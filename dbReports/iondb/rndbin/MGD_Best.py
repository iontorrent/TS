#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

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

def best_getByChip(db, project_name, chipType, daterange, timeStart, timeEnd):
    results = models.Experiment.objects.using(db).filter(chipType__icontains=chipType)
    print 'Found %s %s runs at %s' % (len(results), chipType, db)

    report_storage = models.ReportStorage.objects.using(db).all().order_by('id')[0]

    gc = models.GlobalConfig.objects.using(db).all()[0]
    web_root = gc.web_root
    if len(web_root) > 0:
        if web_root[-1] == '/':
            web_root = web_root[:len(web_root)-1]

    if daterange:
        results = results.filter(date__range=(timeStart,timeEnd))

    return (db,results,report_storage,web_root)

def best_getByProject(db, project_name, chipType, daterange, timeStart, timeEnd):
    results = models.Experiment.objects.using(db).filter(project__istartswith=project_name).filter(chipType__icontains=chipType)
    print 'Found %s %s %s runs at %s' % (len(results), chipType, project_name, db)

    report_storage = models.ReportStorage.objects.using(db).all().order_by('id')[0]

    gc = models.GlobalConfig.objects.using(db).all()[0]
    web_root = gc.web_root
    if len(web_root) > 0:
        if web_root[-1] == '/':
            web_root = web_root[:len(web_root)-1]

    if daterange:
        results = results.filter(date__range=(timeStart,timeEnd))

    return (db,results,report_storage,web_root)

def dump(project_name, place):
    # res is an array of results (best report for each experiment)
    res = []
    res200Q17 = []
    for exp in place[1]:
        # get the best report (some experiments get re-analyzed so there could be multiple reports)
        rep = exp.results_set.all()
        bestNumReads = 0
        bestNumReads200Q17 = 0
        for r in rep:
            try:
                libmetrics = r.libmetrics_set.all()[0] # ok, there's really only one libmetrics set per result, but we still need to get at it
            except IndexError:
                continue

#            if libmetrics.align_sample == 1:
#                numReads = libmetrics.extrapolated_100q17_reads
#                numQ17Bases = libmetrics.extrapolated_mapped_bases_in_q17_alignments
#            else:
            if libmetrics.align_sample != 1:
                numReads = libmetrics.i100Q17_reads
                numReads200Q17 = libmetrics.i200Q17_reads

                if numReads > bestNumReads:
                    bestNumReads = numReads
                    bestResult = r
                    bestNumQ17Bases = libmetrics.q17_mapped_bases
                if numReads200Q17 > bestNumReads200Q17:
                    bestNumReads200Q17 = numReads200Q17
                    bestResult200Q17 = r
                    bestNumQ17Bases200Q17 = libmetrics.q17_mapped_bases

        if bestNumReads != 0: #best report per experiment
            res.append((bestNumReads,bestNumQ17Bases,bestResult.timeStamp, place[3]+bestResult.reportLink)) #TUPLE! - important for sorting
        if bestNumReads200Q17 != 0: #best report per experiment
            res200Q17.append((bestNumReads200Q17,bestNumQ17Bases200Q17,bestResult200Q17.timeStamp, place[3]+bestResult200Q17.reportLink)) #TUPLE! - important for sorting

    return res,res200Q17

def print_results(metricTitle, big_list_of_tuples):
    big_list_of_tuples.sort(reverse=True)
    for top in big_list_of_tuples[:6]:
        print '%s: %s Q17 bases: %s Date: %s URL: %s' % (metricTitle, top[0], top[1], top[2], top[3])

if __name__ == '__main__':
    project_name = 'Best316Runs'

    # get the start and end date
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

    # Top5 314 runs - build the list of all project best results
    beverly,beverly200Q17 = dump(project_name, best_getByChip("beverly", project_name, '314', DoWeekly, timeStart,timeEnd))
    west,west200Q17 = dump(project_name, best_getByChip("ionwest", project_name, '314', DoWeekly, timeStart,timeEnd))
    east,east200Q17 = dump(project_name, best_getByChip("ioneast", project_name, '314', DoWeekly, timeStart, timeEnd))
    pbox,pbox200Q17 = dump(project_name, best_getByChip("pbox", project_name, '314', DoWeekly, timeStart, timeEnd))
    ioncarlsbad,ioncarlsbad200Q17 = dump(project_name, best_getByChip("ioncarlsbad", project_name, '314', DoWeekly, timeStart, timeEnd))
    big_list = beverly + west + east + pbox + ioncarlsbad
    print_results('100Q17 reads', big_list)
    big_list200Q17 = beverly200Q17 + west200Q17 + east200Q17 + pbox200Q17 + ioncarlsbad200Q17
    print_results('200Q17 reads', big_list200Q17)

    # Top5 316 runs - build the list of all project best results
    beverly,beverly200Q17 = dump(project_name, best_getByChip("beverly", project_name, '316', DoWeekly, timeStart,timeEnd))
    west,west200Q17 = dump(project_name, best_getByChip("ionwest", project_name, '316', DoWeekly, timeStart,timeEnd))
    east,east200Q17 = dump(project_name, best_getByChip("ioneast", project_name, '316', DoWeekly, timeStart, timeEnd))
    pbox,pbox200Q17 = dump(project_name, best_getByChip("pbox", project_name, '316', DoWeekly, timeStart, timeEnd))
    ioncarlsbad,ioncarlsbad200Q17 = dump(project_name, best_getByChip("ioncarlsbad", project_name, '316', DoWeekly, timeStart, timeEnd))
    big_list = beverly + west + east + pbox + ioncarlsbad
    print_results('100Q17 reads', big_list)
    big_list200Q17 = beverly200Q17 + west200Q17 + east200Q17 + pbox200Q17 + ioncarlsbad200Q17
    print_results('200Q17 reads', big_list200Q17)

    # Top5 318 runs - build the list of all project best results
    beverly,beverly200Q17 = dump(project_name, best_getByChip("beverly", project_name, '318', DoWeekly, timeStart,timeEnd))
    west,west200Q17 = dump(project_name, best_getByChip("ionwest", project_name, '318', DoWeekly, timeStart,timeEnd))
    east,east200Q17 = dump(project_name, best_getByChip("ioneast", project_name, '318', DoWeekly, timeStart, timeEnd))
    pbox,pbox200Q17 = dump(project_name, best_getByChip("pbox", project_name, '318', DoWeekly, timeStart, timeEnd))
    ioncarlsbad,ioncarlsbad200Q17 = dump(project_name, best_getByChip("ioncarlsbad", project_name, '318', DoWeekly, timeStart, timeEnd))
    big_list = beverly + west + east + pbox + ioncarlsbad
    print_results('100Q17 reads', big_list)
    big_list200Q17 = beverly200Q17 + west200Q17 + east200Q17 + pbox200Q17 + ioncarlsbad200Q17
    print_results('200Q17 reads', big_list200Q17)

