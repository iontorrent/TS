#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

# find_historical_best - dumps the best 50Q17/100Q17 run (date, #reads, link) recorded each week

import datetime
from iondb.bin.djangoinit import *
from iondb.rundb import models

# this function returns the best report of multiple reports per experiment
def getBestResultsRange(sortfield,startTime,ret_num):
    '''Returns the `ret_num` best analysis for all runs in date range'''
    timeEnd = datetime.datetime.now() - datetime.timedelta(days=startTime)
    timeStart = datetime.datetime.now() - datetime.timedelta(days=startTime+7)
    print 'start time: %s end time: %s' % (timeStart, timeEnd)
    exp = models.Experiment.objects.filter(date__range=(timeStart,timeEnd))
    # get best result for each experiment
    res = [e.best_result(sortfield) for e in exp if e.best_result(sortfield) is not None]
    # sort the best results of the best experiments and return the best one (or more, depends on ret_num)
    if ret_num is None:
        res = sorted(res,key=lambda r: getattr(r.best_lib_by_value(sortfield),sortfield),reverse=True)
    else:
        res = sorted(res,key=lambda r: getattr(r.best_lib_by_value(sortfield),sortfield),reverse=True)[:ret_num]
    return res


def listHistoricalBest50Q17():
    sortfield = 'i50Q17_reads'
    # go back one year, looking weekly
    for t in range(365, 0, -7):
        res = getBestResultsRange(sortfield, t, 1)
        for r in res:
            print '50Q17 reads: %s from %s' % (r.best_lib_metrics.i50Q17_reads, r.reportLink)

def listHistoricalBest100Q17():
    sortfield = 'i100Q17_reads'
    # go back one year, looking weekly
    for t in range(365, 0, -7):
        res = getBestResultsRange(sortfield, t, 1)
        for r in res:
            print '100Q17 reads: %s from %s' % (r.best_lib_metrics.i100Q17_reads, r.reportLink)


if __name__=="__main__":
    listHistoricalBest50Q17()
    listHistoricalBest100Q17()

