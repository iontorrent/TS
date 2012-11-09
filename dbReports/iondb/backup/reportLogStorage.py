#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import os
import datetime
import string
from datetime import timedelta


def getReport(days):
    fileName = "/var/log/ion/reportsLog.log"
    file = open(fileName)
    f = file.readlines()
    str = []
    dates = []
    for line in f:
        if "Size saved:* " in line:
            dates.append(getDate(line[:10]))
            s = line[string.find(line, 'Size saved:* ')+13:]
            s = s[:string.find(s, '.')]
            if 'KB' in line:
                str.append(int(s))
            else:
                str.append(int(s)/1024)
    
    file.close()
    
    total = 0
    if days == 0:
        for st in str:
            total += st
    else:
        i = 0
        for dat in dates:
            t1 = timedelta(weeks=datetime.date.today().year*52+datetime.date.today().month, days=datetime.datetime.today().day)
            t2 = timedelta(weeks=dat.year*52+dat.month, days=dat.day)
            t3 = t1-t2
            if t3.days <= days:
                total += str[i]
            i += 1
    return total
            
def getDate(dateStr):
    year = int(dateStr[:4])
    str = dateStr[5:]
    month = int(str[:2])
    str = str[3:]
    day = int(str)
    return datetime.date(year, month, day)