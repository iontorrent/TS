#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
'''Print a list of Runs marked either Keep or Archive.'''

import sys
import os
sys.path.append("/opt/ion/iondb")
from iondb.bin import djangoinit
from iondb.rundb import models
from django.db.models import Q

def main():
    '''main function'''
    
    #dbase query for list of Raw Data Experiments marked Keep or Archive
    query = Q(storage_options="KI") | Q(storage_options="A")
    runs = models.Experiment.objects.filter(query).order_by('date')

    #------------------
    # Prints to stdout
    #------------------
    for i, run in enumerate(runs, start=1):
        print ("(%d) RUN: %s" % (i, run.expName))
        print ("DATE: %s" % run.date.strftime("%B %d, %Y"))
        print ("STORAGE: %s" % ("Keep" if run.storage_options == "KI" else "Archive"))
        print ("SIZE: %s mb" % run.diskusage)
        print ("NOTES: %s" % run.notes)
        try:
            projects = run.results_set.values_list(
                'projects__name', flat=True).distinct()
            projects = [project for project in projects if project is not None]
            print ("PROJECTS: %s" % ",".join(projects))
        except:
            print ("PROJECTS: %s" % projects)
        print ("DIRECTORY: %s" % os.path.dirname(run.expDir))
        print ("-----")

    #--------------------
    # Writes to csv file
    #--------------------
    with open('runlist.csv', 'w') as fout:
        print ("Writing output file: runlist.csv")
        fout.write("DATE,NAME,STORAGE,SIZE,NOTE,PROJECT,DIRECTORY\n")
        for i, run in enumerate(runs, start=1):

            projects = run.results_set.values_list(
                'projects__name', flat=True).distinct()
            projects = [project for project in projects if project is not None]

            fout.write("%s,%s,%s,%s,%s,%s,%s\n" %
                    (run.date.strftime("%Y-%m-%d"),
                     run.expName,
                     "Keep" if run.storage_options == "KI" else "Archive",
                     run.diskusage,
                     run.notes.replace(',', ' '),
                     ";".join(projects),
                     os.path.dirname(run.expDir)))

    #-----------------
    # Writes xls file
    #-----------------
    try:
        import xlwt
    except ImportError:
        print
        print "To write an Excel spreadsheet file, install python-xlwt:"
        print "  sudo apt-get install python-xlwt"
        print
    else:
        book = xlwt.Workbook(encoding="utf-8")
        sheet1 = book.add_sheet("%s" % 'Runlist')
        sheet1.write(0, 0, "DATE")
        sheet1.write(0, 1, "NAME")
        sheet1.write(0, 2, "STORAGE")
        sheet1.write(0, 3, "SIZE")
        sheet1.write(0, 4, "NOTE")
        sheet1.write(0, 5, "PROJECT")
        sheet1.write(0, 6, "DIRECTORY")
    
        for i, run in enumerate(runs, start=1):
    
            projects = run.results_set.values_list(
                'projects__name', flat=True).distinct()
            projects = [project for project in projects if project is not None]
    
            sheet1.write(i, 0, (run.date.strftime("%Y-%m-%d")))
            sheet1.write(i, 1, run.expName)
            sheet1.write(
                i, 2, "Keep" if run.storage_options == "KI" else "Archive")
            sheet1.write(i, 3, run.diskusage)
            sheet1.write(i, 4, run.notes.replace(',', ' '))
            sheet1.write(i, 5, ";".join(projects))
            sheet1.write(i, 6, os.path.dirname(run.expDir))
        print ("Writing output file: runlist.xls")
        book.save("runlist.xls")

if __name__ == '__main__':
    sys.exit(main())
