#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
'''Print a list of Runs filtered on storage designation'''

import sys
import os
sys.path.append("/opt/ion/iondb")
from iondb.bin import djangoinit
from iondb.rundb import models
from django.db.models import Q

# Convert list of lists into dictionary
storage = {}
for item in models.Experiment.STORAGE_CHOICES:
    storage[item[0]] = item[1]

def show_keepers():
    # All SigProc Stat objects
    sigprocstats = models.DMFileStat.objects.filter(dmfileset__type="Signal Processing Input").order_by('created')
    print "All SigProc objects count: %d" % sigprocstats.count()

    # Limit to objects marked keep
    sigprocstats = sigprocstats.filter(result__experiment__storage_options = 'KI')
    print "All SigProc objects Local marked Keep count: %d" % (sigprocstats.count())

    # Limit to objects with files on filesystem
    sigprocstats = sigprocstats.filter(action_state='L')
    print "All SigProc objects Local count: %d" % sigprocstats.count()

    #--------------------
    # Writes to csv file
    #--------------------
    with open('runlist.csv', 'w') as fout:
        print ("Writing output file: runlist.csv")
        fout.write("DATE,NAME,STORAGE,SIZE,NOTE,PROJECT,DIRECTORY\n")

        dupes_list = []

        for stat in sigprocstats:

            run = stat.result.experiment

            # Only need single instance of a Run Name in the output
            if run.expName in dupes_list:
                continue
            else:
                dupes_list.append(run.expName)

            projects = run.results_set.values_list(
                    'projects__name', flat=True).distinct()
            projects = [project for project in projects if project is not None]

            fout.write("%s,%s,%s,%s,%s,%s,%s\n" %
                       (run.date.strftime("%Y-%m-%d"),
                        run.expName,
                        'Keep',
                        run.diskusage,
                        run.notes.replace(',', ' ') if run.notes != None else '',
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

        dupes_list = []

        num_items = 0
        for stat in sigprocstats:

            run = stat.result.experiment

            # Only need single instance of a Run Name in the output
            if run.expName in dupes_list:
                continue
            else:
                num_items += 1
                dupes_list.append(run.expName)
                projects = run.results_set.values_list(
                    'projects__name', flat=True).distinct()
                projects = [project for project in projects if project is not None]

                sheet1.write(num_items, 0, (run.date.strftime("%Y-%m-%d")))
                sheet1.write(num_items, 1, run.expName)
                sheet1.write(num_items, 2, storage[run.storage_options])
                sheet1.write(num_items, 3, run.diskusage)
                sheet1.write(num_items, 4, run.notes.replace(',', ' ') if run.notes != None else '')
                sheet1.write(num_items, 5, ";".join(projects))
                sheet1.write(num_items, 6, os.path.dirname(run.expDir))
        print ("Writing output file: runlist.xls")
        book.save("runlist.xls")

if __name__ == '__main__':
    sys.exit(show_keepers())
