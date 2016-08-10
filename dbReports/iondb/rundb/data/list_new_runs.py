#!/usr/bin/env python
# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
'''
 * Counts the number of new datasets created per day for the given number of days.
 * Uses the number of Experiment objects created in each time period.
 * Note: Experiment object's date field gets initialized to the time the object is updated
 * in the database.  Typically 1-2 minutes after start time on the sequencer.
 * Displays the approximate disk space used per day for the new datasets.
'''
import sys
import json
from iondb.bin import djangoinit
from iondb.rundb import models
from datetime import timedelta
from django.utils import timezone
from iondb.rundb.data.dmactions_types import FILESET_TYPES
DETAILS = False


class RunLister(object):

    def __init__(self, days=7):
        self.mydatadict = {}
        self.day_limit = days
        self._reset_data()
        # Query the database and populate the data dictionary
        self.get_data()

    def _reset_data(self):
        self.mydatadict = {
            'num_days': 0,
            'day_data': [],
            'age_threshold': {},
        }
        self._get_thresholds()

    def set_day_limit(self, _days):
        if isinstance(_days, str):
            if _days.lower() in ['max', 'all']:
                # Find oldest Experiment and set day_limit to number of days old.
                query_set = models.Experiment.objects.filter(status='run').order_by('date')
                self.day_limit = (timezone.now().replace(
                    hour=23, minute=59, second=59, microsecond=0) - query_set[0].date).days
                # Add 1 day to include the actual oldest day
                self.day_limit += 1
        else:
            self.day_limit = _days

    def get_data(self):
        '''Organized per-day.  Names of Experiments added on a given day.'''
        self._reset_data()
        midnight_tonight = timezone.now().replace(hour=23, minute=59, second=59, microsecond=0)
        for i in range(0, self.day_limit, 1):
            # Normalize to midnight of the current day to get 'per day' numbers
            query_set = models.Experiment.objects.filter(status='run',
                                                         date__lt=midnight_tonight - timedelta(days=i),
                                                         date__gt=midnight_tonight - timedelta(days=(i + 1)))
            thisday = []
            tot_disk_usage = 0
            for run in query_set.order_by('-date'):
                # Get the name of Reports associated with the Experiment
                report_set = run.sorted_results_with_reports()
                report_names = [report.resultsName for report in report_set]
                thisday.append({
                    'name': run.displayName,
                    'timestamp': str(timezone.localtime(run.date)),
                    'report_names': report_names,
                })
                # Get the total disk space used by this Experiment and Reports
                # This is complicated.  But there are some rules we can agree on to get a good approximation
                # without over-estimating disk usage per day.  We want a number as close to truth without
                # going over.
                # 1) Signal Processing Input. Raw data. Each Experiment object provides disk usage
                # for the raw data. So, use a single dmfilestat object that points to a Report that
                # uses the Experiment.
                # 2) Basecalling Input. At least one Report will have the actual files. Any
                # re-analysis will be symbolic links to original report - on Proton and S5. PGM can
                # either be symbolic link if its a "from-wells" analysis or it will be actual files
                # if its "from-raw". Thumbnail analysis also can either be "from-wells" or
                # "from-raw".
                # 3) Output Files.  Easy.  Count every instance, add them up.
                # 4) Intermediate Files.  Easy.  Count every instance, add them up.
                #
                # 1) Signal Processing Input
                if report_set:
                    dmfilestat = report_set[0].get_filestat("Signal Processing Input")
                    tot_disk_usage += dmfilestat.diskspace if dmfilestat.diskspace is not None else 0
                # 2) Basecalling Input
                # Naive code: just take the first Report.
                if report_set:
                    dmfilestat = report_set[0].get_filestat("Basecalling Input")
                    tot_disk_usage += dmfilestat.diskspace if dmfilestat.diskspace is not None else 0
                # 3) Output Files
                disk_usage = 0
                for report in report_set:
                    dmfilestat = report.get_filestat("Output Files")
                    disk_usage += dmfilestat.diskspace if dmfilestat.diskspace is not None else 0
                tot_disk_usage += disk_usage
                # 4) Intermediate Files
                disk_usage = 0
                for report in report_set:
                    dmfilestat = report.get_filestat("Intermediate Files")
                    disk_usage += dmfilestat.diskspace if dmfilestat.diskspace is not None else 0
                tot_disk_usage += disk_usage

            self.mydatadict['num_days'] = (i)
            self.mydatadict['day_data'].append(
                {
                    'label': '%2d Days Ago' % (i) if i > 0 else '      Today',
                    'date': str((midnight_tonight - timedelta(days=i)).strftime("%x %a")),
                    'num_runs': query_set.count(),
                    'runs': thisday,
                    'disk_usage': tot_disk_usage,
                }
            )

    def get_json(self):
        '''Returns serialized data'''
        return json.dumps(self.mydatadict, sort_keys=False, indent=2, separators=(',', ': '))

    def show_runs(self):
        '''Prints to stdout'''
        for thisday in self.mydatadict['day_data']:
            sys.stdout.write("%s: %2d  %9d MB   %s\n" %
                             (thisday['label'], thisday['num_runs'], int(thisday['disk_usage']), thisday['date']))
            if DETAILS:
                for run in thisday['runs']:
                    sys.stdout.write("  %s %s\n" % (run['timestamp'], run['name']))
                    for report in run['report_names']:
                        sys.stdout.write("    %s\n" % report)

    def _get_thresholds(self):
        'Query database for age thresholds of the 4 dataset categories'
        # We cheat here by assuming the most recent four dmfileset objects are the current ones
        # which will have the correct age thresholds.
        dmfilesets = models.DMFileSet.objects.all().order_by('-id')
        for dmtype in FILESET_TYPES:
            self.mydatadict['age_threshold'][dmtype] = [
                dm.auto_trigger_age for dm in dmfilesets[:4] if dm.type == dmtype][0]

if __name__ == '__main__':
    ListerMaker = RunLister()
# ListerMaker.set_day_limit('max')    # 'max' will return list for all time.  Use an integer to set day limit.
    ListerMaker.set_day_limit(31)
    ListerMaker.get_data()
    ListerMaker.show_runs()
#    print ListerMaker.get_json()
