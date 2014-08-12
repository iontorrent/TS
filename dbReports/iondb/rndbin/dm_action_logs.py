#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.bin import djangoinit
from iondb.rundb.models import DMFileStat, Results, EventLog

class DMLogs:
    
    def get_all_reports(self):
        """Return list of all Report objects reverse chronological order."""
        reports = Results.objects.all().order_by('-timeStamp')
        return reports
    
    
    def get_log_entries(self, report):
        """Return all log entries for a given Report object."""
        something = EventLog.objects.filter(object_pk=report.id).order_by('-created')
        return something
    
    
    def main(self):
        for report in self.get_all_reports():
            print "Report: %s" % report.resultsName
            for log in self.get_log_entries(report):
                print "\t(%s)%s" % (log.created, log.text)


if __name__ == '__main__':
    foo = DMLogs()
    foo.main()