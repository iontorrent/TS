#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.bin import djangoinit
from iondb.rundb import models
from django.db.models import Q

def restore_in_process():
    query = Q(action_state="AG") | \
            Q(action_state="DG") | \
            Q(action_state="EG")

    dmfilestats = models.DMFileStat.objects.filter(query)
    print (" %6d Currently in-process" % dmfilestats.count())

    if dmfilestats.count() > 0:

        dmfilestats.update(action_state='L')
        print ("Changing action_state to Local")

        dmfilestats = models.DMFileStat.objects.filter(query)
        print (" %6d Currently in-process" % dmfilestats.count())


def restore_error():
    dmfilestats = models.DMFileStat.objects.filter(action_state='E')
    print (" %6d Currently in error" % dmfilestats.count())

    if dmfilestats.count() > 0:
        dmfilestats.update(action_state='L')
        print ("Changing action_state to Local")

        dmfilestats = models.DMFileStat.objects.filter(action_state='E')
        print (" %6d Currently in error" % dmfilestats.count())


def main():
    restore_in_process()
    restore_error()

if __name__ == "__main__":
    main()
