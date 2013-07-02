#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from iondb.bin import djangoinit
#from django.core.cache import get_cache
from django.core.cache import cache

class TaskLock:

    def __init__(self, lock_id):
        self.lock_id = lock_id
        #self.cache = get_cache('dm_action')
        self.cache = cache

    def lock(self):
        val = self.cache.add(self.lock_id, "fubar", 86400)
        return val

    def unlock(self):
        self.cache.delete(self.lock_id)

# Test main routine
if __name__ == '__main__':
    import time
    applock = TaskLock ('Test.lock')
    if (applock.lock ()):
        # Hint: try running 2nd program instance while this instance sleeps
        print ("Obtained lock, sleeping 10 seconds")
        time.sleep (10)
        print ("Unlocking")
        applock.unlock ()
    else:
        print ("Unable to obtain lock, exiting")