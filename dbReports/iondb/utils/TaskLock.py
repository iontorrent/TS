#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import uuid
from iondb.bin import djangoinit
from django.core.cache import get_cache

def test_cache(cache):
    # test to make sure the cache is accessible to store locks
    key = str(uuid.uuid4())
    cache.add(key, 'testing')
    if cache.get(key) is None:
        raise Exception('Unable to create TaskLock in cache')
    else:
        cache.delete(key)


class TaskLock(object):

    def __init__(self, lock_id, timeout=None):
        self.lock_id = lock_id
        try:
            self.cache = get_cache('file')
            test_cache(self.cache)
        except:
            self.cache = get_cache('default')
            test_cache(self.cache)

        if timeout:
            self.timeout = timeout
        else:
            self.timeout = 86400

    def lock(self):
        val = self.cache.add(self.lock_id, "init", self.timeout)
        return val

    def update(self, value):
        '''Creates, or updates this key'''
        val = self.cache.set(self.lock_id, value, self.timeout)
        return val

    def get(self):
        '''Show cache value'''
        val = self.cache.get(self.lock_id)
        return val

    def unlock(self):
        self.cache.delete(self.lock_id)

# Test main routine
if __name__ == '__main__':
    import time
    applock = TaskLock('Test.lock')
    if applock.lock():
        # Hint: try running 2nd program instance while this instance sleeps
        print "Obtained lock, sleeping 10 seconds"
        time.sleep(10)
        print "Unlocking"
        applock.unlock()
    else:
        print "Unable to obtain lock, exiting"

'''
To clear a specific lock manually:
start python shell
from iondb.bin import djangoinit
from django.core.cache import cache
cache.delete(<lock id>)

or direct on command line:
python -c "from iondb.bin import djangoinit; from django.core.cache import get_cache; cache=get_cache('file'); cache.delete(<lock id>)"

to delete everything from cache:
python -c "from iondb.bin import djangoinit; from django.core.cache import cache; cache.clear()"
'''
