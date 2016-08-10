#!/usr/bin/env python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
import os
import sys
import time


def file_exists(filepath, block=30, delay=1):
    """
    Returns True when file exists.  block is time in seconds to wait for the file.
    delay is time interval in seconds between checking for file
    """
    try:
        got_time = time.time() + block
        finished = False
        while not finished:
            if os.path.exists(filepath):
                return True
            else:
                finished = time.time() > got_time
                time.sleep(delay)
        return False
    except:
        raise

if __name__ == '__main__':
    sys.exit(file_exists(sys.argv[1]))
