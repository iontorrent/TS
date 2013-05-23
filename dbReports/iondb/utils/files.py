# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
#!/usr/bin/env python
import os
import statvfs

from ion.utils.timeout import timeout

# Times out after 60 seconds
#@timeout(60,None)
def disk_attributes(directory):
    '''returns disk attributes'''
    try:
        resDir = os.statvfs(directory)
    except:
        raise
    else:
        totalSpace = resDir.f_blocks
        availSpace = resDir.f_bavail
        freeSpace = resDir.f_bfree
        blocksize = resDir.f_bsize

    return (totalSpace, availSpace, freeSpace, blocksize)


def percent_full(directory):
    '''returns percentage of disk in-use'''
    try:
        totalSpace, availSpace, f, b = disk_attributes(directory)
    except:
        raise
    else:
        if not (totalSpace > 0):
            return (name, 0)
        percent_full = 100-(float(availSpace)/float(totalSpace)*100)

    return percent_full


def test_sigproc_infinite_regression(directory):
    '''
    When pre-3.0 Reports are re-analyzed, a symbolic link is created in the
    report directory named 'sigproc_results' which points to it's parent directory.
    When copying the Report directory, and following this link, ends up in an
    infinite regression.
    We detect this situation and delete the link file.
    '''
    testfile = os.path.join(directory,'sigproc_results')
    if os.path.islink(testfile):
        if os.path.samefile(directory, testfile):
            os.unlink(testfile)
    return


def getSpaceKB(drive_path):
    s = os.statvfs(drive_path)
    freebytes = s[statvfs.F_BSIZE] * s[statvfs.F_BAVAIL]
    return float(freebytes)/1024


def getSpaceMB(drive_path):
    s = os.statvfs(drive_path)
    freebytes = s[statvfs.F_BSIZE] * s[statvfs.F_BAVAIL]
    return float(freebytes)/(1024*1024)


@timeout(30, None)
def getdeviceid(dirpath):
    return os.stat(dirpath)[2]