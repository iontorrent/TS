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


def getdiskusage(directory):
    # Try an all-python solution here - in case the suprocess spawning is causing grief.  We could be opening
    # hundreds of instances of shells above.
    def dir_size (start):
        if not start or not os.path.exists(start):
            return 0

        file_walker = (
            os.path.join(root, f)
            for root, _, files in os.walk( start )
            for f in files
        )
        total = 0L
        for f in file_walker:
            if os.path.isdir(f):
                total += dir_size(f)
                continue
            if not os.path.isfile(f):
                continue
            try:
                total += os.lstat(f).st_size
            except OSError:
                pass
        return total
    # Returns size in MB
    return dir_size(directory)/(1024*1024)
