# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
#!/usr/bin/env python
import os
import statvfs
import zipfile
import os.path
import subprocess

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
    """Return free space in kilobytes"""
    s = os.statvfs(drive_path)
    freebytes = s[statvfs.F_BSIZE] * s[statvfs.F_BAVAIL]
    return float(freebytes)/1024


def getSpaceMB(drive_path):
    """Return free space in megabytes"""
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


def get_common_prefix(files):
    """For a list of files, a common path prefix and a list file names with
    the prefix removed.

    Returns a tuple (prefix, relative_files):
        prefix: Longest common path to all files in the input. If input is a
                single file, contains full file directory.  Empty string is
                returned of there's no common prefix.
        relative_files: String containing the relative paths of files, skipping
                        the common prefix.
    """
    # Handle empty input
    if not files or not any(files):
        return '', []
    # find the common prefix in the directory names.
    directories = [os.path.dirname(f) for f in files]
    prefix = os.path.commonprefix(directories)
    start = len(prefix)
    if all(f[start] == "/" for f in files):
        start += 1
    relative_files = [f[start:] for f in files]
    return prefix, relative_files


def valid_files(files):
    black_list = [lambda f: "__MACOSX" in f]
    absolute_paths = [os.path.isabs(d) for d in files]
    if any(absolute_paths) and not all(absolute_paths):
        raise ValueError("Archive contains a mix of absolute and relative paths.")
    return [f for f in files if not any(reject(f) for reject in black_list)]


def make_relative_directories(root, files):
    directories = [os.path.dirname(f) for f in files]
    for directory in directories:
        path = os.path.join(root, directory)
        if not os.path.exists(path):
            os.makedirs(path)


def unzip_archive(root, data):
    zip_file = zipfile.ZipFile(data, 'r')
    namelist = zip_file.namelist()
    namelist = valid_files(namelist)
    prefix, files = get_common_prefix(namelist)
    make_relative_directories(root, files)
    out_names = [(n, f) for n, f in zip(namelist, files) if
                                                    os.path.basename(f) != '']
    for key, out_name in out_names:
        if os.path.basename(out_name) != "":
            full_path = os.path.join(root, out_name)
            contents = zip_file.open(key, 'r')
            try:
                output_file = open(full_path, 'wb')
                output_file.write(contents.read())
                output_file.close()
            except IOError as err:
                print("For zip's '%s', could not open '%s'" % (key, full_path))
    return [f for n, f in out_names]


def ismountpoint(directory):
    # Call shell command to run mountpoint tool
    cmd = ['/bin/mountpoint',directory]
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p1.communicate()
    return p1.returncode
