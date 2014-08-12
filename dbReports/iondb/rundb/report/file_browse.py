# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
#!/usr/bin/env python
from os.path import dirname
import os
import datetime
import errno
import logging
import time
import urllib
from operator import attrgetter
from cStringIO import StringIO


logger = logging.getLogger(__name__)

suffixes = ('k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')


def format_units(quantity, unit="B", base=1000):
    if quantity < base:
        return '%d  %s' % (quantity, unit)

    for i, suffix in enumerate(suffixes):
        magnitude = base ** (i + 2)
        if quantity < magnitude:
            return '%.1f %s%s' % ((base * quantity / float(magnitude)),
                suffix, unit)

    return '%.1f %s%s' % ((base * quantity / float(magnitude)), suffix, unit)


def dir_size(path):
    return len(os.listdir(path))


def list_directory(path):
    dirs, files = [], []
    for name in os.listdir(path):
        full = os.path.join(path, name)
        try :
            stats = os.stat(full)
        except OSError as e:
            if e.errno == errno.ENOENT and os.path.islink(full):
                stats = os.lstat(full)
                logger.error("Path %s is a broken symlink" % full)
            else:
                logger.exception("OS Error in metal UI")
                raise e
        info = [name, full, stats]
        if os.path.isdir(full):
            dirs.append(info)
        else:
            files.append(info)
    return dirs, files


def bread_crumb_path(path):
    path = path.strip('/')
    crumbs = []
    while path:
        head, tail = os.path.split(path)
        crumbs.insert(0, (path, tail))
        path = head
    return crumbs


def get_size(start_path='.'):
    """Return the total size of a folder's contents"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size


def ellipsize_file(fname, chunk_size=1000000):
    """If a file is more than 2.5 times chunk_size, output the first and last
    chunk_size number of bytes instead of the whole file
    """
    stats = os.stat(fname)
    size = stats.st_size
    if size < chunk_size * 2.5:
        handler = open(fname)
    else:
        handler = StringIO()
        with open(fname) as file:
            handler.write(file.read(chunk_size))
            handler.write("\n... File truncated to first and last %s ...\n\n" % format_units(chunk_size))
            file.seek(size-chunk_size)
            handler.write(file.read(chunk_size))
        handler.seek(0)
    return handler
