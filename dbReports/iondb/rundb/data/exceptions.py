#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

class FilePermission(Exception):
    def __init__(self,_msg):
        self.tag = "file_permission_fail"
        self.message = str(_msg)
    def __str__(self):
        return repr(self.message)

class InsufficientDiskSpace(Exception):
    def __init__(self,_msg):
        self.tag = "disk_space_fail"
        self.message = str(_msg)
    def __str__(self):
        return repr(self.message)

class MediaNotSet(Exception):
    def __init__(self,_msg):
        self.tag = "media_undefined"
        self.message = str(_msg)
    def __str__(self):
        return repr(self.message)

class MediaNotAvailable(Exception):
    def __init__(self,_msg):
        self.tag = "media_fail"
        self.message = str(_msg)
    def __str__(self):
        return repr(self.message)

class FilesInUse(Exception):
    def __init__(self,_msg):
        self.tag = "files_in_use_fail"
        self.message = str(_msg)
    def __str__(self):
        return repr(self.message)

class FilesMarkedKeep(Exception):
    def __init__(self,_msg):
        self.tag = "files_marked_keep"
        self.message = str(_msg)
    def __str__(self):
        return repr(self.message)

class NoDMFileStat(Exception):
    def __init__(self,_msg):
        self.tag = "no_dmfilestat_objects_available"
        self.message = str(_msg)
    def __str__(self):
        return repr(self.message)

class SrcDirDoesNotExist(Exception):
    def __init__(self,_msg):
        self.tag = "does_not_exist_%s" % str(_msg)
        self.message = str(_msg)
    def __str__(self):
        return repr(self.message)
