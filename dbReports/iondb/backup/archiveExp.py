#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
import os

# Definition of archiveExperiment object used by archive tool


class Experiment:
    def __init__(self,
                 _exp,
                 _name,
                 _date,
                 _star,
                 _storage_options,
                 _user_ack,
                 _dir,
                 _pk,
                 _rawdatastyle,
                 _diskusage):
        
        self.prettyname = _exp.pretty_print()
        self.name = _name
        self.date = _date
        self.star = _star
        self.store_opt = _storage_options
        self.user_ack = _user_ack
        self.dir = _dir
        self.pk = _pk
        self.rawdatastyle = _rawdatastyle
        self.diskusage = _diskusage

    def get_exp_path(self):
        return self.dir

    def is_starred(self):
        def to_bool(value):
            if value == 'True':
                return True
            else:
                return False
        return to_bool(self.star)

    def get_folder_size(self):
        dir_size = 0
        for (path, dirs, files) in os.walk(self.dir):
            for item in files:
                filename = os.path.join(path, item)
                dir_size += os.path.getsize(filename)
        return dir_size

    def get_exp_name(self):
        return self.name

    def get_pk(self):
        return self.pk

    def get_storage_option(self):
        return self.store_opt
