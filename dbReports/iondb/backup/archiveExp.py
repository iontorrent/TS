#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
import os

# Definition of archiveExperiment object used by archive tool
class Experiment:
    def __init__(self, exp, name, date, star, storage_options, user_ack, dir, pk, rawdatastyle):
        self.prettyname = exp.pretty_print()
        self.name = name
        self.date = date
        self.star = star
        self.store_opt = storage_options
        self.user_ack = user_ack
        self.dir = dir
        self.pk = pk
        self.rawdatastyle = rawdatastyle

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
        dir = self.dir
        dir_size = 0
        for (path, dirs, files) in os.walk(self.dir):
            for file in files:
                filename = os.path.join(path, file)
                dir_size += os.path.getsize(filename)
        return dir_size

    def get_exp_name(self):
        return self.name

    def get_pk(self):
        return self.pk

    def get_storage_option(self):
        return self.store_opt
    
