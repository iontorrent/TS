#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
# First stage plugin driver script that uses user input to generate document.json with the help of django template
# This template takes user.json in the form of individual contents - table, images

__version__ = '1.0'

import os, sys, logging
from optparse import OptionParser
import json
from glob import glob
from django.template.loader import render_to_string
from django.conf import settings

import csv
import json
import random
import shutil

from distutils.sysconfig import get_python_lib
from ion.plugin.utils.Utility import Utility

class Validatehtml(Utility):
        """Check for proper json format in the html section"""
        def expandhtml(self,section):
            self.section = section
            self.section["id"] = self.generateID()
            if "title" not in self.section:
                 logging.error("title field for section type %s missing. Using default one" % self.section["type"])
                 self.section["title"] = "HTML Section"


            if "content" not in self.section:
                self.section["content"] = ""
            logging.debug("HTML section generated %s" % self.section["content"])
            return self.section

if __name__ == "__main__":
    print "in main block"
