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


class Utility(object):

    """Class containing utility methods or helper methods used by other classes"""

    def generateID(self):
        """Generate a ID number using random module of python. And ID should be unique"""
        return random.choice('abcdefghijklmn') + str(random.randint(1, 50))

    def parseJsonFile(self, jsonFile):
        try:
            with open(jsonFile, 'r') as f:
                content = f.read()
                try:
                    result = json.loads(content)
                except:
                    logging.error("Invalid Json file %s " % jsonFile)
                    sys.exit(0)
                return result
        except IOError:
            logging.error("FATAL ERROR. Can't open file %s " % jsonFile)

    def generateHTML(self, template, context):
        # self.InitializeTemplates()
        try:
            content = render_to_string(template, context)
            return content
        except Exception, e:
            logging.error("Report Generation failed for %s " % template, e)
