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


class ParseMessages(object):

    def parse_errors(self, fh):
        """ Parses the drmaa_stdout.txt for errors and include that in the document json"""
        errors = []
        for row in fh:
            if "ERROR" in row or "Error" in row:
                errors.append(row)
        return errors

    def parse_warnings(self, fh):
        """Parses the drmaa_stdout.txt for warnings and include that in the document json"""
        warnings = []
        for row in fh:
            if "WARNING" in row or "Warning" in row:
                warnings.append(row)
        return warnings

    def parse_success(self, fh):
        """Parse any success message(s) in the drmaa_stdout.txt and inclues that in document.json"""
        success = []
        for row in fh:
            if "success" in row or "successfully" in row:
                success.append(row)
        return success

    def parsemessages(self, log_file):
        self.stdout_file = os.path.join(self.resultsDir, "drmaa_stdout.txt")
        logfiles = []
        logfiles.append(self.stdout_file)
        if log_file:
            if os.path.isfile(log_file):
                logfiles = logfiles.append(os.path.join(self.resultsDir, log_file))
        for logfilename in logfiles:
            logging.debug(logfilename)
            for message in ["errors", "warnings", "success"]:
                if os.path.isfile(logfilename):
                    logging.debug("log file found %s" % logfilename)
                    try:
                        with open(logfilename, 'r') as f:
                            method = getattr(self, "parse_%s" % message, None)
                            self.documentJson[message] = method(f)
                    except:
                        logging.error("Can't open stdout file %s " % logfilename)
                else:
                    logging.error("File %s does not exist" % logfilename)
