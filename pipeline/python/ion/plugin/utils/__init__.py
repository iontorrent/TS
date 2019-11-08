#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
# Plugin driver script that uses user input to generate html report with the help of django template

__version__ = "1.0"

import os, sys, logging
from optparse import OptionParser
import json
from glob import glob
from django.template.loader import render_to_string
from django.conf import settings

from distutils.sysconfig import get_python_lib


def parseJsonFile(jsonFile):
    with open(jsonFile, "r") as f:
        content = f.read()
        result = json.loads(content)
    return result


def generateHTML(template, context):
    content = render_to_string(template, context)
    output = os.path.join(opt.results_dir, template)
    with open(output, "w") as f:
        f.write(content)
