#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
# First stage plugin driver script that uses user input to generate document.json with the help of django template
# This template takes user.json in the form of individual contents - table, images

__version__ = '1.0'

import os, sys, logging
from optparse import OptionParser
import json
from glob import glob
from django.conf import settings
from django.template.loader import render_to_string

import json
import random
import shutil
import logging

from ion.plugin.utils.CreateDocument import CreateDocument


def InitializeLogging():
    pass

if __name__ == "__main__":
    #logging.basicConfig()
    logging.basicConfig(level=logging.INFO,
                        format=' %(asctime)s %(funcName)s %(lineno)d [%(levelname)s] - %(message)s'
                        )

    parser = OptionParser(usage="%prog [options]", version="%%prog %s" % __version__)
    parser.add_option("-d", dest="results_dir", help="specify the plugin's results directory containing startplugin.json and output.json. DEFAULT is environment variable ${TSP_FILEPATH_PLUGIN_DIR}")
    parser.add_option("-s", dest="startplugin_json", help="specify name for startplugin.json file in the plugin output directory. DEFAULT is startplugin.json ", default="startplugin.json")
    parser.add_option("-f", dest="user_json", help="specify path to userjson file in the plugin output directory. DEFAULT is output.json", default="output.json")
    (opt, args) = parser.parse_args()

    # TO DO: Better way of defining variables
    path=opt.results_dir
    startpluginjson = os.path.join(path, opt.startplugin_json)
    userprovidedjson = os.path.join(path, opt.user_json)
    logging.debug("result directory %s" % path)

    if not os.path.isdir(path):
        logging.fatal("%s does not exist" % path)
        sys.exit(1)

    if not os.path.isfile(startpluginjson):
        logging.error("%s not found" % startpluginjson)
        sys.exit(1)

    if not os.path.isfile(userprovidedjson):
        logging.error("%s not found" % userprovidedjson)
        sys.exit(1)

    logging.debug("Initializing standard/global templates")
    CreateDocument(opt.results_dir, userprovidedjson, startpluginjson)



