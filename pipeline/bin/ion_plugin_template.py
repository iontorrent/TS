#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
# Plugin driver script that uses user input to generate html report with the help of django template

__version__ = '1.0'

import os, sys, logging
from optparse import OptionParser
import json
from glob import glob
from django.template.loader import render_to_string
from django.conf import settings

from distutils.sysconfig import get_python_lib


def parseJsonFile(jsonFile):
    with open(jsonFile, 'r') as f:
        content = f.read()
        result = json.loads(content)
    return result

def generateHTML(template, context):
        content = render_to_string(template, context)
        output = os.path.join(opt.results_dir, template)
        with open(output, "w") as f:
            f.write(content)


if __name__ == "__main__":

    parser = OptionParser(usage="%prog [options]", version="%%prog %s" % __version__)
    parser.add_option("-j", dest="json_file", help="specify startplugin.json file that contains plugin directory path. DEPRECATED")
    parser.add_option("-d", dest="results_dir", help="specify the plugin's results directory")
    parser.add_option("-n", dest="plugin_name", help="specify plugin name without any white spaces")
    parser.add_option("-t", dest="template_dir", help="specify template directory that storesplugin's results directory.DEPRECATED")
    parser.add_option("-r", dest="results_json_file", help="specify the json file that stores the output of plugin to be displayed by template")
    parser.add_option("-i", dest="plugin_report_html", help="specify template html file rendering the results.json.DEPRECATED")

    (opt, args) = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(opt.plugin_name)
    #log.propagate = False

    #Get the python package for pipeline and append the templates directory to it
    pythonPackage = get_python_lib()
    path = os.path.join(pythonPackage, "ion/plugin/templates")
    settings.configure(TEMPLATE_DIRS=(path, ''), TEMPLATE_DEBUG=True, DEBUG=True)

    results = parseJsonFile(opt.results_json_file)
    log.info("Generating Plugin Report")
    generateHTML('ion_plugin_report_template.html',{'data': results})
