#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
# Main Plugin run script that integrates results from each plugin and renders to a html template

import os, sys, logging
from optparse import OptionParser
import json
from glob import glob

# Import settings, so TEMPLATE_DIRS can be overridden
import iondb.settings as settings

os.environ["DJANGO_SETTINGS_MODULE"] = "iondb.settings"
from django.template.loader import render_to_string


def parseJsonFile(jsonFile):
    with open(jsonFile, "r") as f:
        content = f.read()
        result = json.loads(content)
    return result


def generateHTML(template, context):
    content = render_to_string(template, context)
    output = os.path.join(pluginparams["runinfo"]["results_dir"], template)
    # .rstrip(".tmpl")) ## tmpl is optional, best to omit it.
    with open(output, "w") as f:
        f.write(content)


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option(
        "-v", dest="version", action="count", help="print version and exit"
    )
    parser.add_option(
        "-j",
        dest="json_file",
        help="specify startplugin.json file that contains plugin directory path",
    )
    parser.add_option(
        "-n", dest="plugin_name", help="specify plugin name without any white spaces"
    )
    parser.add_option(
        "-t",
        dest="template_dir",
        help="specify template directory that stores django templates",
    )
    parser.add_option(
        "-r",
        dest="results_json_file",
        help="specify the results.json file that stores the output of plugin to be displayed",
    )
    parser.add_option(
        "-h",
        dest="plugin_report_html",
        help="specify template html file rendering the results.json ",
    )
    (opt, args) = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(opt.plugin_name)
    log.propagate = False

    # TO DO - Instead of json_file, take the plugin directory path as input
    if opt.json_file:
        pluginparams = parseJsonFile(opt.json_file)
    else:
        log.fatal("No Plugin JSON file specified")
        sys.exit(1)
    """
    if pluginparams['globalconfig']['debug']:
        logging.basicConfig(level=logging.DEBUG)
    log.debug(pluginparams)
    """
    settings.TEMPLATE_DIRS += (
        os.path.join(pluginparams["runinfo"]["plugin_dir"], opt.template_dir),
    )
    data = []
    # results.json is assumed to be the output file for plugin.
    # TO DO: context.json
    myResults = parseJsonFile(opt.results_json_file)
    log.info("Generating Plugin Report")
    generateHTML(opt.plugin_report_html, {"data": myResults})
