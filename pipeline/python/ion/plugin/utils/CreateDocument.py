#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
# First stage plugin driver script that uses user input to generate document.json with the help of django template
# This template takes user.json in the form of individual contents - table, images

__version__ = "1.0"

import os, sys, logging
from optparse import OptionParser
import json
from glob import glob
from django.conf import settings
from django.template.loader import render_to_string

import csv
import json
import random
import shutil

from distutils.sysconfig import get_python_lib

from ion.plugin.utils.Validatetable import Validatetable
from ion.plugin.utils.Validateimage import Validateimage
from ion.plugin.utils.Validatehtml import Validatehtml
from ion.plugin.utils.ParseMessages import ParseMessages
from ion.plugin.utils.Utility import Utility


class CreateDocument(
    Validatetable, Validateimage, Validatehtml, ParseMessages, Utility
):
    def __init__(self, results_dir, json_file, startplugin_json):
        self.resultsDir = results_dir
        self.log_file = ""
        self.template_file = "ion_plugin_report_template.html"
        self.documentJson = self.parseJsonFile(json_file)
        self.startplugin_json = self.parseJsonFile(startplugin_json)
        if (
            "name" in self.startplugin_json["runinfo"]
            or "plugin_name" in self.startplugin_json["runinfo"]
        ):
            self.documentJson["meta_data"]["plugin_name"] = self.startplugin_json[
                "runinfo"
            ]["plugin_name"]
        """
        if "version" in self.startplugin_json["runinfo"]["plugin"]:
            self.documentJson["meta_data"]["plugin_version"] = self.startplugin_json["runinfo"]["plugin"]["version"]
        """
        self.url_root = self.startplugin_json["runinfo"]["url_root"]
        self.results_dir = self.startplugin_json["runinfo"]["results_dir"]
        self.InitializeTemplates()
        self.EvaluateMetaData()
        self.EvaluateSections()
        self.CreateJson()

    def InitializeTemplates(self):
        if not settings.configured:
            settings.configure(
                DEBUG=False,
                TEMPLATE_DEBUG=False,
                TEMPLATE_DIRS=(
                    os.path.abspath(
                        os.path.join(os.path.dirname(__file__), "..", "templates")
                    ),
                    self.results_dir,
                    os.path.join(self.results_dir, "templates"),
                ),
            )

    def EvaluateMetaData(self):
        """Look for the meta_data section that contains plugin level details"""
        logging.debug("Expanding the meta_data section")
        self.documentJson.setdefault("meta_data", {})
        meta = self.documentJson["meta_data"]

        if "log_file" in meta and meta["log_file"]:
            self.log_file = os.path.join(self.resultsDir, meta["log_file"])

        if "template_file" in meta and meta["template_file"]:
            template_file = os.path.join(self.resultsDir, meta["template_file"])
            if os.path.isfile(template_file):
                self.template_file = template_file
                if not settings.configured:
                    settings.configure(
                        TEMPLATE_DIRS=(
                            os.path.join(self.resultsDir, self.template_file),
                            "",
                        ),
                        TEMPLATE_DEBUG=False,
                        DEBUG=False,
                    )
                else:
                    settings.TEMPLATE_DIRS += (
                        os.path.join(self.resultsDir, self.template_file),
                        "",
                    )
                    # settings.TEMPLATE_DEBUG=True

        # logging.debug("logfile %s templatefile %s" % self.log_file % self.template_file)

    def EvaluateSections(self):
        """Look for expandable sections such as csv to table and images etc"""
        logging.debug("Expanding user.json sections")

        array = []
        for sec in self.documentJson["sections"]:
            temp_dict = {}
            # Validate type
            if sec["type"] not in ["image", "html", "table"]:
                logging.error("Unknown type: %s" % sec["type"])
                continue
            method = getattr(self, "expand%s" % sec["type"], None)
            temp_dict = method(sec)
            array.append(temp_dict)

        # Replace the sections block with the newly updated sections
        self.documentJson["sections"] = array
        if (
            "plugin_about" in self.documentJson["meta_data"]
            and self.documentJson["meta_data"]
        ):
            self.documentJson["sections"].append(
                {
                    "type": "html",
                    "title": "About",
                    "content": "<p>"
                    + self.documentJson["meta_data"]["plugin_about"]
                    + "</p>",
                    "id": self.generateID(),
                }
            )
        self.parsemessages(self.log_file)

    def CreateJson(self):
        """Constructs the json file to be written to disk"""
        logging.debug(
            "Constructing document.json file.Look for filename ion_report.html"
        )
        fw = open(os.path.join(self.resultsDir, "document.json"), "w")
        fw.write(json.dumps(self.documentJson))
        self.documentJson["plugin_name"] = self.startplugin_json["runinfo"][
            "plugin_name"
        ]
        sections = {"data": self.documentJson}
        # TO-DO - the template name is harcoded here
        logging.debug("Template file is %s" % self.template_file)
        html_content = self.generateHTML(self.template_file, sections)
        # TO-DO - report name is harcode here
        with open(
            os.path.join(
                self.resultsDir, "ion_%s_report.html" % self.documentJson["plugin_name"]
            ),
            "w",
        ) as f:
            if html_content:
                f.write(html_content)
            else:
                logging.error("html content empty")


if __name__ == "__main__":
    parser = OptionParser(usage="%prog [options]", version="%%prog %s" % __version__)
    parser.add_option(
        "-d",
        dest="results_dir",
        help="specify the plugin's results directory containing startplugin.json and user.json. REQUIRED",
    )
    parser.add_option(
        "-s",
        dest="startplugin_json",
        help="specify path to startplugin.json file. REQUIRED. ",
    )
    parser.add_option(
        "-f",
        dest="user_json",
        help="specify path to userjson file containing sections. REQUIRED",
    )
    (opt, args) = parser.parse_args()

    # TO DO: Better way of defining variables
    path = opt.results_dir
    path1 = os.path.join(path, opt.startplugin_json)
    path2 = os.path.join(path, opt.user_json)
    if os.path.isdir(path):
        if os.path.isfile(path1) and os.path.isfile(path2):
            CreateDocument(opt.results_dir, path2, path1)
