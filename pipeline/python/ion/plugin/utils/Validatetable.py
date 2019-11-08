#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
# First stage plugin driver script that uses user input to generate document.json with the help of django template
# This template takes user.json in the form of individual contents - table, images

__version__ = "1.0"

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


class Validatetable(Utility):

    """Check for proper json format in the table section"""

    def expandtable(self, section):
        """Set the Default values for the section"""
        self.section = section
        self.section["id"] = self.generateID()
        i = 0
        # self.section["id"] = Utility.generateID() #ID is generated radomly

        if "title" not in self.section:
            logging.error(
                "title field for section type %s not provided. Using a default title."
                % self.section["type"]
            )
            self.section["title"] = "Table Section"

        if "filename" not in self.section:
            logging.error(
                "Comma Seperated (csv) file for section type %s not provided. Refer to a standard Json structure."
                % self.section["type"]
            )
            self.section["content"] = ""
            return self.section

        else:
            self.temp_dict = {}
            try:
                with open(self.section["filename"], "r") as f:
                    reader = csv.DictReader(f)
                    out = [row for row in reader]
                    # TO-DO- if headers is not specified, then render then generate table for entire csv file
                    if "columns" not in self.section:
                        logging.warning(
                            "columns fields not provided. Generating table for the first six columns of csv file."
                        )
                        allColumns = [list(i.keys()) for i in out][1]
                        self.section["columns"] = allColumns[1:6]
                    table_block = []
                    for row in out:
                        temp_dict = {}
                        for header in self.section["columns"]:
                            temp_dict[header] = row[header]
                        table_block.append(temp_dict)

                    self.temp_dict = {
                        "data": table_block,
                        "headers": list(table_block[0].keys()),
                    }
                    print(self.temp_dict)

            except IOError as e:
                logging.error(e.errno)
                logging.error(
                    "Can't open csv file %s or partial file or truncated file"
                    % self.section["filename"]
                )
            self.section["content"] = self.generateHTML(
                "ion_table_template.html", self.temp_dict
            )
            logging.debug("Table section generated - %s" % self.section["content"])
        return self.section


if __name__ == "__main__":
    print("In main section")
