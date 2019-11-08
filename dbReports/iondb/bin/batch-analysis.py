#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Look for old experiments and analysis them again

Input is a text file where each line is the name of a report to rerun.
The name of this text file must be provided as the first arg.
"""

import datetime
import json
import os
from os import path
import re
import sys

from djangoinit import *
from django import db
from iondb.rundb import models, views
import gc
import httplib, urllib

# get django models
from iondb.rundb import models


def get_name_from_json(exp, key):
    data = exp.log
    name = data.get(key, False)
    if not name:
        return "Auto_%s_%s" % (exp.pretty_print(), exp.pk)
    else:
        return "%s_%s" % (str(name), exp.pk)


def generate_http_post(exp, server):
    params = urllib.urlencode(
        {
            "report_name": get_name_from_json(exp, "autoanalysisname") + "_V7",
            "tf_config": "",
            "path": exp.expDir,
            "qname": settings.SGEQUEUENAME,
            "submit": ["Start Analysis"],
        }
    )
    print(params)
    headers = {
        "Content-type": "text/html",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        f = urllib.urlopen(
            "http://%s/rundb/newanalysis/%s/0" % (server, exp.pk), params
        )
    except Exception:
        f = urllib.urlopen(
            "https://%s/rundb/newanalysis/%s/0" % (server, exp.pk), params
        )
    return


def get_exp_from_report_name(report):
    """get the exp name from the report name"""
    exp = models.Experiment.objects.filter(results__resultsName__exact=report)
    return exp[0]


def parse_input(input_file):
    # make a list where each element is a line from the input file
    parsed = input_file.readlines()
    # get rid of extra characters
    parsed = [exp.strip() for exp in parsed]

    return parsed


if __name__ == "__main__":

    try:
        print(sys.argv[1])
        input_file = open(sys.argv[1])
        reports = parse_input(input_file)
    except IndexError:
        print(
            "Please provide the path to a text file where each line is the name of a report as the first arg"
        )
        sys.exit(1)
    except IOError:
        print("Could not open text file!")
        sys.exit(1)
    except:
        print("Fatal Error")
        sys.exit(1)

    # make a set of exps
    exp_list = set()

    # get the exp names from the report names
    for report in reports:
        exp = get_exp_from_report_name(report)
        exp_list.add(exp)

    # now start the analysis, we are using a set of that each exp only has to be reran once.
    for exp in exp_list:
        print("Starting analysis for : ", exp)
        generate_http_post(exp, "localhost")
