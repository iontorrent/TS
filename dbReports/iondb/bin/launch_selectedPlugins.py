#!/usr/bin/env python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

from iondb.bin.djangoinit import *
from iondb.rundb import models
import time
import datetime
import sys
import requests
import json
import traceback


def print_help():
    print()
    print("Usage: launch_selectedPlugins.py report_id")
    print()


if __name__ == "__main__":

    try:
        report_id = int(sys.argv[1])  # tests for valid input - single integer
        report_id = str(report_id)  # needs to be a string
    except Exception:
        print_help()
        sys.exit(1)

    try:
        result = models.Results.objects.get(pk=report_id)
        plugins = list(result.eas.selectedPlugins.keys())
    except Exception:
        traceback.print_exception()
        sys.exit(1)

    host = "http://localhost"
    headers = {"Content-type": "application/json", "Accept": "application/json"}
    url = host + "/rundb/api/v1/results/" + report_id + "/plugin/"

    print(
        "Start Time: %s"
        % (datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))
    )
    print("Report ID: %s Name:%s" % (report_id, result.resultsName))
    for plugin in plugins:
        data = json.dumps({"plugin": plugin})
        try:
            r = requests.post(url, data=data, headers=headers)
            print("Started: %s" % plugin)
        except Exception:
            print("Failed: %s" % plugin)
