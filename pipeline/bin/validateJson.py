#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
# Python script that validates the user input json file for Plugin SDK

import json
import csv


@staticmethod
def parseJsonFile(jsonFile):
    with open(jsonFile, "r") as f:
        content = f.read()
        result = json.loads(content)
        return result


class VerifyJson(object):
    def __init__(self, json_file):
        self.json_file = json_file

    def checkJson(self):
        pass

    def checkImages(self):
        pass

    def checkCSV(self):
        pass

    def checkHTML(self):
        pass

    def checkError(self):
        pass

    def checkWarnings(self):
        pass

    def checkTemplates(self):
        pass


if __name__ == "__main__":

    obj = VerifyJson(json_filename)
