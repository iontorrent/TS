#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
# First stage plugin driver script that uses user input to generate document.json with the help of django template
# This template takes user.json in the form of individual contents - table, images

__version__ = '1.0'

import os, sys, logging
# from ion.plugin.utils.pluginsLogger import *
from optparse import OptionParser
import json
from glob import glob
from django.template.loader import render_to_string
from django.conf import settings

import types
import csv
import json
import random
import shutil

from distutils.sysconfig import get_python_lib
from ion.plugin.utils.Utility import Utility


class Validateimage(Utility):

        """Check for proper json format in the image section"""

        def expandimage(self, section):
            """Set the Default values of the section"""
            # vimagelog = logging.getLogger("imagelogging")
            self.section = section
            self.section["id"] = self.generateID()
            imagesinDir = []

            # print "in Validimage constructor"
            if "title" not in self.section:
                logging.error("title not provided for field section type %s. Using the default title. " % self.section["type"])
                self.section["title"] = "Image Section"
            if "resultsDir" in self.section:
                logging.debug("Looking for images in resultsDir %s" % self.section["resultsDir"])
                imagesinDir = glob(os.path.join(self.results_dir, self.section["resultsDir"], "*.png"))
                # imagesinDir.append(glob.glob("*.jpg"))
                imageList = [os.path.join(self.section["resultsDir"], os.path.basename(img)) for img in imagesinDir]
                self.section["content"] = imageList
                logging.debug("Found images %s in results directory" % self.section["content"])

            self.imageArray = []
            if "content" in self.section and self.section["content"]:
                if isinstance(self.section["content"][0], types.DictType):
                    logging.debug("Image block - Dictionary Type")
                    for image in self.section["content"]:
                        if os.path.isfile(os.path.join(self.results_dir, image["source"])):
                            imageJson = {}
                            imageJson["source"] = os.path.join(self.url_root, "plugin_out", os.path.basename(self.results_dir), image["source"])
                            # print "IMAGEJSON", imageJson["source"]
                            if "caption" in image and image["caption"]:
                                try:
                                    imageJson["caption"] = image["caption"]
                                except:
                                    imageJson["caption"] = '.'.join(os.path.basename(image["source"]).split('.')[0:-1])  # Default caption for images
                                    logging.debug("Using default filename as caption")
                            else:
                                logging.debug("No image caption found. Using the default filename as caption")
                                imageJson["caption"] = '.'.join(os.path.basename(image["source"]).split('.')[0:-1])  # Default caption for imag

                            if "description" in image and image["description"]:
                                try:
                                    imageJson["description"] = image["description"]
                                except:
                                    imageJson["description"] = ""  # blank is the default description
                            else:
                                logging.debug("No description found. Using the default description as blank")
                                imageJson["description"] = ""  # blank if the default description

                            self.imageArray.append(imageJson)
                        else:
                            logging.error("Image %s does not exist" % os.path.join(self.results_dir, image["source"]))
                    self.section["content"] = self.imageArray

                elif isinstance(self.section["content"][0], types.UnicodeType):
                    logging.debug("Image block - List Type")
                    for images in self.section["content"]:
                        if os.path.isfile(os.path.join(self.results_dir, images)):
                            imageJson = {}
                            imageJson["source"] = os.path.join(self.url_root, "plugin_out", os.path.basename(self.results_dir), images)
                            imageJson["caption"] = ""
                            self.imageArray.append(imageJson)
                    self.section["content"] = self.imageArray
                else:
                    print self.section["content"]
                    logging.error("Image block - Format unrecognizable")

            else:
                logging.debug("No content section found. Checking for images in the results directory")
                imagesinDir = glob(os.path.join(self.results_dir, "*.png"))
                # imagesinDir.append(glob.glob("*.jpg"))
                if imagesinDir:
                    for image in imagesinDir:
                        imageJson = {}
                        print image
                        imageJson["source"] = os.path.join(self.url_root, "plugin_out", os.path.basename(self.results_dir), os.path.basename(image))
                        imageJson["caption"] = os.path.basename(image)
                        self.imageArray.append(imageJson)
                    self.section["content"] = self.imageArray
                    logging.debug("Found images %s in results directory" % self.section["content"])
                else:
                    logging.debug("Found no images in results directory")
                    self.section["content"] = ""

            logging.debug("Image section generated %s" % self.section)
            return self.section

if __name__ == "__main__":
    print "in main block"
