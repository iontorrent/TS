#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
# First stage plugin driver script that uses user input to generate document.json with the help of django template
# This template takes user.json in the form of individual contents - table, images

__version__ = '1.0'

import os, sys, logging
from optparse import OptionParser
import json
from glob import glob
from django.template.loader import render_to_string
from django.conf import settings

import csv
import json
import random

from distutils.sysconfig import get_python_lib

class Utility(object):
    """Class containing utility methods or helper methods used by other classes"""

    @staticmethod
    def generateID():
        """Generate a ID number using random module of python. And ID should be unique"""
        return random.choice('abcdefghijklmn') + str(random.randint(1,50))

    @staticmethod
    def generateHTML(template, context):
        try:
            content = render_to_string(template, context)
        except:
            print "Report Generation failed for %s " % template
        return content



class ValidateHTML(object):
        """Check for proper json format in the html section"""
        def __init__(self,section):
            self.section = section
            if "title" not in self.section:
                print "ERROR FATAL. Please provide a title field for section type %s " % self.section["type"]
                sys.exit(0)
            self.section["id"] = Utility.generateID()

        def expandSection(self):
            """Return the html expanded section"""
            if "content" in self.section and self.section["content"]:
                return self.section
            else:
                self.section["content"] = "<p>No Content Specified</p>"
                return self.section

class ValidateImage(object):
        """Check for proper json format in the image section"""

        def __init__(self,section):
            """Set the Default values of the section"""
            self.section = section
            if "title" not in self.section:
                print "ERROR FATAL. Please provide a title field for section type %s. Refer to a standard Json structure. " % self.section["type"]
                sys.exit(0)
            if "resultsDir" not in self.section:
                print "ERROR FATAL. Please provide a results directory for images for the section type %s. Refer to standard Json Structure." % self.section["type"]
            self.section["id"] = Utility.generateID()
            self.array=[]
            self.parseImage()

        def parseImage(self):
            if "content" in self.section and self.section["content"]:
                for image in self.section["content"]:
                    if os.path.isfile(os.path.join(self.section["resultsDir"],image)):
                        imageJson = {}
                        imageJson["source"] = os.path.relpath(image)
                        imageJson["caption"] = '.'.join(image.split('.')[0:-1])
                        self.array.append(imageJson)
                    else:
                        print "Image %s does not exist" % image
            else:
                self.array = "<p>No Images Specified</p>"

        def expandSection(self):
            self.section["content"] = self.array
            return self.section


class ValidateTable(object):
        """Check for proper json format in the table section"""

        def __init__(self, section):
            """Set the Default values for the section"""
            self.section = section
            self.section["id"] = Utility.generateID() #ID is generated radomly
            if "title" not in self.section:
               print "ERROR FATAL. Please provide a title field for section type %s  Refer to a standard Json structure." % self.section["type"]
               sys.exit(0)

            if "filename" not in self.section:
                print "ERROR FATAL. Please provide a Comma Seperated (csv) file for section type %s. Refer to a standard Json structure." % self.section["type"]
                sys.exit(0)

            if "columns" not in self.section:
                print "ERROR FATAL. Please provide a list of *columns* fields occuring in the CSV file for section type %s  Refer to a standard Json structure." % self.section["type"]
                sys.exit(0)


            self.temp_dict={}
            self.parseCSV()
            self.generateHTML()

        def parseCSV(self):
            """Parse out the csv contents into a dictionary. Pass the temp_dict as context to Django template for table"""
            try:
                with open(self.section["filename"],'r') as f:
                    reader = csv.DictReader(f)
                    out = [row for row in reader]
                    #TO-DO- if headers is not specified, then render then generate table for entire csv file
                    self.temp_dict = {'data': out, 'headers': self.section["columns"]}

            except IOError, e:
                print e.errno
                print "Can't open csv file %s " % self.section["filename"]

        def generateHTML(self):
            """Call the generateHTML method to render table"""
            try:
                if "data" in self.temp_dict:
                   self.section["content"] = render_to_string("table_template.html", self.temp_dict)
            except:
                print "template generation failed..."

        def expandSection(self):
            """Return the final section block"""
            return self.section

class CreateDocument(ValidateTable, ValidateImage, ValidateHTML):
    def __init__(self, results_dir, json_file, startplugin_json):
        self.resultsDir = results_dir
        self.stdout_file = os.path.join(self.resultsDir,"drmaa_stdout.txt")
        self.documentJson = CreateDocument.parseJsonFile(json_file)
        self.startplugin_json = CreateDocument.parseJsonFile(startplugin_json)
        #self.documentJson.update(user_data)
        self.InitializeTemplates()#Singleton
        self.EvaluateSections()
        self.parseMessages()
        self.CreateJson()

    def parse_errors(self,fh):
        """ Parses the drmaa_stdout.txt for errors and include that in the document json"""
        errors = []
        for row in fh:
            if "ERROR" in row or "Error" in row:
                errors.append(row)
        return errors

    def parse_warnings(self,fh):
        """Parses the drmaa_stdout.txt for warnings and include that in the document json"""
        warnings = []
        for row in fh:
            if "WARNING" in row or "Warning" in row:
                warnings.append(row)
        return warnings

    def parse_success(self,fh):
        """Parse any success message(s) in the drmaa_stdout.txt and inclues that in document.json"""
        success=[]
        for row in fh:
            if "success" in row or "successfully" in row:
               success.append(row)
        return success

    def parseMessages(self):
        for message in ["errors", "warnings", "success"]:
            if os.path.isfile(self.stdout_file):
                try:
                    with open(self.stdout_file,'r') as f:
                        method = getattr(self, "parse_%s" % message, None)
                        self.documentJson[message] = method(f)
                except:
                    print "Can't open stdout file %s " % self.stdout_file
            else:
                print "File %s does not exist" % self.stdout_file

    @staticmethod
    def parseJsonFile(jsonFile):
        try:
            with open(jsonFile, 'r') as f:
                  content = f.read()
                  try:
                      result = json.loads(content)
                  except:
                      print "Invalid Json file %s " % jsonFile
                      sys.exit(0)
                  return result
        except IOError:
            print "FATAL ERROR. Can't open file %s " % jsonFile


    def InitializeTemplates(self):
        pythonPackage = get_python_lib()
        path = os.path.join(pythonPackage, "ion/plugin/templates")
        settings.configure(TEMPLATE_DIRS=(path, ''), TEMPLATE_DEBUG=True, DEBUG=True)


    def Evaluate_table(self, section):
        """Return the html table generated from the csv file and wrapped into a django table template"""
        obj = ValidateTable(section)
        table_sec = obj.expandSection()
        return table_sec

    def Evaluate_image(self, section):
        """Return the content section with image source and caption listing """
        obj = ValidateImage(section)
        image_sec = obj.expandSection()
        return image_sec

    def Evaluate_html(self, section):
        """simply return the original content"""
        obj = ValidateHTML(section)
        html_sec = obj.expandSection()
        return html_sec

    def EvaluateSections(self):
        """Look for expandable sections such as csv to table and images etc"""
        array = []
        for sec in self.documentJson["sections"]:
            temp_dict = {}
            # Validate type
            method = getattr(self, "Evaluate_%s" % sec["type"], None)
            if not method:
                print "ERROR: Unknown type: %s" % sec["type"]
                continue
            temp_dict = method(sec)
            array.append(temp_dict)

        #Replace the sections block with the newly updated sections
        self.documentJson["sections"] = array


    def CreateJson(self):
        """Constructs the json file to be written to disk"""
        self.documentJson["plugin_name"] = self.startplugin_json["runinfo"]["plugin_name"]
        sections = {'data': self.documentJson}
        # TO-DO - the template name is harcoded here
        html_content = Utility.generateHTML("ion_plugin_report_template.html", sections)
        # TO-DO - report name is harcode here
        with open(os.path.join(self.resultsDir, "ion_%s_report.html" % self.documentJson["plugin_name"]),'w') as f:
            f.write(html_content)



if __name__ == "__main__":

    parser = OptionParser(usage="%prog [options]", version="%%prog %s" % __version__)
    parser.add_option("-d", dest="results_dir", help="specify the plugin's results directory containing startplugin.json and user.json. REQUIRED")
    parser.add_option("-s", dest="startplugin_json", help="specify path to startplugin.json file. REQUIRED. ")
    parser.add_option("-f", dest="user_json", help="specify path to userjson file containing sections. REQUIRED")
    (opt, args) = parser.parse_args()

    # TO DO: Better way of defining variables
    path=opt.results_dir
    path1 = os.path.join(path, opt.startplugin_json)
    path2 = os.path.join(path, opt.user_json)
    if os.path.isdir(path):
        if os.path.isfile(path1) and os.path.isfile(path2):
            CreateDocument(opt.results_dir, path2, path1)



