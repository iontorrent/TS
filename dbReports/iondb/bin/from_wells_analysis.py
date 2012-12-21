#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
#
# PURPOSE: Command line utility which initiates from-wells analysis given a directory
# path to 1.wells data.
#
# USE CASE: Importing sig proc results to a different TS.
# Requires that the given argument is a filesystem path to a directory containing:
#
# explog.txt
# sigproc_results/1.wells
# sigproc_results/bfmask.bin
# sigproc_results/bfmask.stats
# sigproc_results/Bead_density_contour.png
# sigproc_results/avgNukeTrace_*.txt
#
################################################################################
import os
import sys
import json
import glob
import time
import copy
import string
import random
import urllib
import argparse
import datetime
import traceback
import logging
logging.basicConfig(level=logging.DEBUG)
from djangoinit import *
from iondb.rundb import models
from iondb.rundb.report.views import get_default_cmdline_args
from ion.utils.explogparser import load_log_path
from ion.utils.explogparser import parse_log
from iondb.utils.crawler_utils import getFlowOrder
from iondb.utils.crawler_utils import folder_mtime
from iondb.bin.crawler import exp_kwargs

TIMESTAMP_RE = models.Experiment.PRETTY_PRINT_RE
def extract_rig(folder):
    """Given the name of a folder storing experiment data, return the name
    of the PGM from which the date came."""
    #return os.path.basename(os.path.dirname(folder))
    return "uploads"


# Extracted from crawler.py and stem modified to be Reanalysis instead of Auto
def get_name_from_json(exp, key, thumbnail_analysis):
    data = exp.log
    name = data.get(key, False)
    twig = ''
    if thumbnail_analysis:
        twig = '_tn'
    # also ignore name if it has the string value "None"
    if not name or name == "None":
        uniq = ''.join(random.choice(string.letters + string.digits) for i in xrange(4))
        return 'Reanalysis_%s_%s_%s%s' % (exp.pretty_print().replace(" ","_"),exp.pk,uniq,twig)
    else:
        return '%s_%s%s' % (str(name),exp.pk,twig)


# Extracted from crawler.py and modified to launch fromWells analysis
def generate_http_post(exp, projectName, data_path, thumbnail_analysis=False):
    try:
        GC = models.GlobalConfig.objects.all().order_by('pk')[0]
        base_recalibrate = GC.base_recalibrate
    except models.GlobalConfig.DoesNotExist:
        base_recalibrate = False

    default_args = get_default_cmdline_args(exp.chipType)
    if thumbnail_analysis:
        analysisArgs = default_args['thumbnailAnalysisArgs']
    else:
        analysisArgs = default_args['analysisArgs']
        
    # Force the from-wells option here
    analysisArgs = analysisArgs + " --from-wells %s" % os.path.join(data_path,"1.wells")

    report_name = get_name_from_json(exp,'autoanalysisname',thumbnail_analysis)

    params = urllib.urlencode({'report_name':report_name,
                            'tf_config':'',
                            'path':exp.expDir,
                            'args':analysisArgs,
                            'submit': ['Start Analysis'],
                            'do_thumbnail':"%r" % thumbnail_analysis,
                            'blockArgs':'fromWells',
                            'previousReport':os.path.join(data_path),
                            'project_names':projectName,
                            'do_base_recal':base_recalibrate
                            })
    
    status_msg = report_name
    try:
        connection_url = 'http://127.0.0.1/report/analyze/%s/0/' % (exp.pk)
        f = urllib.urlopen(connection_url, params)
    except IOError:
        logger.errors.debug('could not make connection %s' % connection_url)
        try:
            connection_url = 'https://127.0.0.1/report/analyze/%s/0/' % (exp.pk)
            f = urllib.urlopen(connection_url, params)
        except IOError:
            logger.errors.error(" !! Failed to start analysis.  could not connect to %s" % connection_url)
            status_msg = "Failure to generate POST"
            f = None
    
    if f:
        error_code = f.getcode()
        if error_code is not 200:
            logger.errors.error(" !! Failed to start analysis. URL failed with error code %d for %s" % (error_code, f.geturl()))
            status_msg = "Failure to generate POST"

    return status_msg

def newExperiment(explog_path):
    '''Create Experiment record'''
    # Parse the explog.txt file
    text = load_log_path(explog_path)
    dict = parse_log(text)
    
    # Create the Experiment Record
    folder = os.path.dirname(explog_path)
    
    # Test if Experiment object already exists
    try:
        newExp = models.Experiment.objects.get(unique=folder)
    except:
        newExp = None
        
    if newExp is None:
        try:
            expArgs,st = exp_kwargs(dict,folder)
            newExp = models.Experiment(**expArgs)
            newExp.save()
        except:
            newExp = None
            print traceback.format_exc()
    
    return newExp

def newReport(exp, data_path):
    '''Submit analysis job'''
    projectName = ''
    output = generate_http_post(exp,projectName,data_path)
    return output

def getReportURL(report_name):
    URLString = None
    try:
        report = models.Results.objects.get(resultsName=report_name)
        URLString = report.reportLink
    except models.Results.DoesNotExist:
        URLString = "Not found"
    except:
        print traceback.format_exc()
    finally:
        return URLString

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initiate from-wells analysis Report")
    parser.add_argument("directory",metavar="directory",help="Path to data to analyze")
    
    # If no arguments, print help and exit
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
        
    # Parse command line
    args = parser.parse_args()
    
    # Test inputs
    if not os.path.isdir(args.directory):
        print "Does not exist: %s" % args.directory
        sys.exit(1)
    src_dir = args.directory
    
    # Validate existence of prerequisite files
    explog_path = os.path.join(src_dir,'explog.txt')
    if not os.path.isfile(explog_path):
        print "Does not exist: %s" % explog_path
        print "Cannot create environment for re-analysis to take place"
        print "STATUS: Error"
        sys.exit(1)
        
    wells_path = os.path.join(src_dir,'sigproc_results','1.wells')
    if not os.path.isfile(wells_path):
        print "Does not exist: %s" % wells_path
        print "Cannot basecall without output from signal processing"
        print "STATUS: Error"
        sys.exit(1)
        
    testpath = os.path.join(src_dir,'sigproc_results','analysis.bfmask.bin')
    if not os.path.isfile(testpath):
        testpath = os.path.join(src_dir,'sigproc_results','bfmask.bin')
        if not os.path.isfile(testpath):
            print "Does not exist: %s" % testpath
            print "Cannot basecall without bfmask.bin from signal processing"
            print "STATUS: Error"
            sys.exit(1)
    
    testpath = os.path.join(src_dir,'sigproc_results','analysis.bfmask.stats')
    if not os.path.isfile(testpath):
        testpath = os.path.join(src_dir,'sigproc_results','bfmask.stats')
        if not os.path.isfile(testpath):
            print "Does not exist: %s" % testpath
            print "Cannot basecall without bfmask.stats from signal processing"
            print "STATUS: Error"
            sys.exit(1)
    
    # Missing these files just means key signal graph will not be generated
    testpath = os.path.join(src_dir,'sigproc_results','avgNukeTrace_ATCG.txt')
    if not os.path.isfile(testpath):
        print "Does not exist: %s" % testpath
        print "Cannot create TF key signal graph without %s file" % 'avgNukeTrace_ATCG.txt'
        
    testpath = os.path.join(src_dir,'sigproc_results','avgNukeTrace_TCAG.txt')
    if not os.path.isfile(testpath):
        print "Does not exist: %s" % testpath
        print "Cannot create Library key signal graph without %s file" % 'avgNukeTrace_TACG.txt'
    
    # Create Experiment record
    newExp = newExperiment(explog_path)
    if newExp is None:
        print "STATUS: Error"
        sys.exit(1)
    
    # Submit analysis job URL
    report_name = newReport(newExp, src_dir)
    if report_name is None:
        print "STATUS: Error"
        sys.exit(1)
    
    # Test for Report Object
    count = 0
    delay = 1
    retries = 60
    while count < retries:
        count += 1
        reportURL = getReportURL(report_name)
        if reportURL is None:
            print "STATUS: Error"
            sys.exit(1)
        elif reportURL == "Not found":
            print "Retry %d of %d in %d second" % (count,retries, delay)
            time.sleep(delay)
        else:
            count = retries
        
    if reportURL is "Not found":
        print "STATUS: Error"
    else:
        print "STATUS: Success"
    print "REPORT-URL: %s" % reportURL
    
    
