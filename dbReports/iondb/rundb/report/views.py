# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.contrib.auth.decorators import login_required
from django.template import RequestContext
from iondb.rundb import models
from iondb.rundb import forms
from iondb.backup import makePDF
from iondb.anaserve import client
from ion.utils import makeCSA
from django import http
import json
import os
import csv
import re
import socket
import traceback

from cStringIO import StringIO

import ConfigParser
import logging
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponsePermanentRedirect, HttpResponseRedirect, HttpResponse
from django.core import serializers
from django.core.serializers.json import DjangoJSONEncoder
from django.core.exceptions import ObjectDoesNotExist
from django.core.servers.basehttp import FileWrapper
from django.core.urlresolvers import reverse
from django.shortcuts import get_object_or_404, redirect, render_to_response
from django.conf import settings
import numpy
import math
import urllib
from datetime import datetime, date
import file_browse
import xmlrpclib
import iondb.backup.tasks
import fnmatch
from iondb.backup import ion_archiveResult
import glob

logger = logging.getLogger(__name__)

#TODO do something fancy to keep track of the versions


def url_with_querystring(path, **kwargs):
    return path + '?' + urllib.urlencode(kwargs)


@login_required
def getCSA(request, pk):
    '''Python replacement for the pipeline/web/db/writers/csa.php script'''
    try:
        # From the database record, get the report information
        result = models.Results.objects.get(pk=pk)
        reportDir = result.get_report_dir()
        rawDataDir = result.experiment.expDir
        
        # Generate report PDF file.
        # This will create a file named backupPDF.pdf in results directory
        makePDF.latex(pk, reportDir)
        
        csaFullPath = makeCSA.makeCSA(reportDir,rawDataDir)
        csaFileName = os.path.basename(csaFullPath)
        
        response = http.HttpResponse(FileWrapper (open(csaFullPath)), mimetype='application/zip')
        response['Content-Disposition'] = 'attachment; filename=%s' % csaFileName
        
    except:
        logger.error(traceback.format_exc())
        response = http.HttpResponse(status=500)
        
    return response


def getPDF(request, pk):
    ret = get_object_or_404(models.Results, pk=pk)
    filename = "%s"%ret.resultsName
    
    response = http.HttpResponse(makePDF.getPDF(pk), mimetype="application/pdf")
    response['Content-Disposition'] = 'attachment; filename=%s-full.pdf'%filename
    return response

def getlatex(request, pk):
    ret = get_object_or_404(models.Results, pk=pk)
    
    response = http.HttpResponse(makePDF.getlatex(pk), mimetype="application/pdf")
    response['Content-Disposition'] = 'attachment; filename=' + ret.resultsName + '.pdf'
    return response

def getPlugins(request, pk):
    ret = get_object_or_404(models.Results, pk=pk)
    pluginPDF = makePDF.getPlugins(pk)
    if pluginPDF:
        response = http.HttpResponse(pluginPDF, mimetype="application/pdf")
        response['Content-Disposition'] = 'attachment; filename=' + ret.resultsName + '-plugins.pdf'
        return response
    else:
        return HttpResponseRedirect(url_with_querystring(reverse('report', args=[pk]), noplugins="True"))

def percent(q, d):
    return "%04.1f%%" % (100 * float(q) / float(d)) 


def load_ini(report,subpath,filename,namespace="global"):
    parse = ConfigParser.ConfigParser()
    parse.optionxform = str # preserve the case
    try:
        parse.read(os.path.join(report.get_report_dir(), subpath , filename))
        parse = parse._sections.copy()
        return parse[namespace]
    except:
        return False

# TODO, this function should just read from the database instead of reading from the filesystem (TS 3.6)
def getBlockStatus(report,blockdir,namespace="global"):

    STATUS = ""
    try:
        if not os.path.exists(os.path.join(report.get_report_dir(), blockdir, 'sigproc_results')):
            STATUS = "Transfer..."
            return STATUS

        analysis_return_code = os.path.join(report.get_report_dir(), blockdir, 'sigproc_results', 'analysis_return_code.txt')
        if os.path.exists(analysis_return_code):
            with open(analysis_return_code, 'r') as f:
                text = f.read()
                if not '0' in text:
                    STATUS = "Analysis error %s" % text
                    return STATUS

        blockstatusfile = os.path.join(report.get_report_dir(), blockdir, 'blockstatus.txt')
        if os.path.exists( blockstatusfile ):
            f = open(blockstatusfile)
            text = f.readlines()
            f.close()
            for line in text:
                [component, status] = line.split('=')
                print component, status
                if int(status) != 0:
                    if component == 'Beadfind':
                        if int(status) == 2:
                            STATUS = 'Checksum Error'
                        elif int(status) == 3:
                            STATUS = 'No Live Beads'
                        else:
                            STATUS = "Error in Beadfind"
                    elif component == 'Analysis':
                        if int(status) == 2:
                            STATUS = 'Checksum Error'
                        elif int(status) == 3:
                            STATUS = 'No Live Beads'
                        else:
                            STATUS = "Error in Analysis"
                    elif component == 'Recalibration':
                        STATUS = "Skip Recal."
                    else:
                        STATUS = "Error in %s" % component
                    return STATUS

        old_progress_file = os.path.join(report.get_report_dir(), blockdir, 'progress.txt')
        if os.path.exists(old_progress_file):
            f = open(old_progress_file)
            text = f.readlines()
            f.close()
            for line in text:
                [component, status] = line.split('=')
                if "yellow" in status:
                    if "wellfinding" in component:
                        STATUS = "Beadfind"
                    elif "signalprocessing" in component:
                        STATUS = "Sigproc"
                    elif "basecalling" in component:
                        STATUS = "Basecalling"
                    elif "alignment" in component:
                        STATUS = "Alignment"
                    else:
                        STATUS = "%s" % component
                    return STATUS

        if os.path.exists(os.path.join(report.get_report_dir(), blockdir, 'badblock.txt')):
            STATUS = "unknown error"
            return STATUS


        if not os.path.exists( blockstatusfile ):
            STATUS = "Pending"

    except:
        STATUS = "Exception"

    return STATUS


def load_json(report,subpath,filename):
    """shortcut to load the json"""
    try:
        if subpath:
            f =  open(os.path.join(report.get_report_dir(), subpath , filename), mode='r')
        else:
            f =  open(os.path.join(report.get_report_dir() , filename), mode='r')   
        return json.loads(f.read())
    except:
        return False


def datasets_read(report):
    datasets = load_json(report,"basecaller_results","datasets_basecaller.json")
    if not datasets: 
        return None
    else:
        return datasets


def testfragments_read(report):
    testfragments = load_json(report,"basecaller_results","TFStats.json")
    if not testfragments: 
        return None

    try:
        for tf_name,tf_data in testfragments.iteritems():
            num_reads = int(tf_data.get("Num",0))
            num_50AQ17 = int(tf_data.get("50Q17",0))
            conversion_50AQ17 = "N/A"
            if num_reads > 0:
                conversion_50AQ17 = (100*num_50AQ17/num_reads)
            testfragments[tf_name]["conversion_50AQ17"] = conversion_50AQ17
            testfragments[tf_name]["histogram_filename"] = "new_Q17_%s.png" % tf_name
            testfragments[tf_name]["num_reads"] = num_reads
        
    except KeyError:
        pass

    return testfragments

def barcodes_read(report):
    """get the list of barcodes"""
    barcodes = load_json(report,"basecaller_results","datasets_basecaller.json")
    if not barcodes: return False

    try:
        for dataset in barcodes.get("datasets",[]):
            file_prefix = dataset["file_prefix"]
            for rg in dataset.get("read_groups",[]):
                if rg in barcodes.get("read_groups",{}):
                    barcodes["read_groups"][rg]["file_prefix"] = file_prefix
    except KeyError:
        return False

    barcode_list = []
    try:
        read_groups = barcodes.get("read_groups",{})
        for key,value in read_groups.iteritems():
            try:
                value["mean_read_length"] = str(int(float(value["total_bases"])/float(value["read_count"]))) + " bp"
            except:
                value["mean_read_length"] = "N/A"
            barcode_list.append(value)
        return sorted(barcode_list, key = lambda x: x.get('index',0))
    except KeyError:
        return False

def csv_barcodes_read(report):
    filename = 'alignment_barcode_summary.csv'
    csv_barcodes = []
    try:
        with open(os.path.join(report.get_report_dir() , filename), mode='rU') as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [key.replace(' ','_') for key in reader.fieldnames]
            d = list(reader)
        return d
    except:
        return False        

def basecaller_read(report):
    basecaller = load_json(report,"basecaller_results","BaseCaller.json")
    if not basecaller: return None

    bcd = {}
    #  final lib reads / total library isp
    try:
        bcd["total_reads"] = filtered_library_isps = basecaller["Filtering"]["LibraryReport"]["final_library_reads"]
        bcd["polyclonal"] = basecaller["Filtering"]["LibraryReport"]["filtered_polyclonal"]
        bcd["low_quality"] = basecaller["Filtering"]["LibraryReport"]["filtered_low_quality"]
        bcd["primer_dimer"] = basecaller["Filtering"]["LibraryReport"]["filtered_primer_dimer"]
        bcd["bases_called"] = basecaller["Filtering"]["BaseDetails"]["final"]

        return bcd
    except KeyError, IOError:
        #"Error parsing BaseCaller.json, the version of BaseCaller used for this report is too old."
        return None
    
def dataset_basecaller_read(report):
    """Read datasets_basecaller.json for read count data"""

    datasets_basecaller = load_json(report,"basecaller_results","datasets_basecaller.json")
    if not datasets_basecaller: return False

    bcd = {}
    # mappable output
    try:
        bcd["total_Q20_bases"] = 0
        bcd["total_Q0_bases"] = 0
        bcd["percent_Q20_bases"] = "N/A"
        for rg_id, rg_values in datasets_basecaller["read_groups"].items():
            bcd["total_Q20_bases"] += rg_values["Q20_bases"]
            bcd["total_Q0_bases"] += rg_values["total_bases"]
        if bcd["total_Q0_bases"]:
            bcd["percent_Q20_bases"] = (100 * bcd["total_Q20_bases"] / bcd["total_Q0_bases"])
    except KeyError as err:
        logger.error("datasets_basecaller.json format error: %s" % str(err))
    return bcd

def report_plan(report):
    if report.experiment.plan:
        plan = report.experiment.plan.pk
    else:
        plan = False
    return plan

def alignStats_read(report):
    """read the alignStats json file"""
    #alignStats = load_json(report, "" ,"alignTable.json")
    alignStats = load_json(report, "" ,"alignStats_err.json")
    if not alignStats: 
        return False
    else:
        return alignStats
    
# From beadDensityPlot.py
def getFormatForVal(value):
    if(numpy.isnan(value)): value = 0 #Value here doesn't really matter b/c printing nan will always result in -nan
    if(value==0): value=0.01
    precision = 2 - int(math.ceil(math.log10(value)))
    if(precision>4): precision=4;
    frmt = "%%0.%df" % precision
    return frmt
    
def report_status_read(report):
    if report.status == "Completed":
        return  False
    else:
        return report.status


def report_version(report):
    simple_version = re.compile(r"^(\d+\.?\d*)")
    try:
        versions = dict(v.split(':') for v in report.analysisVersion.split(",") if v)
        logger.debug(str(versions))
        version = simple_version.match(versions['db']).group(1)
    except Exception as err:
        try:
            with open(os.path.join(report.get_report_dir(), "version.txt")) as f:
                line = f.readline()
            key, version = line.strip().split("=")
        except IOError as err:
            logger.exception("version failure: '%s'" % report.analysisVersion)
            #just fail to 2.2
            version = "2.2"
    return version


def report_version_display(report):
    versiontxt_blacklist = set([
        "ion-onetouchupdater",
        "ion-pgmupdates",
        "ion-publishers",
        "ion-referencelibrary",
        "ion-sampledata",
        "ion-docs",
        "ion-tsconfig",
        "ion-rsmts",
        "ion-usbmount",
    ])
    explogtxt_whitelist  = [
        ("script_version", "Script"),
        ("liveview_version", "LiveView"),
        ("datacollect_version", "DataCollect"),
        ("oia_version", "OIA"),
        ("os_version", "OS"),
        ("graphics_version", "Graphics"),
    ]
    versions = []

    try:
        path = os.path.join(report.get_report_dir(), "version.txt")
        with open(path) as f:
            for line in f:
                item, version = line.strip().split('=')
                if item not in versiontxt_blacklist:
                    versions.append((item, version))
    except IOError as err:
        logger.error("Report %s could not open version.txt in %s" %(report, path))

    for key, label in explogtxt_whitelist:
        if key in report.experiment.log:
            value = report.experiment.log.get(key)
            if isinstance(value, list):
                value = ".".join(map(str, value))
            versions.append((label, value))
    return versions


def find_output_file_groups(report, datasets, barcodes):
    output_file_groups = []

    web_link = report.reportWebLink()
    report_path = report.get_report_dir() 
    if os.path.exists(report_path + "/download_links"):
        download_dir = "/download_links"
    else:
        download_dir = ""

    #Links
    prefix_tuple = (web_link, download_dir, report.experiment.expName, report.resultsName)
    
    current_group = {"name":"Library"}
    current_group["basecaller_bam"] = "%s%s/%s_%s.basecaller.bam" % prefix_tuple
    current_group["sff"]            = "%s%s/%s_%s.sff" % prefix_tuple
    current_group["fastq"]          = "%s%s/%s_%s.fastq" % prefix_tuple
    current_group["bam"]            = "%s%s/%s_%s.bam" % prefix_tuple
    current_group["bai"]            = "%s%s/%s_%s.bam.bai" % prefix_tuple
    output_file_groups.append(current_group)

    #Barcodes
    if report.resultsType != 'CombinedAlignments' and "barcode_config" in datasets:
        current_group = {"name":"Barcodes"}
        current_group["basecaller_bam"] = "%s%s/%s_%s.barcode.basecaller.bam.zip" % prefix_tuple
        current_group["sff"]            = "%s%s/%s_%s.barcode.sff.zip" % prefix_tuple
        current_group["fastq"]          = "%s%s/%s_%s.barcode.fastq.zip" % prefix_tuple
        current_group["bam"]            = "%s%s/%s_%s.barcode.bam.zip" % prefix_tuple
        current_group["bai"]            = "%s%s/%s_%s.barcode.bam.bai.zip" % prefix_tuple
        output_file_groups.append(current_group)

        # links for barcodes.html: mapped bam links if aligned to reference, unmapped otherwise
        for barcode in barcodes:
            if report.reference != 'none':
                barcode['bam_link'] = "%s%s/%s_%s_%s.bam" % (web_link, download_dir, barcode['file_prefix'].rstrip('_rawlib'), report.experiment.expName, report.resultsName)
                barcode['bai_link'] = "%s%s/%s_%s_%s.bam.bai" % (web_link, download_dir, barcode['file_prefix'].rstrip('_rawlib'), report.experiment.expName, report.resultsName)
            else:
                barcode['bam_link'] = "%s/basecaller_results/%s.basecaller.bam" % (web_link, barcode['file_prefix'])
                barcode['bai_link'] = "%s/basecaller_results/%s.basecaller.bam.bai" % (web_link, barcode['file_prefix'])            

    #Dim buttons if files don't exist
    for output_group in output_file_groups:
        keys = [k for k in output_group if k!='name']
        for key in keys:
            file_path = output_group[key].replace(web_link, report_path)
            output_group[key] = {"link": output_group[key], "exists": os.path.isfile(file_path)}

    return output_file_groups

@login_required
def report_display(request, report_pk):
    """Show the main report for an data analysis result.
    """
    report = get_object_or_404(models.Results, pk=report_pk)
    if report.reportStatus == ion_archiveResult.ARCHIVED:
        error = "report_archived"
        return HttpResponseRedirect(url_with_querystring(reverse('report_log', 
                                        kwargs={'pk':report_pk}), error=error))
    elif 'Error' in report.status:
        error = report.status
        return HttpResponseRedirect(url_with_querystring(reverse('report_log', 
                                        kwargs={'pk':report_pk}), error=error))

    version = report_version(report)
    if report.status == 'Completed' and version < "3.0":
        error = "old_report"
        return HttpResponseRedirect(url_with_querystring(reverse('report_log', 
                                        kwargs={'pk':report_pk}), error=error))

    experiment = report.experiment
    otherReports = report.experiment.results_set.exclude(pk=report_pk).order_by("-timeStamp")

    #the loading status
    report_status = report_status_read(report)

    plan = report_plan(report)
    try:
        reference = models.ReferenceGenome.objects.filter(short_name = report.reference).order_by("-index_version")[0]
    except IndexError, IOError:
        reference = False

    #find the major blocks from the important plugins
    major_plugins = {}
    has_major_plugins = False
    pluginList = models.PluginResult.objects.filter(result__pk=report_pk)
    for major_plugin in pluginList:
        if major_plugin.plugin.majorBlock:
            #list all of the _blocks for the major plugins, just use the first one
            try:
                majorPluginFiles = glob.glob(os.path.join(major_plugin.path(),"*_block.html"))[0]
                has_major_plugins = True
            except IndexError:
                majorPluginFiles = False

            major_plugins[major_plugin.plugin.name] = majorPluginFiles
    
    #TODO: encapuslate all vars into their parent block to make it easy to build the API maybe put
    #all of this in the model? 
    basecaller = basecaller_read(report)
    barcodes = barcodes_read(report)
    datasets = datasets_read(report)
    testfragments = testfragments_read(report)

    software_versions = report_version_display(report)

    # special case: combinedAlignments output doesn't have any basecaller results
    if report.resultsType and report.resultsType == 'CombinedAlignments':            
        report.experiment.expName = "CombineAlignments"            
        CA_barcodes = csv_barcodes_read(report)
        try:
            paramsJson = load_json(report, "" ,"ion_params_00.json")
            parents = [(pk,name) for pk,name in zip(paramsJson["parentIDs"],paramsJson["parentNames"])]
            CA_warnings = paramsJson.get("warnings","")
        except:
            logger.exception("Cannot read info from ion_params_00.json.")     

    beadfind = load_ini(report,"sigproc_results","analysis.bfmask.stats")
    alignStats = alignStats_read(report)
    dbr = dataset_basecaller_read(report)
    
    try:
        qcTypes = dict((qc.qcType.qcName, qc.threshold) for qc in 
                            report.experiment.plan.plannedexperimentqc_set.all())
    except:
        qcTypes = {}

    key_signal = "N/A"
    key_signal_threshold = qcTypes.get("Key Signal (1-100)", 0)
    try:
        f =  open(os.path.join(report.get_report_dir() , "raw_peak_signal"), mode='r')
        for line in f.readlines():
            if str(line).startswith("Library"):
                key_signal = str(line)[10:]
        f.close()
    except:
        pass

    # Beadfind
    try:
        bead_loading = 100 * float(beadfind["Bead Wells"]) / (float(beadfind["Total Wells"]) - float(beadfind["Excluded Wells"]))
        bead_loading = int(round(bead_loading))
        bead_loading_threshold = qcTypes.get("Bead Loading (%)", 0)

        beadsummary = {}
        beadsummary["total_addressable_wells"] = int(beadfind["Total Wells"]) - int(beadfind["Excluded Wells"])
        beadsummary["bead_wells"] = beadfind["Bead Wells"]
        beadsummary["p_bead_wells"] = percent(beadfind["Bead Wells"], beadsummary["total_addressable_wells"])
        beadsummary["live_beads"] = beadfind["Live Beads"]
        beadsummary["p_live_beads"] = percent(beadfind["Live Beads"], beadfind["Bead Wells"])
        beadsummary["test_fragment_beads"] = beadfind["Test Fragment Beads"]
        beadsummary["p_test_fragment_beads"] = percent(beadfind["Test Fragment Beads"], beadfind["Live Beads"])
        beadsummary["library_beads"] = beadfind["Library Beads"]
        beadsummary["p_library_beads"] = percent(beadfind["Library Beads"], beadfind["Live Beads"])
    except:
        logger.exception("Failed to build Beadfind report content.")

    #Basecaller
    try:
        usable_sequence = basecaller and int(round(100.0 * 
            float(basecaller["total_reads"]) / float(beadfind["Library Beads"])))
        usable_sequence_threshold = qcTypes.get("Usable Sequence (%)", 0)
        quality = load_ini(report,"basecaller_results","quality.summary")
        
        if float(dbr['total_Q0_bases']) > 0:
            mappable_output = int(round( float(alignStats["total_mapped_target_bases"]) / float(dbr['total_Q0_bases'])  * 100))
        else:
            mappable_output = 0
        
        basecaller["p_polyclonal"] = percent(basecaller["polyclonal"], beadfind["Library Beads"])
        basecaller["p_low_quality"] = percent(basecaller["low_quality"], beadfind["Library Beads"])
        basecaller["p_primer_dimer"] = percent(basecaller["primer_dimer"], beadfind["Library Beads"])
        basecaller["p_total_reads"] = percent(basecaller["total_reads"], beadfind["Library Beads"])
    except:
        logger.exception("Failed to build Basecaller report content.")       
    
    #Alignment
    try:
        if report.reference != 'none':
            if reference:
                avg_coverage_depth_of_target = round( float(alignStats["total_mapped_target_bases"]) / reference.genome_length(),1  )
                avg_coverage_depth_of_target = str(avg_coverage_depth_of_target) + "X"
            else:
                avg_coverage_depth_of_target = "N/A"
        
            if float(alignStats["accuracy_total_bases"]) > 0:
                raw_accuracy = round( (1 - float(alignStats["accuracy_total_errors"]) / float(alignStats["accuracy_total_bases"])) * 100, 1)
            else:
                raw_accuracy = 0.0
            alignment_ini = load_ini(report,".","alignment.summary")
            alignment = dict([(k.replace(' ','_'), v) for k, v in alignment_ini.items()])
        else:
            reference = 'none'
    except:
        logger.exception("Failed to build Alignment report content.")       

    class ProtonResultBlock:
        def __init__(self,directory,status_msg):
            self.directory = directory
            self.status_msg = status_msg

    try:
        if os.path.exists("/opt/ion/.ion-internal-server"):
            isInternalServer = True
        else:
            isInternalServer = False
    except:
        logger.exception("Failed to create isInternalServer variable")

    try:
        # TODO
        isThumbnail = report.metaData.get("thumb", False)
        if isInternalServer and report.experiment.log['blocks'] > 0 and not isThumbnail:
            proton_log_blocks = report.experiment.log['blocks']
            proton_block_tuples = []
            for b in proton_log_blocks:
                if not "thumbnail" in b:
                    xoffset = b.split(',')[0].strip()
                    yoffset = b.split(',')[1].strip()
                    directory = "%s_%s" % (xoffset,yoffset)
                    proton_block_tuples.append( (int(xoffset[1:]),int(yoffset[1:]),directory) )
            # sort by 2 keys
            sorted_block_tuples = sorted(proton_block_tuples, key=lambda block: (-block[1], block[0]))
            # get the directory names as a list
            pb = list(zip(*sorted_block_tuples)[2])

            proton_blocks = []
            for block in pb:
                blockdir = "block_"+block
                proton_blocks.append( ProtonResultBlock(blockdir, getBlockStatus(report,blockdir) ) )
    except:
        logger.exception("Failed to create proton block content.")  

    output_file_groups = [] 
    try:
        output_file_groups = find_output_file_groups(report, datasets, barcodes)
    except Exception as err:
        logger.exception("Could not generate output file links")

    noheader = request.GET.get("no_header",False)
    latex = request.GET.get("latex",False)
    noplugins = request.GET.get("noplugins",False)
    # This is both awesome and playing with fire.  Should be made explicit soon
    ctxd = locals()
    ctx = RequestContext(request, ctxd)
    if not latex:
        return render_to_response("rundb/reports/report.html", context_instance=ctx)
    else:
        return render_to_response("rundb/reports/printreport.html", context_instance=ctx)


@login_required
def report_log(request, pk):
    report = models.Results.objects.select_related('reportstorage').get(pk=pk)
    error = request.GET.get("error", None)
    contents = ""
    root_path = report.get_report_dir()
    log_data = []
    log_names = [
        "sigproc_results/sigproc.log",
        "basecaller_results/basecaller.log",
        "alignment.log",
        "ReportLog.html",
        "drmaa_stdout.txt",
        "drmaa_stdout_block.txt",
    ]
    report_link = report.reportWebLink()
    file_links = []
    if report.reportStatus != ion_archiveResult.ARCHIVED:
        file_links = [ (os.path.exists(os.path.join(root_path, p)), 
                        "%s/%s" % (report_link, p), t) for p, t in (
            ("Default_Report.php", "Classic Report"),
        )]
    file_links.extend( (os.path.exists(os.path.join(root_path, p)), 
                       reverse('report_metal', args=[report.pk, p]) , t) for p, t in (
        ("drmaa_stdout.txt", "TLScript: drmaa_stdout.txt"),
        ("drmaa_stdout_block.txt", "BlockTLScript: drmaa_stdout_block.txt"),
        ("drmaa_stderr_block.txt", "BlockTLScript: drmaa_stderr_block.txt"),
    ))

    for root, dirnames, filenames in os.walk(root_path):
        for filename in fnmatch.filter(filenames, 'drmaa_stderr_block.txt'):
            name = os.path.relpath(os.path.join(root, filename), root_path)
            log_names.append(name)

    log_paths = [os.path.join(root_path, name) for name in log_names]

    for name, path in zip(log_names, log_paths):
        if os.path.exists(path):
            file = file_browse.ellipsize_file(path)
            log = (name, file.read())
        else:
            log = (name, None)
        log_data.append(log)

    context = {
        "report": report,
        "report_link": report_link,
        "log_data" : log_data,
        "error" : error,
        "file_links": file_links,
    }

    return render_to_response("rundb/reports/report_log.html", 
        context, RequestContext(request))


def get_initial_arg(pk):
    """
    Builds the initial arg string for rerunning from wells
    """
    if int(pk) != 0:
        try:
            report = models.Results.objects.get(pk=pk)
            ret = report.get_report_dir()
            return ret
        except models.Results.DoesNotExist:
            return ""
    else:
        return ""


def get_project_names(rpf, exp):
    names = ''
    # get projects from form
    try:
      names = rpf.cleaned_data['project_names']
    except:
      pass  
    if len(names) > 1: return names
    # get projects from previous report
    if len(exp.sorted_results()) > 0: 
      names = exp.sorted_results()[0].projectNames()
    if len(names) > 1: return names           
    # get projects from Plan
    if exp.plan:
        try:
            names = [p.name for p in exp.plan.projects.all()]  
            names = ','.join(names)
        except:
            pass
        
    if len(names) > 1: return names    
    # last try: get from explog  
    try:
      names = exp.log['project']  
    except:
      pass
    return names


def create_tf_conf(tfConfig):
    """
    Build the contents of the report TF file (``DefaultTFs.conf``)
    using the contents of an uploaded file.
    """
    def tfs_from_db():
        """
        Build the contents of a report TF file (``DefaultTFs.conf``),
        using the TF template data stored in the database.
        """
        tfs = models.Template.objects.all()
        lines = ["%s,%s,%s" % (tf.name, tf.key, tf.sequence,) for tf in tfs if tf.isofficial]
        return lines
    fname = "DefaultTFs.conf"
    if tfConfig is not None:
        if tfConfig.size > 1024 * 1024 * 1024:
            raise ValueError("Uploaded TF config file too large (%d bytes)"
                             % tfConfig.size)
        buf = StringIO()
        for chunk in tfConfig.chunks():
            buf.write(chunk)
        ret = (fname, buf.getvalue())
        buf.close()
        return ret
    else:
        lines = tfs_from_db()
        return (fname, "\n".join(lines))


def get_plugins_dict(pg, plan={}, exclude_plugins = False):
    """
    Build a list containing dictionaries of plugin information.
    will only put the plugins in the list that are selected in the 
    interface
    """
    ret = []
    if len(pg) > 0:
        planPlugins = {}        
        if plan and 'selectedPlugins' in plan.keys():
          selectedPlugins = json.loads(plan['selectedPlugins'])
          selected = selectedPlugins.get('planplugins',[]) + selectedPlugins.get('planuploaders',[])          
          for pl in selected:
              planPlugins[pl['name']] = pl.get('userInput','')

        for p in pg:          
          if exclude_plugins: 
              #exclude plugins from launching automatically by pipeline
              #plugins must be marked autorun or selected during planning, otherwise excluded
              if not (p.autorun or p.name in planPlugins.keys()):
                  continue            
        
          params = {'name':p.name,
                'path':p.path,
                'version':p.version,
                'id':p.id,
                'autorun':p.autorun,
                'pluginconfig': json.dumps(p.config),
                'userInput':  planPlugins.get(p.name, '')
                } 
                
          for key in p.pluginsettings.keys():
              params[key] = p.pluginsettings[key]
          
          ret.append(params)   
    return ret


def get_default_cmdline_args(chipType):
    chips = models.Chip.objects.all()
    for c in chips:
        if chipType.startswith(c.name):
            beadfindArgs    = c.beadfindargs
            analysisArgs    = c.analysisargs
            basecallerArgs  = c.basecallerargs
            thumbnailBeadfindArgs   = c.thumbnailbeadfindargs
            thumbnailAnalysisArgs   = c.thumbnailanalysisargs
            thumbnailBasecallerArgs = c.thumbnailbasecallerargs
            break
    # loop finished without finding chipType: provide basic defaults
    else:
        beadfindArgs = thumbnailBeadfindArgs = 'justBeadFind'
        analysisArgs = thumbnailAnalysisArgs = 'Analysis'
        basecallerArgs = thumbnailBasecallerArgs = 'BaseCaller'        
    
    args = {
      'beadfindArgs':   beadfindArgs,
      'analysisArgs':   analysisArgs,      
      'basecallerArgs': basecallerArgs,
      'thumbnailBeadfindArgs':    thumbnailBeadfindArgs,
      'thumbnailAnalysisArgs':    thumbnailAnalysisArgs,
      'thumbnailBasecallerArgs':  thumbnailBasecallerArgs
    }
    return args


def makeParams(exp, beadfindArgs, analysisArgs, basecallerArgs, blockArgs, doThumbnail, resultsName, result, align_full, libraryKey,
                                url_path, aligner_opts_extra, mark_duplicates,
                                runid, previousReport,tfKey,
                                thumbnailBeadfindArgs, thumbnailAnalysisArgs, thumbnailBasecallerArgs, doBaseRecal):
    """Build a dictionary of analysis parameters, to be passed to the job
    server when instructing it to run a report.  Any information that a job
    will need to be run must be constructed here and included inside the return.  
    This includes any special instructions for flow control in the top level script."""
    gc = models.GlobalConfig.objects.all().order_by('id')[0]    
    pathToData = os.path.join(exp.expDir)
    if doThumbnail and exp.chipType == "900":
        pathToData = os.path.join(pathToData,'thumbnail')
    defaultLibKey = gc.default_library_key

    ##logger.debug("...views.makeParams() gc.default_library_key=%s;" % defaultLibKey)
    expName = exp.expName

    #get the exp data for sam metadata
    exp_filter = models.Experiment.objects.filter(pk=exp.pk)
    exp_json = serializers.serialize("json", exp_filter)
    exp_json = json.loads(exp_json)
    exp_json = exp_json[0]["fields"]

    #now get the plan and return that
    try:
        if exp.plan:
            planObj = [exp.plan]
        else:
            # Fallback to explog data... crawler should be setting this up

            #check plan's GUId in explog first
            planGUId = exp.log.get("planned_run_guid", {})
            if planGUId:
                planObj = models.PlannedExperiment.objects.filter(planGUID=planGUId)
            else:                                    
                planId = exp.log.get("pending_run_short_id",exp.log.get("planned_run_short_id", {}))
                if planId:
                    planObj = models.PlannedExperiment.objects.filter(planShortID=planId)
                    # Broken - ShortID is not unique once plan is used, may get wrong plan here.
                else:
                    planObj = []
    except: ## (KeyError, ValueError, TypeError):
        logger.exception("Failed to extract plan data from exp '%s'", exp.expName)
        planObj = []

    if planObj:
        plan_json = serializers.serialize("json", planObj)
        plan_json = json.loads(plan_json)
        plan = plan_json[0]["fields"]
    else:
        plan = {}

    try:
        pg = models.Plugin.objects.filter(selected=True,active=True).exclude(path='')
        plugins = get_plugins_dict(pg, plan, True)
    except:
        logger.exception("Failed to get list of active plugins")
        plugins = ""

    site_name = gc.site_name
    barcode_args = gc.barcode_args

    libraryName = result.reference

    skipchecksum = False
    fastqpath = result.fastqLink.strip().split('/')[-1]

    #TODO: remove the libKey from the analysis args, assign this in the TLScript. To make this more fluid

    #if the librayKey was set by createReport use that value. If not use the value from the PGM
    if not libraryKey:
        if exp.isReverseRun:
            libraryKey = exp.reverselibrarykey
        else:
            libraryKey = exp.libraryKey

    if libraryKey == None or len(libraryKey) < 1:
        libraryKey = defaultLibKey

    # floworder field sometimes has whitespace appended (?)  So strip it off
    flowOrder = exp.flowsInOrder.strip()
    # Set the default flow order if its not stored in the dbase.  Legacy support
    if flowOrder == '0' or flowOrder == None or flowOrder == '':
        flowOrder = "TACG"

    # Set the barcodeId
    if exp.barcodeId:
        barcodeId = exp.barcodeId
    else:
        barcodeId = ''
    project = ','.join(p.name for p in result.projects.all())
    sample = exp.sample
    
    # get barcoded sample names from Plan
    barcodeSamples = ''
    if barcodeId and plan:
        barcodeSamples = plan.get('barcodedSamples','')
    
    chipType = exp.chipType
    #net_location = gc.web_root
    #get the hostname try to get the name from global config first
    if gc.web_root:
        net_location = gc.web_root
    else:
        #if a hostname was not found in globalconfig.webroot then use what the system reports
        net_location = "http://" + str(socket.getfqdn())
    
    # Get the 3' adapter sequence
    adapterSequence = exp.forward3primeadapter
    if exp.isReverseRun:
        adapterSequence = exp.reverse3primeadapter
        
    try:
        adapter_primer_dicts = models.ThreePrimeadapter.objects.filter(sequence=adapterSequence)
    except:
        adapter_primer_dicts = None
        
    #the adapter_primer_dicts should not be empty or none
    if not adapter_primer_dicts or adapter_primer_dicts.count() == 0:
        if exp.isReverseRun:
            try:
                adapter_primer_dict = models.ThreePrimeadapter.objects.get(direction="Reverse", isDefault=True)
            except (models.ThreePrimeadapter.DoesNotExist,
                    models.ThreePrimeadapter.MultipleObjectsReturned):
                    
                #ok, there should be a default in db, but just in case... I'm keeping the previous logic for fail-safe
                adapter_primer_dict = {'name':'Reverse Ion Kit',
                                       'sequence':'CTGAGTCGGAGACACGCAGGGATGAGATGG',
                                       'direction': 'Reverse'
                                        }                
        else:     
            try:         
                adapter_primer_dict = models.ThreePrimeadapter.objects.get(direction="Forward", isDefault=True)
            except (models.ThreePrimeadapter.DoesNotExist,
                    models.ThreePrimeadapter.MultipleObjectsReturned):
                
                #ok, there should be a default in db, but just in case... I'm keeping the previous logic for fail-safe
                adapter_primer_dict = {'name':'Ion Kit',
                                       'sequence':'ATCACCGACTGCCCATAGAGAGGCTGAGAC',
                                       'direction': 'Forward'
                                        }
    else:
        adapter_primer_dict = adapter_primer_dicts[0]


    rawdatastyle = exp.rawdatastyle

    #if args are passed use them, if not use global defaults
    default_args = get_default_cmdline_args(exp.chipType)
    if doThumbnail:
        beadfindArgs = thumbnailBeadfindArgs if thumbnailBeadfindArgs else default_args['thumbnailBeadfindArgs']
        analysisArgs = thumbnailAnalysisArgs if thumbnailAnalysisArgs else default_args['thumbnailAnalysisArgs']
        basecallerArgs = thumbnailBasecallerArgs if thumbnailBasecallerArgs else default_args['thumbnailBasecallerArgs']
    else:
        beadfindArgs = beadfindArgs if beadfindArgs else default_args['beadfindArgs'] 
        analysisArgs = analysisArgs if analysisArgs else default_args['analysisArgs']
        basecallerArgs = basecallerArgs if basecallerArgs else default_args['basecallerArgs']
    
    ret = {'pathToData':pathToData,
           'beadfindArgs':beadfindArgs,
           'analysisArgs':analysisArgs,
           'basecallerArgs' : basecallerArgs,
           'blockArgs':blockArgs,
           'libraryName':libraryName,
           'resultsName':resultsName,
           'expName':expName,
           'libraryKey':libraryKey,
           'plugins':plugins,
           'fastqpath':fastqpath,
           'skipchecksum':skipchecksum,
           'flowOrder':flowOrder,
           'align_full' : align_full,
           'project':project,
           'sample':sample,
           'chiptype':chipType,
           'barcodeId':barcodeId,
           'barcodeSamples': barcodeSamples,
           'net_location':net_location,
           'exp_json': json.dumps(exp_json,cls=DjangoJSONEncoder),
           'site_name': site_name,
           'url_path':url_path,
           'reverse_primer_dict':adapter_primer_dict,
           'rawdatastyle':rawdatastyle,
           'aligner_opts_extra':aligner_opts_extra,
           'mark_duplicates' : mark_duplicates,
           'plan': plan,
           'flows':exp.flows,
           'pgmName':exp.pgmName,
           'isReverseRun':exp.isReverseRun,
           'barcode_args':json.dumps(barcode_args,cls=DjangoJSONEncoder),
           'tmap_version':settings.TMAP_VERSION,
           'runid':runid,
           'previousReport':previousReport,
           'tfKey': tfKey,
           'doThumbnail' : doThumbnail,
           'sam_parsed' : True if os.path.isfile('/opt/ion/.ion-internal-server') else False,
           'doBaseRecal':doBaseRecal,
    }
    
    return ret


def build_result(experiment, name, server, location):
    """Initialize a new `Results` object named ``name``
    representing an analysis of ``experiment``. ``server`` specifies
    the ``models.reportStorage`` for the location in which the report output
    will be stored, and ``location`` is the
    ``models.Location`` object for that file server's location.
    """
    # Final "" element forces trailing '/'
    # reportLink is used in calls to dirname, which would otherwise resolve to parent dir
    link = os.path.join(server.webServerPath, location.name, "%s_%%03d" % name, "")
    j = lambda l: os.path.join(link, l)
    storages = models.ReportStorage.objects.all()
    storage = storages.filter(default=True)[0]   #Select default ReportStorage obj.
        
    kwargs = {
        "experiment":experiment,
        "resultsName":name,
        "sffLink":j("%s_%s.sff" % (experiment, name)),
        "fastqLink": os.path.join(link,"basecaller_results", "%s_%s.fastq" % (experiment, name)),
        "reportLink": link, # Default_Report.php is implicit via Apache DirectoryIndex
        "status":"Pending", # Used to be "Started"
        "tfSffLink":j("%s_%s.tf.sff" % (experiment, name)),
        "tfFastq":"_",
        "log":j("log.html"),
        "analysisVersion":"_",
        "processedCycles":"0",
        "processedflows":"0",
        "framesProcessed":"0",
        "timeToComplete":0,
        "reportstorage":storage,
        }
    ret = models.Results(**kwargs)
    ret.save()
    for k, v in kwargs.iteritems():
        if hasattr(v, 'count') and v.count("%03d") == 1:
            v = v % ret.pk
            setattr(ret, k, v)
    ret.save()
    return ret


def _createReport(request, pk, reportpk):
    """
    Send a report to the job server.
    
    If ``createReport`` receives a `GET` request, it displays a form
    to the user.

    If ``createReport`` receives a `POST` request, it will attempt
    to validate a ``RunParamsForm``. If the form fails to validate, it
    re-displays the form to the user, with error messages explaining why
    the form did not validate (using standard Django form error messages).

    If the ``RunParamsForm`` is valid, ``createReport`` will go through
    the following process. If at any step the process fails, ``createReport``
    raises and then catches ``BailException``, which causes an error message
    to be displayed to the user.

    * Attempt to contact the job server. If this does not raise a socket
      error or an ``xmlrpclib.Fault`` exception, then ``createReport`` will
      check with job server to make sure the job server can write to the
      report's intended working directory.
    * If the user uploaded a template file (for use as ``DefaultTFs.conf``),
      then ``createReport`` will check that the file is under 1MB in size.
      If the file is too big, ``createReport`` bails.
    * Finally, ``createReport`` contacts the job server and instructs it
      to run the report.

    When contacting the job server, ``createReport`` will attempt to
    figure out where the appropriate job server is listening. First,
    ``createReport`` checks to see if these is an entry in
    ``settings.JOB_SERVERS`` for the report's location. If it doesn't
    find an entry in ``settings.JOB_SERVERS``, it attempts to connect
    to `127.0.0.1` on the port given by ``settings.JOBSERVER_PORT``.
    """
    def bail(result, err):
        result.status = err
        raise BailException(err)
    
    def create_bc_conf(barcodeId,fname):
        """
        Creates a barcodeList file for use in barcodeSplit binary.
        
        Danger here is if the database returns a blank, or no lines, then the
        file will be written with no entries.  The use of this empty file later
        will generate no fastq files, except for the nomatch.fastq file.
        
        See C source code BarCode.h for list of valid keywords
        """
        # Retrieve the list of barcodes associated with the given barcodeId
        db_barcodes = models.dnaBarcode.objects.filter(name=barcodeId).order_by("index")
        lines = []
        for db_barcode in db_barcodes:
            lines.append('barcode %d,%s,%s,%s,%s,%s,%d,%s' % (db_barcode.index,db_barcode.id_str,db_barcode.sequence,db_barcode.adapter,db_barcode.annotation,db_barcode.type,db_barcode.length,db_barcode.floworder))
        if db_barcodes:
            lines.insert(0,"file_id %s" % db_barcodes[0].name)
            lines.insert(1,"score_mode %s" % str(db_barcodes[0].score_mode))
            lines.insert(2,"score_cutoff %s" % str(db_barcodes[0].score_cutoff))
        return (fname, "\n".join(lines))

    def create_pk_conf(pk):
        """
        Build the contents of the report primary key file (``primary.key``).
        """
        text = "ResultsPK = %d" % pk
        return ("primary.key", text)
    
    class BailException(Exception):
        """
        Raised when an error is encountered with report creation. These errors
        may include failure to contact the job server, or attempting to create
        an analysis in a directory which can't be read from or written to
        by the job server.
        """
        def __init__(self, msg):
            super(BailException, self).__init__()
            self.msg = msg

    def flattenString(string):
        return string.replace("\n"," ").replace("\r"," ").strip()

    def create_meta(experiment, result):
    
        """Build the contents of a report metadata file (``expMeta.dat``)."""
        def get_chipcheck_status(exp):
            """
            Load the explog stored in the log field in the experiment
            table into a python dict.  Check if `calibratepassed` is set
            """
            data = exp.log
            if data.get('calibratepassed', 'Not Found'):
                return 'Passed'
            else:
                return 'Failed'    
        lines = ("Run Name = %s" % experiment.expName,
                 "Run Date = %s" % experiment.date,
                 "Run Cycles = %s" % experiment.cycles,
                 "Run Flows = %s" % experiment.flows,
                 "Project = %s" % ','.join(p.name for p in result.projects.all()),
                 "Sample = %s" % experiment.sample,
                 "Library = %s" % result.reference,
                 "PGM = %s" % experiment.pgmName,
                 "Flow Order = %s" % (experiment.flowsInOrder.strip() if experiment.flowsInOrder.strip() != '0' else 'TACG'),
                 "Library Key = %s" % (experiment.libraryKey if experiment.libraryKey != "" else experiment.reverselibrarykey),
                 "TF Key = %s" % "ATCG", # TODO, is it really unique?
                 "Chip Check = %s" % get_chipcheck_status(experiment),
                 "Chip Type = %s" % experiment.chipType,
                 "Chip Data = %s" % experiment.rawdatastyle,
                 "Notes = %s" % experiment.notes,
                 "Barcode Set = %s" % experiment.barcodeId,
                 "Analysis Name = %s" % result.resultsName,
                 "Analysis Date = %s" % date.today(),
                 "Analysis Flows = %s" % result.processedflows,
                 "runID = %s" % result.runid,
                 )
        return ('expMeta.dat', '\n'.join(lines))

    def create_runid(name):
        '''Returns 5 char string hashed from input string'''
        #Copied from TS/Analysis/file-io/ion_util.c
        def DEKHash(key):
            hash = len(key)
            for i in key:
                hash = ((hash << 5) ^ (hash >> 27)) ^ ord(i)
            return (hash & 0x7FFFFFFF)
    
        def base10to36(num):
            str = ''
            for i in range(5):
                digit=num % 36
                if digit < 26:
                    str = chr(ord('A') + digit) + str
                else:
                    str = chr(ord('0') + digit - 26) + str
                num /= 36
            return str
            
        return base10to36(DEKHash(name))     
       
    exp = get_object_or_404(models.Experiment, pk=pk)
    try:
        rig = models.Rig.objects.get(name=exp.pgmName)
        loc = rig.location
    except ObjectDoesNotExist:
        #If there is a rig try to use the location set by it
        loc = models.Location.objects.filter(defaultlocation=True)
        if not loc:
            #if there is not a default, just take the first one
            loc = models.Location.objects.all().order_by('pk')
        if loc:
            loc = loc[0]
        else:
            logger.critical("There are no Location objects, at all.")
            raise ObjectDoesNotExist("There are no Location objects, at all.")

    # Always use the default ReportStorage object
    storages = models.ReportStorage.objects.all()
    storage = storages.filter(default=True)[0]   #Select default ReportStorage obj.
    start_error = None
    
    javascript = ""
    
    # alignment reference
    references = [("none","none")]
    for r in models.ReferenceGenome.objects.filter(index_version=settings.TMAP_VERSION, enabled=True):
      references.append((r.short_name, r.short_name + " (" + r.name + ")"))        

    isProton = True if exp.chipType == "900" else False

    #get the list of report addresses
    resultList = models.Results.objects.filter(experiment=exp).order_by("timeStamp")
    previousReports = []
    previousThumbReports = []
    simple_version = re.compile(r"^(\d+\.?\d*)")
    for r in resultList:
        #try to get the version the major version the report was generated with
        try:
            versions = dict(v.split(':') for v in r.analysisVersion.split(",") if v)
            version = simple_version.match(versions['db']).group(1)
        except Exception:
            #just fail to 2.2
            version = "2.2"
        isThumbnail = r.metaData.get("thumb", False)
        result_choice = ( r.get_report_dir(), 
                        r.resultsName + " [" + str(r.get_report_dir()) + "]", 
                        version
        )
        if isThumbnail:
            previousThumbReports.append(result_choice)
        else:
            previousReports.append(result_choice)

    if request.method == 'POST':
        rpf = forms.RunParamsForm(request.POST, request.FILES)
        
        rpf.fields['previousReport'].widget.choices = previousReports
        rpf.fields['previousThumbReport'].widget.choices = previousThumbReports
        rpf.fields['reference'].widget.choices = references

        #send some js to the page
        previousReportDir = get_initial_arg(reportpk)
        if previousReportDir:
            rpf.fields['blockArgs'].initial = "fromWells"
            javascript = """
            $("#fromWells").click();
            """

        # validate the form
        if rpf.is_valid():
            chiptype_arg = exp.chipType
            ufResultsName = rpf.cleaned_data['report_name']
            resultsName = ufResultsName.strip().replace(' ', '_')

            result = build_result(exp, resultsName, storage, loc)

            webRootPath = result.web_root_path(loc)
            tfConfig = rpf.cleaned_data['tf_config']
            tfKey = rpf.cleaned_data['tfKey']
            blockArgs = rpf.cleaned_data['blockArgs']
            doThumbnail = rpf.cleaned_data['do_thumbnail']
            doBaseRecal = rpf.cleaned_data['do_base_recal']
            ts_job_type = ""
            if doThumbnail:
                ts_job_type = 'thumbnail'
                result.metaData["thumb"] = 1
                previousReport = rpf.cleaned_data['previousThumbReport']
            else:
                previousReport = rpf.cleaned_data['previousReport']

            beadfindArgs = flattenString(rpf.cleaned_data['beadfindArgs'])
            analysisArgs = flattenString(rpf.cleaned_data['analysisArgs'])
            basecallerArgs = flattenString(rpf.cleaned_data['basecallerArgs'])
            thumbnailBeadfindArgs = flattenString(rpf.cleaned_data['thumbnailBeadfindArgs'])
            thumbnailAnalysisArgs = flattenString(rpf.cleaned_data['thumbnailAnalysisArgs'])
            thumbnailBasecallerArgs = flattenString(rpf.cleaned_data['thumbnailBasecallerArgs'])

            #do a full alignment?
            align_full = True
            #If libraryKey was set, then override the value taken from the explog.txt on the PGM
            libraryKey = rpf.cleaned_data['libraryKey']
            #ionCrawler may modify the path to raw data in the path variable passed thru URL
            exp.expDir = rpf.cleaned_data['path']
            aligner_opts_extra = rpf.cleaned_data['aligner_opts_extra']
            mark_duplicates = rpf.cleaned_data['mark_duplicates']
            result.runid = create_runid(resultsName + "_" + str(result.pk))
            
            if rpf.cleaned_data['reference']:
              result.reference = rpf.cleaned_data['reference']
            else:
              result.reference = exp.library
            
            #attach project(s)
            projectNames = get_project_names(rpf, exp)                    
            username = request.user.username
            for name in projectNames.split(','):
              if name:                                  
                try:
                  p = models.Project.objects.get(name=name)            
                except models.Project.DoesNotExist:              
                  p = models.Project()
                  p.name = name    
                  p.creator = models.User.objects.get(username=username) 
                  p.save()
                  models.EventLog.objects.add_entry(p, "Created project name= %s during report creation." % p.name, request.user.username)  
                result.projects.add(p)
                models.EventLog.objects.add_entry(p, "Add result (%s) during report creation." % result.pk, request.user.username)  
            
            result.save()
            try:
                # Default control script definition
                scriptname='TLScript.py'
                
                scriptpath=os.path.join('/usr/lib/python2.6/dist-packages/ion/reports',scriptname)
                try:
                    with open(scriptpath,"r") as f:
                        script=f.read()
                except Exception as error:
                    bail(result,"Error reading %s\n%s" % (scriptpath,error.args))
                
                # check if path to raw data is there
                files = []
                try:
                    bk = models.Backup.objects.get(experiment=exp)
                except:
                    bk = False

                #------------------------------------------------
                # Tests to determine if raw data still available:
                #------------------------------------------------
                # Data directory is located on this server
                logger.debug("Start Analysis on %s" % exp.expDir)
                if ts_job_type == "thumbnail":
                    # thumbnail raw data is special in that despite there being a backup object for the dataset, thumbnail data is not deleted.
                    if rpf.cleaned_data['blockArgs'] != "fromWells" and rpf.cleaned_data['blockArgs'] != "fromSFF" and not os.path.exists(os.path.join(exp.expDir,'thumbnail')):
                        bail(result, "No path to raw data")
                else:
                    if bk and (rpf.cleaned_data['blockArgs'] != "fromWells" and rpf.cleaned_data['blockArgs'] != "fromSFF"):
                        if str(bk.backupPath) == 'DELETED':
                            bail(result, "The analysis cannot start because the raw data has been deleted.")
                            logger.warn("The analysis cannot start because the raw data has been deleted.")
                        else:
                            try:
                                datfiles = os.listdir(exp.expDir)
                                logger.debug("Got a list of files")
                            except:
                                logger.debug(traceback.format_exc())
                                bail(result,
                                     "The analysis cannot start because the raw data has been archived to %s.  Please mount that drive to make the data available." % (str(bk.backupPath),))
                    if rpf.cleaned_data['blockArgs'] != "fromWells" and rpf.cleaned_data['blockArgs'] != "fromSFF" and not os.path.exists(exp.expDir):
                        bail(result, "No path to raw data")
                
                try:
                    host = "127.0.0.1"
                    conn = client.connect(host, settings.JOBSERVER_PORT)
                    to_check = os.path.dirname(webRootPath)
                except (socket.error, xmlrpclib.Fault):
                    bail(result, "Failed to contact job server.")
                # prepare the directory in which the results' outputs will
                # be written
                # copy TF config to new path if it exists
                try:
                    files.append(create_tf_conf(tfConfig))
                except ValueError as ve:
                    bail(result, str(ve))
                # write meta data to folder for report
                files.append(create_meta(exp, result))
                files.append(create_pk_conf(result.pk))
                # write barcodes file to folder
                if exp.barcodeId and exp.barcodeId is not '':
                    files.append(create_bc_conf(exp.barcodeId,"barcodeList.txt"))
                # tell the analysis server to start the job
                params = makeParams(exp, beadfindArgs, analysisArgs, basecallerArgs, blockArgs, doThumbnail, resultsName, result, align_full, libraryKey,
                                                        os.path.join(storage.webServerPath, loc.name), aligner_opts_extra,
                                                        mark_duplicates, result.runid, previousReport, tfKey,
                                                        thumbnailBeadfindArgs, thumbnailAnalysisArgs, thumbnailBasecallerArgs, doBaseRecal)
                chip_dict = {}
                try:
                    chips = models.Chip.objects.all()
                    chip_dict = dict((c.name, '-pe ion_pe %s' % str(c.slots)) for c in chips)
                except:
                    chip_dict = {} # just in case we can't read from the db
                try:
                    conn.startanalysis(resultsName, script, params, files,
                                       webRootPath, result.pk, chiptype_arg, chip_dict, ts_job_type)
                except (socket.error, xmlrpclib.Fault):
                    bail(result, "Failed to contact job server.")
                # redirect the user to the report started page
                return result
            except BailException as be:
                start_error = be.msg
                logger.exception("Aborted createReport for result %d: '%s'", result.pk, start_error)
                result.delete()
    # fall through if not valid...

    if request.method == 'GET':

        rpf = forms.RunParamsForm()
        rpf.fields['path'].initial = os.path.join(exp.expDir)
        rpf.fields['align_full'].initial = True
        #rpf.fields['mark_duplicates'].initial = False

        #if there is a library Key for the exp use that instead of the default
        if exp.isReverseRun:
            if exp.reverselibrarykey:
                rpf.fields['libraryKey'].initial = exp.reverselibrarykey
        else:
            if exp.libraryKey:
                rpf.fields['libraryKey'].initial = exp.libraryKey

        # initialize with default cmdline arguments for Analysis and BaseCaller
        default_args = get_default_cmdline_args(exp.chipType)
        rpf.fields['beadfindArgs'].initial = default_args['beadfindArgs']
        rpf.fields['analysisArgs'].initial = default_args['analysisArgs']
        rpf.fields['basecallerArgs'].initial = default_args['basecallerArgs']
        rpf.fields['thumbnailBeadfindArgs'].initial = default_args['thumbnailBeadfindArgs']
        rpf.fields['thumbnailAnalysisArgs'].initial = default_args['thumbnailAnalysisArgs']
        rpf.fields['thumbnailBasecallerArgs'].initial = default_args['thumbnailBasecallerArgs']

        rpf.fields['previousReport'].widget.choices = previousReports
        rpf.fields['previousThumbReport'].widget.choices = previousThumbReports
        rpf.fields['reference'].widget.choices = references        
        rpf.fields['reference'].initial = exp.library        
        rpf.fields['project_names'].initial = get_project_names(rpf, exp)
        rpf.fields['do_base_recal'].initial = models.GlobalConfig.objects.all()[0].base_recalibrate
        
        #send some js to the page
        previousReportDir = get_initial_arg(reportpk)
        if previousReportDir:
            rpf.fields['blockArgs'].initial = "fromWells"
            javascript = """
            $("#fromWells").click();
            """
            javascript += '$("#id_previousReport").val("'+previousReportDir +'");'


    ctx = {"rpf": rpf, "expName":exp.pretty_print_no_space, "start_error":start_error, "javascript" : javascript,
           "isProton":isProton, "pk":pk, "reportpk":reportpk, "isexpDir": os.path.exists(exp.expDir)}
    ctx = RequestContext(request, ctx)
    return ctx


def _report_started(request, pk):
    """
    Inform the user if a report sent to the job server was successfully
    started.
    """
    try:
        pk = int(pk)
    except (TypeError, ValueError):
        return http.HttpResponseNotFound()
    result = get_object_or_404(models.Results, pk=pk)
    report = result.reportLink
    log = os.path.join(os.path.dirname(result.reportLink), "log.html")
    ctxd = {"name":result.resultsName, "pk":result.pk,
            "link":report, "log":log,
            "status":result.status}
    ctx = RequestContext(request, ctxd)
    return ctx


@login_required
@csrf_exempt
def analyze(request, exp_pk, report_pk):
    result = _createReport(request, exp_pk, report_pk)
    if isinstance(result, RequestContext):
        return render_to_response("rundb/reports/analyze.html",
                                        context_instance=result)
    if (request.method == 'POST'):
        ctx = _report_started(request, result.pk)
        return render_to_response("rundb/reports/analysis_started.html",
                                        context_instance=ctx)


def show_png_image(full_path):
    return HttpResponse(open(full_path, 'rb'), mimetype='image/png')


def show_csv(full_path):
    reader = csv.reader(open(full_path))
    return render_to_response("rundb/reports/metal/show_csv.html", {
        "table_rows" : reader,
    })


def show_whitespace_csv(full_path):
    reader = ((c for c in r.strip().split()) for r in open(full_path))
    return render_to_response("rundb/reports/metal/show_csv.html", {
        "table_rows" : reader,
    })


def show_config(full_path):
    parser = ConfigParser.RawConfigParser()
    try:
        parser.read(full_path)
        reader = ((section, parser.items(section)) for section in parser.sections())
    except ConfigParser.MissingSectionHeaderError as err:
        reader = [(" ", (l.split('=', 1) for l in open(full_path)))]
    return render_to_response("rundb/reports/metal/show_config.html", {
        "config": reader,
    })


def indent_json(full_path):
    json_obj = json.load(open(full_path))
    formatted_json = json.dumps(json_obj, sort_keys=True, indent=4)
    return HttpResponse(formatted_json, mimetype='text/plain')


BINARY_FILTER=''.join([
        chr(x) if 32 <= x <= 126 else '.' for x in range(256)
    ])


def show_binary(full_path, max_read=100000):
    data = open(full_path, 'rb').read(max_read)
    length = 8
    result = StringIO()
    for i in xrange(0, len(data), length):
       row = data[i:i+length]
       hexa = ' '.join(["%02X" % ord(x) for x in row])
       printable = row.translate(BINARY_FILTER)
       result.write("%05X   %-*s   %s\n" % (i, length*3, hexa, printable))
    if len(data) == max_read:
        result.write("\n\n%s\nTruncated after %d bytes" % (full_path, max_read))
    else:
        result.write("\n\n%s\nEnd of file at %d bytes" % (full_path, len(data)))
    result.seek(0)
    return HttpResponse(result, mimetype='text/plain')


def plain_text(full_path):
    return HttpResponse(FileWrapper(open(full_path, 'rb')), mimetype='text/plain')



# These handlers are in strict priority order, i.e. as soon as one matches
# the search for a match halts.  More specific patterns must preceed more
# general patterns that might incorrectly match them.
FILE_HANDLERS = [ (re.compile(r), h) for r, h in (
    (r'barcodeFilter\.txt$', show_csv),
    (r'flowQVtable\.txt$', show_csv),
    (r'alignmentQC_out\.txt$', plain_text),
    (r'processParameters.txt$', show_config),
    (r'expMeta.dat$', show_config),

    (r'basecaller_results/.*\.txt$', show_whitespace_csv),
    (r'sigproc_results/.*\.txt$', show_whitespace_csv),
    (r'\.dat$', show_whitespace_csv),

    (r'\.bin$', show_binary),
    (r'\.bam$', show_binary),
    (r'\.bai$', show_binary),
    (r'\.h5$', show_binary),
    (r'\.sff$', show_binary),
    (r'\.wells$', show_binary),
    
    (r'\.stats$', show_config),
    (r'\.summary$', show_config),
    
    (r'\.png$', show_png_image),
    (r'\.csv$', show_csv),
    (r'\.json$', indent_json),
    (r'.', plain_text),
)]


def show_file(report, pk, path, root, full_path):
    for pattern, handle in FILE_HANDLERS:
        if pattern.search(path):
            try:
                return handle(full_path)
            except Exception as err:
                raise err


def show_directory(report, pk, path, root, full_path):
    breadcrumbs = file_browse.bread_crumb_path(path)
    dirs, files = file_browse.list_directory(full_path)
    dirs.sort()
    files.sort()
    dir_info, file_info = [], []
    for name, full_dir_path, stat in dirs:
        dir_path = os.path.join(path, name)
        date = datetime.fromtimestamp(stat.st_mtime)
        size = file_browse.dir_size(full_dir_path)
        dir_info.append((name, dir_path, date, size))
    for name, full_file_path, stat in files:
        file_path = os.path.join(path, name)
        date = datetime.fromtimestamp(stat.st_mtime)
        size = file_browse.format_units(stat.st_size)
        file_info.append((name, file_path, date, size))

    return render_to_response("rundb/reports/metal.html", {
        "report": report,
        "root": root,
        "path": path,
        "full_path": full_path,
        "breadcrumbs": breadcrumbs,
        "dirs": dir_info,
        "files": file_info
    })


@login_required
def metal(request, pk, path):
    path = path.strip('/')
    report = get_object_or_404(models.Results, pk=pk)
    root = report.get_report_dir()
    full_path = os.path.join(root, path)
    if os.path.isdir(full_path):
        return show_directory(report, pk, path, root, full_path)
    else:
        return show_file(report, pk, path, root, full_path)


@login_required
def report_action(request, pk, action):
    logger.info("report_action: request '%s' on report: %s" % (action, pk))

    if request.method != "POST":
        return http.HttpResponseNotAllowed(['POST'])

    comment = request.POST.get("comment", "No Comment")
    async_task_result = None
    if action == 'Z':
        ret = get_object_or_404(models.Results, pk=pk)
        ret.autoExempt = not ret.autoExempt
        ret.save()
    elif action == 'A':
        async_task_result = iondb.backup.tasks.archive_report.delay(request.user, pk, comment)
    elif action == 'E':
        async_task_result = iondb.backup.tasks.export_report.delay(request.user, pk, comment)
    elif action == "P":
        async_task_result = iondb.backup.tasks.prune_report.delay(request.user, pk, comment)
    #elif action == "D":
    #    proxy.delete_report(pk, comment)
    #    async_task_result = delete_report.delay(request.user, pk, comment)
    if async_task_result: 
        logger.info(async_task_result)

    return http.HttpResponse()
