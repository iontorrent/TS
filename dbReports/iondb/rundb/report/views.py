# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.contrib.auth.decorators import login_required
from django.template import RequestContext
from iondb.rundb import models
from iondb.rundb import forms
from iondb.utils import makePDF
from iondb.anaserve import client
from ion.utils import makeCSA
from django import http
import json
import os
import csv
import re
import socket
import traceback
import fnmatch
import glob
import ast
import file_browse
import xmlrpclib
import numpy
import math
import urllib
import ConfigParser
import logging
import subprocess
from cStringIO import StringIO
from django.db import transaction
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponsePermanentRedirect, HttpResponseRedirect, HttpResponse, StreamingHttpResponse, HttpResponseServerError
from django.core import serializers
from django.core.serializers.json import DjangoJSONEncoder
from django.core.exceptions import ObjectDoesNotExist
from django.core.servers.basehttp import FileWrapper
from django.core.urlresolvers import reverse
from django.shortcuts import get_object_or_404, redirect, render_to_response
from django.conf import settings
from datetime import datetime, date
from iondb.plugins.launch_utils import get_plugins_dict
from iondb.rundb.data import dmactions_types

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
        # This will create a file named report.pdf in results directory
        makePDF.write_report_pdf(pk)

        csaFullPath = makeCSA.makeCSA(reportDir,rawDataDir)
        csaFileName = os.path.basename(csaFullPath)

        response = http.HttpResponse(FileWrapper (open(csaFullPath)), mimetype='application/zip')
        response['Content-Disposition'] = 'attachment; filename=%s' % csaFileName

    except:
        logger.error(traceback.format_exc())
        response = http.HttpResponse(status=500)

    return response


@login_required
def get_summary_pdf(request, pk):
    '''Report Page PDF + Plugins Pages PDF'''
    ret = get_object_or_404(models.Results, pk=pk)
    filename = "%s"%ret.resultsName
    pdf_file_contents = makePDF.get_summary_pdf(pk)
    if pdf_file_contents:
        response = http.HttpResponse(pdf_file_contents, mimetype="application/pdf")
        response['Content-Disposition'] = 'attachment; filename=' + ret.resultsName + '-full.pdf'
    else:
        return HttpResponseRedirect(url_with_querystring(reverse('report', args=[pk]), nosummarypdf="True"))


@login_required
def get_report_pdf(request, pk):
    '''Report Page PDF'''
    ret = get_object_or_404(models.Results, pk=pk)
    pdf_file_contents = makePDF.get_report_pdf(pk)
    if pdf_file_contents:
        response = http.HttpResponse(pdf_file_contents, mimetype="application/pdf")
        response['Content-Disposition'] = 'attachment; filename=' + ret.resultsName + '.pdf'
        return response
    else:
        return HttpResponseRedirect(url_with_querystring(reverse('report', args=[pk]), nosummarypdf="True"))


@login_required
def get_plugin_pdf(request, pk):
    '''Plugin Pages PDF'''
    ret = get_object_or_404(models.Results, pk=pk)
    pdf_file_contents = makePDF.get_plugin_pdf(pk)
    if pdf_file_contents:
        response = http.HttpResponse(pdf_file_contents, mimetype="application/pdf")
        response['Content-Disposition'] = 'attachment; filename=' + ret.resultsName + '-plugins.pdf'
        return response
    else:
        return HttpResponseRedirect(url_with_querystring(reverse('report', args=[pk]), noplugins="True"))


def percent(q, d):
    return "%04.1f%%" % (100 * float(q) / float(d))


def load_ini(report,subpath,filename,namespace="global"):
    parse = ConfigParser.ConfigParser()
    parse.optionxform = str # preserve the case
    report_dir = report.get_report_dir()
    try:
        if os.path.exists(os.path.join(report_dir, subpath , filename)):
            parse.read(os.path.join(report_dir, subpath , filename))
        else:
            # try in top dir
            parse.read(os.path.join(report_dir, filename))
        parse = parse._sections.copy()
        return parse[namespace]
    except:
        return False


def getBlockStatus(report,blockdir,namespace="global"):

    STATUS = ""
    try:
        
        analysis_return_code = os.path.join(report.get_report_dir(), blockdir, 'sigproc_results', 'analysis_return_code.txt')
        if os.path.exists(analysis_return_code):
            with open(analysis_return_code, 'r') as f:
                text = f.read()
                if not '0' in text:
                    STATUS = "Analysis error %s" % text
                    return STATUS


        try:
            f = open(os.path.join(report.get_report_dir(), blockdir, 'blockstatus.txt'))
            text = f.readlines()
            f.close()
    
            for line in text:
                [component, status] = line.split('=')
                if int(status) != 0:
                    if component == 'Beadfind':
                        if int(status) == 2:
                            STATUS = 'Checksum Error'
                        elif int(status) == 3:
                            STATUS = 'No Live Beads'
                        else:
                            STATUS = "Beadfind error %s" % int(status)
                    elif component == 'Analysis':
                        if int(status) == 2:
                            STATUS = 'Checksum Error'
                        elif int(status) == 3:
                            STATUS = 'No Live Beads'
                        else:
                            STATUS = "Analysis error %s" % int(status)
                    elif component == 'Recalibration':
                        STATUS = "Skip Recal."
                    else:
                        STATUS = "Error in %s" % component

        except IOError:
            STATUS = "Transfer..."

        old_progress_file = os.path.join(report.get_report_dir(), blockdir, 'progress.txt')
        if STATUS == "" and os.path.exists(old_progress_file):
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

    except:
        STATUS = "Exception"
        logger.exception(traceback.format_exc())

    return STATUS


def load_json(report, *subpaths):
    """shortcut to load the json"""
    path = os.path.join(report.get_report_dir(), *subpaths)
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            logger.exception("Failed to read JSON: %s" % path)
    return None


def testfragments_read(report):
    testfragments = load_json(report,"basecaller_results","TFStats.json")
    if not testfragments:
        return None

    try:
        for tf_name,tf_data in testfragments.iteritems():
            num_reads = int(tf_data.get("Num",0))
            num_50AQ17 = int(tf_data.get("50Q17",0))
            conversion_50AQ17 = "N/A"
            
            #since 100Q17 is a new attribute in TFStats.json, old file will not have this attribute
            is_100Q17_key_found = False
            if "100Q17" in tf_data.keys():
                num_100AQ17 = int(tf_data.get("100Q17",0))
                is_100Q17_key_found = True
            conversion_100AQ17 = "---"
            
            if num_reads > 0:
                conversion_50AQ17 = (100*num_50AQ17/num_reads)
                if is_100Q17_key_found:
                    conversion_100AQ17 = (100*num_100AQ17/num_reads)
                    if conversion_100AQ17 < 1:
                        conversion_100AQ17 = "---"
            testfragments[tf_name]["conversion_50AQ17"] = conversion_50AQ17
            testfragments[tf_name]["conversion_100AQ17"] = conversion_100AQ17
            testfragments[tf_name]["histogram_filename"] = "new_Q17_%s.png" % tf_name
            testfragments[tf_name]["num_reads"] = num_reads

    except KeyError:
        pass

    return testfragments

def get_barcodes(datasets):
    """get the list of barcodes"""
    barcode_list = []
    if not datasets:
        return []

    try:
        for dataset in datasets.get("datasets",[]):
            file_prefix = dataset["file_prefix"]
            for rg in dataset.get("read_groups",[]):
                if rg in datasets.get("read_groups",{}):
                    datasets["read_groups"][rg]["file_prefix"] = file_prefix
    
        for key,value in datasets.get("read_groups",{}).iteritems():
            if value.get('filtered', False):
                continue
            try:
                value["mean_read_length"] = "%d bp" % round(float(value["total_bases"])/float(value["read_count"]))
            except:
                value["mean_read_length"] = "N/A"
            barcode_list.append(value)
        return sorted(barcode_list, key = lambda x: x.get('index',0))

    except KeyError:
        return []

def csv_barcodes_read(report):
    def convert_to_long_or_float(value):
        try:
            v = long(value)
            return v
        except:
            try:
                v = float(value)
                return v
            except:
                return value
    def convert_values_to_long_or_float(dict):
        for row in d:
            for k in row:
                if k:
                    #logger.info('"%s" "%s"' % (k, row[k]))
                    row[k] = convert_to_long_or_float(row[k])
    filename = 'alignment_barcode_summary.csv'
    csv_barcodes = []
    try:
        with open(os.path.join(report.get_report_dir() , filename), mode='rU') as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [key.replace(' ','_') for key in reader.fieldnames]
            d = list(reader)
        convert_values_to_long_or_float(d)
        return d
    except:
        return {}

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

def report_plan(report):
    if report.experiment.plan:
        plan = report.experiment.plan.pk
    else:
        plan = False
    return plan


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

    if report.experiment.chefPackageVer:
        versions.append(("Ion_Chef", report.experiment.chefPackageVer))
    
    return versions


def report_chef_display(report):
    """ 
    Returns data to be displayed in Chef Summary tab
    """
    chef_info = []

    if not report.experiment.chefInstrumentName:
        return chef_info
     
    chef_infoList  = [
        ("chefInstrumentName", "Chef Instrument Name"),
        ("chefSamplePos", "Sample Position"),
        ("chefTipRackBarcode", "Tip Rack Barcode"),
        ("chefChipType1", "Chip Type 1"),
        ("chefChipType2", "Chip Type 2"),           
        ("chefChipExpiration1", "Chip Expiration 1"),
        ("chefChipExpiration2", "Chip Expiration 2"),    
        #("chefLotNumber", "Lot Number"),
        #("chefManufactureDate", "Manufacturing Date"),            
        ("chefKitType", "Templating Kit Type"),
        #("chefReagentID", "Reagent Id"),        
        ("chefReagentsExpiration", "Reagent Expiration"),          
        ("chefReagentsLot", "Reagent Lot Number"),
        ("chefReagentsPart", "Reagent Part Number"),
        ("chefSolutionsLot", "Solution Lot Number"),
        ("chefSolutionsPart", "Solution Part Number"), 
        ("chefSolutionsExpiration", "Solution Expiration"),             
        ("chefPackageVer", "Chef Version"),
        ("chefLastUpdate", "Chef Software Last Update Date"),        
    ]
    
    
    for key, label in chef_infoList:
        value = getattr(report.experiment, key)

        if key in ["chefChipType1", 'chefChipType2']:
            chips = models.Chip.objects.filter(name = value)
            if chips:
                value = chips[0].description
                
        chef_info.append((label, value))
    
    return chef_info


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
    current_group["vcf"]            = "%s/plugin_out/variantCaller_out/TSVC_variants.vcf.gz" % (web_link,)
    output_file_groups.append(current_group)

    #Barcodes
    if datasets and "barcode_config" in datasets:
        # links for barcodes.html: mapped bam links if aligned to reference, unmapped otherwise
        for barcode in barcodes:
            barcode['basecaller_bam_link'] = "%s/basecaller_results/%s.basecaller.bam" % (web_link, barcode['file_prefix'])
            if report.eas.reference or report.eas.get_barcoded_samples_reference_names():
                barcode['bam_link'] = "%s%s/%s_%s_%s.bam" % (web_link, download_dir, re.sub('_rawlib$', '', barcode['file_prefix']), report.experiment.expName, report.resultsName)
                barcode['bai_link'] = "%s%s/%s_%s_%s.bam.bai" % (web_link, download_dir, re.sub('_rawlib$', '', barcode['file_prefix']), report.experiment.expName, report.resultsName)
            else:
                barcode['bam_link'] = None
                barcode['bai_link'] = "%s/basecaller_results/%s.basecaller.bam.bai" % (web_link, barcode['file_prefix'])

            barcode['vcf_link'] = None
            for key in ['basecaller_bam_link', 'bam_link', 'bai_link']:
                if not (barcode[key] and os.path.exists(barcode[key].replace(web_link, report_path))):
                    barcode[key] = None

    #Dim buttons if files don't exist
    for output_group in output_file_groups:
        keys = [k for k in output_group if k!='name']
        for key in keys:
            file_path = output_group[key].replace(web_link, report_path)
            output_group[key] = {"link": output_group[key], "exists": os.path.isfile(file_path)}

    return output_file_groups


def get_recalibration_panel(datasets):
    panel_recal = []
    if datasets and "IonControl" in datasets:
        sorted_groups = sorted(datasets['IonControl']['read_groups'].values(), key=lambda d: d.get('barcode_name', None))
        panel_recal = sorted_groups
    return panel_recal



def find_source_files(report, files, subfolders):
    source_files = {}
    report_dir = report.get_report_dir()
    reportWebLink = report.reportWebLink()
    for filename in files:
        for folder in subfolders:
            if os.path.isfile(os.path.join(report_dir,folder,filename)):
                source_files[filename] =  os.path.normpath(os.path.join(reportWebLink,folder,filename))
                break
    return source_files


def ionstats_histogram_median(data):
    cumulative_reads = numpy.cumsum(data)
    half = cumulative_reads[-1] / 2.0
    median_index = numpy.searchsorted(cumulative_reads, half, 'right')
    return median_index


def ionstats_histogram_mode(data):
    return numpy.argmax(data)


def ionstats_histogram_mean(full):
    bases = float(full.get('num_bases', None))
    reads = full.get('num_reads', None)
    if bases is None or reads is None:
        return None
    elif reads == 0 or bases == 0:
        return 0
    return bases / reads


def ionstats_compute_stats(stats):
    data = stats.get("read_length_histogram", None)
    read_stats = {
        'mean_length': stats and ionstats_histogram_mean(stats),
        'median_length': data and ionstats_histogram_median(data),
        'mode_length': data and ionstats_histogram_mode(data),
    }
    return read_stats


def ionstats_read_stats(report):
    ionstats = load_json(report, "basecaller_results", "ionstats_basecaller.json") or {}
    full = ionstats.get("full", {})
    return ionstats_compute_stats(full)


def _report_context(request, report_pk):
    """Show the main report for an data analysis result.
    """
    # QUIRK
    #
    # Set the JSON Encoder FLOAT Formatter to avoid float misrepresentation. Our version of Python 2.6.5 , see http://stackoverflow.com/a/1447581
    # >>> repr(float('3.88'))
    # '3.8799999999999999'
    # instead we want
    # '3.88'
    #
    # QUIRK
    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.15g')

    # Each of these is fetched later, listing here avoids extra query
    report_extra_tables = [ 'experiment', 'experiment__plan', 'experiment__samples', 'eas', 'libmetrics']
    qs = models.Results.objects.select_related(*report_extra_tables)
    report = get_object_or_404(qs, pk=report_pk)
    globalconfig = models.GlobalConfig.get()

    noheader = request.GET.get("no_header",False)
    latex = request.GET.get("latex",False)
    noplugins = request.GET.get("noplugins",False)

    error = None
    if not latex:
        dmfilestat = report.get_filestat(dmactions_types.OUT)
        if dmfilestat.isarchived():
            error = "report_archived"
        elif dmfilestat.isdeleted():
            error = "report_deleted"
        elif "User Aborted" == report.experiment.ftpStatus:
            error = "user_aborted"
        elif 'Error' in report.status:
            error = report.status
        elif report.status == 'Completed' and report_version(report) < "3.0":
            error = "old_report"
    if error is not None:
        return error, None


    experiment = report.experiment
    otherReports = report.experiment.results_set.exclude(pk=report_pk).order_by("-timeStamp")

    #the loading status
    report_status = report_status_read(report)

    plan = report_plan(report)

    #find the major blocks from the important plugins
    major_plugins = {}
    major_plugins_images = {}
    has_major_plugins = False
    pluginList = report.pluginresult_set.all().select_related('result', 'result__reportstorage', 'plugin')
    for major_plugin in pluginList:
        if major_plugin.plugin.majorBlock:
            #list all of the _blocks for the major plugins, just use the first one
            try:
                majorPluginFiles = glob.glob(os.path.join(major_plugin.path(),"*_block.html"))[0]
                majorPluginImages = glob.glob(os.path.join(report.get_report_dir() , "pdf",
                                    "slice_" + major_plugin.plugin.name + "*"))
                has_major_plugins = True
            except IndexError:
                majorPluginFiles = False
                majorPluginImages = False

            major_plugins[major_plugin.plugin.name] = majorPluginFiles
            if majorPluginImages:
                major_plugins_images[major_plugin.plugin.name] = sorted(majorPluginImages)
            else:
                major_plugins_images[major_plugin.plugin.name] = majorPluginImages

    #TODO: encapuslate all vars into their parent block to make it easy to build the API maybe put
    #all of this in the model?
    basecaller = basecaller_read(report)        # basecaller_results/BaseCaller.json
    read_stats = ionstats_read_stats(report)    # basecaller_results/ionstats_basecaller.json
    datasets = load_json(report, "basecaller_results", "datasets_basecaller.json")
    testfragments = testfragments_read(report)  # basecaller_results/TFStats.json
    beadfind = load_ini(report,"sigproc_results","analysis.bfmask.stats")
    software_versions = report_version_display(report)  # version.txt
    chef_info = report_chef_display(report)     # chef info
    
    # special case: combinedAlignments output doesn't have any basecaller results
    if report.resultsType and report.resultsType == 'CombinedAlignments':
        report.experiment.expName = "CombineAlignments"

        CA_barcodes_json = []
        try:
            CA_barcodes_json_path = os.path.join(report.get_report_dir(), 'CA_barcode_summary.json')
            if os.path.exists(CA_barcodes_json_path):
                CA_barcodes_json = json.load(open(CA_barcodes_json_path))
            else:
                # compatibility <TS3.6
                CA_barcodes = csv_barcodes_read(report)
                for CA_barcode in CA_barcodes:
                    CA_barcodes_json.append({
                        "barcode_name": CA_barcode["ID"],
                        "AQ7_num_bases": CA_barcode["Filtered_Mapped_Bases_in_Q7_Alignments"],
                        "full_num_reads": CA_barcode["Total_number_of_Reads"],
                        "AQ7_mean_read_length": CA_barcode["Filtered_Q7_Mean_Alignment_Length"]
                    })
        except:
            pass
        CA_barcodes_json = json.dumps(CA_barcodes_json)

        try:
            paramsJson = load_json(report ,"ion_params_00.json")
            parents = [(pk,name) for pk,name in zip(paramsJson["parentIDs"],paramsJson["parentNames"])]
            CA_warnings = paramsJson.get("warnings","")
        except:
            logger.exception("Cannot read info from ion_params_00.json.")


    try:
        qcTypes = dict(qc for qc in
                       report.experiment.plan.plannedexperimentqc_set.all().values_list('qcType__qcName', 'threshold'))
    except:
        qcTypes = {}

    key_signal_threshold = qcTypes.get("Key Signal (1-100)", 0)

    # Beadfind
    try:
        if "Adjusted Addressable Wells" in beadfind:
            addressable_wells = int(beadfind["Adjusted Addressable Wells"])
        else:
            addressable_wells = int(beadfind["Total Wells"]) - int(beadfind["Excluded Wells"])

        bead_loading = 100 * float(beadfind["Bead Wells"]) / float(addressable_wells)
        bead_loading = int(round(bead_loading))
        bead_loading_threshold = qcTypes.get("Bead Loading (%)", 0)

        beadsummary = {}
        beadsummary["total_addressable_wells"] = addressable_wells
        beadsummary["bead_wells"] = beadfind["Bead Wells"]
        beadsummary["p_bead_wells"] = percent(beadfind["Bead Wells"], beadsummary["total_addressable_wells"])
        beadsummary["live_beads"] = beadfind["Live Beads"]
        beadsummary["p_live_beads"] = percent(beadfind["Live Beads"], beadfind["Bead Wells"])
        beadsummary["test_fragment_beads"] = beadfind["Test Fragment Beads"]
        beadsummary["p_test_fragment_beads"] = percent(beadfind["Test Fragment Beads"], beadfind["Live Beads"])
        beadsummary["library_beads"] = beadfind["Library Beads"]
        beadsummary["p_library_beads"] = percent(beadfind["Library Beads"], beadfind["Live Beads"])
    except:
        logger.warn("Failed to build Beadfind report content for %s." % report.resultsName)

    #Basecaller
    try:
        usable_sequence = basecaller and int(round(100.0 *
            float(basecaller["total_reads"]) / float(beadfind["Library Beads"])))
        usable_sequence_threshold = qcTypes.get("Usable Sequence (%)", 0)
        #quality = load_ini(report,"basecaller_results","quality.summary")

        basecaller["p_polyclonal"] = percent(basecaller["polyclonal"], beadfind["Library Beads"])
        basecaller["p_low_quality"] = percent(basecaller["low_quality"], beadfind["Library Beads"])
        basecaller["p_primer_dimer"] = percent(basecaller["primer_dimer"], beadfind["Library Beads"])
        basecaller["p_total_reads"] = percent(basecaller["total_reads"], beadfind["Library Beads"])
    except:
        logger.warn("Failed to build Basecaller report content for %s." % report.resultsName)


    # Special alignment backward compatibility code
    ionstats_alignment = load_json(report,"ionstats_alignment.json")

    if not ionstats_alignment:
        ionstats_alignment = {}
        alignStats = load_json(report, "alignStats_err.json")
        alignment_ini = load_ini(report,".","alignment.summary")
        if alignStats and alignment_ini:
            ionstats_alignment['aligned'] = {'num_bases' : alignStats["total_mapped_target_bases"] }
            ionstats_alignment['error_by_position'] = [alignStats["accuracy_total_errors"],]
            ionstats_alignment['AQ17'] = {'num_bases'           : alignment_ini["Filtered Mapped Bases in Q17 Alignments"],
                                          'mean_read_length'    : alignment_ini["Filtered Q17 Mean Alignment Length"],
                                          'max_read_length'     : alignment_ini["Filtered Q17 Longest Alignment"]}
            ionstats_alignment['AQ20'] = {'num_bases'           : alignment_ini["Filtered Mapped Bases in Q20 Alignments"],
                                          'mean_read_length'    : alignment_ini["Filtered Q20 Mean Alignment Length"],
                                          'max_read_length'     : alignment_ini["Filtered Q20 Longest Alignment"]}
            ionstats_alignment['AQ47'] = {'num_bases'           : alignment_ini["Filtered Mapped Bases in Q47 Alignments"],
                                          'mean_read_length'    : alignment_ini["Filtered Q47 Mean Alignment Length"],
                                          'max_read_length'     : alignment_ini["Filtered Q47 Longest Alignment"]}
        del alignStats
        del alignment_ini


    eas_reference = report.eas.reference
    barcodedSamples_reference_names = ""
    barcodedSamples_reference_name_count = 0
    if not eas_reference:
        barcodedSamples_reference_names = report.eas.get_barcoded_samples_reference_names()
        if barcodedSamples_reference_names:
            eas_reference = barcodedSamples_reference_names[0]
            barcodedSamples_reference_name_count = len(barcodedSamples_reference_names)
            
    #Alignment
    try:
        reference = models.ReferenceGenome.objects.filter(short_name = eas_reference).order_by("-index_version")[0]
        genome_length = reference.genome_length()
    except (IndexError, IOError):
        reference = report.eas.reference if report.eas.reference != 'none' else ''
        genome_length = None

    if reference:
        try:
            if genome_length:
                avg_coverage_depth_of_target = round( float(ionstats_alignment['aligned']['num_bases']) / genome_length, 1  )
                avg_coverage_depth_of_target = str(avg_coverage_depth_of_target) + "X"
                for c in ['AQ17', 'AQ20', 'AQ47']:
                    try:
                        ionstats_alignment[c]['mean_coverage'] = round( float(ionstats_alignment[c]['num_bases']) / genome_length, 1  )
                    except:
                        pass

            if float(ionstats_alignment['aligned']['num_bases']) > 0:
                raw_accuracy = round( (1 - float(sum(ionstats_alignment['error_by_position'])) / float(ionstats_alignment['aligned']['num_bases'])) * 100.0, 1)
            else:
                raw_accuracy = 0.0

            try:
                for c in ['aligned']:
                    ionstats_alignment[c]['p_num_reads'] = 100.0 * ionstats_alignment[c]['num_reads'] / float(ionstats_alignment['full']['num_reads'])
                    ionstats_alignment[c]['p_num_bases'] = 100.0 * ionstats_alignment[c]['num_bases'] / float(ionstats_alignment['full']['num_reads'])

                ionstats_alignment['unaligned'] = {}
                ionstats_alignment['unaligned']['num_reads'] = int(ionstats_alignment['full']['num_reads']) - int(ionstats_alignment['aligned']['num_reads'])
                # close enough, and ensures they sum to 100 despite rounding
                ionstats_alignment['unaligned']['p_num_reads'] = 100.0 - ionstats_alignment['aligned']['p_num_reads']
                #ionstats_alignment['unaligned']['p_num_reads'] = 100.0 * ionstats_alignment['unaligned']['num_reads']) / float(ionstats_alignment['full']['num_reads']
            except:
                logger.exception("Failed to compute percent alignment values")
        except:
            logger.warn("Failed to build Alignment report content for %s." % report.resultsName)

    duplicate_metrics = {}
    if report.libmetrics and report.libmetrics.duplicate_reads is not None:
        duplicate_metrics['duplicate_reads'] = report.libmetrics.duplicate_reads
        if report.libmetrics.totalNumReads:
            duplicate_metrics['duplicate_read_percentage'] = 100.0 * report.libmetrics.duplicate_reads / float(report.libmetrics.totalNumReads)
        else:
            # totalNumReads is None or 0
            duplicate_metrics['duplicate_read_percentage'] = None


    class ProtonResultBlock:
        def __init__(self,directory,status_msg):
            self.directory = directory
            self.status_msg = status_msg

    try:
        isInternalServer = os.path.exists("/opt/ion/.ion-internal-server")
    except:
        logger.exception("Failed to create isInternalServer variable")

    try:
        # TODO
        if isInternalServer and len(report.experiment.log.get('blocks','')) > 0 and not report.isThumbnail:
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
    except Exception as err:
        logger.exception("Failed to create proton block content for report [%s]." % report.get_report_dir())



    # Barcodes and Output
    barcodes = get_barcodes(datasets)
    output_file_groups = []
    try:
        output_file_groups = find_output_file_groups(report, datasets, barcodes)
        # Convert the barcodes to JSON for use by Kendo UI Grid
        barcodes_json = json.dumps(barcodes)
    except Exception as err:
        logger.exception("Could not generate output file links")

    # This is both awesome and playing with fire.  Should be made explicit soon
    # This is the list of keys returned by locals in ordinary GET request
    # [
    #    "ProtonResultBlock", "addressable_wells", "avg_coverage_depth_of_target", "barcodes", "barcodes_json",
    #    "basecaller", "bead_loading", "bead_loading_threshold", "beadfind", "beadsummary", "c", "datasets",
    #    "dmfilestat", "duplicate_metrics", "encoder", "error", "experiment", "genome_length", "globalconfig",
    #    "has_major_plugins", "ionstats_alignment", "isInternalServer", "key_signal_threshold", "latex",
    #    "major_plugins", "major_plugins_images", "noheader", "noplugins", "otherReports", "output_file_groups", "plan",
    #    "pluginList", "qcTypes", "qs", "raw_accuracy", "read_stats", "reference", "report", "report_extra_tables",
    #    "report_pk", "report_status", "request", "software_versions", "testfragments", "usable_sequence",
    #    "usable_sequence_threshold"
    # ]

    # This is where we collect all local variables in the current scope for
    # our template context.  This is bad.
    context = locals()

    # On this line and below, we'll start explicitly adding only that which
    # we actually require for the report and it's different sections
    context['panel_recal'] = get_recalibration_panel(datasets)

    return None, context


@login_required
def report_display(request, report_pk):
    latex = request.GET.get("latex", False)

    error, ctxd = _report_context(request, report_pk)
    if error is not None:
        return HttpResponseRedirect(url_with_querystring(reverse('report_log',
                                            kwargs={'pk':report_pk}), error=error))
    ctx = RequestContext(request, ctxd)
    if not latex:
        return render_to_response("rundb/reports/report.html", context_instance=ctx)
    else:
        return render_to_response("rundb/reports/printreport.tex", context_instance=ctx)


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
    report_php = os.path.exists(os.path.join(root_path,"Default_Report.php"))
    report_pdf = os.path.exists(os.path.join(root_path,"backupPDF.pdf"))

    file_links = []
    dmfilestat = report.get_filestat(dmactions_types.OUT)
    if not dmfilestat.isdisposed():
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
            #file = file_browse.ellipsize_file(path)
            #instead of load the file and pushing it through Django, load it with jQuery
            log = (name, name)
        else:
            log = (name, None)
        log_data.append(log)

    archive_files = {}
    listdir = os.listdir(root_path) if os.path.exists(root_path) else []
    for filename in listdir:
        if filename.endswith(".support.zip"):
            archive_files["csa"] = filename
        # Following should be mutually exclusive; but prefer latter if both exist
        if filename.endswith("backupPDF.pdf"):
            archive_files["report_pdf"] = filename
        if filename.endswith("-full.pdf"):
            archive_files["report_pdf"] = filename

    archive_restore = False
    if dmfilestat.action_state == 'AD':
        serialized_json_path = os.path.join(dmfilestat.archivepath, "serialized_%s.json" % report.resultsName)
        if os.path.exists(serialized_json_path):
            archive_restore = json.dumps([{
                "name": report.resultsName,
                "Output Files": serialized_json_path,
                "copy_report": "on"
            }])

    context = {
        "report": report,
        "report_link": report_link,
        "report_pdf": report_pdf,
        "report_php": report_php,
        "log_data" : log_data,
        "error" : error,
        "file_links": file_links,
        "archive_files": archive_files,
        "archive_restore": archive_restore,
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


def get_project_names(exp, names=''):
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
    if tfConfig:
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


def get_default_cmdline_args(plan):
    if plan:
        args = plan.get_default_cmdline_args()
    else:
        args = {
                'beadfindargs':   'justBeadFind',
                'analysisargs':   'Analysis',
                'basecallerargs': 'BaseCaller',
                'prebasecallerargs': 'BaseCaller',
                'calibrateargs': 'calibrate',
                'alignmentargs': '',
                'thumbnailbeadfindargs':    'justBeadFind',
                'thumbnailanalysisargs':    'Analysis',
                'thumbnailbasecallerargs':  'BaseCaller',
                'prethumbnailbasecallerargs':  'BaseCaller',
                'thumbnailcalibrateargs': 'calibrate',
                'thumbnailalignmentargs': ''
        }
    return args

def make_barcodeInfo(eas, exp, doBaseRecal):
    # Generate a table of per-barcode info for pipeline use
    barcodeInfo = {}
    barcodeId = eas.barcodeKitName if eas.barcodeKitName else ''
    no_bc_sample = exp.get_sample() if not barcodeId else 'none'
    
    barcodeInfo['no_barcode'] = {
        'sample': no_bc_sample or 'none',
        'referenceName': eas.reference,
        'calibrate': False if barcodeId else doBaseRecal
    }
    
    if barcodeId:
        for barcode in models.dnaBarcode.objects.filter(name=barcodeId).values('index','id_str','sequence','adapter'):
            barcodeInfo[barcode['id_str']] = barcode
            barcodeInfo[barcode['id_str']]['sample'] = 'none'
            barcodeInfo[barcode['id_str']]['referenceName'] = eas.reference
            barcodeInfo[barcode['id_str']]['calibrate'] = doBaseRecal
        
        if eas.barcodedSamples:
            for sample, value in eas.barcodedSamples.items():
                try:
                    info = value.get('barcodeSampleInfo',{})
                    dna_rna_sample = set([v.get('nucleotideType','') for v in info.values()]) == set(['DNA','RNA'])
                    for bcId in value['barcodes']:
                        barcodeInfo[bcId]['sample'] = sample
                        
                        if 'reference' in info.get(bcId,{}):
                            barcodeInfo[bcId]['referenceName'] = info[bcId]['reference']
                        
                        # exclude RNA barcodes from recalibration (Compendia project RNA/DNA sample)
                        if dna_rna_sample and info.get(bcId,{}).get('nucleotideType','') == 'RNA':
                            barcodeInfo[bcId]['calibrate'] = False
                except:
                    pass
                
    return barcodeInfo

def makeParams(exp, result, blockArgs, doThumbnail, align_full,
                                url_path, mark_duplicates,
                                pathToData, previousReport, plugins_list,
                                doBaseRecal, realign, username):
    """Build a dictionary of analysis parameters, to be passed to the job
    server when instructing it to run a report.  Any information that a job
    will need to be run must be constructed here and included inside the return.
    This includes any special instructions for flow control in the top level script."""

    # defaults from GlobalConfig
    gc = models.GlobalConfig.get()
    site_name = gc.site_name
    barcode_args = gc.barcode_args
    #get the hostname try to get the name from global config first
    if gc.web_root:
        net_location = gc.web_root
    else:
        #if a hostname was not found in globalconfig.webroot then use what the system reports
        net_location = "http://" + str(socket.getfqdn())

    # floworder field sometimes has whitespace appended (?)  So strip it off
    flowOrder = exp.flowsInOrder.strip()
    # Set the default flow order if its not stored in the dbase.  Legacy support
    if flowOrder == '0' or flowOrder == None or flowOrder == '':
        flowOrder = "TACG"

    #get the exp data for sam metadata
    exp_json = serializers.serialize("json", [exp])
    exp_json = json.loads(exp_json)
    exp_json = exp_json[0]["fields"]

    # ExperimentAnalysisSettings
    eas = result.eas
    eas_json = serializers.serialize("json", [eas])
    eas_json = json.loads(eas_json)
    eas_json = eas_json[0]["fields"]

    # Get the 3' adapter sequence
    adapterSequence = eas.threePrimeAdapter
    try:
        adapter_primer_dict = models.ThreePrimeadapter.objects.filter(sequence=adapterSequence)[0]
    except:
        adapter_primer_dict = None

    #the adapter_primer_dicts should not be empty or none
    if not adapter_primer_dict:
        try:
            adapter_primer_dict = models.ThreePrimeadapter.objects.get(direction="Forward", isDefault=True)
        except (models.ThreePrimeadapter.DoesNotExist,
                models.ThreePrimeadapter.MultipleObjectsReturned):

            #ok, there should be a default in db, but just in case... I'm keeping the previous logic for fail-safe
            adapter_primer_dict = {'name':'Ion Kit',
                                   'sequence':'ATCACCGACTGCCCATAGAGAGGCTGAGAC',
                                   'direction': 'Forward'
                                    }

    barcodeId = eas.barcodeKitName if eas.barcodeKitName else ''

    # Plan
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
        # add SampleSet name to be passed to plugins
        if planObj[0].sampleSet:
            plan['sampleSet_name'] = planObj[0].sampleSet.displayedName

    else:
        plan = {}

    # Plugins
    plugins = get_plugins_dict(plugins_list, eas.selectedPlugins)

    # Samples
    sampleInfo = {}
    for sample in exp.samples.all():
        sampleInfo[sample.name] = {
            'name': sample.name,
            'displayedName': sample.displayedName,
            'externalId': sample.externalId,
            'description': sample.description,
            'attributes': {}
        }
        for attributeValue in sample.sampleAttributeValues.all():
            sampleInfo[sample.name]['attributes'][attributeValue.sampleAttribute.displayedName] = attributeValue.value

    
    barcodedSamples_reference_names = eas.get_barcoded_samples_reference_names()
    #logger.debug("report.views.makeParams() barcodedSamples_reference_names=%s" %(barcodedSamples_reference_names))

    #use barcodedSamples' selected reference if NO plan default reference is specified
    reference = eas.reference
    if not eas.reference and barcodedSamples_reference_names:
        reference = barcodedSamples_reference_names[0]
        
    if doThumbnail:
        beadfindArgs = eas.thumbnailbeadfindargs
        analysisArgs = eas.thumbnailanalysisargs
        basecallerArgs = eas.thumbnailbasecallerargs
        prebasecallerArgs = eas.prethumbnailbasecallerargs
        recalibArgs = eas.thumbnailcalibrateargs
        alignmentArgs = eas.thumbnailalignmentargs
    else:
        beadfindArgs = eas.beadfindargs
        analysisArgs = eas.analysisargs
        basecallerArgs = eas.basecallerargs
        prebasecallerArgs = eas.prebasecallerargs
        recalibArgs = eas.calibrateargs
        alignmentArgs = eas.alignmentargs

    ret = {'pathToData':pathToData,
           'beadfindArgs':beadfindArgs,
           'analysisArgs':analysisArgs,
           'prebasecallerArgs' : prebasecallerArgs,
           'basecallerArgs' : basecallerArgs,
           'blockArgs':blockArgs,
           'referenceName':reference,
           'resultsName':result.resultsName,
           'expName': exp.expName,
           'libraryKey':eas.libraryKey,
           'tfKey': eas.tfKey,
           'plugins':plugins,
           'fastqpath': result.fastqLink.strip().split('/')[-1],
           'skipchecksum': False,
           'flowOrder':flowOrder,
           'align_full' : align_full,
           'project': ','.join(p.name for p in result.projects.all()),
           'sample': exp.get_sample(),
           'chiptype':exp.chipType,
           'barcodeId': barcodeId,
           'barcodeSamples': json.dumps(eas.barcodedSamples,cls=DjangoJSONEncoder) if barcodeId else "{}",
           "barcodeSamples_referenceNames" : barcodedSamples_reference_names,           
           'net_location':net_location,
           'exp_json': json.dumps(exp_json,cls=DjangoJSONEncoder),
           'site_name': site_name,
           'url_path':url_path,
           'reverse_primer_dict':adapter_primer_dict,
           'rawdatastyle':exp.rawdatastyle,
           'aligner_opts_extra':alignmentArgs,
           'mark_duplicates' : mark_duplicates,
           'plan': plan,
           'flows':exp.flows,
           'pgmName':exp.pgmName,
           'barcode_args':json.dumps(barcode_args,cls=DjangoJSONEncoder),
           'tmap_version':settings.TMAP_VERSION,
           'runid': result.runid,
           'previousReport':previousReport,
           'doThumbnail' : doThumbnail,
           'sam_parsed' : True if os.path.isfile('/opt/ion/.ion-internal-server') else False,
           'doBaseRecal':doBaseRecal,
           'realign':realign,
           'experimentAnalysisSettings': eas_json,
           'username':username,
           'recalibArgs': recalibArgs,
           'sampleInfo': sampleInfo,
           'barcodeInfo': make_barcodeInfo(eas, exp, doBaseRecal)
    }

    return ret

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

def build_result(experiment, resultsName, server, location, doThumbnail=False):
    """Initialize a new `Results` object named ``name``
    representing an analysis of ``experiment``. ``server`` specifies
    the ``models.reportStorage`` for the location in which the report output
    will be stored, and ``location`` is the
    ``models.Location`` object for that file server's location.
    """
    # Final "" element forces trailing '/'
    # reportLink is used in calls to dirname, which would otherwise resolve to parent dir
    link = os.path.join(server.webServerPath, location.name, "%s_%%03d" % resultsName, "")
    j = lambda l: os.path.join(link, l)

    kwargs = {
        "experiment":experiment,
        "resultsName":resultsName,
        "sffLink":j("%s_%s.sff" % (experiment, resultsName)),
        "fastqLink": os.path.join(link,"basecaller_results", "%s_%s.fastq" % (experiment, resultsName)),
        "reportLink": link, # Default_Report.php is implicit via Apache DirectoryIndex
        "status":"Pending", # Used to be "Started"
        "tfSffLink":j("%s_%s.tf.sff" % (experiment, resultsName)),
        "tfFastq":"_",
        "log":j("log.html"),
        "analysisVersion":"_",
        "processedCycles":"0",
        "processedflows":"0",
        "framesProcessed":"0",
        "timeToComplete":0,
        "reportstorage":server,
        }
    result = models.Results(**kwargs)
    result.save() # generate the pk

    result.runid = create_runid(resultsName + "_" + str(result.pk))
    if doThumbnail:
        result.metaData["thumb"] = 1

    # What does this do?
    for k, v in kwargs.iteritems():
        if hasattr(v, 'count') and v.count("%03d") == 1:
            v = v % result.pk
            setattr(result, k, v)

    result.save()
    return result

def update_experiment_analysis_settings(eas, **kwargs):
    """
    Check whether ExperimentAnalysisSettings need to be updated:
    if settings were changed on re-analysis page save a new EAS with isOneTimeOverride = True
    """

    analysisArgs = ['beadfindargs','thumbnailbeadfindargs','analysisargs','thumbnailanalysisargs','prebasecallerargs','prethumbnailbasecallerargs',
                    'calibrateargs', 'thumbnailcalibrateargs','basecallerargs','thumbnailbasecallerargs','alignmentargs','thumbnailalignmentargs']

    override = False
    fill_in_args = True

    for key, new_value in kwargs.items():
        value = getattr(eas,key)
        if key in analysisArgs and value:
            fill_in_args = False
        if value != new_value:
            setattr(eas,key,new_value)
            if key in analysisArgs:
                # set isOneTimeOverride only if old args were not blank
                if value:
                    override = True
                else:
                    pass
            else:
                override = True

    if override:
        eas.isOneTimeOverride = True
        eas.isEditable = False
        eas.date = datetime.now()
        eas.pk = None # this will create a new instance
        eas.save()
    elif fill_in_args:
        # special case to reduce duplication when args were not saved with Plan
        # if no change other than filling in empty args: create new EAS and allow it to be reused
        eas.isOneTimeOverride = False
        eas.isEditable = False
        eas.date = datetime.now()
        eas.pk = None # this will create a new instance
        eas.save()

    return eas

def _createReport(exp, eas, resultsName, **kwargs):
    """
    Create result and send to the job server.
    Re-analyze page and crawler POST both end up here.

    * Attempt to contact the job server. If this does not raise a socket
      error or an ``xmlrpclib.Fault`` exception, then ``createReport`` will
      check with job server to make sure the job server can write to the
      report's intended working directory.
    * If the user uploaded a template file (for use as ``DefaultTFs.conf``),
      then ``createReport`` will check that the file is under 1MB in size.
      If the file is too big, ``createReport`` bails.
    * Finally, ``createReport`` contacts the job server and instructs it
      to run the report.
    """
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
                 "Run Flows = %s" % experiment.flows,
                 "Project = %s" % ','.join(p.name for p in result.projects.all()),
                 "Sample = %s" % experiment.get_sample(),
                 "Library = N/A",
                 "Reference = %s" % result.eas.reference,
                 "Instrument = %s" % experiment.pgmName,
                 "Flow Order = %s" % (experiment.flowsInOrder.strip() if experiment.flowsInOrder.strip() != '0' else 'TACG'),
                 "Library Key = %s" % result.eas.libraryKey,
                 "TF Key = %s" % result.eas.tfKey,
                 "Chip Check = %s" % get_chipcheck_status(experiment),
                 "Chip Type = %s" % experiment.chipType,
                 "Chip Data = %s" % experiment.rawdatastyle,
                 "Notes = %s" % experiment.notes,
                 "Barcode Set = %s" % result.eas.barcodeKitName,
                 "Analysis Name = %s" % result.resultsName,
                 "Analysis Date = %s" % date.today(),
                 "Analysis Flows = %s" % result.processedflows,
                 "runID = %s" % result.runid,
                 )

        return ('expMeta.dat', '\n'.join(lines))


    # inputs:
    blockArgs = kwargs.get('blockArgs','fromRaw')
    doThumbnail = kwargs.get('do_thumbnail', False)
    previousReport = kwargs.get('previousThumbReport','') if doThumbnail else kwargs.get('previousReport','')
    username = kwargs.get('username','')
    tfConfig = kwargs.get('tf_config')
    project_names = kwargs.get('project_names','')
    plugins_list = kwargs.get('plugins',[])

    mark_duplicates = kwargs['mark_duplicates'] if ('mark_duplicates' in kwargs) else eas.isDuplicateReads
    doBaseRecal = kwargs['do_base_recal'] if ('do_base_recal' in kwargs) else eas.base_recalibration_mode    
    realign= kwargs['realign'] if ('realign' in kwargs) else eas.realign

    #do a full alignment?
    align_full = True

    loc = exp.location()
    if not loc:
        raise ObjectDoesNotExist("There are no Location objects, at all.")

    # Always use the default ReportStorage object
    storage = models.ReportStorage.objects.filter(default=True)[0]

    try:
        with transaction.atomic():
            result = build_result(exp, resultsName, storage, loc, doThumbnail)
        
            # make sure we have a set of cmdline args for analysis
            if doThumbnail:
                have_args = bool(eas.thumbnailbeadfindargs) and bool(eas.thumbnailanalysisargs) and bool(eas.thumbnailbasecallerargs) and bool(eas.thumbnailcalibrateargs)
            else:
                have_args = bool(eas.beadfindargs) and bool(eas.analysisargs) and bool(eas.basecallerargs) and bool(eas.calibrateargs)
            if not have_args:
                default_args = get_default_cmdline_args(exp.plan)
                for key,value in default_args.items():
                    if not getattr(eas,key):
                        setattr(eas, key, value)
                eas.save()
        
            # Don't allow EAS to be edited once analysis has started
            if eas.isEditable:
                eas.isEditable = False
                eas.save()
        
            result.eas = eas
            result.reference = eas.reference
        
            #attach project(s)
            projectNames = get_project_names(exp, project_names)
            for name in projectNames.split(','):
                if name:
                    try:
                        p = models.Project.objects.get(name=name)
                    except models.Project.DoesNotExist:
                        p = models.Project()
                        p.name = name
                        p.creator = models.User.objects.get(username='ionadmin')
                        p.save()
                        models.EventLog.objects.add_entry(p, "Created project name= %s during report creation." % p.name, 'ionadmin')
                    result.projects.add(p)
                    models.EventLog.objects.add_entry(p, "Add result (%s) during report creation." % result.pk, username)
        
            result.save()
    
    except Exception as e:
        logger.exception("Aborted createReport for result %d: '%s'", result.pk, e)
        raise
    
    try:
        # Default control script definition
        scriptname='TLScript.py'

        from distutils.sysconfig import get_python_lib;
        python_lib_path=get_python_lib()
        scriptpath=os.path.join(python_lib_path,'ion/reports',scriptname)
        try:
            with open(scriptpath,"r") as f:
                script=f.read()
        except Exception as error:
            raise Exception("Error reading %s\n%s" % (scriptpath,error.args))

        pathToData = os.path.join(exp.expDir)
        if doThumbnail:
            pathToData = os.path.join(pathToData,'thumbnail')

        # Determine if data has been archived or deleted
        if blockArgs == "fromWells":
            dmactions_type = dmactions_types.BASE
            dmfilestat = result.get_filestat(dmactions_type)
            selected_previous_pk = None
            try:
                selected_previous_pk = int(previousReport.strip('/').split('_')[-1])
            except ValueError:
                # TorrentSuiteCloud plugin 3.4.2 uses reportName for this value
                previous_obj = models.Results.objects.filter(resultsName = os.path.basename(previousReport))
                if previous_obj:
                    selected_previous_pk = previous_obj[0].pk
            if selected_previous_pk:
                dmfilestat = models.Results.objects.get(pk=selected_previous_pk).get_filestat(dmactions_type)
                # replace dmfilestat
                result.dmfilestat_set.filter(dmfileset__type=dmactions_types.BASE).delete()
                dmfilestat.pk = None
                dmfilestat.result = result
                dmfilestat.save()
        else:
            dmactions_type = dmactions_types.SIG
            dmfilestat = result.get_filestat(dmactions_type)

        if dmfilestat:
            if dmfilestat.action_state in ['DG','DD']:
                raise Exception("Analysis cannot start because %s data has been deleted." % dmactions_type)
            elif dmfilestat.action_state in ['AG','AD']:
                # replace paths with archived locations
                try:
                    datfiles = os.listdir(dmfilestat.archivepath)
                    logger.debug("Got a list of files in %s" % dmfilestat.archivepath)
                    if dmactions_type == dmactions_types.SIG:
                        pathToData = dmfilestat.archivepath
                        if doThumbnail:
                            pathToData = os.path.join(pathToData,'thumbnail')
                    elif dmactions_type == dmactions_types.BASE:
                        previousReport = dmfilestat.archivepath
                        # on-instrument analysis Basecalling Input data is in onboard_results folder
                        if exp.log.get('oninstranalysis','') == "yes" and not doThumbnail:
                            archived_onboard_path = os.path.join(dmfilestat.archivepath, 'onboard_results')
                            if os.path.exists(archived_onboard_path):
                                previousReport = archived_onboard_path
                except:
                    raise Exception("Analysis cannot start because %s data has been archived to %s.  Please mount that drive to make the data available."
                            % (dmactions_type, dmfilestat.archivepath) )
        else:
            raise Exception("Analysis cannot start because DMFileStat objects refuse to instantiate.  Please know its not your fault!")
        
        # check data input folder exists
        if blockArgs == "fromWells" and previousReport:
            data_input_folder = os.path.join(previousReport, 'sigproc_results')
        else:
            data_input_folder = pathToData
        
        if not os.path.exists(data_input_folder):
            raise Exception("Analysis cannot start because data folder is missing: %s" % data_input_folder)

        msg = 'Started from %s %s %s.' % (dmfilestat.get_action_state_display(), dmactions_type, previousReport or pathToData)
        models.EventLog.objects.add_entry(result, msg, username)

        logger.debug("Start Analysis on %s" % exp.expDir)

        # create params
        params = makeParams(exp, result, blockArgs, doThumbnail, align_full,
                                                os.path.join(storage.webServerPath, loc.name),
                                                mark_duplicates, pathToData, previousReport, plugins_list,
                                                doBaseRecal, realign, username)

        # test job server connection
        webRootPath = result.web_root_path(loc)
        try:
            host = "127.0.0.1"
            conn = client.connect(host, settings.JOBSERVER_PORT)
            to_check = os.path.dirname(webRootPath)
        except (socket.error, xmlrpclib.Fault):
            raise Exception("Failed to contact job server.")

        # prepare the directory in which the results' outputs will be written
        # copy TF config to new path if it exists
        files = []
        try:
            files.append(create_tf_conf(tfConfig))
        except ValueError as ve:
            raise Exception(str(ve))
        # write meta data to folder for report
        files.append(create_meta(exp, result))
        files.append(create_pk_conf(result.pk))
        # write barcodes file to folder
        if eas.barcodeKitName and eas.barcodeKitName is not '':
            files.append(create_bc_conf(eas.barcodeKitName,"barcodeList.txt"))

        # tell the analysis server to start the job
        try:
            chips = models.Chip.objects.all()
            chip_dict = dict((c.name, '-pe ion_pe %s' % str(c.slots)) for c in chips)
        except:
            chip_dict = {} # just in case we can't read from the db

        try:
            ts_job_type = 'thumbnail' if doThumbnail else ''
            conn.startanalysis(resultsName, script, params, files,
                               webRootPath, result.pk, exp.chipType, chip_dict, ts_job_type)
        except (socket.error, xmlrpclib.Fault):
            raise Exception("Failed to contact job server.")
        # redirect the user to the report started page

        return result

    except Exception as e:
        logger.exception("Unable to launch analysis for result %d: '%s'", result.pk, e)
        result.delete()
        if eas.isOneTimeOverride and eas.results_set.count()==0:
            eas.delete()
        raise


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
    
    allowed = ['GET', 'POST']
    if request.method not in allowed:
        return http.HttpResponseNotAllowed(allowed)

    exp = get_object_or_404(models.Experiment, pk=exp_pk)
    # get ExperimentAnalysisSettings to attach to new report, prefer latest editable EAS if available
    if exp.plan and exp.plan.latestEAS:
        eas = exp.plan.latestEAS
    else:
        eas = exp.get_EAS(editable=True,reusable=True)
        if not eas:
            eas, eas_created = exp.get_or_create_EAS(reusable=True)

    # get list of plugins for the pipeline to run, include plugins marked autorun or selected during planning
    plugins = models.Plugin.objects.filter(selected=True,active=True).exclude(path='')
    selected_names = [pl['name'] for pl in eas.selectedPlugins.values()]
    plugins_list = list(plugins.filter(name__in=selected_names) | plugins.filter(autorun=True))

    re_analysis = request.POST.get('re-analysis',False)
    if request.method == 'GET' or re_analysis:
        # this is a reanalysis web page request
        ctxd, eas, post_dict = reanalyze(request, exp, eas, plugins_list, report_pk)
    else:
        # this is new analysis POST request (e.g. from crawler)
        post_dict = {}
        for key,val in request.POST.items():
            if str(val).strip().lower() == 'false':
                post_dict[key] = False
            elif str(val).strip().lower() == 'true':
                post_dict[key] = True
            else:
                post_dict[key] = val

        post_dict['plugins'] = plugins_list
        if exp.plan and exp.plan.username:
            post_dict['username'] = exp.plan.username
        else:
            post_dict['username'] = request.user.username

        #ionCrawler may modify the path to raw data in the path variable passed thru URL?
        if post_dict.get('path',False):
            exp.expDir = post_dict['path']

    if post_dict:
        # create new result and launch analysis
        try:
            ufResultsName = post_dict['report_name']
            resultsName = ufResultsName.strip().replace(' ', '_')
            result = _createReport(exp, eas, resultsName, **post_dict)
            ctx = _report_started(request, result.pk)
            return render_to_response("rundb/reports/analysis_started.html", context_instance=ctx)
        except Exception as e:
            if re_analysis:
                ctxd['start_error'] = str(e)
                return render_to_response("rundb/reports/analyze.html", context_instance=RequestContext(request, ctxd))
            else:
                # TODO: could add banner msg to alert user analysis was not able to start
                logger.exception(traceback.format_exc())
                return HttpResponseServerError(str(e))
    else:
        # render the re-analysis web page
        return render_to_response("rundb/reports/analyze.html", context_instance=RequestContext(request, ctxd))


def reanalyze(request, exp, eas, plugins_list, start_from_report=None):
    """
    Process re-analyse web page POST and GET requests.
    GET: returns dict for making RequestContext to render reanalyse web page.
    POST: validates request, if successfull fills in and returns parameters dict for _createReport.
    Also returns ExperimentAnalysisSettings (eas) object:
        if any eas fields are changed on web page, it will create new non-reusable eas with updated values.
    """

    def flattenString(string):
        return string.replace("\n"," ").replace("\r"," ").strip()

    params = False
    javascript = ""
    isProton = exp.isProton

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

        result_choice = (
            r.get_report_dir(),
            r.resultsName + " [" + str(r.get_report_dir()) + "]",
            r.pk,
            version
        )
        if r.isThumbnail:
            previousThumbReports.append(result_choice)
        else:
            previousReports.append(result_choice)

    # plugins user input json from Planning
    pluginsUserInput = {}
    for plugin in plugins_list:
        pluginsUserInput[str(plugin.id)] = eas.selectedPlugins.get(plugin.name, {}).get('userInput','')

    # warnings for missing or archived data
    warnings = {}
    if resultList:
        dmfilestat = resultList[0].get_filestat(dmactions_types.SIG)
        if dmfilestat.isdisposed():
            warnings['sigproc'] = "Warning: Signal Processing Input data is %s" % dmfilestat.get_action_state_display()
        elif not os.path.exists(exp.expDir):
            warnings['sigproc'] = "Warning: Signal Processing Input data is missing"
        
        for r in resultList:
            dmfilestat = r.get_filestat(dmactions_types.BASE)
            if dmfilestat.isdisposed():
                warnings[r.pk] = "Warning: Basecalling Input data is %s" % dmfilestat.get_action_state_display()
            else:
                report_dir = r.get_report_dir()
                if not os.path.exists(report_dir) or not os.path.exists(os.path.join(report_dir,'sigproc_results')):
                    warnings[r.pk] = "Warning: Basecalling Input data is missing"

    globalConfig = models.GlobalConfig.get()
    
    # when samples are defined during Planning, references per-barcode can be selected
    barcodesWithSamples = []
    if eas.barcodedSamples:
        for sample, value in eas.barcodedSamples.items():
            try:
                info = value.get('barcodeSampleInfo',{})
                for bcId in value['barcodes']:
                    barcodesWithSamples.append({
                        'sample': sample,
                        'barcodeId': bcId,
                        'reference': info.get(bcId,{}).get('reference') if 'reference' in info.get(bcId,{}) else eas.reference,
                        'hotSpotRegionBedFile': info.get(bcId,{}).get('hotSpotRegionBedFile') if 'hotSpotRegionBedFile' in info.get(bcId,{}) else eas.hotSpotRegionBedFile,
                        'targetRegionBedFile': info.get(bcId,{}).get('targetRegionBedFile') if 'targetRegionBedFile' in info.get(bcId,{}) else eas.targetRegionBedFile,

                        'nucType': info.get(bcId,{}).get('nucleotideType')
                    })
            except:
                pass
        barcodesWithSamples.sort(key=lambda item: item['barcodeId'])

    if request.method == 'POST':
        rpf = forms.RunParamsForm(request.POST, request.FILES)
        eas_form = forms.AnalysisSettingsForm(request.POST)
                       
        # validate the form
        if rpf.is_valid() and eas_form.is_valid():
            '''
            Process input forms
            '''
            beadfindArgs = flattenString(rpf.cleaned_data['beadfindArgs'])
            analysisArgs = flattenString(rpf.cleaned_data['analysisArgs'])
            prebasecallerArgs = flattenString(rpf.cleaned_data['prebasecallerArgs'])
            recalibArgs = flattenString(rpf.cleaned_data['recalibArgs'])
            basecallerArgs = flattenString(rpf.cleaned_data['basecallerArgs'])
            alignmentArgs = flattenString(rpf.cleaned_data['alignmentArgs'])
            thumbnailBeadfindArgs = flattenString(rpf.cleaned_data['thumbnailBeadfindArgs'])
            thumbnailAnalysisArgs = flattenString(rpf.cleaned_data['thumbnailAnalysisArgs'])
            prethumbnailBasecallerArgs = flattenString(rpf.cleaned_data['prethumbnailBasecallerArgs'])
            thumbnailRecalibArgs = flattenString(rpf.cleaned_data['thumbnailRecalibArgs'])
            thumbnailBasecallerArgs = flattenString(rpf.cleaned_data['thumbnailBasecallerArgs'])
            thumbnailAlignmentArgs = flattenString(rpf.cleaned_data['thumbnailAlignmentArgs'])

            # need to update selectedPlugins field if 1) plugins added/removed by user 2) changes in any plugin configuration
            form_plugins_list = list(eas_form.cleaned_data['plugins'])
            form_pluginsUserInput = json.loads(eas_form.cleaned_data['pluginsUserInput']) if eas_form.cleaned_data['pluginsUserInput'] else {}
            if set(plugins_list) != set(form_plugins_list) or pluginsUserInput != form_pluginsUserInput:
                plugins_list = form_plugins_list
                selectedPlugins = {}
                for plugin in form_plugins_list:
                    selectedPlugins[plugin.name] = {
                         "id" : str(plugin.id),
                         "name" : plugin.name,
                         "version" : plugin.version,
                         "features": plugin.pluginsettings.get('features',[]),
                         "userInput": form_pluginsUserInput.get(str(plugin.id),'')
                    }
            else:
                selectedPlugins = eas.selectedPlugins

            # from-BaseCalling reanalysis needs to copy Beadfind and Analysis args from previous report
            if rpf.cleaned_data['blockArgs'] == "fromWells":
                try:
                    previousReport = rpf.cleaned_data['previousThumbReport'] if rpf.cleaned_data.get('do_thumbnail') else rpf.cleaned_data['previousReport']
                    selected_previous_pk = int(previousReport.strip('/').split('_')[-1])
                    previousEAS = resultList.get(pk=selected_previous_pk).eas
                    beadfindArgs = previousEAS.beadfindargs
                    analysisArgs = previousEAS.analysisargs
                    thumbnailBeadfindArgs = previousEAS.thumbnailbeadfindargs
                    thumbnailAnalysisArgs = previousEAS.thumbnailanalysisargs
                except:
                    pass

            # Selected reference per barcode (for Compendia project RNA/DNA sample)
            barcodedSamples = eas.barcodedSamples
            barcodedReferences = eas_form.cleaned_data.get('barcodedReferences')
            if barcodedReferences:
                barcodedReferences = json.loads(barcodedReferences)
                for sample in barcodedSamples.values():
                    for barcode, info in sample.get('barcodeSampleInfo',{}).items():
                        info['reference'] = barcodedReferences[barcode]['reference'] if barcodedReferences[barcode]['reference'] != 'none' else ''
                        info['nucleotideType'] = barcodedReferences[barcode]['nucType']
            
            eas_kwargs = {
                'libraryKey': rpf.cleaned_data['libraryKey'] or globalConfig.default_library_key,
                'tfKey': rpf.cleaned_data['tfKey'] or globalConfig.default_test_fragment_key,
                'reference':  eas_form.cleaned_data['reference'] if eas_form.cleaned_data['reference']!= 'none' else '',
                'targetRegionBedFile':  eas_form.cleaned_data['targetRegionBedFile'],
                'hotSpotRegionBedFile': eas_form.cleaned_data['hotSpotRegionBedFile'],
                'barcodeKitName': eas_form.cleaned_data['barcodeKitName'],
                'barcodedSamples': barcodedSamples,
                'threePrimeAdapter': eas_form.cleaned_data['threePrimeAdapter'],
                'selectedPlugins': selectedPlugins,
                'isDuplicateReads': rpf.cleaned_data['mark_duplicates'],
                'base_recalibration_mode': rpf.cleaned_data['do_base_recal'],                
                'realign': rpf.cleaned_data['realign'],
                'beadfindargs': beadfindArgs,
                'thumbnailbeadfindargs': thumbnailBeadfindArgs,
                'analysisargs': analysisArgs,
                'thumbnailanalysisargs': thumbnailAnalysisArgs,
                'prebasecallerargs': prebasecallerArgs,
                'prethumbnailbasecallerargs': prethumbnailBasecallerArgs,
                'calibrateargs': recalibArgs,
                'thumbnailcalibrateargs': thumbnailRecalibArgs,
                'basecallerargs': basecallerArgs,
                'thumbnailbasecallerargs': thumbnailBasecallerArgs,
                'alignmentargs': alignmentArgs,
                'thumbnailalignmentargs': thumbnailAlignmentArgs
            }
            eas = update_experiment_analysis_settings(eas, **eas_kwargs)

            # Ready to launch analysis pipeline
            # create parameters needed by _createReport function
            params = rpf.cleaned_data
            params['username'] = request.user.username
            params['plugins'] = plugins_list
            if not isProton:
                params['do_thumbnail'] = False

    if request.method == 'GET':

        rpf = forms.RunParamsForm()
        rpf.fields['align_full'].initial = True

        # initialize with default cmdline arguments
        default_args = get_default_cmdline_args(exp.plan)
        rpf.fields['beadfindArgs'].initial = default_args['beadfindargs']
        rpf.fields['analysisArgs'].initial = default_args['analysisargs']
        rpf.fields['prebasecallerArgs'].initial = default_args['prebasecallerargs']
        rpf.fields['recalibArgs'].initial = default_args['calibrateargs']
        rpf.fields['basecallerArgs'].initial = default_args['basecallerargs']
        rpf.fields['alignmentArgs'].initial = default_args['alignmentargs']
        rpf.fields['thumbnailBeadfindArgs'].initial = default_args['thumbnailbeadfindargs']
        rpf.fields['thumbnailAnalysisArgs'].initial = default_args['thumbnailanalysisargs']
        rpf.fields['thumbnailBasecallerArgs'].initial = default_args['thumbnailbasecallerargs']
        rpf.fields['prethumbnailBasecallerArgs'].initial = default_args['prethumbnailbasecallerargs']
        rpf.fields['thumbnailRecalibArgs'].initial = default_args['thumbnailcalibrateargs']
        rpf.fields['thumbnailAlignmentArgs'].initial = default_args['thumbnailalignmentargs']
        
        rpf.fields['previousReport'].widget.choices = previousReports
        rpf.fields['previousThumbReport'].widget.choices = previousThumbReports
        rpf.fields['project_names'].initial = get_project_names(exp)

        rpf.fields['do_base_recal'].widget.choices = eas.get_base_recalibration_mode_choices()
        rpf.fields['do_base_recal'].initial = eas.base_recalibration_mode
        
        rpf.fields['mark_duplicates'].initial = eas.isDuplicateReads
        rpf.fields['realign'].initial = eas.realign

        rpf.fields['libraryKey'].initial = eas.libraryKey
        rpf.fields['tfKey'].initial = eas.tfKey or globalConfig.default_test_fragment_key

        # Analysis settings form
        eas_form = forms.AnalysisSettingsForm(instance=eas)
        eas_form.fields['plugins'].initial = [plugin.id for plugin in plugins_list]
        eas_form.fields['pluginsUserInput'].initial = json.dumps(pluginsUserInput, cls=DjangoJSONEncoder)

        #send some js to the page
        previousReportDir = get_initial_arg(start_from_report)
        if previousReportDir:
            rpf.fields['blockArgs'].initial = "fromWells"
            javascript = """
            $("#fromWells").click();
            """
            javascript += '$("#id_previousReport").val("'+previousReportDir +'");'


    ctxd = {"rpf": rpf, "eas_form": eas_form, "expName":exp.pretty_print_no_space, "javascript" : javascript,
           "isProton":isProton, "pk":exp.pk, "reportpk":start_from_report, "warnings": json.dumps(warnings),
           "barcodesWithSamples": barcodesWithSamples}

    return ctxd, eas, params


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


def show_pdf(full_path):
    return HttpResponse(FileWrapper(open(full_path, 'rb')), mimetype='application/pdf')


def html_text(full_path):
    return HttpResponse(FileWrapper(open(full_path, 'rb')), mimetype='text/html')


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
    (r'\.pdf$', show_pdf),
    (r'\.html$', html_text),
    (r'.', plain_text),
)]


def show_file(report, pk, path, root, full_path):
    for pattern, handle in FILE_HANDLERS:
        if pattern.search(path):
            try:
                return handle(full_path)
            except Exception as err:
                raise err
    raise OSError("File path matches no pattern, path = '%' full_path='%s'" % (path, full_path))


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
        "files": file_info,
        "can_upload": bool(settings.AWS_ACCESS_KEY)
    })


@login_required
def metal(request, pk, path):
    path = path.strip('/')
    report = get_object_or_404(models.Results, pk=pk)
    root = report.get_report_dir()
    full_path = os.path.join(root, path)
    if not os.path.exists(full_path):
        return http.HttpResponseNotFound("This path no longer exists:<br/>" + full_path)
    if os.path.isdir(full_path):
        return show_directory(report, pk, path, root, full_path)
    else:
        return show_file(report, pk, path, root, full_path)


def getZip(request, pk):
	try:
		# Goal: zip up the target pk's _bamOut directory and send it over.
		# Later goal: zip up a certain subdirectory of _bamOut and send it.
		prePath = '/results/analysis/output/'
		prePathList = os.listdir(prePath)
		fullPath = ""
		for prePathEntry in prePathList:
			pathList = os.listdir(os.path.join(prePath, prePathEntry))
			for path in pathList:
				if (('%s'%path).endswith('%s'%pk)):
					fullPath = os.path.join(prePath, prePathEntry, path)
		fullPath = os.path.join(fullPath, 'plugin_out', 'FileExporter_out', 'downloads.zip')
		if not os.path.isfile(fullPath):
			causeException = 19/0
		zipFile = open(fullPath, 'r')
		zip_filename = 'bamFiles_%s.zip'%pk
		#ctx = template.RequestContext(request, {"pk" : pk, "path" : fullPath})
		#response = StreamingHttpResponse(zs)
		response = HttpResponse(zipFile)
		response['Content-type'] = 'application/zip'
		response['Content-Disposition'] = 'attachment; filename="%s"'%zip_filename
		return response
		#return render_to_response("rundb/data/zipGet.html", context_instance=ctx)
	except:
		# Just return nothing if it fails.
		return http.HttpResponseRedirect("/report/%s"%pk)

def locate(pattern, root):
	for path, dirs, files in os.walk(os.path.abspath(root)):
		#for filename in fnmatch.filter(files, pattern):
		for fileName in files:
			#sys.stderr.write('LOC: %s\n'%fileName)
			if fileName == pattern:
				yield os.path.join(path, fileName)

def getVCF(request, pk):
	prePath = '/results/analysis/output/'
	prePathList = os.listdir(prePath)
	fullPath = ""
	for prePathEntry in prePathList:
		pathList = os.listdir(os.path.join(prePath, prePathEntry))
		for path in pathList:
			if (('%s'%path).endswith('%s'%pk)):
				fullPath = os.path.join(prePath, prePathEntry, path)
	fullPath = os.path.join(fullPath, 'plugin_out')
	plStr = ''
	VCPaths = []
	VCNames = []
	if (os.path.isdir(os.path.join(fullPath, 'FileExporter_out'))):
		nameFile = open(os.path.join(fullPath, 'FileExporter_out', 'FN.log'), 'r')
		nameVals = nameFile.read().split('\n')
		toName = nameVals[0]
		toDelim = nameVals[1]
		#plStr += '%s<br/>%s<br/><br/>\n'%(toName, toDelim)
	pluginPaths = os.listdir(fullPath)
	for path in pluginPaths:
		if 'variantCaller' in path:
			VCPaths.append(os.path.join(fullPath, path))
			VCNames.append(path)
	for V in VCNames:
		plStr += '<b>%s:</b><br/>\n'%V
		for VC in VCPaths:
			if V in VC:
				mkCmd = subprocess.Popen(['mkdir', os.path.join(fullPath, 'FileExporter_out', V)])
				mkOut, mkErr = mkCmd.communicate()
				for fn in locate('TSVC_variants.vcf', VC):
					useablePath = fn[len(fullPath)+1:]
					useablePath = useablePath[len(V)+1:]
					if '/' in useablePath:
						useablePath = useablePath.split('/')
						outFP = toName.replace('@BARINFO@', useablePath[0])
						newFP = os.path.join(fullPath, 'FileExporter_out', V, useablePath[0], '%s.vcf'%outFP)
						mkCmd = subprocess.Popen(['mkdir', os.path.join(fullPath, 'FileExporter_out', V, useablePath[0])])
						mkOut, mkErr = mkCmd.communicate()
					else:
						outFP = toName.replace('@BARINFO@', '')
						outFP = outFP.replace('%s%s'%(toDelim, toDelim), toDelim)
						newFP = os.path.join(fullPath, 'FileExporter_out', V, '%s.vcf'%outFP)
					lnCmd = subprocess.Popen(['ln', '-sf', fn, newFP])
					lnOut, lnErr = lnCmd.communicate()
					plStr += 'VCF Link: <a href=%s>%s</a><br/>\n'%(newFP[newFP.find('/results/analysis')+len('/results/analysis'):], newFP[len(fullPath):])
	return HttpResponse(plStr)
