# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.contrib.auth.decorators import login_required
from django.template import RequestContext
from iondb.rundb import models
from iondb.backup import makePDF
from django import http
import json
import os
import csv

import ConfigParser
import logging
from django.views.decorators.csrf import csrf_exempt
from iondb.rundb.views import _createReport, _report_started
from django.http import HttpResponsePermanentRedirect, HttpResponseRedirect, HttpResponse
from django.core.urlresolvers import reverse
from django.shortcuts import get_object_or_404, redirect, render_to_response
import numpy
import math
import urllib

logger = logging.getLogger(__name__)

#TODO do something fancy to keep track of the versions


def url_with_querystring(path, **kwargs):
    return path + '?' + urllib.urlencode(kwargs)


def getPDF(request, pk):
    ret = get_object_or_404(models.Results, pk=pk)
    filename = "%s"%ret.resultsName
    
    response = http.HttpResponse(makePDF.getPDF(pk), mimetype="application/pdf")
    response['Content-Disposition'] = 'attachment; filename=%s.pdf'%filename
    return response


def percent(q, d):
    return "%04.1f%%" % (100 * float(q) / float(d)) 


def load_ini(report,subpath,filename,namespace="global"):
    parse = ConfigParser.ConfigParser()
    #TODO preseve the case
    try:
        parse.read(os.path.join(report.get_report_dir(), subpath , filename))
        parse = parse._sections.copy()
        return parse[namespace]
    except:
        return False


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
        return False
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
    #get the plan data
    try:
        plan = models.PlannedExperiment.objects.get(planShortID=report.planShortID())
        return plan.pk
    except models.PlannedExperiment.DoesNotExist:
        return False

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

@login_required
def report_display(request, report_pk):
    """
    get the plan
    get the functions from the mobile site too
    """
    report = get_object_or_404(models.Results, pk=report_pk)
    otherReports = report.experiment.results_set.exclude(pk=report_pk).order_by("-timeStamp")

    #the loading status
    report_status = report_status_read(report)

    plan = report_plan(report)
    try:
        reference = models.ReferenceGenome.objects.filter(short_name = report.reference).order_by("-index_version")[0]
    except IndexError, IOError:
        reference = False

    #find the major blocks from the important pliugins
    major_plugins = models.Plugin.objects.filter(majorBlock=True)
    
    #TODO: encapuslate all vars into their parent block to make it easy to build the API maybe put
    #all of this in the model? 
    basecaller = basecaller_read(report)
    barcodes = barcodes_read(report)
    datasets = datasets_read(report)
    testfragments = testfragments_read(report)

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
    # redirect to old report if no basecaller_results/datasets_basecaller.json found
    elif (not barcodes) and (report.status == 'Completed' or 'Error' in report.status):      
        #TODO: better text  
        error = "old_report"
        return HttpResponseRedirect(url_with_querystring(reverse('report_log', 
                                        kwargs={'pk':report_pk}), error=error))

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
        bead_loading = 100 * float(beadfind["bead wells"]) / (float(beadfind["total wells"]) - float(beadfind["excluded wells"]))
        bead_loading = int(round(bead_loading))
        bead_loading_threshold = qcTypes.get("Bead Loading (%)", 0)

        beadsummary = {}
        beadsummary["total_addressable_wells"] = int(beadfind["total wells"]) - int(beadfind["excluded wells"])
        beadsummary["bead_wells"] = beadfind["bead wells"]
        beadsummary["p_bead_wells"] = percent(beadfind["bead wells"], beadsummary["total_addressable_wells"])
        beadsummary["live_beads"] = beadfind["live beads"]
        beadsummary["p_live_beads"] = percent(beadfind["live beads"], beadfind["bead wells"])
        beadsummary["test_fragment_beads"] = beadfind["test fragment beads"]
        beadsummary["p_test_fragment_beads"] = percent(beadfind["test fragment beads"], beadfind["live beads"])
        beadsummary["library_beads"] = beadfind["library beads"]
        beadsummary["p_library_beads"] = percent(beadfind["library beads"], beadfind["live beads"])
    except:
        logger.exception("Failed to build Beadfind report content.")

    #Basecaller
    try:
        usable_sequence = basecaller and int(round(100.0 * 
            float(basecaller["total_reads"]) / float(beadfind["library beads"])))
        usable_sequence_threshold = qcTypes.get("Usable Sequence (%)", 0)
        quality = load_ini(report,"basecaller_results","quality.summary")
        
        if float(dbr['total_Q0_bases']) > 0:
            mappable_output = int(round( float(alignStats["total_mapped_target_bases"]) / float(dbr['total_Q0_bases'])  * 100))
        else:
            mappable_output = 0
        
        basecaller["p_polyclonal"] = percent(basecaller["polyclonal"], beadfind["library beads"])
        basecaller["p_low_quality"] = percent(basecaller["low_quality"], beadfind["library beads"])
        basecaller["p_primer_dimer"] = percent(basecaller["primer_dimer"], beadfind["library beads"])
        basecaller["p_total_reads"] = percent(basecaller["total_reads"], beadfind["library beads"])
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
                raw_accuracy = round( (1- float(alignStats["accuracy_total_errors"]) / float(alignStats["accuracy_total_bases"])) * 100, 1)
            else:
                raw_accuracy = 0.0
        else:
            reference = 'none'
    except:
        logger.exception("Failed to build Alignment report content.")       
    
    #Links
    try:
        prefix_tuple = (report.reportWebLink(),report.experiment.expName,report.resultsName)
        
        output_file_groups = []
        
        current_group = {"name":"Library"}
        current_group["basecaller_bam"] = "%s/basecaller_results/%s_%s.basecaller.bam" % prefix_tuple
        current_group["sff"]            = "%s/basecaller_results/%s_%s.sff.zip" % prefix_tuple
        current_group["fastq"]          = "%s/basecaller_results/%s_%s.fastq.zip" % prefix_tuple
        current_group["bam"]            = "%s/%s_%s.bam" % prefix_tuple
        current_group["bai"]            = "%s/%s_%s.bam.bai" % prefix_tuple
        output_file_groups.append(current_group)
    except:
        logger.exception("Failed to build Links report content.")
        
    #Barcodes
    try:
        if "barcode_config" in datasets:
            current_group = {"name":"Barcodes"}
            current_group["basecaller_bam"] = "%s/%s_%s.barcode.basecaller.bam.zip" % prefix_tuple
            current_group["sff"]            = "%s/%s_%s.barcode.sff.zip" % prefix_tuple
            current_group["fastq"]          = "%s/%s_%s.barcode.fastq.zip" % prefix_tuple
            current_group["bam"]            = "%s/%s_%s.barcode.bam.zip" % prefix_tuple
            current_group["bai"]            = "%s/%s_%s.barcode.bam.bai.zip" % prefix_tuple
            output_file_groups.append(current_group)
    
            #current_group = {"name":"Test Fragments"}
            #current_group["basecaller_bam"] = "%s/basecaller_results/%s_%s.tf.basecaller.bam" % prefix_tuple
            #current_group["sff"]            = "%s/basecaller_results/%s_%s.tf.sff.zip" % prefix_tuple
            #current_group["bam"]            = "%s/basecaller_results/%s_%s.tf.bam" % prefix_tuple
            #output_file_groups.append(current_group)
            
            # links for barcodes.html: mapped bam links if aligned to reference, unmapped otherwise
            report_link = report.reportWebLink()
            for barcode in barcodes:
                if report.reference != 'none':
                    barcode['bam_link'] = "%s/%s.bam" % (report_link, barcode['file_prefix'])
                    barcode['bai_link'] = "%s/%s.bam.bai" % (report_link, barcode['file_prefix'])
                else:
                    barcode['bam_link'] = "%s/basecaller_results/%s.basecaller.bam" % (report_link, barcode['file_prefix'])
                    barcode['bai_link'] = "%s/basecaller_results/%s.basecaller.bam.bai" % (report_link, barcode['file_prefix'])            
            
    except Exception as err:
        logger.exception("Failed to build Barcodes report content.")
        #gotta catch em all

    #Dim buttons if files don't exist
    try:
        report_link = report.reportWebLink()
        report_path = report.get_report_path()        
        for output_group in output_file_groups:
            keys = [k for k in output_group if k!='name']
            for key in keys:                
                file_path = output_group[key].replace(report_link, report_path)
                output_group[key+'_dim'] = 1.0 if os.path.isfile(file_path) else 0.3
    except:
        pass    

    noheader = request.GET.get("no_header",False)
    # This is both awesome and playing with fire.  Should be made explicit soon
    ctxd = locals()
    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/reports/report.html", context_instance=ctx)

@login_required
def report_log(request, pk):
    report = models.Results.objects.select_related('reportstorage').get(pk=pk)
    error = request.GET.get("error", None)

    contents = ""

    logs = [ os.path.join(report.get_report_path(),"sigproc_results/sigproc.log"),
            os.path.join(report.get_report_path(),"basecaller_results/basecaller.log"),
            os.path.join(report.get_report_path(),"alignment.log"),
            os.path.join(report.get_report_path(),"ReportLog.html")
            ]

    for log_name in logs:
        if os.path.exists(log_name):
            with open(log_name) as log_file:
                contents += log_file.read()

    context = {
        "contents": contents,
        "report": report,
        "report_link": report.reportWebLink(),
        "logs" : logs,
        "error" : error
    }

    return render_to_response("rundb/reports/report_log.html", 
        context, RequestContext(request))

@login_required
@csrf_exempt
def analyze(request, exp_pk, report_pk):
    templateName = "rundb/reports/analyze.html"
    get_object_or_404(models.Experiment, pk=exp_pk)
    result = _createReport(request, exp_pk, report_pk)
    if isinstance(result, RequestContext):
        return render_to_response(templateName,
                                        context_instance=result)
    if (request.method == 'POST'):
        ctx = _report_started(request, result.pk)
        return render_to_response("rundb/reports/analysis_started.html",
                                        context_instance=ctx)
