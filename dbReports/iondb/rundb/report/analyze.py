# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
import json
import os
import traceback
import socket
import xmlrpclib
import re
from datetime import date
from django.conf import settings
from django.db import transaction
from django.forms.models import model_to_dict
from django.core.serializers.json import DjangoJSONEncoder
from django.core.exceptions import ObjectDoesNotExist

from iondb.anaserve import client
from iondb.rundb import models
from iondb.plugins.launch_utils import get_plugins_dict
from iondb.rundb.data import dmactions_types

import logging
logger = logging.getLogger(__name__)


def createReport_and_launch(exp, eas, resultsName, **kwargs):
    """
    Create result and send to the job server.
    Re-analyze page and crawler POST both end up here.
    """
    blockArgs = kwargs.get('blockArgs', 'fromRaw')
    doThumbnail = kwargs.get('do_thumbnail', False)
    username = kwargs.get('username', '')
    plugins_list = kwargs.get('plugins', [])

    if blockArgs == 'fromWells':
        previousReport = kwargs.get('previousThumbReport', '') if doThumbnail else kwargs.get('previousReport', '')
    else:
        previousReport = ''

    result = _createReport(exp, eas, resultsName, doThumbnail, previousReport)

    try:
        dmactions_type = dmactions_types.BASE if blockArgs == "fromWells" else dmactions_types.SIG
        dmfilestat = result.get_filestat(dmactions_type)

        pathToData, previousReport = find_data_files(exp, dmfilestat, doThumbnail, previousReport)

        msg = 'Started from %s %s %s.' % (dmfilestat.get_action_state_display(), dmactions_type, previousReport or pathToData)
        models.EventLog.objects.add_entry(result, msg, username)

        # create params
        params = makeParams(exp, eas, result, blockArgs, doThumbnail, pathToData, previousReport, plugins_list, username)
        params = json.dumps(params, cls=DjangoJSONEncoder, indent=1)

        logger.debug("Start Analysis on %s" % exp.expDir)
        launch_analysis_job(result, params, doThumbnail)

        return result

    except Exception as e:
        result.delete()
        if eas.isOneTimeOverride and eas.results_set.count() == 0:
            eas.delete()
        raise


def _createReport(exp, eas, resultsName, doThumbnail, previousReport):
    ''' create Result and related objects '''

    loc = exp.location()
    if not loc:
        raise ObjectDoesNotExist("There are no Location objects, at all.")

    # error out if cmdline args are missing
    if not eas.have_args(thumbnail=doThumbnail):
        raise Exception("Analysis cannot start because of missing Command Line Args.")

    # Always use the default ReportStorage object
    storage = models.ReportStorage.objects.filter(default=True)[0]

    try:
        with transaction.atomic():
            result = build_result(exp, resultsName, storage, loc, doThumbnail)

            # Don't allow EAS to be edited once analysis has started
            if eas.isEditable:
                eas.isEditable = False
                eas.save()

            result.eas = eas
            result.reference = eas.reference

            # attach project(s)
            projectNames = get_project_names(exp)
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

            result.save()

            # handle from-Basecalling reAnalysis
            if previousReport:
                parent_obj = None
                try:
                    selected_previous_pk = int(previousReport.strip('/').split('_')[-1])
                    parent_obj = models.Results.objects.get(pk=selected_previous_pk)
                except:
                    # TorrentSuiteCloud plugin 3.4.2 uses reportName for this value
                    try:
                        parent_obj = models.Results.objects.get(resultsName=os.path.basename(previousReport))
                    except:
                        pass

                if parent_obj:
                    result.parentResult = parent_obj
                    result.save()
                    # replace dmfilestat
                    dmfilestat = parent_obj.get_filestat(dmactions_types.BASE)
                    result.dmfilestat_set.filter(dmfileset__type=dmactions_types.BASE).delete()
                    dmfilestat.pk = None
                    dmfilestat.result = result
                    dmfilestat.save()

            return result

    except Exception as e:
        logger.exception("Aborted createReport for result %d: '%s'", result.pk, e)
        raise


def find_data_files(exp, dmfilestat, doThumbnail, previousReport=''):

    pathToData = os.path.join(exp.expDir)
    if doThumbnail:
        pathToData = os.path.join(pathToData, 'thumbnail')

    # Determine if data has been archived or deleted
    if dmfilestat:
        dmactions_type = dmfilestat.dmfileset.type
        if dmfilestat.action_state in ['DG', 'DD']:
            raise Exception("Analysis cannot start because %s data has been deleted." % dmactions_type)
        elif dmfilestat.action_state in ['AG', 'AD']:
            # replace paths with archived locations
            try:
                datfiles = os.listdir(dmfilestat.archivepath)
                logger.debug("Got a list of files in %s" % dmfilestat.archivepath)
                if dmactions_type == dmactions_types.SIG:
                    pathToData = dmfilestat.archivepath
                    if doThumbnail:
                        pathToData = os.path.join(pathToData, 'thumbnail')
                elif dmactions_type == dmactions_types.BASE:
                    previousReport = dmfilestat.archivepath
                    # on-instrument analysis Basecalling Input data is in onboard_results folder
                    if exp.log.get('oninstranalysis', '') == "yes" and not doThumbnail:
                        archived_onboard_path = os.path.join(dmfilestat.archivepath, 'onboard_results')
                        if os.path.exists(archived_onboard_path):
                            previousReport = archived_onboard_path
            except:
                logger.error(traceback.format_exc())
                raise Exception("Analysis cannot start because %s data has been archived to %s.  Please mount that drive to make the data available."
                                % (dmactions_type, dmfilestat.archivepath))
    else:
        raise Exception("Analysis cannot start because DMFileStat objects refuse to instantiate.  Please know its not your fault!")

    # check data input folder exists
    if previousReport:
        data_input_folder = os.path.join(previousReport, 'sigproc_results')
    else:
        data_input_folder = pathToData

    if not os.path.exists(data_input_folder):
        raise Exception("Analysis cannot start because data folder is missing: %s" % data_input_folder)

    return pathToData, previousReport


def launch_analysis_job(result, params, doThumbnail):
    ''' Create files and send to jobServer '''

    def create_tf_conf():
        """
        Build the contents of the report TF file (``DefaultTFs.conf``)
        """
        fname = "DefaultTFs.conf"
        tfs = models.Template.objects.filter(isofficial=True).order_by('name')
        lines = ["%s,%s,%s" % (tf.name, tf.key, tf.sequence,) for tf in tfs]

        return (fname, "\n".join(lines))

    def create_bc_conf(barcodeId, fname):
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
            lines.append('barcode %d,%s,%s,%s,%s,%s,%d,%s' % (db_barcode.index, db_barcode.id_str, db_barcode.sequence, db_barcode.adapter, db_barcode.annotation, db_barcode.type, db_barcode.length, db_barcode.floworder))
        if db_barcodes:
            lines.insert(0, "file_id %s" % db_barcodes[0].name)
            lines.insert(1, "score_mode %s" % str(db_barcodes[0].score_mode))
            lines.insert(2, "score_cutoff %s" % str(db_barcodes[0].score_cutoff))
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

    # Default control script definition
    scriptname = 'TLScript.py'

    from distutils.sysconfig import get_python_lib;
    python_lib_path = get_python_lib()
    scriptpath = os.path.join(python_lib_path, 'ion/reports', scriptname)
    try:
        with open(scriptpath, "r") as f:
            script = f.read()
    except Exception as error:
        raise Exception("Error reading %s\n%s" % (scriptpath, error.args))

    # test job server connection
    webRootPath = result.get_report_path()
    try:
        host = "127.0.0.1"
        conn = client.connect(host, settings.JOBSERVER_PORT)
        to_check = os.path.dirname(webRootPath)
    except (socket.error, xmlrpclib.Fault):
        raise Exception("Failed to contact job server.")

    # the following files will be written into result's directory
    files = []
    files.append(create_tf_conf())          # DefaultTFs.conf
    files.append(create_meta(result.experiment, result))  # expMeta.dat
    files.append(create_pk_conf(result.pk))  # primary.key
    # barcodeList.txt
    barcodeKitName = result.eas.barcodeKitName
    if barcodeKitName:
        files.append(create_bc_conf(barcodeKitName, "barcodeList.txt"))

    try:
        chips = models.Chip.objects.all()
        chip_dict = dict((c.name, '-pe ion_pe %s' % str(c.slots)) for c in chips)
    except:
        chip_dict = {}  # just in case we can't read from the db

    try:
        ts_job_type = 'thumbnail' if doThumbnail else ''
        conn.startanalysis(result.resultsName, script, params, files,
                           webRootPath, result.pk, result.experiment.chipType, chip_dict, ts_job_type)
    except (socket.error, xmlrpclib.Fault):
        raise Exception("Failed to contact job server.")


def makeParams(exp, eas, result, blockArgs, doThumbnail, pathToData, previousReport='', plugins_list=[], username=''):
    """Build a dictionary of analysis parameters, to be passed to the job
    server when instructing it to run a report.  Any information that a job
    will need to be run must be constructed here and included inside the return.
    This includes any special instructions for flow control in the top level script."""

    # defaults from GlobalConfig
    gc = models.GlobalConfig.get()
    site_name = gc.site_name
    # get the hostname try to get the name from global config first
    if gc.web_root:
        net_location = gc.web_root
    else:
        # if a hostname was not found in globalconfig.webroot then use what the system reports
        net_location = "http://" + str(socket.getfqdn())

    storage = models.ReportStorage.objects.get(default=True)
    url_path = os.path.join(storage.webServerPath, exp.location().name)

    # floworder field sometimes has whitespace appended (?)  So strip it off
    flowOrder = exp.flowsInOrder.strip()
    # Set the default flow order if its not stored in the dbase.  Legacy support
    if flowOrder == '0' or flowOrder == None or flowOrder == '':
        flowOrder = "TACG"

    # Experiment
    exp_json = model_to_dict(exp)

    # ExperimentAnalysisSettings
    eas_json = model_to_dict(eas)
    # remove selectedPlugins, it's duplicated in the plugins dict
    del eas_json['selectedPlugins']

    # Get the 3' adapter primer
    try:
        threePrimeadapter = models.ThreePrimeadapter.objects.filter(sequence=eas.threePrimeAdapter)
        if threePrimeadapter:
            threePrimeadapter = threePrimeadapter[0]
        else:
            threePrimeadapter = models.ThreePrimeadapter.objects.get(direction="Forward", isDefault=True)

        adapter_primer_dict = model_to_dict(threePrimeadapter)
    except:
        adapter_primer_dict = {'name': 'Ion Kit',
                               'sequence': 'ATCACCGACTGCCCATAGAGAGGCTGAGAC',
                               'direction': 'Forward'}

    if exp.plan:
        plan_json = model_to_dict(exp.plan)
    else:
        plan_json = {}

    # Plugins
    plugins = get_plugins_dict(plugins_list, eas.selectedPlugins)

    # Samples
    sampleInfo = {}
    for sample in exp.samples.all():
        sampleInfo[sample.displayedName] = {
            'name': sample.name,
            'displayedName': sample.displayedName,
            'externalId': sample.externalId,
            'description': sample.description,
            'attributes': {}
        }
        for attributeValue in sample.sampleAttributeValues.all():
            sampleInfo[sample.displayedName]['attributes'][attributeValue.sampleAttribute.displayedName] = attributeValue.value

    barcodedSamples_reference_names = eas.barcoded_samples_reference_names
    # use barcodedSamples' selected reference if NO plan default reference is specified
    reference = eas.reference
    if not eas.reference and barcodedSamples_reference_names:
        reference = barcodedSamples_reference_names[0]

    doBaseRecal = eas.base_recalibration_mode

    if doThumbnail:
        beadfindArgs = eas.thumbnailbeadfindargs
        analysisArgs = eas.thumbnailanalysisargs
        basecallerArgs = eas.thumbnailbasecallerargs
        prebasecallerArgs = eas.prethumbnailbasecallerargs
        recalibArgs = eas.thumbnailcalibrateargs
        alignmentArgs = eas.thumbnailalignmentargs
        ionstatsArgs = eas.thumbnailionstatsargs
    else:
        beadfindArgs = eas.beadfindargs
        analysisArgs = eas.analysisargs
        basecallerArgs = eas.basecallerargs
        prebasecallerArgs = eas.prebasecallerargs
        recalibArgs = eas.calibrateargs
        alignmentArgs = eas.alignmentargs
        ionstatsArgs = eas.ionstatsargs

    # Special case: add selected regions file to tmap --bed-file parameter
    tmap_bedfile_option = "--bed-file"
    if eas.targetRegionBedFile and tmap_bedfile_option in alignmentArgs:
        alignmentArgs = alignmentArgs.replace(tmap_bedfile_option, "%s %s" % (tmap_bedfile_option, eas.targetRegionBedFile))

    # special case: override blocks
    chipBlocksOverride = ''
    m = re.search(' --chip.\w+', basecallerArgs)
    if m:
        option = m.group()
        chipBlocksOverride = option.split()[1]
        basecallerArgs = basecallerArgs.replace(option, "")

    ret = {
        'exp_json': exp_json,
        'plan': plan_json,
        'experimentAnalysisSettings': eas_json,

        'beadfindArgs': beadfindArgs,
        'analysisArgs': analysisArgs,
        'prebasecallerArgs': prebasecallerArgs,
        'basecallerArgs': basecallerArgs,
        'aligner_opts_extra': alignmentArgs,
        'recalibArgs': recalibArgs,
        'ionstatsArgs': ionstatsArgs,

        'resultsName': result.resultsName,
        'runid': result.runid,
        'expName': exp.expName,
        'sample': exp.get_sample(),
        'chiptype': exp.chipType,
        'rawdatastyle': exp.rawdatastyle,
        'flowOrder': flowOrder,
        'flows': exp.flows,
        'instrumentName': exp.pgmName,
        'platform': exp.getPlatform,
        'referenceName': reference,
        'libraryKey': eas.libraryKey,
        'tfKey': eas.tfKey,
        'reverse_primer_dict': adapter_primer_dict,
        'barcodeId': eas.barcodeKitName if eas.barcodeKitName else '',
        "barcodeSamples_referenceNames": barcodedSamples_reference_names,
        'barcodeInfo': make_barcodeInfo(eas, exp, doBaseRecal),
        'sampleInfo': sampleInfo,
        'plugins': plugins,
        'project': ','.join(p.name for p in result.projects.all()),

        'pathToData': pathToData,
        'blockArgs': blockArgs,
        'previousReport': previousReport,
        'skipchecksum': False,
        'doThumbnail': doThumbnail,
        'mark_duplicates': eas.isDuplicateReads,
        'doBaseRecal': doBaseRecal,
        'realign': eas.realign,
        'align_full': True,

        'net_location': net_location,
        'site_name': site_name,
        'url_path': url_path,
        'tmap_version': settings.TMAP_VERSION,
        'sam_parsed': True if os.path.isfile('/opt/ion/.ion-internal-server') else False,
        'username': username,
    }

    if chipBlocksOverride:
        ret['chipBlocksOverride'] = chipBlocksOverride

    return ret


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
        for barcode in models.dnaBarcode.objects.filter(name=barcodeId).values('index', 'id_str', 'sequence', 'adapter'):
            barcodeInfo[barcode['id_str']] = barcode
            barcodeInfo[barcode['id_str']]['sample'] = 'none'
            barcodeInfo[barcode['id_str']]['referenceName'] = eas.reference
            barcodeInfo[barcode['id_str']]['calibrate'] = doBaseRecal
            barcodeInfo[barcode['id_str']]['controlType'] = ''

        if eas.barcodedSamples:
            for sample, value in eas.barcodedSamples.items():
                try:
                    info = value.get('barcodeSampleInfo', {})
                    dna_rna_sample = set([v.get('nucleotideType', '') for v in info.values()]) == set(['DNA', 'RNA'])
                    for bcId in value['barcodes']:
                        barcodeInfo[bcId]['sample'] = sample

                        if 'reference' in info.get(bcId, {}):
                            barcodeInfo[bcId]['referenceName'] = info[bcId]['reference']

                        # exclude RNA barcodes from recalibration (Compendia project RNA/DNA sample)
                        if dna_rna_sample and info.get(bcId, {}).get('nucleotideType', '') == 'RNA':
                            barcodeInfo[bcId]['calibrate'] = False

                        # get the controlType from the barcode.json
                        barcodeInfo[bcId]['controlType'] = info[bcId].get('controlType', '')
                except:
                    pass

    return barcodeInfo


def create_runid(name):
    '''Returns 5 char string hashed from input string'''
    # Copied from TS/Analysis/file-io/ion_util.c
    def DEKHash(key):
        hash = len(key)
        for i in key:
            hash = ((hash << 5) ^ (hash >> 27)) ^ ord(i)
        return (hash & 0x7FFFFFFF)

    def base10to36(num):
        str = ''
        for i in range(5):
            digit = num % 36
            if digit < 26:
                str = chr(ord('A') + digit) + str
            else:
                str = chr(ord('0') + digit - 26) + str
            num /= 36
        return str

    return base10to36(DEKHash(name))


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
        "experiment": experiment,
        "resultsName": resultsName,
        "reportLink": link,  # Default_Report.php is implicit via Apache DirectoryIndex
        "status": "Pending",  # Used to be "Started"
        "log": j("log.html"),
        "analysisVersion": "_",
        "processedCycles": "0",
        "processedflows": "0",
        "framesProcessed": "0",
        "timeToComplete": 0,
        "reportstorage": server,
        }
    result = models.Results(**kwargs)
    result.save()  # generate the pk

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
