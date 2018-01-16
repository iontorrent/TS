# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
import json
import glob
import traceback

sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from django.db import connection, transaction, IntegrityError
from django.utils import timezone
from iondb.rundb import models
from iondb.rundb.report.views import ionstats_compute_stats
import subprocess
from ion.reports import parseBeadfind
from ion.utils.blockprocessing import parse_metrics
from ion.utils.textTo import fileToDict
import logging

logger = logging.getLogger(__name__)


def getCurrentAnalysis(procMetrics, res):
    def get_current_version():
        ver_map = {'analysis': 'an', 'alignment': 'al', 'dbreports': 'db', 'tmap': 'tm'}
        a = subprocess.Popen('ion_versionCheck.py', shell=True, stdout=subprocess.PIPE)
        ret = a.stdout.readlines()
        ver = {}
        for i in ret:
            ver[i.split('=')[0].strip()] = i.split('=')[1].strip()
        ret = []
        for name, shortname in ver_map.iteritems():
            if name in ver:
                ret.append('%s:%s,' % (shortname, ver[name]))
        return "".join(ret)

    res.analysisVersion = get_current_version()
    if procMetrics != None:
        res.processedflows = procMetrics.get('numFlows', 0)
        res.processedCycles = procMetrics.get('cyclesProcessed', 0)
        res.framesProcessed = procMetrics.get('framesProcessed', 0)
    res.save()
    return res


@transaction.commit_on_success
def addTfMetrics(tfMetrics, keyPeak, BaseCallerMetrics, res):
    # Populate metrics for each TF###

    if tfMetrics == None:
        return

    for tf, metrics in tfMetrics.iteritems():

        hpAccNum = metrics.get('Per HP accuracy NUM', [0])
        hpAccDen = metrics.get('Per HP accuracy DEN', [0])

        kwargs = {'report': res,
                  'name': tf,
                  'sequence': metrics.get('TF Seq', 'None'),
                  'number': metrics.get('Num', 0.0),
                  'keypass': metrics.get('Num', 0.0),   # Deprecated, populating by next best thing
                  'aveKeyCount': keyPeak.get('Test Fragment', '0'),
                  'SysSNR': metrics.get('System SNR', 0.0),

                  'Q10Histo': ' '.join(map(str, metrics.get('Q10', [0]))),
                  'Q10Mean': metrics.get('Q10 Mean', 0.0),
                  'Q10ReadCount': metrics.get('50Q10', 0.0),

                  'Q17Histo': ' '.join(map(str, metrics.get('Q17', [0]))),
                  'Q17Mean': metrics.get('Q17 Mean', 0.0),
                  'Q17ReadCount': metrics.get('50Q17', 0.0),

                  'HPAccuracy': ', '.join('%d : %d/%d' % (x, y[0], y[1]) for x, y in enumerate(zip(hpAccNum, hpAccDen))),
                  }

        tfm, created = models.TFMetrics.objects.get_or_create(report=res, name=tf,
                                                              defaults=kwargs)
        if not created:
            for key, value in kwargs.items():
                setattr(tfm, key, value)
            tfm.save()


@transaction.commit_on_success
def addAnalysisMetrics(beadMetrics, BaseCallerMetrics, res):
    # print 'addAnalysisMetrics'
    kwargs = {
        'report': res,
        'libLive': 0,
        'libKp': 0,
        'libFinal': 0,
        'tfLive': 0,
        'tfKp': 0,
        'tfFinal': 0,
        'lib_pass_basecaller': 0,
        'lib_pass_cafie': 0,
        'empty': 0,
        'bead': 0,
        'live': 0,
        'dud': 0,
        'amb': 0,
        'tf': 0,
        'lib': 0,
        'pinned': 0,
        'ignored': 0,
        'excluded': 0,
        'washout': 0,
        'washout_dud': 0,
        'washout_ambiguous': 0,
        'washout_live': 0,
        'washout_test_fragment': 0,
        'washout_library': 0,
        'keypass_all_beads': 0,
        'sysCF': 0.0,
        'sysIE': 0.0,
        'sysDR': 0.0,
        'libFinal': 0,
        'tfFinal': 0,
        'libMix': 0,
        'tfMix': 0,
        'total': 0,
        'adjusted_addressable': 0,
        'loading': 0.0,
    }

    bead_metrics_map = {
        'empty': 'Empty Wells',
        'bead': 'Bead Wells',
        'live': 'Live Beads',
        'dud': 'Dud Beads',
        'amb': 'Ambiguous Beads',
        'tf': 'Test Fragment Beads',
        'lib': 'Library Beads',
        'pinned': 'Pinned Wells',
        'ignored': 'Ignored Wells',
        'excluded': 'Excluded Wells',
        'washout': 'Washout Wells',
        'washout_dud': 'Washout Dud',
        'washout_ambiguous': 'Washout Ambiguous',
        'washout_live': 'Washout Live',
        'washout_test_fragment': 'Washout Test Fragment',
        'washout_library': 'Washout Library',
        'keypass_all_beads': 'Keypass Beads',
        'total': 'Total Wells',
        'adjusted_addressable': 'Adjusted Addressable Wells',
    }

    if BaseCallerMetrics:
        try:
            kwargs["libFinal"] = BaseCallerMetrics["Filtering"]["ReadDetails"]["lib"]["valid"]
            kwargs["tfFinal"] = BaseCallerMetrics["Filtering"]["ReadDetails"]["tf"]["valid"]
        except Exception as err:
            print("During AnalysisMetrics creation, reading from BaseCaller.json: %s", err)

    if beadMetrics:
        for dbname, key in bead_metrics_map.iteritems():
            kwargs[dbname] = set_type(beadMetrics.get(key, 0))

        if not kwargs['adjusted_addressable']:
            kwargs['adjusted_addressable'] = kwargs['total'] - kwargs["excluded"]
        if kwargs['adjusted_addressable']:
            kwargs['loading'] = 100 * float(kwargs['bead']) / kwargs['adjusted_addressable']

    if BaseCallerMetrics:
        try:
            kwargs['sysCF'] = 100.0 * BaseCallerMetrics['Phasing']['CF']
            kwargs['sysIE'] = 100.0 * BaseCallerMetrics['Phasing']['IE']
            kwargs['sysDR'] = 100.0 * BaseCallerMetrics['Phasing']['DR']
        except Exception as err:
            print("During AnalysisMetrics creation, reading from BaseCaller.json: %s", err)

    analysismetrics = res.analysismetrics or models.AnalysisMetrics()
    for key, value in kwargs.items():
        setattr(analysismetrics, key, value)
    analysismetrics.save()
    res.analysismetrics = analysismetrics
    res.save()


def updateAnalysisMetrics(beadPath, primarykeyPath):
    """Create or Update AnalysisMetrics with only bfmask.stats info"""
    result = None
    for line in open(primarykeyPath):
        if line.startswith("ResultsPK"):
            rpk = int(line.split("=")[1])
            result = models.Results.objects.get(pk=rpk)
            break
    if not result:
        logger.error("Primary key %s not available", primarykeyPath)
    beadMetrics = parseBeadfind.generateMetrics(beadPath)
    try:
        addAnalysisMetrics(beadMetrics, None, result)
    except Exception as err:
        logger.exception()
        return str(err)
    return "Alls well that ends well"


def set_type(num_string):
    val = 0
    if num_string == "NA":
        return -1
    if num_string:
        try:
            if '.' in num_string:
                val = float(num_string)
            else:
                val = int(num_string)
        except ValueError:
            val = str(num_string)
    return val


@transaction.commit_on_success
def addLibMetrics(genomeinfodict, ionstats_alignment, ionstats_basecaller, keyPeak, BaseCallerMetrics, res, extra):
    print 'addlibmetrics'

    if keyPeak != None:
        aveKeyCount = float(keyPeak.get('Library', 0.0))
    else:
        aveKeyCount = 0.0

    align_sample = 0
    if ionstats_alignment == None:
        align_sample = -1
    # check to see if this is a samled or full alignment
    # if libMetrics.has_key('Total number of Sampled Reads'):
    #    align_sample = 1
    if res.metaData.get('thumb', '0') == 1:
        align_sample = 2

    kwargs = {'report': res, 'aveKeyCounts': aveKeyCount, 'align_sample': align_sample}

    kwargs['Genome_Version'] = genomeinfodict['genome_version']
    kwargs['Index_Version'] = genomeinfodict['index_version']
    kwargs['genome'] = genomeinfodict['genome_name']
    kwargs['genomesize'] = genomeinfodict['genome_length']

    kwargs['totalNumReads'] = ionstats_basecaller['full']['num_reads']

    quallist = ['7', '10', '17', '20', '47']  # TODO Q30
    bplist = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    if ionstats_alignment != None:
        kwargs['total_mapped_reads'] = ionstats_alignment['aligned']['num_reads']
        kwargs['total_mapped_target_bases'] = ionstats_alignment['aligned']['num_bases']

        for q in quallist:
            kwargs['q%s_longest_alignment' % q] = ionstats_alignment['AQ'+q]['max_read_length']
            kwargs['q%s_mean_alignment_length' % q] = ionstats_alignment['AQ'+q]['mean_read_length']
            kwargs['q%s_mapped_bases' % q] = ionstats_alignment['AQ'+q]['num_bases']
            kwargs['q%s_alignments' % q] = ionstats_alignment['AQ'+q]['num_reads']
            kwargs['q%s_coverage_percentage' % q] = 0.0  # 'N/A' # TODO
#           'Filtered %s Mean Coverage Depth' % q, '%.1f' % (float(ionstats_alignment['AQ'+q]['num_bases'])/float(3095693981)) ) TODO
            for bp in bplist:
                kwargs['i%sQ%s_reads' % (bp, q)] = sum(ionstats_alignment['AQ'+q]['read_length_histogram'][bp:])

        try:
            raw_accuracy = round((1 - float(sum(ionstats_alignment['error_by_position'])) / float(ionstats_alignment['aligned']['num_bases'])) * 100.0, 2)
            kwargs['raw_accuracy'] = raw_accuracy
        except:
            kwargs['raw_accuracy'] = 0.0
    else:
        kwargs['total_mapped_reads'] = 0
        kwargs['total_mapped_target_bases'] = 0

        for q in quallist:
            for bp in bplist:
                kwargs['i%sQ%s_reads' % (bp, q)] = 0
            kwargs['q%s_longest_alignment' % q] = 0
            kwargs['q%s_mean_alignment_length' % q] = 0
            kwargs['q%s_mapped_bases' % q] = 0
            kwargs['q%s_alignments' % q] = 0
            kwargs['q%s_coverage_percentage' % q] = 0.0
            kwargs['raw_accuracy'] = 0.0

    try:
        kwargs['sysSNR'] = ionstats_basecaller['system_snr']
        kwargs['cf'] = 100.0 * BaseCallerMetrics['Phasing']['CF']
        kwargs['ie'] = 100.0 * BaseCallerMetrics['Phasing']['IE']
        kwargs['dr'] = 100.0 * BaseCallerMetrics['Phasing']['DR']
    except:
        kwargs['sysSNR'] = 0.0
        kwargs['cf'] = 0.0
        kwargs['ie'] = 0.0
        kwargs['dr'] = 0.0

    if extra:
        kwargs["duplicate_reads"] = extra.get("duplicate_reads", None)

    libmetrics = res.libmetrics or models.LibMetrics()
    for key, value in kwargs.items():
        setattr(libmetrics, key, value)
    libmetrics.save()
    res.libmetrics = libmetrics
    res.save()


def get_some_quality_metrics(quality, ionstats, qv_histogram):
    kwargs = {}

    read_length_histogram = ionstats['read_length_histogram']
    for size in [50, 100, 150]:
        if len(read_length_histogram) > size:
            kwargs["q%d_%dbp_reads" % (quality, size)] = sum(read_length_histogram[size:])
        else:
            kwargs["q%d_%dbp_reads" % (quality, size)] = 0

    stats = ionstats_compute_stats(ionstats)
    kwargs["q%d_bases" % quality] = sum(qv_histogram[quality:])
    kwargs["q%d_reads" % quality] = ionstats["num_reads"]
    kwargs["q%d_max_read_length" % quality] = ionstats["max_read_length"]
    kwargs["q%d_mean_read_length" % quality] = stats['mean_length']
    kwargs["q%d_median_read_length" % quality] = stats['median_length']
    kwargs["q%d_mode_read_length" % quality] = stats['mode_length']

    return kwargs


@transaction.commit_on_success
def addQualityMetrics(ionstats_basecaller, res):
    kwargs = {'report': res}
    for quality, key in [(0, 'full'), (17, 'Q17'), (20, 'Q20')]:
        try:
            metrics = get_some_quality_metrics(quality, ionstats_basecaller["full"],
                                               ionstats_basecaller["qv_histogram"])
            logger.debug("\n".join(str(l) for l in sorted(metrics.items())))
            kwargs.update(metrics)
        except Exception as err:
            logger.exception("During %s QualityMetrics creation: %s" % (key, err))

    logger.info("\n".join(str(l) for l in sorted(kwargs.items())))
    qualitymetrics = res.qualitymetrics or models.QualityMetrics()
    for key, value in kwargs.items():
        setattr(qualitymetrics, key, value)
    qualitymetrics.save()
    res.qualitymetrics = qualitymetrics
    res.save()


@transaction.commit_on_success
def pluginStoreInsert(pluginDict):
    """insert plugin data into the django database"""
    print "Insert Plugin data into database"
    f = open('primary.key', 'r')
    pk = f.readlines()
    pkDict = {}
    for line in pk:
        parsline = line.strip().split("=")
        key = parsline[0].strip().lower()
        value = parsline[1].strip()
        pkDict[key] = value
    f.close()
    rpk = pkDict['resultspk']
    res = models.Results.objects.get(pk=rpk)
    res.pluginStore = json.dumps(pluginDict)
    res.save()


@transaction.commit_on_success
def updateStatus(primarykeyPath, status, reportLink=False):
    """
    A function to update the status of reports
    """
    f = open(primarykeyPath, 'r')
    pk = f.readlines()
    pkDict = {}
    for line in pk:
        parsline = line.strip().split("=")
        key = parsline[0].strip().lower()
        value = parsline[1].strip()
        pkDict[key] = value
    f.close()

    rpk = pkDict['resultspk']
    res = models.Results.objects.get(pk=rpk)

    # by default make this "Started"
    if status:
        res.status = status
    else:
        res.status = "Started"

    if not reportLink:
        res.reportLink = res.log
    else:
        # Commenting out because reportLink is already set correctly in this case
        # I want to be able to call this function with reportLink == True
        # res.reportLink = reportLink
        pass

    res.save()


def writeDbFromFiles(tfPath, procPath, beadPath, ionstats_alignment_json_path, ionParamsPath, status, keyPath, ionstats_basecaller_json_path, BaseCallerJsonPath, primarykeyPath, uploadStatusPath, cwd):

    return_message = ""

    afile = open(ionParamsPath, 'r')
    ionparams = json.load(afile)
    afile.close()

    procParams = None
    if os.path.exists(procPath):
        procParams = fileToDict(procPath)

    beadMetrics = None
    if os.path.isfile(beadPath):
        try:
            beadMetrics = parseBeadfind.generateMetrics(beadPath)
        except:
            beadMetrics = None
            return_message += traceback.format_exc()
    else:
        beadMetrics = None
        return_message += 'ERROR: generating beadMetrics failed - file %s is missing\n' % beadPath

    tfMetrics = None
    if os.path.exists(tfPath):
        try:
            file = open(tfPath, 'r')
            tfMetrics = json.load(file)
            file.close()
        except:
            tfMetrics = None
            return_message += traceback.format_exc()
    else:
        tfMetrics = None
        return_message += 'ERROR: generating tfMetrics failed - file %s is missing\n' % tfPath

    keyPeak = {
        'Test Fragment': 0,
        'Library': 0
    }
    if os.path.exists(keyPath):
        keyPeak.update(parse_metrics(keyPath))

    BaseCallerMetrics = None
    if os.path.exists(BaseCallerJsonPath):
        try:
            afile = open(BaseCallerJsonPath, 'r')
            BaseCallerMetrics = json.load(afile)
            afile.close()
        except:
            BaseCallerMetrics = None
            return_message += traceback.format_exc()
    else:
        BaseCallerMetrics = None
        return_message += 'ERROR: generating BaseCallerMetrics failed - file %s is missing\n' % BaseCallerJsonPath

    ionstats_basecaller = None
    if os.path.exists(ionstats_basecaller_json_path):
        try:
            afile = open(ionstats_basecaller_json_path, 'r')
            ionstats_basecaller = json.load(afile)
            afile.close()
        except:
            ionstats_basecaller = None
            return_message += traceback.format_exc()
    else:
        ionstats_basecaller = None
        return_message += 'ERROR: generating ionstats_basecaller failed - file %s is missing\n' % ionstats_basecaller_json_path

    ionstats_alignment = None
    if ionparams['referenceName']:
        if os.path.exists(ionstats_alignment_json_path):
            try:
                afile = open(ionstats_alignment_json_path, 'r')
                ionstats_alignment = json.load(afile)
                afile.close()
            except:
                ionstats_alignment = None
                return_message += traceback.format_exc()
        else:
            ionstats_alignment = None
            return_message += 'ERROR: generating ionstats_alignment failed - file %s is missing\n' % ionstats_alignment_json_path

    genomeinfodict = {}
    try:
        if ionparams['referenceName']:
            genomeinfofilepath = '/results/referenceLibrary/%s/%s/%s.info.txt' % (ionparams['tmap_version'], ionparams['referenceName'], ionparams['referenceName'])
            with open(genomeinfofilepath) as genomeinfofile:
                for line in genomeinfofile:
                    key, value = line.partition("\t")[::2]
                    genomeinfodict[key.strip()] = value.strip()
        else:
            genomeinfodict['genome_version'] = 'None'
            genomeinfodict['index_version'] = 'None'
            genomeinfodict['genome_name'] = 'None'
            genomeinfodict['genome_length'] = 0
    except:
        return_message += traceback.format_exc()

    extra_files = glob.glob(os.path.join(cwd, 'BamDuplicates*.json'))
    extra = {u'reads_with_adaptor': 0, u'duplicate_reads': 0, u'total_reads': 0, u'fraction_with_adaptor': 0, u'fraction_duplicates': 0}
    for extra_file in extra_files:
        if os.path.exists(extra_file):
            try:
                with open(extra_file, 'r') as afile:
                    val = json.load(afile)
                    for key in extra.keys():
                        extra[key] += val.get(key, 0)
            except:
                return_message += traceback.format_exc()
        else:
            return_message += 'INFO: generating extra metrics failed - file %s is missing\n' % extra_file
    else:
        # No data, send None/null instead of zeros
        extra = {u'reads_with_adaptor': None, u'duplicate_reads': None, u'total_reads': None, u'fraction_with_adaptor': None, u'fraction_duplicates': None}

    try:
        if extra[u'total_reads']:
            extra[u'fraction_with_adaptor'] = extra[u'reads_with_adaptor'] / float(extra[u'total_reads'])
            extra[u'fraction_duplicates'] = extra[u'duplicate_reads'] / float(extra[u'total_reads'])
    except:
        return_message += traceback.format_exc()

    # print "BamDuplicates stats: ", extra

    writeDbFromDict(tfMetrics, procParams, beadMetrics, ionstats_alignment, genomeinfodict, status, keyPeak, ionstats_basecaller, BaseCallerMetrics, primarykeyPath, uploadStatusPath, extra)

    return return_message


def writeDbFromDict(tfMetrics, procParams, beadMetrics, ionstats_alignment, genomeinfodict, status, keyPeak, ionstats_basecaller, BaseCallerMetrics, primarykeyPath, uploadStatusPath, extra):
    print "writeDbFromDict"

    # We think this will fix "DatabaseError: server closed the connection unexpectedly"
    # which happens rather randomly.
    connection.close()

    f = open(primarykeyPath, 'r')
    pk = f.readlines()
    pkDict = {}
    for line in pk:
        parsline = line.strip().split("=")
        key = parsline[0].strip().lower()
        value = parsline[1].strip()
        pkDict[key] = value
    f.close()

    e = open(uploadStatusPath, 'w')

    rpk = pkDict['resultspk']
    res = models.Results.objects.get(pk=rpk)
    if status != None:
        res.status = status
    res.timeStamp = timezone.now()
    res.save()
    experiment = res.experiment
    experiment.resultDate = res.timeStamp
    experiment.save()

    try:
        e.write('Updating Analysis\n')
        getCurrentAnalysis(procParams, res)
    except:
        e.write("Failed getCurrentAnalysis\n")
        logger.exception("Failed getCurrentAnalysis")
    try:
        e.write('Adding TF Metrics\n')
        addTfMetrics(tfMetrics, keyPeak, BaseCallerMetrics, res)
    except IntegrityError:
        e.write("Failed addTfMetrics\n")
        logger.exception("Failed addTfMetrics")
    except:
        e.write("Failed addTfMetrics\n")
        logger.exception("Failed addTfMetrics")
    try:
        e.write('Adding Analysis Metrics\n')
        addAnalysisMetrics(beadMetrics, BaseCallerMetrics, res)
    except IntegrityError:
        e.write("Failed addAnalysisMetrics\n")
        logger.exception("Failed addAnalysisMetrics")
    except:
        e.write("Failed addAnalysisMetrics\n")
        logger.exception("Failed addAnalysisMetrics")
    try:
        e.write('Adding Library Metrics\n')
        addLibMetrics(genomeinfodict, ionstats_alignment, ionstats_basecaller, keyPeak, BaseCallerMetrics, res, extra)
    except IntegrityError:
        e.write("Failed addLibMetrics\n")
        logger.exception("Failed addLibMetrics")
    except:
        e.write("Failed addLibMetrics\n")
        logger.exception("Failed addLibMetrics")

    # try to add the quality metrics
    try:
        e.write('Adding Quality Metrics\n')
        if ionstats_basecaller:
            addQualityMetrics(ionstats_basecaller, res)
        else:
            e.write("Failed to add QualityMetrics, missing object ionstats_basecaller\n")
    except IntegrityError:
        e.write("Failed addQualityMetrics\n")
        logger.exception("Failed addQualityMetrics")
    except:
        e.write("Failed addQualityMetrics\n")
        logger.exception("Failed addQualityMetrics")

    e.close()

if __name__ == '__main__':
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    logger.info("Started")
    if len(sys.argv) < 2:
        folderPath = os.getcwd()
        logger.warn("No path specified. Assuming cwd: '%s'", folderPath)
    else:
        folderPath = sys.argv[1]

    SIGPROC_RESULTS = "sigproc_results"
    BASECALLER_RESULTS = "basecaller_results"
    ALIGNMENT_RESULTS = "./"

    print "processing: " + folderPath

    status = None

    tfPath = os.path.join(folderPath, BASECALLER_RESULTS, 'TFStats.json')
    procPath = os.path.join(SIGPROC_RESULTS, "processParameters.txt")
    beadPath = os.path.join(folderPath, SIGPROC_RESULTS, 'analysis.bfmask.stats')
    ionstats_alignment_json_path = os.path.join(folderPath, ALIGNMENT_RESULTS, 'ionstats_alignment.json')
    ionParamsPath = os.path.join(folderPath, 'ion_params_00.json')
    primarykeyPath = os.path.join(folderPath, 'primary.key')
    BaseCallerJsonPath = os.path.join(folderPath, BASECALLER_RESULTS, 'BaseCaller.json')
    ionstats_basecaller_json_path = os.path.join(folderPath, BASECALLER_RESULTS, 'ionstats_basecaller.json')
    keyPath = os.path.join(folderPath, 'raw_peak_signal')
    uploadStatusPath = os.path.join(folderPath, 'status.txt')

    ret_messages = writeDbFromFiles(tfPath,
                                    procPath,
                                    beadPath,
                                    ionstats_alignment_json_path,
                                    ionParamsPath,
                                    status,
                                    keyPath,
                                    ionstats_basecaller_json_path,
                                    BaseCallerJsonPath,
                                    primarykeyPath,
                                    uploadStatusPath,
                                    folderPath)

    print("messages: '%s'" % ret_messages)
