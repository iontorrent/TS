# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
import json
import glob
import datetime
import traceback

sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from django.db import connection, transaction, IntegrityError
from iondb.rundb import models
import subprocess
from ion.reports import parseBeadfind
from ion.utils.blockprocessing import parse_metrics
from ion.utils.textTo import fileToDict
import logging

def getCurrentAnalysis(procMetrics, res):
    def get_current_version():
        ver_map = {'analysis':'an','alignment':'al','dbreports':'db', 'tmap' : 'tm' }
        a = subprocess.Popen('ion_versionCheck.py', shell=True, stdout=subprocess.PIPE)
        ret = a.stdout.readlines()
        ver = {}
        for i in ret:
            ver[i.split('=')[0].strip()]=i.split('=')[1].strip()
        ret = []
        for name, shortname in ver_map.iteritems():
            if name in ver:
                ret.append('%s:%s,' % (shortname,ver[name]))
        return "".join(ret)

    res.analysisVersion = get_current_version()
    if procMetrics != None:
        res.processedflows =  procMetrics.get('numFlows',0)
        res.processedCycles = procMetrics.get('cyclesProcessed',0)
        res.framesProcessed = procMetrics.get('framesProcessed',0)
    res.save()
    return res

@transaction.commit_on_success
def addTfMetrics(tfMetrics, keyPeak, BaseCallerMetrics, res):
    ###Populate metrics for each TF###
    
    if tfMetrics == None:
        return

    for tf, metrics in tfMetrics.iteritems():
        
        hpAccNum = metrics.get('Per HP accuracy NUM',[0])
        hpAccDen = metrics.get('Per HP accuracy DEN',[0])
        
        kwargs = {'report'                  : res,
                  'name'                    : tf,
                  'sequence'                : metrics.get('TF Seq','None'),
                  'number'                  : metrics.get('Num',0.0),
                  'keypass'                 : metrics.get('Num',0.0),   # Deprecated, populating by next best thing
                  'aveKeyCount'             : keyPeak.get('Test Fragment','0'),
                  'SysSNR'                  : metrics.get('System SNR',0.0),

                  'Q10Histo'                : ' '.join(map(str,metrics.get('Q10',[0]))),
                  'Q10Mean'                 : metrics.get('Q10 Mean',0.0),
                  'Q10ReadCount'            : metrics.get('50Q10',0.0),
                  
                  'Q17Histo'                : ' '.join(map(str,metrics.get('Q17',[0]))),
                  'Q17Mean'                 : metrics.get('Q17 Mean',0.0),
                  'Q17ReadCount'            : metrics.get('50Q17',0.0),

                  'HPAccuracy'              : ', '.join('%d : %d/%d' % (x,y[0],y[1]) for x,y in enumerate(zip(hpAccNum,hpAccDen))),
                  }
        
        tfm, created = models.TFMetrics.objects.get_or_create(report=res, name=tf,
                                                                defaults=kwargs)
        if not created:
            for key, value in kwargs.items():
                setattr(tfm, key, value)
            tfm.save()


@transaction.commit_on_success
def addAnalysisMetrics(beadMetrics, BaseCallerMetrics, res):
    #print 'addAnalysisMetrics'
    analysis_metrics_map = {
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
        'sysCF' : 0.0,
        'sysIE' : 0.0,
        'sysDR' : 0.0,
        'libFinal' : 0,
        'tfFinal' : 0,
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
        'keypass_all_beads': 'Keypass Beads'
    }

    kwargs = {'report':res,'libMix':0,'tfMix':0,'sysCF':0.0,'sysIE':0.0,'sysDR':0.0}

    if BaseCallerMetrics:
        try:
            analysis_metrics_map["libFinal"] = BaseCallerMetrics["Filtering"]["ReadDetails"]["lib"]["valid"]
            analysis_metrics_map["tfFinal"] = BaseCallerMetrics["Filtering"]["ReadDetails"]["tf"]["valid"]
        except Exception as err:
            print("During AnalysisMetrics creation, reading from BaseCaller.json: %s", err)

    kwargs.update(analysis_metrics_map)

    if beadMetrics:
        for dbname, key in bead_metrics_map.iteritems():
            kwargs[dbname]=set_type(beadMetrics.get(key, 0))
    
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
        logging.error("Primary key %s not available", primarykeyPath)
    beadMetrics = parseBeadfind.generateMetrics(beadPath)
    try:
        addAnalysisMetrics(beadMetrics, None, result)
    except Exception as err:
        logging.exception()
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
    metric_map = {
    'genomelength':'Genomelength',
    'rNumAlignments':'Filtered relaxed BLAST Alignments',
    'rMeanAlignLen':'Filtered relaxed BLAST Mean Alignment Length',
    'rLongestAlign':'Filtered relaxed BLAST Longest Alignment',
    'rCoverage':'Filtered relaxed BLAST coverage percentage',
    'r50Q10':'Filtered relaxed BLAST 50Q10 Reads',
    'r100Q10':'Filtered relaxed BLAST 100Q10 Reads',
    'r200Q10':'Filtered relaxed BLAST 200Q10 Reads',
    'r50Q17':'Filtered relaxed BLAST 50Q17 Reads',
    'r100Q17':'Filtered relaxed BLAST 100Q17 Reads',
    'r200Q17':'Filtered relaxed BLAST 200Q17 Reads',
    'r50Q20':'Filtered relaxed BLAST 50Q20 Reads',
    'r100Q20':'Filtered relaxed BLAST 100Q20 Reads',
    'r200Q20':'Filtered relaxed BLAST 200Q20 Reads',
    'sNumAlignments':'Filtered strict BLAST Alignments',
    'sMeanAlignLen':'Filtered strict BLAST Mean Alignment Length',
    'sCoverage':'Filtered strict BLAST coverage percentage',
    'sLongestAlign':'Filtered strict BLAST Longest Alignment',
    's50Q10':'Filtered strict BLAST 50Q10 Reads',
    's100Q10':'Filtered strict BLAST 100Q10 Reads',
    's200Q10':'Filtered strict BLAST 200Q10 Reads',
    's50Q17':'Filtered strict BLAST 50Q17 Reads',
    's100Q17':'Filtered strict BLAST 100Q17 Reads',
    's200Q17':'Filtered strict BLAST 200Q17 Reads',
    's50Q20':'Filtered strict BLAST 50Q20 Reads',
    's100Q20':'Filtered strict BLAST 100Q20 Reads',
    's200Q20':'Filtered strict BLAST 200Q20 Reads',
    "q7_qscore_bases":"Filtered Mapped Q7 Bases",
    "q10_qscore_bases":"Filtered Mapped Q10 Bases",
    "q17_qscore_bases":"Filtered Mapped Q17 Bases",
    "q20_qscore_bases":"Filtered Mapped Q20 Bases",
    "q47_qscore_bases":"Filtered Mapped Q47 Bases",
    'total_number_of_sampled_reads':'Total number of Sampled Reads',
    'sampled_q7_coverage_percentage':'Sampled Filtered Q7 Coverage Percentage',
    'sampled_q7_mean_coverage_depth':'Sampled Filtered Q7 Mean Coverage Depth',
    'sampled_q7_alignments':'Sampled Filtered Q7 Alignments',
    'sampled_q7_mean_alignment_length':'Sampled Filtered Q7 Mean Alignment Length',
    'sampled_mapped_bases_in_q7_alignments':'Sampled Filtered Mapped Bases in Q7 Alignments',
    'sampled_q7_longest_alignment':'Sampled Filtered Q7 Longest Alignment',
    'sampled_50q7_reads':'Sampled Filtered 50Q7 Reads',
    'sampled_100q7_reads':'Sampled Filtered 100Q7 Reads',
    'sampled_200q7_reads':'Sampled Filtered 200Q7 Reads',
    'sampled_300q7_reads':'Sampled Filtered 300Q7 Reads',
    'sampled_400q7_reads':'Sampled Filtered 400Q7 Reads',
    'sampled_q10_coverage_percentage':'Sampled Filtered Q10 Coverage Percentage',
    'sampled_q10_mean_coverage_depth':'Sampled Filtered Q10 Mean Coverage Depth',
    'sampled_q10_alignments':'Sampled Filtered Q10 Alignments',
    'sampled_q10_mean_alignment_length':'Sampled Filtered Q10 Mean Alignment Length',
    'sampled_mapped_bases_in_q10_alignments':'Sampled Filtered Mapped Bases in Q10 Alignments',
    'sampled_q10_longest_alignment':'Sampled Filtered Q10 Longest Alignment',
    'sampled_50q10_reads':'Sampled Filtered 50Q10 Reads',
    'sampled_100q10_reads':'Sampled Filtered 100Q10 Reads',
    'sampled_200q10_reads':'Sampled Filtered 200Q10 Reads',
    'sampled_300q10_reads':'Sampled Filtered 300Q10 Reads',
    'sampled_400q10_reads':'Sampled Filtered 400Q10 Reads',
    'sampled_q17_coverage_percentage':'Sampled Filtered Q17 Coverage Percentage',
    'sampled_q17_mean_coverage_depth':'Sampled Filtered Q17 Mean Coverage Depth',
    'sampled_q17_alignments':'Sampled Filtered Q17 Alignments',
    'sampled_q17_mean_alignment_length':'Sampled Filtered Q17 Mean Alignment Length',
    'sampled_mapped_bases_in_q17_alignments':'Sampled Filtered Mapped Bases in Q17 Alignments',
    'sampled_q17_longest_alignment':'Sampled Filtered Q17 Longest Alignment',
    'sampled_50q17_reads':'Sampled Filtered 50Q17 Reads',
    'sampled_100q17_reads':'Sampled Filtered 100Q17 Reads',
    'sampled_200q17_reads':'Sampled Filtered 200Q17 Reads',
    'sampled_300q17_reads':'Sampled Filtered 300Q17 Reads',
    'sampled_400q17_reads':'Sampled Filtered 400Q17 Reads',
    'sampled_q20_coverage_percentage':'Sampled Filtered Q20 Coverage Percentage',
    'sampled_q20_mean_coverage_depth':'Sampled Filtered Q20 Mean Coverage Depth',
    'sampled_q20_alignments':'Sampled Filtered Q20 Alignments',
    'sampled_q20_mean_alignment_length':'Sampled Filtered Q20 Mean Alignment Length',
    'sampled_mapped_bases_in_q20_alignments':'Sampled Filtered Mapped Bases in Q20 Alignments',
    'sampled_q20_longest_alignment':'Sampled Filtered Q20 Longest Alignment',
    'sampled_50q20_reads':'Sampled Filtered 50Q20 Reads',
    'sampled_100q20_reads':'Sampled Filtered 100Q20 Reads',
    'sampled_200q20_reads':'Sampled Filtered 200Q20 Reads',
    'sampled_300q20_reads':'Sampled Filtered 300Q20 Reads',
    'sampled_400q20_reads':'Sampled Filtered 400Q20 Reads',
    'sampled_q47_coverage_percentage':'Sampled Filtered Q47 Coverage Percentage',
    'sampled_q47_mean_coverage_depth':'Sampled Filtered Q47 Mean Coverage Depth',
    'sampled_q47_alignments':'Sampled Filtered Q47 Alignments',
    'sampled_q47_mean_alignment_length':'Sampled Filtered Q47 Mean Alignment Length',
    'sampled_mapped_bases_in_q47_alignments':'Sampled Filtered Mapped Bases in Q47 Alignments',
    'sampled_q47_longest_alignment':'Sampled Filtered Q47 Longest Alignment',
    'sampled_50q47_reads':'Sampled Filtered 50Q47 Reads',
    'sampled_100q47_reads':'Sampled Filtered 100Q47 Reads',
    'sampled_200q47_reads':'Sampled Filtered 200Q47 Reads',
    'sampled_300q47_reads':'Sampled Filtered 300Q47 Reads',
    'sampled_400q47_reads':'Sampled Filtered 400Q47 Reads',
    'extrapolated_from_number_of_sampled_reads':'Extrapolated from number of Sampled Reads',
    'extrapolated_q7_coverage_percentage':'Extrapolated Filtered Q7 Coverage Percentage',
    'extrapolated_q7_mean_coverage_depth':'Extrapolated Filtered Q7 Mean Coverage Depth',
    'extrapolated_q7_alignments':'Extrapolated Filtered Q7 Alignments',
    'extrapolated_q7_mean_alignment_length':'Extrapolated Filtered Q7 Mean Alignment Length',
    'extrapolated_mapped_bases_in_q7_alignments':'Extrapolated Filtered Mapped Bases in Q7 Alignments',
    'extrapolated_q7_longest_alignment':'Extrapolated Filtered Q7 Longest Alignment',
    'extrapolated_50q7_reads':'Extrapolated Filtered 50Q7 Reads',
    'extrapolated_100q7_reads':'Extrapolated Filtered 100Q7 Reads',
    'extrapolated_200q7_reads':'Extrapolated Filtered 200Q7 Reads',
    'extrapolated_300q7_reads':'Extrapolated Filtered 300Q7 Reads',
    'extrapolated_400q7_reads':'Extrapolated Filtered 400Q7 Reads',
    'extrapolated_q10_coverage_percentage':'Extrapolated Filtered Q10 Coverage Percentage',
    'extrapolated_q10_mean_coverage_depth':'Extrapolated Filtered Q10 Mean Coverage Depth',
    'extrapolated_q10_alignments':'Extrapolated Filtered Q10 Alignments',
    'extrapolated_q10_mean_alignment_length':'Extrapolated Filtered Q10 Mean Alignment Length',
    'extrapolated_mapped_bases_in_q10_alignments':'Extrapolated Filtered Mapped Bases in Q10 Alignments',
    'extrapolated_q10_longest_alignment':'Extrapolated Filtered Q10 Longest Alignment',
    'extrapolated_50q10_reads':'Extrapolated Filtered 50Q10 Reads',
    'extrapolated_100q10_reads':'Extrapolated Filtered 100Q10 Reads',
    'extrapolated_200q10_reads':'Extrapolated Filtered 200Q10 Reads',
    'extrapolated_300q10_reads':'Extrapolated Filtered 300Q10 Reads',
    'extrapolated_400q10_reads':'Extrapolated Filtered 400Q10 Reads',
    'extrapolated_q17_coverage_percentage':'Extrapolated Filtered Q17 Coverage Percentage',
    'extrapolated_q17_mean_coverage_depth':'Extrapolated Filtered Q17 Mean Coverage Depth',
    'extrapolated_q17_alignments':'Extrapolated Filtered Q17 Alignments',
    'extrapolated_q17_mean_alignment_length':'Extrapolated Filtered Q17 Mean Alignment Length',
    'extrapolated_mapped_bases_in_q17_alignments':'Extrapolated Filtered Mapped Bases in Q17 Alignments',
    'extrapolated_q17_longest_alignment':'Extrapolated Filtered Q17 Longest Alignment',
    'extrapolated_50q17_reads':'Extrapolated Filtered 50Q17 Reads',
    'extrapolated_100q17_reads':'Extrapolated Filtered 100Q17 Reads',
    'extrapolated_200q17_reads':'Extrapolated Filtered 200Q17 Reads',
    'extrapolated_300q17_reads':'Extrapolated Filtered 300Q17 Reads',
    'extrapolated_400q17_reads':'Extrapolated Filtered 400Q17 Reads',
    'extrapolated_q20_coverage_percentage':'Extrapolated Filtered Q20 Coverage Percentage',
    'extrapolated_q20_mean_coverage_depth':'Extrapolated Filtered Q20 Mean Coverage Depth',
    'extrapolated_q20_alignments':'Extrapolated Filtered Q20 Alignments',
    'extrapolated_q20_mean_alignment_length':'Extrapolated Filtered Q20 Mean Alignment Length',
    'extrapolated_mapped_bases_in_q20_alignments':'Extrapolated Filtered Mapped Bases in Q20 Alignments',
    'extrapolated_q20_longest_alignment':'Extrapolated Filtered Q20 Longest Alignment',
    'extrapolated_50q20_reads':'Extrapolated Filtered 50Q20 Reads',
    'extrapolated_100q20_reads':'Extrapolated Filtered 100Q20 Reads',
    'extrapolated_200q20_reads':'Extrapolated Filtered 200Q20 Reads',
    'extrapolated_300q20_reads':'Extrapolated Filtered 300Q20 Reads',
    'extrapolated_400q20_reads':'Extrapolated Filtered 400Q20 Reads',
    'extrapolated_q47_coverage_percentage':'Extrapolated Filtered Q47 Coverage Percentage',
    'extrapolated_q47_mean_coverage_depth':'Extrapolated Filtered Q47 Mean Coverage Depth',
    'extrapolated_q47_alignments':'Extrapolated Filtered Q47 Alignments',
    'extrapolated_q47_mean_alignment_length':'Extrapolated Filtered Q47 Mean Alignment Length',
    'extrapolated_mapped_bases_in_q47_alignments':'Extrapolated Filtered Mapped Bases in Q47 Alignments',
    'extrapolated_q47_longest_alignment':'Extrapolated Filtered Q47 Longest Alignment',
    'extrapolated_50q47_reads':'Extrapolated Filtered 50Q47 Reads',
    'extrapolated_100q47_reads':'Extrapolated Filtered 100Q47 Reads',
    'extrapolated_200q47_reads':'Extrapolated Filtered 200Q47 Reads',
    'extrapolated_300q47_reads':'Extrapolated Filtered 300Q47 Reads',
    'extrapolated_400q47_reads':'Extrapolated Filtered 400Q47 Reads',
    'duplicate_reads': 'Count of Duplicate Reads',
    }

    if keyPeak != None:
        aveKeyCount = float(keyPeak.get('Library',0.0))
    else:
        aveKeyCount = 0.0
        
    align_sample = 0
    if ionstats_alignment == None:
        align_sample = -1
    #check to see if this is a samled or full alignment 
    #if libMetrics.has_key('Total number of Sampled Reads'):
    #    align_sample = 1
    if res.metaData.get('thumb','0') == 1:
        align_sample = 2

    kwargs = {'report':res, 'aveKeyCounts':aveKeyCount, 'align_sample': align_sample }

    for dbname, key in metric_map.iteritems():
        kwargs[dbname] = set_type('0')

    kwargs['Genome_Version'] = genomeinfodict['genome_version']
    kwargs['Index_Version'] = genomeinfodict['index_version']
    kwargs['genome'] = genomeinfodict['genome_name']
    kwargs['genomesize'] = genomeinfodict['genome_length']

    quallist = ['7', '10', '17', '20', '47'] #TODO Q30
    bplist = [50,100,150,200,250,300,350,400,450,500,550,600]
    if ionstats_alignment != None:
        kwargs['totalNumReads'] = ionstats_alignment['full']['num_reads']
        kwargs['total_mapped_reads'] = ionstats_alignment['aligned']['num_reads']
        kwargs['total_mapped_target_bases'] = ionstats_alignment['aligned']['num_bases']

        for q in quallist:
            kwargs['q%s_longest_alignment' % q] = ionstats_alignment['AQ'+q]['max_read_length']
            kwargs['q%s_mean_alignment_length' % q] = ionstats_alignment['AQ'+q]['mean_read_length']
            kwargs['q%s_mapped_bases' % q] = ionstats_alignment['AQ'+q]['num_bases']
            kwargs['q%s_alignments' % q] = ionstats_alignment['AQ'+q]['num_reads']
            kwargs['q%s_coverage_percentage' % q] = 0.0 #'N/A' # TODO
#           'Filtered %s Mean Coverage Depth' % q, '%.1f' % (float(ionstats_alignment['AQ'+q]['num_bases'])/float(3095693981)) ) TODO
            for bp in bplist:
                kwargs['i%sQ%s_reads' % (bp,q)] = sum(ionstats_alignment['AQ'+q]['read_length_histogram'][bp:])

        try:
            raw_accuracy = round( (1 - float(sum(ionstats_alignment['error_by_position'])) / float(ionstats_alignment['aligned']['num_bases'])) * 100.0, 1)
            kwargs['raw_accuracy'] = raw_accuracy
        except:
            kwargs['raw_accuracy'] = 0.0
    else:
        kwargs['totalNumReads'] = 0
        kwargs['total_mapped_reads'] = 0
        kwargs['total_mapped_target_bases'] = 0

        for q in quallist:
            for bp in bplist:
                kwargs['i%sQ%s_reads' % (bp,q)] = 0
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


@transaction.commit_on_success
def addQualityMetrics(ionstats_basecaller, res):

    kwargs = {'report':res }

    try:
        kwargs["q0_50bp_reads"]        = 0
        kwargs["q0_100bp_reads"]       = 0
        kwargs["q0_150bp_reads"]       = 0
        kwargs["q0_bases"]             = sum(ionstats_basecaller["qv_histogram"])
        kwargs["q0_reads"]             = ionstats_basecaller["full"]["num_reads"]
        kwargs["q0_max_read_length"]   = ionstats_basecaller["full"]["max_read_length"]
        kwargs["q0_mean_read_length"]  = ionstats_basecaller["full"]["mean_read_length"]
        kwargs["q17_50bp_reads"]       = 0
        kwargs["q17_100bp_reads"]      = 0
        kwargs["q17_150bp_reads"]      = 0
        kwargs["q17_bases"]            = sum(ionstats_basecaller["qv_histogram"][17:])
        kwargs["q17_reads"]            = ionstats_basecaller["Q17"]["num_reads"]
        kwargs["q17_max_read_length"]  = ionstats_basecaller["Q17"]["max_read_length"]
        kwargs["q17_mean_read_length"] = ionstats_basecaller["Q17"]["mean_read_length"]
        kwargs["q20_50bp_reads"]       = 0
        kwargs["q20_100bp_reads"]      = 0
        kwargs["q20_150bp_reads"]      = 0
        kwargs["q20_bases"]            = sum(ionstats_basecaller["qv_histogram"][20:])
        kwargs["q20_reads"]            = ionstats_basecaller["Q20"]["num_reads"]
        kwargs["q20_max_read_length"]  = ionstats_basecaller["Q20"]["max_read_length"]
        kwargs["q20_mean_read_length"] = ionstats_basecaller["Q20"]["mean_read_length"]

        read_length_histogram = ionstats_basecaller['full']['read_length_histogram']
        if len(read_length_histogram) > 50:
            kwargs["q0_50bp_reads"] = sum(read_length_histogram[50:])
        if len(read_length_histogram) > 100:
            kwargs["q0_100bp_reads"] = sum(read_length_histogram[100:])
        if len(read_length_histogram) > 150:
            kwargs["q0_150bp_reads"] = sum(read_length_histogram[150:])

        read_length_histogram = ionstats_basecaller['Q17']['read_length_histogram']
        if len(read_length_histogram) > 50:
            kwargs["q17_50bp_reads"] = sum(read_length_histogram[50:])
        if len(read_length_histogram) > 100:
            kwargs["q17_100bp_reads"] = sum(read_length_histogram[100:])
        if len(read_length_histogram) > 150:
            kwargs["q17_150bp_reads"] = sum(read_length_histogram[150:])

        read_length_histogram = ionstats_basecaller['Q20']['read_length_histogram']
        if len(read_length_histogram) > 50:
            kwargs["q20_50bp_reads"] = sum(read_length_histogram[50:])
        if len(read_length_histogram) > 100:
            kwargs["q20_100bp_reads"] = sum(read_length_histogram[100:])
        if len(read_length_histogram) > 150:
            kwargs["q20_150bp_reads"] = sum(read_length_histogram[150:])


    except Exception as err:
        print("During QualityMetrics creation: %s", err)

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
        pkDict[key]=value
    f.close()
    rpk = pkDict['resultspk']
    res = models.Results.objects.get(pk=rpk)
    res.pluginStore = json.dumps(pluginDict)
    res.save()


@transaction.commit_on_success
def updateStatus(primarykeyPath, status, reportLink = False):
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
        pkDict[key]=value
    f.close()

    rpk = pkDict['resultspk']
    res = models.Results.objects.get(pk=rpk)

    #by default make this "Started"
    if status:
        res.status = status
    else:
        res.status = "Started"
   
    if not reportLink:
        res.reportLink = res.log
    else:
        #Commenting out because reportLink is already set correctly in this case
        #I want to be able to call this function with reportLink == True
        #res.reportLink = reportLink
        pass

    res.save()

def writeDbFromFiles(tfPath, procPath, beadPath, ionstats_alignment_json_path, ionParamsPath, status, keyPath, ionstats_basecaller_json_path, BaseCallerJsonPath, primarykeyPath, uploadStatusPath,cwd):

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

    if os.path.exists(keyPath):
        keyPeak = parse_metrics(keyPath)
    else:
        keyPeak = {}
        keyPeak['Test Fragment'] = 0
        keyPeak['Library'] = 0

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
    if ionparams['libraryName'] != 'none':
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
        if ionparams['libraryName'] != 'none' and ionstats_alignment != None:
            genomeinfofilepath = '/results/referenceLibrary/%s/%s/%s.info.txt' % (ionparams['tmap_version'], ionparams['libraryName'], ionparams['libraryName'])
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

    extra_files = glob.glob(os.path.join(cwd,'BamDuplicates*.json'))
    extra = { u'reads_with_adaptor':0, u'duplicate_reads':0, u'total_reads':0, u'fraction_with_adaptor':0, u'fraction_duplicates':0 }
    for extra_file in extra_files:
      if os.path.exists(extra_file):
          try:
              with open(extra_file, 'r') as afile:
                val = json.load(afile)
                for key in extra.keys():
                    extra[key] += val.get(key,0)
          except:
              return_message += traceback.format_exc()
      else:
          return_message += 'INFO: generating extra metrics failed - file %s is missing\n' % extra_file
    else:
        # No data, send None/null instead of zeros
        extra = { u'reads_with_adaptor':None, u'duplicate_reads':None, u'total_reads':None, u'fraction_with_adaptor':None, u'fraction_duplicates':None }

    try:
        if extra[u'total_reads']:
            extra[u'fraction_with_adaptor'] = extra[u'reads_with_adaptor'] / float( extra[u'total_reads'] )
            extra[u'fraction_duplicates'] = extra[u'duplicate_reads'] / float( extra[u'total_reads'] )
    except:
        return_message += traceback.format_exc()
        
    #print "BamDuplicates stats: ", extra
    
    writeDbFromDict(tfMetrics, procParams, beadMetrics, ionstats_alignment, genomeinfodict, status, keyPeak, ionstats_basecaller, BaseCallerMetrics, primarykeyPath, uploadStatusPath,extra)

    return return_message

def writeDbFromDict(tfMetrics, procParams, beadMetrics, ionstats_alignment, genomeinfodict, status, keyPeak, ionstats_basecaller, BaseCallerMetrics, primarykeyPath, uploadStatusPath,extra):
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
        pkDict[key]=value
    f.close()

    e = open(uploadStatusPath, 'w')

    rpk = pkDict['resultspk']
    res = models.Results.objects.get(pk=rpk)
    if status != None:
        res.status = status
        if not 'Completed' in status:
           res.reportLink = res.log
    res.timeStamp = datetime.datetime.now()
    res.save()

    try:
        e.write('Updating Analysis\n')
        getCurrentAnalysis(procParams, res)
    except:
        e.write("Failed getCurrentAnalysis\n")
        print traceback.format_exc()
        print sys.exc_info()[0]
    try:
        e.write('Adding TF Metrics\n')
        addTfMetrics(tfMetrics, keyPeak, BaseCallerMetrics, res)
    except IntegrityError:
        e.write("Failed addTfMetrics\n")
        logging.exception()
    except:
        e.write("Failed addTfMetrics\n")
        print traceback.format_exc()
        print sys.exc_info()[0]
    try:
        e.write('Adding Analysis Metrics\n')
        addAnalysisMetrics(beadMetrics, BaseCallerMetrics, res)
    except IntegrityError:
        e.write("Failed addAnalysisMetrics\n")
        logging.exception()
    except:
        e.write("Failed addAnalysisMetrics\n")
        print traceback.format_exc()
        print sys.exc_info()[0]
    try:
        e.write('Adding Library Metrics\n')
        addLibMetrics(genomeinfodict, ionstats_alignment, ionstats_basecaller, keyPeak, BaseCallerMetrics, res, extra)
    except IntegrityError:
        e.write("Failed addLibMetrics\n")
        logging.exception()
    except:
        e.write("Failed addLibMetrics\n")
        print traceback.format_exc()
        print sys.exc_info()[0] 

    #try to add the quality metrics
    try:
        e.write('Adding Quality Metrics\n')
        if ionstats_basecaller:
            addQualityMetrics(ionstats_basecaller, res)
        else:
            e.write("Failed to add QualityMetrics, missing object ionstats_basecaller\n")
    except IntegrityError:
        e.write("Failed addQualityMetrics\n")
        logging.exception()
    except:
        e.write("Failed addQualityMetrics\n")
        print traceback.format_exc()
        print sys.exc_info()[0] 

    e.close()

if __name__=='__main__':
    logging.basicConfig()
    if len(sys.argv) < 2:
        folderPath = os.getcwd()
        logging.warn("No path specified. Assuming cwd: '%s'", folderPath)
    else:
        folderPath = sys.argv[1]

    SIGPROC_RESULTS="sigproc_results"
    BASECALLER_RESULTS="basecaller_results"
    ALIGNMENT_RESULTS="./"

    print "processing: " + folderPath

    status = None

    tfPath = os.path.join(folderPath, BASECALLER_RESULTS, 'TFStats.json')
    procPath = os.path.join(SIGPROC_RESULTS,"processParameters.txt")
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
