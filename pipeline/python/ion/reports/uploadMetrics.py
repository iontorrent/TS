# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
import json
import datetime
import traceback

sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from django.db import models
from django.db import connection
from iondb.rundb import models
from os import path
import subprocess
from ion.reports import parseBeadfind
from ion.utils.blockprocessing import parse_metrics
from ion.utils.textTo import textToDict

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

def addTfMetrics(tfMetrics, keyPeak, BaseCallerMetrics, res):
    ###Populate metrics for each TF###
    
    if tfMetrics == None:
        return

    for tf, metrics in tfMetrics.iteritems():
        
        HPSNR = metrics.get('Raw HP SNR',[0])
        hpAccNum = metrics.get('Per HP accuracy NUM',[0])
        hpAccDen = metrics.get('Per HP accuracy DEN',[0])
        avgIonogram = [a/float(b) for a,b in zip(metrics.get('Avg Ionogram NUM',[0]),metrics.get('Avg Ionogram DEN',[0]))]
        corIonogram = [a/float(b) for a,b in zip(metrics.get('Corrected Avg Ionogram NUM',[0]),metrics.get('Corrected Avg Ionogram DEN',[0]))]
        
        kwargs = {'report'                  : res,
                  'name'                    : tf,
                  'sequence'                : metrics.get('TF Seq','None'),
                  'number'                  : metrics.get('Num',0.0),
                  'keypass'                 : metrics.get('Num',0.0),   # Deprecated, populating by next best thing
                  'aveKeyCount'             : keyPeak.get('Test Fragment','0'),
                  'SysSNR'                  : metrics.get('System SNR',0.0),

                  'Q10Histo'                : ' '.join(map(str,metrics.get('Q10',[0]))),
                  'Q10Mean'                 : metrics.get('Q10 Mean',0.0),
                  'Q10Mode'                 : metrics.get('Q10 Mean',0.0), # Deprecated, populating by next best thing
                  'Q10ReadCount'            : metrics.get('50Q10',0.0),
                  
                  'Q17Histo'                : ' '.join(map(str,metrics.get('Q17',[0]))),
                  'Q17Mean'                 : metrics.get('Q17 Mean',0.0),
                  'Q17Mode'                 : metrics.get('Q17 Mean',0.0), # Deprecated, populating by next best thing
                  'Q17ReadCount'            : metrics.get('50Q17',0.0),

                  'HPAccuracy'              : ', '.join('%d : %d/%d' % (x,y[0],y[1]) for x,y in enumerate(zip(hpAccNum,hpAccDen))),
                  'HPSNR'                   : ', '.join('%d : %f' % x for x in enumerate(HPSNR)),
                  'rawIonogram'             : ' '.join(map(str,avgIonogram)),
                  'corrIonogram'            : ' '.join(map(str,corIonogram)),
                  
                  'aveHqReadCount'          : 0.0,      # Deprecated
                  'preCorrSNR'              : 0.0,      # Deprecated
                  'postCorrSNR'             : 0.0,      # Deprecated 
                  'rawOverlap'              : '0',      # Deprecated
                  'corOverlap'              : '0',      # Deprecated
                  'error'                   : 0.0,      # Deprecated
                  'matchMismatchHisto'      : '0',      # Deprecated
                  'matchMismatchMean'       : 0.0,      # Deprecated
                  'matchMismatchMode'       : 0.0,      # Deprecated
                  'hqReadCount'             : 0.0,      # Deprecated
                  'aveQ10ReadCount'         : 0.0,      # Deprecated
                  'aveQ17ReadCount'         : 0.0,      # Deprecated
                  }
        
        try:
            kwargs['CF'] = 100.0 * BaseCallerMetrics['Phasing']['CF']   # Deprecated, populating by next best thing
            kwargs['IE'] = 100.0 * BaseCallerMetrics['Phasing']['IE']   # Deprecated, populating by next best thing
            kwargs['DR'] = 100.0 * BaseCallerMetrics['Phasing']['DR']   # Deprecated, populating by next best thing
        except:
            kwargs['CF'] = 0.0
            kwargs['IE'] = 0.0
            kwargs['DR'] = 0.0
        
        tfm = models.TFMetrics(**kwargs)
        tfm.save()


def addAnalysisMetrics(beadMetrics, filterMetrics, BaseCallerMetrics, res):
    #print 'addAnalysisMetrics'
    analysis_metrics_map = {'libLive':'Library live beads',
                            'libKp':'Library keypass filter',
                            'libFinal':'Library final',
                            'tfLive':'TF live beads',
                            'tfKp':'TF keypass filter',
                            'tfFinal':'TF final',
                            'lib_pass_basecaller':'Library pass basecaller',
                            'lib_pass_cafie':'Library pass cafie'
                            }

    bead_metrics_map = {'empty':'Empty Wells',
                        'bead':'Bead Wells',
                        'live':'Live Beads',
                        'dud':'Dud Beads',
                        'amb':'Ambiguous Beads',
                        'tf':'Test Fragment Beads',
                        'lib':'Library Beads',
                        'pinned':'Pinned Wells',
                        'ignored':'Ignored Wells',
                        'excluded':'Excluded Wells',
                        'washout':'Washout Wells',
                        'washout_dud':'Washout Dud',
                        'washout_ambiguous':'Washout Ambiguous',
                        'washout_live':'Washout Live',
                        'washout_test_fragment':'Washout Test Fragment',
                        'washout_library':'Washout Library',
                        'keypass_all_beads':'Keypass Beads'
                        }

    kwargs = {'report':res,'libMix':0,'tfMix':0,'sysCF':0.0,'sysIE':0.0,'sysDR':0.0}

    for dbname, key in analysis_metrics_map.iteritems():
        try:
            kwargs[dbname]=set_type(filterMetrics.get(key, '0'))
        except:
            kwargs[dbname]='0'
    for dbname, key in bead_metrics_map.iteritems():
        kwargs[dbname]=set_type(beadMetrics.get(key, '0'))
    
    try:
        kwargs['sysCF'] = 100.0 * BaseCallerMetrics['Phasing']['CF']
        kwargs['sysIE'] = 100.0 * BaseCallerMetrics['Phasing']['IE']
        kwargs['sysDR'] = 100.0 * BaseCallerMetrics['Phasing']['DR']
    except:
        kwargs['sysCF'] = 0.0
        kwargs['sysIE'] = 0.0
        kwargs['sysDR'] = 0.0
   
    am = models.AnalysisMetrics(**kwargs)
    am.save()

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
    
def addLibMetrics(libMetrics, qualityMetrics, keyPeak, BaseCallerMetrics, res):
    #print 'addlibmetrics'
    metric_map = {'totalNumReads':'Total number of Reads', 
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

    "q7_coverage_percentage":"Filtered Q7 Coverage Percentage",
    "q7_alignments":"Filtered Q7 Alignments",
    "q7_mean_alignment_length":"Filtered Q7 Mean Alignment Length",
    "q7_mapped_bases":"Filtered Mapped Bases in Q7 Alignments",
    "q7_longest_alignment":"Filtered Q7 Longest Alignment",
    "i50Q7_reads":"Filtered 50Q7 Reads",
    "i100Q7_reads":"Filtered 100Q7 Reads",
    "i150Q7_reads":"Filtered 150Q7 Reads",
    "i200Q7_reads":"Filtered 200Q7 Reads",
    "i250Q7_reads":"Filtered 250Q7 Reads",
    "i300Q7_reads":"Filtered 300Q7 Reads",
    "i350Q7_reads":"Filtered 350Q7 Reads",
    "i400Q7_reads":"Filtered 400Q7 Reads",
    "i450Q7_reads":"Filtered 450Q7 Reads",
    "i500Q7_reads":"Filtered 500Q7 Reads",
    "i550Q7_reads":"Filtered 550Q7 Reads",
    "i600Q7_reads":"Filtered 600Q7 Reads",

    "q10_coverage_percentage":"Filtered Q10 Coverage Percentage",
    "q10_alignments":"Filtered Q10 Alignments",
    "q10_mean_alignment_length":"Filtered Q10 Mean Alignment Length",
    "q10_mapped_bases":"Filtered Mapped Bases in Q10 Alignments",
    "q10_longest_alignment":"Filtered Q10 Longest Alignment",
    "i50Q10_reads":"Filtered 50Q10 Reads",
    "i100Q10_reads":"Filtered 100Q10 Reads",
    "i150Q10_reads":"Filtered 150Q10 Reads",
    "i200Q10_reads":"Filtered 200Q10 Reads",
    "i250Q10_reads":"Filtered 250Q10 Reads",
    "i300Q10_reads":"Filtered 300Q10 Reads",
    "i350Q10_reads":"Filtered 350Q10 Reads",
    "i400Q10_reads":"Filtered 400Q10 Reads",
    "i450Q10_reads":"Filtered 450Q10 Reads",
    "i500Q10_reads":"Filtered 500Q10 Reads",
    "i550Q10_reads":"Filtered 550Q10 Reads",
    "i600Q10_reads":"Filtered 600Q10 Reads",

    "q17_coverage_percentage":"Filtered Q17 Coverage Percentage",
    "q17_alignments":"Filtered Q17 Alignments",
    "q17_mean_alignment_length":"Filtered Q17 Mean Alignment Length",
    "q17_mapped_bases":"Filtered Mapped Bases in Q17 Alignments",
    "q17_longest_alignment":"Filtered Q17 Longest Alignment",
    "i50Q17_reads":"Filtered 50Q17 Reads",
    "i100Q17_reads":"Filtered 100Q17 Reads",
    "i150Q17_reads":"Filtered 150Q17 Reads",
    "i200Q17_reads":"Filtered 200Q17 Reads",
    "i250Q17_reads":"Filtered 250Q17 Reads",
    "i300Q17_reads":"Filtered 300Q17 Reads",
    "i350Q17_reads":"Filtered 350Q17 Reads",
    "i400Q17_reads":"Filtered 400Q17 Reads",
    "i450Q17_reads":"Filtered 450Q17 Reads",
    "i500Q17_reads":"Filtered 500Q17 Reads",
    "i550Q17_reads":"Filtered 550Q17 Reads",
    "i600Q17_reads":"Filtered 600Q17 Reads",

    "q20_coverage_percentage":"Filtered Q20 Coverage Percentage",
    "q20_alignments":"Filtered Q20 Alignments",
    "q20_mean_alignment_length":"Filtered Q20 Mean Alignment Length",
    "q20_mapped_bases":"Filtered Mapped Bases in Q20 Alignments",
    "q20_longest_alignment":"Filtered Q20 Longest Alignment",
    "i50Q20_reads":"Filtered 50Q20 Reads",
    "i100Q20_reads":"Filtered 100Q20 Reads",
    "i150Q20_reads":"Filtered 150Q20 Reads",
    "i200Q20_reads":"Filtered 200Q20 Reads",
    "i250Q20_reads":"Filtered 250Q20 Reads",
    "i300Q20_reads":"Filtered 300Q20 Reads",
    "i350Q20_reads":"Filtered 350Q20 Reads",
    "i400Q20_reads":"Filtered 400Q20 Reads",
    "i450Q20_reads":"Filtered 450Q20 Reads",
    "i500Q20_reads":"Filtered 500Q20 Reads",
    "i550Q20_reads":"Filtered 550Q20 Reads",
    "i600Q20_reads":"Filtered 600Q20 Reads",

    "q47_coverage_percentage":"Filtered Q47 Coverage Percentage",
    "q47_alignments":"Filtered Q47 Alignments",
    "q47_mean_alignment_length":"Filtered Q47 Mean Alignment Length",
    "q47_mapped_bases":"Filtered Mapped Bases in Q47 Alignments",
    "q47_longest_alignment":"Filtered Q47 Longest Alignment",
    "i50Q47_reads":"Filtered 50Q47 Reads",
    "i100Q47_reads":"Filtered 100Q47 Reads",
    "i150Q47_reads":"Filtered 150Q47 Reads",
    "i200Q47_reads":"Filtered 200Q47 Reads",
    "i250Q47_reads":"Filtered 250Q47 Reads",
    "i300Q47_reads":"Filtered 300Q47 Reads",
    "i350Q47_reads":"Filtered 350Q47 Reads",
    "i400Q47_reads":"Filtered 400Q47 Reads",
    "i450Q47_reads":"Filtered 450Q47 Reads",
    "i500Q47_reads":"Filtered 500Q47 Reads",
    "i550Q47_reads":"Filtered 550Q47 Reads",
    "i600Q47_reads":"Filtered 600Q47 Reads",

    "q7_qscore_bases":"Filtered Mapped Q7 Bases",
    "q10_qscore_bases":"Filtered Mapped Q10 Bases",
    "q17_qscore_bases":"Filtered Mapped Q17 Bases",
    "q20_qscore_bases":"Filtered Mapped Q20 Bases",
    "q47_qscore_bases":"Filtered Mapped Q47 Bases",
    "Genome_Version":"Genome Version",
    "Index_Version":"Index Version",
    'genome':'Genome',
    'genomesize':'Genomesize',
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
    'extrapolated_400q47_reads':'Extrapolated Filtered 400Q47 Reads'
    }

    quality_metrics_map = {'sysSNR':'System SNR'}

    if keyPeak != None:
        aveKeyCount = float(keyPeak.get('Library',0.0))
    else:
        aveKeyCount = 0.0
        
    align_sample = 0
    if libMetrics == None:
        align_sample = -1
    else:
        #check to see if this is a samled or full alignment 
        if libMetrics.has_key('Total number of Sampled Reads'):
            align_sample = 1
        if libMetrics.has_key('Thumbnail'):
            align_sample = 2

    kwargs = {'report':res, 'aveKeyCounts':aveKeyCount, 'align_sample': align_sample }

    for dbname, key in quality_metrics_map.iteritems():
        if qualityMetrics != None:
            kwargs[dbname] = set_type(qualityMetrics.get(key, '0'))
        else:
            kwargs[dbname] = set_type('0')
    for dbname, key in metric_map.iteritems():
        if libMetrics != None:
            kwargs[dbname] = set_type(libMetrics.get(key, '0'))
        else:
            kwargs[dbname] = set_type('0')
        
    try:
        kwargs['cf'] = 100.0 * BaseCallerMetrics['Phasing']['CF']
        kwargs['ie'] = 100.0 * BaseCallerMetrics['Phasing']['IE']
        kwargs['dr'] = 100.0 * BaseCallerMetrics['Phasing']['DR']
    except:
        kwargs['cf'] = 0.0
        kwargs['ie'] = 0.0
        kwargs['dr'] = 0.0
    
    lib = models.LibMetrics(**kwargs)
    lib.save()


def addQualityMetrics(QualityMetrics, res):
    # print ' add quality metrics'
    metric_map = {'q0_bases' : 'Number of Bases at Q0',
                    'q0_reads' : 'Number of Reads at Q0',
                    'q0_max_read_length' : 'Max Read Length at Q0',
                    'q0_mean_read_length' : 'Mean Read Length at Q0',
                    'q0_50bp_reads' : 'Number of 50BP Reads at Q0',
                    'q0_100bp_reads' : 'Number of 100BP Reads at Q0',
                    'q0_15bp_reads' : 'Number of 150BP Reads at Q0',
                    'q17_bases' : 'Number of Bases at Q17',
                    'q17_reads' : 'Number of Reads at Q17',
                    'q17_max_read_length' :  'Max Read Length at Q17',
                    'q17_mean_read_length' : 'Mean Read Length at Q17',
                    'q17_50bp_reads' : 'Number of 50BP Reads at Q17',
                    'q17_100bp_reads' : 'Number of 100BP Reads at Q17',
                    'q17_150bp_reads' : 'Number of 150BP Reads at Q17',
                    'q20_bases' : 'Number of Bases at Q20',
                    'q20_reads' : 'Number of Reads at Q20',
                    'q20_max_read_length' :  'Max Read Length at Q20',
                    'q20_mean_read_length' :  'Mean Read Length at Q20',
                    'q20_50bp_reads' : 'Number of 50BP Reads at Q20',
                    'q20_100bp_reads' : 'Number of 100BP Reads at Q20',
                    'q20_150bp_reads' : 'Number of 150BP Reads at Q20'
                  }

    kwargs = {'report':res }

    for dbname, key in metric_map.iteritems():
        if QualityMetrics != None:
            kwargs[dbname] = set_type(QualityMetrics.get(key, '0'))
        else:
            kwargs[dbname] = set_type('0')
    quality = models.QualityMetrics(**kwargs)
    quality.save()

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

def writeDbFromFiles(tfPath, procPath, beadPath, filterPath, libPath, status, keyPath, QualityPath, BaseCallerJsonPath, primarykeyPath, uploadStatusPath):

    tfMetrics = None
    if os.path.exists(tfPath):
        try:
            file = open(tfPath, 'r')
            tfMetrics = json.load(file)
            file.close()
        except:
            print "ERROR: failed to create tfMetrics object from file %s" % tfPath
            tfMetrics = None
            traceback.print_exc()
    else:
        print 'ERROR: generating tfMetrics failed - file %s is missing' % tfPath
        beadMetrics = None

    BaseCallerMetrics = None
    if os.path.exists(BaseCallerJsonPath):
        try:
            afile = open(BaseCallerJsonPath, 'r')
            BaseCallerMetrics = json.load(afile)
            afile.close()
        except:
            print "ERROR: failed to create BaseCallerMetrics object"
            BaseCallerMetrics = None
            traceback.print_exc()
    else:
        print 'ERROR: generating BaseCallerMetrics failed - file %s is missing' % BaseCallerJsonPath
        BaseCallerMetrics = None

    beadMetrics = None
    if os.path.isfile(beadPath):
        try:
            beadMetrics = parseBeadfind.generateMetrics(beadPath)
        except:
            print 'ERROR: generating beadMetrics failed'
            beadMetrics = None
            traceback.print_exc()
    else:
        print 'ERROR: generating beadMetrics failed - file %s is missing' % beadPath
        beadMetrics = None

    if filterPath != None:
        filterMetrics = None
        #filterMetrics = parseFilterMetrics.generateParams(filterPath)
    else:
        filterMetrics = None

    libMetrics = None
    if path.exists(libPath):
        libMetrics = parse_metrics(libPath)
    else:
        libMetrics = None

    QualityMetrics = None
    if path.exists(QualityPath):
        #if the quality metrics exist, make a dict out of them
        QualityMetrics = parse_metrics(QualityPath)
    else:
        QualityMetrics = None

    if path.exists(keyPath):
        keyPeak = parse_metrics(keyPath)
    else:
        keyPeak = {}
        keyPeak['Test Fragment'] = 0
        keyPeak['Library'] = 0

    procParams = None
    if os.path.exists(procPath):
        procParams = textToDict(procPath)

    writeDbFromDict(tfMetrics, procParams, beadMetrics, filterMetrics, libMetrics, status, keyPeak, QualityMetrics, BaseCallerMetrics, primarykeyPath, uploadStatusPath)


def writeDbFromDict(tfMetrics, procParams, beadMetrics, filterMetrics, libMetrics, status, keyPeak, QualityMetrics, BaseCallerMetrics, primarykeyPath, uploadStatusPath):
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
        if status != 'Completed':
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
    except:
        e.write("Failed addTfMetrics\n")
        print traceback.format_exc()
        print sys.exc_info()[0]
    try:
        e.write('Adding Analysis Metrics\n')
        addAnalysisMetrics(beadMetrics, filterMetrics, BaseCallerMetrics, res)
    except:
        e.write("Failed addAnalysisMetrics\n")
        print traceback.format_exc()
        print sys.exc_info()[0]
    try:
        e.write('Adding Library Metrics\n')
        addLibMetrics(libMetrics, QualityMetrics, keyPeak, BaseCallerMetrics, res)
    except:
        e.write("Failed addLibMetrics\n")
        print traceback.format_exc()
        print sys.exc_info()[0] 
        
    #try to add the quality metrics
    try:
        e.write('Adding Quality Metrics\n')
        addQualityMetrics(QualityMetrics, res)
    except:
        e.write("Failed addQualityMetrics\n")
        print traceback.format_exc()
        print sys.exc_info()[0] 
    
    e.close()

if __name__=='__main__':

    # TODO: return if argv[1] is not specified

    SIGPROC_RESULTS="./"
    BASECALLER_RESULTS="./"
    ALIGNMENT_RESULTS="./"

    folderPath = sys.argv[1]
    print "processing: " + folderPath

    status = None

    tfPath = os.path.join(folderPath, BASECALLER_RESULTS, 'TFStats.json')
    procPath = os.path.join(SIGPROC_RESULTS,"processParameters.txt")
    beadPath = os.path.join(folderPath, SIGPROC_RESULTS, 'bfmask.stats')
    filterPath = os.path.join(folderPath, SIGPROC_RESULTS, 'filterMetrics.txt')
    alignmentSummaryPath = os.path.join(folderPath, ALIGNMENT_RESULTS, 'alignment.summary')
    primarykeyPath = os.path.join(folderPath, 'primary.key')
    BaseCallerJsonPath = os.path.join(folderPath, BASECALLER_RESULTS, 'BaseCaller.json')
    QualityPath = os.path.join(folderPath, BASECALLER_RESULTS, 'quality.summary')
    keyPath = os.path.join(folderPath, 'raw_peak_signal')
    uploadStatusPath = os.path.join(folderPath, 'status.txt')

    writeDbFromFiles(tfPath,
                     procPath,
                     beadPath,
                     filterPath,
                     alignmentSummaryPath,
                     status,
                     keyPath,
                     QualityPath,
                     BaseCallerJsonPath,
                     primarykeyPath,
                     uploadStatusPath)
