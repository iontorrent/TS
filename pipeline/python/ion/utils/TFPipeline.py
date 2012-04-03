#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import os
import time
import argparse
import traceback
import json
from matplotlib import use
use("Agg")
#from ion.reports import tfGraphs
import pylab
from ion.reports.plotters import plotters
from numpy import mean
from scipy.stats import mode


def printtime(message, *args):
    if args:
        message = message % args
    print "[ " + time.strftime('%X') + " ] " + message



# Primary use cases:
# - Load one TFMetrics.json
# - Load and combine multiple TFMetrics.json
# - Generate uploadMetrics-friendly dictionary with TF metrics
# - Generate report plots

# Secondary use cases:
# - Generate TF plugin plots
# - Generate TF html files




class TFStats:
    
    def __init__(self, name, dataList):
        self.name = name

        if not isinstance(dataList,list):
            dataList = [dataList]
        
        if len(dataList) == 0:
            raise ValueError
        
        # Presume there are multiple records that need merging. Automatically works for one record
        
        self.sequence = dataList[0]['TF Seq']
        self.count = sum([dataItem['Num'] for dataItem in dataList])
        self.keySNR = sum([dataItem['System SNR']*dataItem['Num'] for dataItem in dataList]) / self.count
        
        numHPs = len(dataList[0]['Per HP accuracy NUM'])
        self.hpAccuracyNum = [0] * numHPs
        self.hpAccuracyDen = [0] * numHPs
        self.hpSNRRaw = [0] * numHPs
        self.hpSNRCorrected = [0] * numHPs
        for idx in range(numHPs):
            self.hpAccuracyNum[idx] = sum([dataItem['Per HP accuracy NUM'][idx] for dataItem in dataList])
            self.hpAccuracyDen[idx] = sum([dataItem['Per HP accuracy DEN'][idx] for dataItem in dataList])
            self.hpSNRRaw[idx] = sum([dataItem['Raw HP SNR'][idx]*dataItem['Per HP accuracy DEN'][idx] for dataItem in dataList])
            self.hpSNRCorrected[idx] = sum([dataItem['Corrected HP SNR'][idx]*dataItem['Per HP accuracy DEN'][idx] for dataItem in dataList])
            if self.hpAccuracyDen[idx] > 0:
                self.hpSNRRaw[idx] /= float(self.hpAccuracyDen[idx])
                self.hpSNRCorrected[idx] /= float(self.hpAccuracyDen[idx])

        numQBases = len(dataList[0]['Q10'])
        self.Q10hist = [0] * numQBases
        self.Q17hist = [0] * numQBases
        for idx in range(numQBases):
            self.Q10hist[idx] = sum([dataItem['Q10'][idx] for dataItem in dataList])
            self.Q17hist[idx] = sum([dataItem['Q17'][idx] for dataItem in dataList])

        numIFlows = len(dataList[0]['Avg Ionogram NUM'])
        self.avgIonogramRawNum       = [0] * numIFlows
        self.avgIonogramRawDen       = [0] * numIFlows
        self.avgIonogramCorrectedNum = [0] * numIFlows
        self.avgIonogramCorrectedDen = [0] * numIFlows
        for idx in range(numIFlows):
            self.avgIonogramRawNum[idx] = sum([dataItem['Avg Ionogram NUM'][idx] for dataItem in dataList])
            self.avgIonogramRawDen[idx] = sum([dataItem['Avg Ionogram DEN'][idx] for dataItem in dataList])
            self.avgIonogramCorrectedNum[idx] = sum([dataItem['Corrected Avg Ionogram NUM'][idx] for dataItem in dataList])
            self.avgIonogramCorrectedDen[idx] = sum([dataItem['Corrected Avg Ionogram DEN'][idx] for dataItem in dataList])
        
        
        # TODO: Combine top ionograms here
        
        #self.tfs[tf]['Top reads'].extend(data['Top reads'])

    def getUploadMetrics(self):
        
        metrics = {
            'TF Seq' : self.sequence,
            'Num' : self.count,
            'System SNR' : self.keySNR,
            
            'Q10' : ' '.join(map(str,self.Q10hist)),
            'Avg Q10 read length' : mean(self.Q10hist),
            'Q10Mode' : mode(self.Q10hist)[0][0],
            '50Q10' : sum(self.Q10hist[50:]),
            
            'Q17' : ' '.join(map(str,self.Q17hist)),
            'Avg Q17 read length' : mean(self.Q17hist),
            'Q17Mode' : mode(self.Q17hist)[0][0],
            '50Q17' : sum(self.Q17hist[50:]),
            
            'Raw HP SNR' : ', '.join(['%d : %f' % (idx,self.hpSNRRaw[idx]) for idx in range(len(self.hpSNRRaw))]),
            'Corrected HP SNR' : ', '.join(['%d : %f' % (idx,self.hpSNRCorrected[idx]) for idx in range(len(self.hpSNRCorrected))]),
            'Per HP accuracy' : ', '.join(['%d : %d/%d' % (idx,self.hpAccuracyNum[idx],self.hpAccuracyDen[idx]) for idx in range(len(self.hpAccuracyNum))]),
            
            'Avg Ionogram' : ' '.join(map(str,[float(self.avgIonogramRawNum[idx])/self.avgIonogramRawDen[idx] for idx in range(len(self.avgIonogramRawNum))])),
            'Corrected Avg Ionogram' : ' '.join(map(str,[float(self.avgIonogramCorrectedNum[idx])/self.avgIonogramCorrectedDen[idx] for idx in range(len(self.avgIonogramCorrectedNum))]))

            # TODO: Top ionograms            
        }
        
        return metrics

    def getTFStatsMetrics(self):
        
        metrics = {
            'TF Seq' : self.sequence,
            'Num' : self.count,
            'System SNR' : self.keySNR,
            
            'Per HP accuracy NUM' : self.hpAccuracyNum,
            'Per HP accuracy DEN' : self.hpAccuracyDen,
            'Raw HP SNR' : self.hpSNRRaw,
            'Corrected HP SNR' : self.hpSNRCorrected,
            
            'Q10' : self.Q10hist,
            'Q17' : self.Q17hist,
            
            'Avg Ionogram NUM' : self.avgIonogramRawNum,
            'Avg Ionogram DEN' : self.avgIonogramRawDen,
            'Corrected Avg Ionogram NUM' : self.avgIonogramCorrectedNum,
            'Corrected Avg Ionogram DEN' : self.avgIonogramCorrectedDen
            
            # TODO: Top ionograms
        }

        return metrics

    
#    def generatePlots(self):
#        pass
    


    def plotQ10(self):

        qplot = plotters.QPlot2(self.Q10hist[:100], q=10, expected=self.sequence[:100])
        qplot.render()
        pylab.savefig(os.path.join(os.getcwd(), "Q10_%s.png" % self.name)) 
        pylab.clf()
            
    def plotQ17(self):
        
        qplot = plotters.QPlot2(self.Q17hist[:100], q=17, expected=self.sequence[:100])
        qplot.render()
        pylab.savefig(os.path.join(os.getcwd(), "Q17_%s.png" % self.name)) 
        pylab.clf()
        

    def plotAvgIonograms(self, floworder):

        v = [float(self.avgIonogramRawNum[idx])/self.avgIonogramRawDen[idx] for idx in range(len(self.avgIonogramRawNum))]
        corrIonogram = plotters.IonogramJMR(floworder,v,v,'Average Raw Ionogram')
        corrIonogram.render()
        pylab.savefig(os.path.join(os.getcwd(), "Average Raw Ionogram_%s.png" % self.name))
            
        v = [float(self.avgIonogramCorrectedNum[idx])/self.avgIonogramCorrectedDen[idx] for idx in range(len(self.avgIonogramCorrectedNum))]
        corrIonogram = plotters.IonogramJMR(floworder,v,v,'Average Corrected Ionogram')
        corrIonogram.render()
        pylab.savefig(os.path.join(os.getcwd(), "Average Corrected Ionogram_%s.png" % self.name))


    

def buildTFReference(tfreffasta_filename):

    #
    # Step 1. Build the DefaultTF.fasta from DefaultTF.conf
    #
    # Currently done by a dedicated executable, but simple enough to do manually here (TODO)

    try:
        com = "/home/msikora/Documents/TestFragmentTorturer/TFReferenceGenerator %s" % tfreffasta_filename
        #com = "TFReferenceGenerator %s" % tfreffasta_filename 
        printtime("DEBUG: Calling '%s'" % com)
        os.system(com)
    except:
        printtime("ERROR: TFReferenceGenerator failed")
    

def alignTFs(sff_filename,bam_filename,fasta_filename):


    #
    # Step 2. Build tmap index for DefaultTF.fasta in a temporary directory
    #
   
    indexDir = 'tfref'  # Might instead use a folder in /tmp

    try:
        printtime("DEBUG: Creating directory '%s'" % indexDir)
        os.mkdir(indexDir)
    except:
        printtime("ERROR: mkdir failed")
    
    try:
        com = "cp %s %s/DefaultTF.fasta" % (fasta_filename,indexDir)
        printtime("DEBUG: Calling '%s'" % com)
        os.system(com)
    except:
        printtime("ERROR: cp failed")
        return

    try:
        com = "tmap index -f %s" % os.path.join(indexDir,'DefaultTF.fasta')
        printtime("DEBUG: Calling '%s'" % com)
        os.system(com)
    except:
        printtime("ERROR: tmap index failed")
        return
    
    #
    # Step 3. Perform mapping of the sff file
    #
    
    try:
        com = "tmap mapall -n 12"
        com += " -f %s" % os.path.join(indexDir,'DefaultTF.fasta')
        com += " -r %s" % sff_filename
        com += " -v stage1 map1 map2 map3"
        com += " | samtools view -Sb -o %s -" % bam_filename
        printtime("DEBUG: Calling '%s'" % com)
        os.system(com)
    except:
        printtime("ERROR: mapall index failed")
        return

    #
    # Step 4. Delete index
    #
    
    try:
        com = "rm -rf %s" % indexDir
        printtime("DEBUG: Calling '%s'" % com)
        os.system(com)
    except:
        printtime("ERROR: rm failed")
    
    
def doAlignStats(bam_filename):

    try:
        com = 'alignStats -i %s -p 1 -o TF -a TF.alignTable.txt -n 12' % bam_filename
        printtime("DEBUG: Calling '%s'" % com)
        os.system(com)
    except:
        printtime("ERROR: alignStats failed")


def doTFMapper(tfsff_filename, tfkey, BASECALLER_RESULTS, SIGPROC_RESULTS, tfmapperstats_path, tfstatsjson_path):
    
    # Currently TFMapper does its own TF classification and does not use bam file, but this will change

    try:
        com = "/home/msikora/Documents/TFMapper"
        #com = "TFMapper"
        com += " --logfile TFMapper.log"
        com += " --output-json=%s" % tfstatsjson_path
        com += " --output-dir=%s" % (BASECALLER_RESULTS)
        com += " --wells-dir=%s" % (SIGPROC_RESULTS)
        com += " --sff-dir=%s" % (BASECALLER_RESULTS)
        com += " --tfkey=%s" % tfkey
        com += " %s" % tfsff_filename
        com += " ./"
        com += " > %s" % (tfmapperstats_path)
        printtime("DEBUG: Calling '%s'" % com)
        os.system(com)
    except:
        printtime("ERROR: TFMapper failed")
    


def generatePlots(floworder,tfstatsjson_path):
    
    try:
        # Q17 TF Read Length Plot
        #tfMetrics = tfGraphs.generateMetricsData('TFMapper.stats')
        #tfGraphs.Q17(tfMetrics)
        #tfGraphs.genCafieIonograms(tfMetrics,floworder)
        
        #file = open('TFMapper.stats', 'r')
        #tfGraphs.generateMetrics(file.readlines())
        #file.close()
        
        file = open(tfstatsjson_path, 'r')
        TFStatsJson = json.load(file)
        file.close()
        
        if TFStatsJson == None:
            TFStatsJson = {}
        
        allTFStats = {}
        for k,v in TFStatsJson.iteritems():
            tf = TFStats(k,v)
            tf.plotQ10()
            tf.plotQ17()
            tf.plotAvgIonograms(floworder)
            
        
    except Exception:
        printtime("Metrics Gen Failed")
        traceback.print_exc()





def processBlock(tfsff_filename, BASECALLER_RESULTS, SIGPROC_RESULTS, tfkey, floworder):
    

    # These files will be created
    tfmapperstats_path = os.path.join(BASECALLER_RESULTS,"TFMapper.stats")
    tfstatsjson_path = os.path.join(BASECALLER_RESULTS,"TFStats.json")
    tfbam_filename = os.path.join(BASECALLER_RESULTS,"rawtf.bam")
    tfreffasta_filename = os.path.join(BASECALLER_RESULTS,"DefaultTF.fasta")
    
    # TF analysis in 5 simple steps
    
    buildTFReference(tfreffasta_filename)
    
    alignTFs(tfsff_filename, tfbam_filename, tfreffasta_filename)
    
    doAlignStats(tfbam_filename)
    
    doTFMapper(tfsff_filename, tfkey, BASECALLER_RESULTS, SIGPROC_RESULTS, tfmapperstats_path, tfstatsjson_path)
    
    generatePlots(floworder,tfstatsjson_path)


def mergeBlocks(BASECALLER_RESULTS,dirs,floworder):
    
    # Input: list of blocks
    # Step 1: Read individual TFStats.json and merge
    # Step 2: Generate combined TFStats.json
    # Step 3: Generate combined TFMapper.stats (temporary interface to database upload)
    # Step 4: Generate plots: Q10, Q17, AvgIonograms
    
    recordList = []
    recordKeys = []
    
    for subdir in dirs:
        _subdir = os.path.join(BASECALLER_RESULTS,subdir)
        
        try:
            file = open(os.path.join(_subdir,'TFStats.json'), 'r')
            TFStatsJson = json.load(file)
            file.close()
            if TFStatsJson != None:
                recordList.extend(TFStatsJson.items())
                recordKeys.extend(TFStatsJson.keys())
        except:
            print "Could not process block %s" % subdir
    
    allTFStats = {}
    for tf in recordKeys:
        if tf not in allTFStats:
            allTFStats[tf] = TFStats(tf,[v[1] for v in recordList if v[0] == tf])

    # Output combined TFStats.json and TFMapper.stats

    mergedTFStatsJson = {}
    f = open(os.path.join(BASECALLER_RESULTS,'TFMapper.stats'),'w')
    for k,v in allTFStats.iteritems():
        mergedTFStatsJson[k] = v.getTFStatsMetrics()
        
        f.write('TF Name = %s\n' % k)
        stats = v.getUploadMetrics()
        for a,b in stats.iteritems():
            f.write('%s = %s\n' % (a,b))

    f.close()
    
    f = open(os.path.join(BASECALLER_RESULTS,'TFStats.json'),'w')
    f.write(json.dumps(mergedTFStatsJson, indent=4))
    f.close()

    # Generate plots
    
    for k,v in allTFStats.iteritems():
        v.plotQ10()
        v.plotQ17()
        v.plotAvgIonograms(floworder)

    



if __name__=="__main__":

    # Step 1. Parser command line arguments

    parser = argparse.ArgumentParser(description='Test Fragment evaluation pipeline.')
    parser.add_argument('-i','--input',dest='sff', default='rawtf.sff',
                        help='Input SFF file containing TF reads (Default: rawtf.sff)')
    parser.add_argument('-b','--bam',  dest='bam', default='rawtf.bam',
                        help='Intermediate output BAM file for TF reads (Default: rawtf.bam)')
    parser.add_argument('-k','--key',  dest='key', default='ATCG',
                        help='TF key sequence (Default: ATCG)')
    parser.add_argument('-f','--fasta',dest='fasta', default=None,
                        help='FASTA file with TF sequences. If not specified, '
                             'the pipeline will generate one from DefaultTF.conf')
    parser.add_argument('-d','--dir', dest='analysis_dir', default='.',
                        help='Directory containing 1.wells, BaseCaller.json, and '
                             'processParameters.txt files for this run (Default: current directory)')
    args = parser.parse_args()
    print "TFPipeline args :",args

    # Step 2. If reference fasta file not specified, build one

    if args.fasta == None:
        args.fasta = 'DefaultTF.fasta'
        buildTFReference(args.fasta)

    # Step 3. Perform alignment and generate bam file
    
    alignTFs(args.sff, args.bam, args.fasta)

    # Step 4. Post-processing. Run alignStats and TFMapper

    doAlignStats(args.bam)
    doTFMapper(args.sff, args.key, ".", ".", "TFMapper.stats",'TFStats.json')
    
    # Step 5. Generate TF performance plots

    generatePlots('TACGTACGTCTGAGCATCGATCGATGTACAGC',"TFStats.json")
    
    #executeTFPipeline(tfsff_filename,tfbam_filename,tfkey)




