#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import os, sys
import time
import argparse
import traceback
import json
from matplotlib import use
use("Agg",warn=False)
import pylab
import matplotlib.pyplot as plt
from ion.reports.plotters import plotters
import shutil
import subprocess
from ion.utils.blockprocessing import printtime



# Primary use cases:
# - Load one TFMetrics.json
# - Load and combine multiple TFMetrics.json
# - Generate uploadMetrics-friendly dictionary with TF metrics
# - Generate report plots

# Secondary use cases:
# - Generate TF plugin plots
# - Generate TF html files



class NoTFDataException(Exception):
    def __init__(self, msg):
        self.msg = msg
        
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

        self.topReads = sorted(sum([dataItem['Top Reads'] for dataItem in dataList],[]), key=lambda k: k['metric'])
        self.topReads = self.topReads[:10]


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
            'Q10 Mean' : sum(x[0]*x[1] for x in enumerate(self.Q10hist))/float(sum(self.Q10hist)),
            'Q17 Mean' : sum(x[0]*x[1] for x in enumerate(self.Q17hist))/float(sum(self.Q17hist)),
            '50Q10' : sum(self.Q10hist[50:]),
            '50Q17' : sum(self.Q17hist[50:]),

            'Avg Ionogram NUM' : self.avgIonogramRawNum,
            'Avg Ionogram DEN' : self.avgIonogramRawDen,
            'Corrected Avg Ionogram NUM' : self.avgIonogramCorrectedNum,
            'Corrected Avg Ionogram DEN' : self.avgIonogramCorrectedDen,

            'Top Reads' : self.topReads

        }

        return metrics


    def plotQ10(self):

        qplot = plotters.QPlot2(self.Q10hist[:101], q=10, expected=self.sequence[:100])
        qplot.render()
        pylab.savefig(os.path.join(os.getcwd(), "Q10_%s.png" % self.name))
        pylab.clf()

    def plotQ17(self):

        qplot = plotters.QPlot2(self.Q17hist[:101], q=17, expected=self.sequence[:100])
        qplot.render()
        pylab.savefig(os.path.join(os.getcwd(), "Q17_%s.png" % self.name))
        pylab.clf()


    def plotNewQ17(self):
        
        num_bases_q = len(self.Q17hist)
        num_bases_s = len(self.sequence)
        num_bases = min(num_bases_q,num_bases_s)
        nuc_color = {'A':"#4DAF4A",'C':"#275EB8",'T':"#E41A1C",'G':"#202020"}
        text_offset = -max(self.Q17hist) * 0.1
        
        fig = plt.figure(figsize=(8,1),dpi=100)
        ax = fig.add_subplot(111,frame_on=False,xticks=[],yticks=[],position=[0,0.3,1,0.7])
        ax.bar(range(num_bases_q),self.Q17hist,linewidth=0,width=1,color="#2D4782")
        for idx in range(num_bases):
            nuc = self.sequence[idx]
            ax.text(idx+1,text_offset,nuc,horizontalalignment='center',verticalalignment='center',fontsize=8,family='sans-serif',weight='bold',color=nuc_color[nuc])
            if (idx%10) == 0:
                ax.text(idx+0.5,3*text_offset,str(idx),horizontalalignment='center',verticalalignment='center',fontsize=8,family='sans-serif',weight='bold')
                
        ax.set_xlim(0,num_bases_q)
        fig.patch.set_alpha(0.0)
        plt.savefig("new_Q17_%s.png" % self.name)



    def plotAvgIonograms(self, floworder):

        v = [float(self.avgIonogramCorrectedNum[idx])/self.avgIonogramCorrectedDen[idx] for idx in range(len(self.avgIonogramCorrectedNum))]
        corrIonogram = plotters.IonogramJMR(floworder,v,v,'Average Corrected Ionogram')
        corrIonogram.render()
        pylab.savefig(os.path.join(os.getcwd(), "Average Corrected Ionogram_%s.png" % self.name))


def buildTFReference(tfreffasta_filename,analysis_dir,tfkey):
    '''
    Build the DefaultTFs.fasta from DefaultTFs.conf
    '''

    DefaultTFconfPath = os.path.join(analysis_dir,'DefaultTFs.conf')
    if not os.path.exists(DefaultTFconfPath):
        if not os.path.exists('/opt/ion/config/DefaultTFs.conf'):
            printtime('ERROR: could not locate DefaultTFs.conf (tried %s and /opt/ion/config/DefaultTFs.conf)' % DefaultTFconfPath)
            raise IOError
        DefaultTFconfPath = '/opt/ion/config/DefaultTFs.conf'

    printtime('TFPipeline: Using TF sequences from %s' % DefaultTFconfPath)
    num_tfs = 0
    try:
        confFile = open(DefaultTFconfPath, 'r')
        fastaFile = open(tfreffasta_filename, 'w')

        for confLine in confFile.readlines():
            if len(confLine) == 0:
                continue
            if confLine[0] == '#':
                continue
            confEntries = confLine.split(',')
            if len(confEntries) != 3:
                continue
            if confEntries[1] != tfkey:
                continue

            fastaFile.write('>%s\n' % confEntries[0])
            fastaFile.write('%s\n' % str(confEntries[2]).strip())
            num_tfs += 1

        confFile.close()
        fastaFile.close()

    except Exception as e:
        printtime("ERROR: failed convert %s into %s" % (DefaultTFconfPath, tfreffasta_filename))
        raise e
    
    if num_tfs == 0:
        printtime("No suitable TFs with key %s found in %s" % (tfkey, DefaultTFconfPath))
        raise NoTFDataException('No TF reference sequences')
        



def alignTFs(basecaller_bam_filename,bam_filename,fasta_filename):

    # Step 1. Build tmap index for DefaultTFs.fasta in a temporary directory

    indexDir = 'tfref'  # Might instead use a folder in /tmp
    indexFile = os.path.join(indexDir,'DefaultTFs.fasta')

    printtime("TFPipeline: Building index '%s' and mapping '%s'" % (indexFile,basecaller_bam_filename))

    if not os.path.exists(indexDir):
        os.makedirs(indexDir)

    shutil.copyfile(fasta_filename, indexFile)

    subprocess.check_call("tmap index -f %s" % indexFile, shell=True)

    # Step 2. Perform mapping of the bam file

    com1 = "tmap mapall -n 12 -f %s -r %s -Y -v stage1 map1 map2 map3" % (indexFile, basecaller_bam_filename)
    com2 = "samtools view -Sb -o %s -" % bam_filename
    p1 = subprocess.Popen(com1, stdout=subprocess.PIPE, shell=True)
    p2 = subprocess.Popen(com2, stdin=p1.stdout, shell=True)
    p2.communicate()
    p1.communicate()

    # Step 3. Delete index

    shutil.rmtree(indexDir, ignore_errors=True)

    if p1.returncode != 0:
        raise subprocess.CalledProcessError(p1.returncode, com1)
    if p2.returncode != 0:
        # Assumption: samtools view only fails when there are zero reads.
        printtime("Command %s failed, presumably because there are no TF reads" % (com2))
        raise NoTFDataException('No TF reads found')        
        #raise subprocess.CalledProcessError(p2.returncode, com2)


    # Step 4. Bonus: Make index for the fasta

    try:
        subprocess.check_call("samtools faidx %s" % fasta_filename, shell=True)
    except:
        printtime("WARNING: samtools faidx failed")




def doAlignStats(bam_filename):

    try:
        com = 'alignStats -i %s -p 1 -o TF -a TF.alignTable.txt -n 12' % bam_filename
        printtime("DEBUG: Calling '%s'" % com)
        os.system(com)
    except:
        printtime("ERROR: alignStats failed")


def doTFMapper(tfbam_filename, tfref_filename, tfstatsjson_path):

    try:
        #com = "/home/msikora/Documents/TFMapper"
        com = "TFMapper"
        com += " --output-json %s" % tfstatsjson_path
        com += " --bam %s" % tfbam_filename
        com += " --ref %s" % tfref_filename
        printtime("DEBUG: Calling '%s'" % com)
        os.system(com)
    except:
        printtime("ERROR: TFMapper failed")


def generatePlots(floworder,tfstatsjson_path):

    try:

        jsonfile = open(tfstatsjson_path, 'r')
        TFStatsJson = json.load(jsonfile)
        jsonfile.close()

        if TFStatsJson == None:
            TFStatsJson = {}

        for k,v in TFStatsJson.iteritems():
            tf = TFStats(k,v)
            tf.plotQ10()
            tf.plotQ17()
            tf.plotNewQ17()
            tf.plotAvgIonograms(floworder)

    except Exception:
        printtime("Metrics Gen Failed")
        traceback.print_exc()




def processBlock(tf_basecaller_bam_filename, BASECALLER_RESULTS, tfkey, floworder, analysis_dir):

    try:

        # These files will be created
        tfstatsjson_path = os.path.join(BASECALLER_RESULTS,"TFStats.json")
        tfbam_filename = os.path.join(BASECALLER_RESULTS,"rawtf.bam")
        tfref_filename = os.path.join(BASECALLER_RESULTS,"DefaultTFs.fasta")

        # TF analysis in 5 simple steps

        buildTFReference(tfref_filename,analysis_dir,tfkey)

        alignTFs(tf_basecaller_bam_filename, tfbam_filename, tfref_filename)

        doAlignStats(tfbam_filename)    # Note: alignStats dumps its results to files in current directory

        doTFMapper(tfbam_filename, tfref_filename, tfstatsjson_path)

        generatePlots(floworder,tfstatsjson_path)

    
    except NoTFDataException as e:
        printtime("No data to analyze Test Fragments (%s)" % e.msg)
        f = open(os.path.join(BASECALLER_RESULTS,'TFStats.json'),'w')
        f.write(json.dumps({}))
        f.close()

    except:
        traceback.print_exc()


def mergeBlocks(BASECALLER_RESULTS,dirs,floworder):

    # Input: list of blocks
    # Step 1: Read individual TFStats.json and merge
    # Step 2: Generate combined TFStats.json
    # Step 3: Generate plots: Q10, Q17, AvgIonograms

    recordList = []
    recordKeys = []

    for subdir in dirs:
        _subdir = os.path.join(BASECALLER_RESULTS,subdir)

        try:
            jsonfile = open(os.path.join(_subdir,'TFStats.json'), 'r')
            TFStatsJson = json.load(jsonfile)
            jsonfile.close()
            if TFStatsJson != None:
                recordList.extend(TFStatsJson.items())
                recordKeys.extend(TFStatsJson.keys())
        except:
            print "Could not process block %s" % subdir

    allTFStats = {}
    for tf in recordKeys:
        if tf not in allTFStats:
            allTFStats[tf] = TFStats(tf,[v[1] for v in recordList if v[0] == tf])

    # Output combined TFStats.json

    mergedTFStatsJson = {}
    for k,v in allTFStats.iteritems():
        mergedTFStatsJson[k] = v.getTFStatsMetrics()
    f = open(os.path.join(BASECALLER_RESULTS,'TFStats.json'),'w')
    f.write(json.dumps(mergedTFStatsJson, indent=4))
    f.close()

    # Generate plots

    for k,v in allTFStats.iteritems():
        v.plotQ10()
        v.plotQ17()
        v.plotNewQ17()
        v.plotAvgIonograms(floworder)



if __name__=="__main__":

    # Step 1. Parser command line arguments

    parser = argparse.ArgumentParser(description='Test Fragment evaluation pipeline.')
    parser.add_argument('-i','--input',dest='basecaller_bam', default='rawtf.basecaller.bam',
                        help='Input unmapped BAM file containing TF reads (Default: rawtf.basecaller.bam)')
    parser.add_argument('-b','--bam',  dest='bam', default='rawtf.bam',
                        help='Intermediate output BAM file for TF reads (Default: rawtf.bam)')
    parser.add_argument('-k','--key',  dest='key', default='ATCG',
                        help='TF key sequence (Default: ATCG)')
    parser.add_argument('-f','--ref',dest='ref', default=None,
                        help='FASTA file with TF sequences. If not specified, '
                             'the pipeline will generate one from DefaultTF.conf')
    parser.add_argument('-d','--dir', dest='analysis_dir', default='.',
                        help='Directory searched for DefaultTFs.conf (Default: current directory)')
    args = parser.parse_args()
    print "TFPipeline args :",args

    # Step 2. If reference fasta file not specified, build one

    try:
        if args.ref == None:
            args.ref = 'DefaultTFs.fasta'
            buildTFReference(args.ref,args.analysis_dir,args.key)

        # Step 3. Perform alignment and generate bam file

        alignTFs(args.basecaller_bam, args.bam, args.ref)

        # Step 4. Post-processing. Run alignStats and TFMapper

        doAlignStats(args.bam)
        doTFMapper(args.bam, args.ref, 'TFStats.json')

        # Step 5. Generate TF performance plots

        generatePlots('TACGTACGTCTGAGCATCGATCGATGTACAGC',"TFStats.json")

    except:
        traceback.print_exc()




