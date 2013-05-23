#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
import os
import sys
import json
import barcodeutils

#constants
ANALYSIS_JSON = "/ion_params_00.json"
PLUGIN_JSON = "/startplugin.json"
TMAP_FILE_INDEX_VERSION = "2"
TMAP_REFERENCE_BASE_LOCATION = "/results2/referenceLibrary/tmap-f%s" % (TMAP_FILE_INDEX_VERSION)
BARCODE_LIST = "/barcodeList.txt"


class PluginWrapper(object):
    
    def __init__(self, pluginJSONDir):
        j = open(pluginJSONDir + PLUGIN_JSON, "r")
        self.params = json.load( j )
        #setup fields
        self.runInfo = RunInfo(self.params['runinfo'])
        self.globalConfig = GlobalConfig(self.params['globalconfig'])
        
        self.analysisDir = None
        self.analysisDir = AnalysisDir( self.runInfo.AnalysisDir() )
        
        for key in self.params['pluginconfig']:
            setattr(self, key, self.params['pluginconfig'][key])
        
    def Init(self):
        #will be implemented by a plugin
        pass
    
    def CommandLine(self):
        pass    
    
    def Execute(self):
        os.system( self.CommandLine() )
    
    def GetAnalysisDir(self):
        return self.analysisDir
    
    def OutputDir(self):
        return self.runInfo.ResultsDir() + "/"
    
class JSONBlock(object):
    
    def __init__(self, JSONDict):
        self.jsonDict = JSONDict
        for key in JSONDict:
            setattr(self, key, JSONDict[key])
    
 
    
class GlobalConfig(JSONBlock):
    
    def Debug(self):
        return self.debug
    
class RunInfo(JSONBlock):
    
    def AnalysisDir(self):
        return self.analysis_dir
    
    def ResultsDir(self):
        return self.results_dir
    
    def PluginDir(self):
        return self.plugin_dir
    
    
class ExperimentLog(JSONBlock):
    
    def ExperimentName(self):
        return self.experiment_name
    
class PluginConfig(JSONBlock):
    
    def Blah(self):
        pass
    

class AnalysisDir(object):
        
    def __init__(self, analysisDir):
        j = open(analysisDir + ANALYSIS_JSON, "r")
        self.params = json.load( j )
        self.analysisDir = analysisDir
        for key in self.params:
            #print "\n",key
            #print self.params[key], "\n"
            setattr(self, key, self.params[key])
        #json.loads( exp_dict['log'] )
        self.experimentLog = ExperimentLog( json.loads( self.ExperimentDict()['log'] ))
        if os.path.exists( analysisDir + BARCODE_LIST ):
            bcList = open(analysisDir + "/barcodeList.txt", "r").readlines()
            self.barcodeList = barcodeutils.BarcodeList( bcList )
        else:
            self.barcodeList = None
        
    def __str__(self):
        return self.analysisDir
    #####################
    #   Get functions   #
    #####################
    def SiteName(self):
        return self.site_name
    
    def GetSFF(self):
        return self.GetFastq().replace("fastq", "sff")
    
    
    
    def GetFastq(self):
        return self.fastqpath
    
    def LibraryName(self):
        return self.libraryName
    
    def FlowOrder(self):
        return self.flowOrder
        
    def ResultsPrefix(self):
        return self.resultsName
    
    def LibraryKey(self):
        return self.libraryKey
        
    def UrlPath(self):
        return self.url_path
    
    def PathToData(self):
        return self.pathToData
    
    def ChipType(self):
        return self.chiptype
    
    def Project(self):
        return self.project
    
    def Plugins(self):
        """
            returns a dictionary of plugin names
        """
        return self.plugins
        
    def ExperimentDict(self):
        return json.loads(self.exp_json)
        
    def Sample(self):
        return self.ExperimentDict()['sample']
        
    def MetaData(self):
        return json.loads( self.ExperimentDict()['metaData'] )
    
    def Log(self):
        return json.loads( self.ExperimentDict()['log'] )
    
    def ExperimentLog(self):
        return self.experimentLog  
        

        
    
    def ExperimentName(self):
        exp_dict = self.ExperimentDict()
        log_dict = json.loads( exp_dict['log'] )
        return log_dict['experiment_name']
        
    def DriverVersion(self):
        return self.Log()['driver_version']
    
    def FirmwareVersion(self):
        return self.Log()['firmware_version']
    
    def ChipSerialNumber(self):
        return self.Log()['serial_number']
    
    def ChipBarcode(self):
        return self.Log()['chipbarcode']
        
    def StartTime(self):
        return self.Log()['start_time']
    
    def TotalFlows(self):
        return self.Log()['flows']
    
    
    def GetFastqFiles(self):
        """
            returns a list of all fastq files from the analysis dir
        """
        import fnmatch
        return fnmatch.filter(os.listdir(), "*.fastq")
        
        
    def GetBamFile(self):
        bam = self.GetFastq().replace("fastq", "bam")
        return "%s/%s" % ( str(self), bam)
        
class ReferenceLibrary(object):
    
    def __init__(self, libraryName):
        
        self.fastaDir = "%s/%s" % (TMAP_REFERENCE_BASE_LOCATION, libraryName)
        self.fasta = None
        import fnmatch
        try:
            self.fasta = str( fnmatch.filter( os.listdir( self.fastaDir ), "*.fasta" )[-1] )
        except OSError:
            print "[ReferenceLibrary]: searching for reference library.."
            ref_base = None
            for i in xrange(7):
                ref_base = "/results%s/referenceLibrary/tmap-f%s/%s" % (str(i), TMAP_FILE_INDEX_VERSION, libraryName)
                try:
                    self.fasta = str(fnmatch.filter( os.listdir( ref_base ), "*.fasta" )[-1])
                    break
                except:
                    print "[ReferenceLibrary] not found in ", ref_base
                    
                    
            if not self.fasta:
                try:
                    ref_base = "/results/referenceLibrary/tmap-f%s/%s" % (TMAP_FILE_INDEX_VERSION, libraryName)
                    self.fasta = str( fnmatch.filter( os.listdir( ref_base ), "*.fasta" )[-1] )
                except:
                    print "[ReferenceLibrary] not found in ", ref_base
                    ref_base = None

            self.fastaDir = ref_base
        
    def FastaPath(self):
        return "%s/%s" % (self.fastaDir, self.fasta)
        
    def Fasta(self):
        return self.fasta
    
    def FastaDir(self):
        return self.FastaDir
    
    
if __name__ == '__main__':
    test = AnalysisDir( os.getcwd() )
