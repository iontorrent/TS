# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

import os

class TMAP( object ):
    """
        ex:tmap mapall -Y -n 4 -f /results/referenceLibrary/tmap-f2/hg19/hg19.fasta -r /results2/analysis/output/Home/Auto_HOF-568-r124806-1100rpm_280m-jo_7819_11575/R_2011_08_31_17_10_25_user_HOF-568-r124806-1100rpm_280m-jo_Auto_HOF-568-r124806-1100rpm_280m-jo_7819.fastq.truncated.fastq -v stage1 map1 map2 map3 2>> alignmentQC_out.txt
        
    """
    def __init__(self):
        self.program = "tmap mapall"
        self.defaultParams = "-n 4 -v stage1 map1 map2 map3"
        self.sam = False
        self.pipe = False
    
    
    def SetReference(self, referenceFile):
        self.referenceFile = referenceFile
    
    def SetReadsSFF(self, sff):
        self.readsSFF = sff
    
    def SetSamName(self, sam):
        self.sam = " > %s" % ( sam )
    
    def SetOutputPipeDestination(self, destination):
        self.pipe = " | %s" % (destination)
    
    def AddParam(self, paramName, paramValue=" "):
        self.defaultParams = "-%s %s %s" % (paramName, paramValue, self.defaultParams)
    
    def CommandLine(self):
        returnStr = self.pipe
        if self.sam is not False:
            returnStr = self.sam
        else:
            returnStr = self.pipe
        
        self.commandLine = "%s -f %s -r %s %s %s" % ( self.program, self.referenceFile, self.readsSFF, self.defaultParams, returnStr )
        return self.commandLine
    
        
    def Align(self):
        print self.CommandLine()
        os.system( self.CommandLine() )


class AlignmentQC( object ):
    
    def __init__( self, drmaa_stdout_file ):
        lines = open(drmaa_stdout_file, "r").readlines()        
        self.AutoBuildCommandLine(lines)
        tokens = self.CommandLine().split()
        self.paramsDict = {}
        for i in xrange(1, len( tokens[1:] ), 2 ):
            #--param arg
            self.paramsDict[ tokens[i] ] = tokens[ i + 1 ]
            
        
        
    def CommandLine( self ):
        """
            returns command line for alignmentQC.pl
        """
            
        return self.commandLine
    
    def Align( self ):
        os.system( self.CommandLine() )
        
    
    def SetReference( self ):
        pass
        
    
    def ReplaceParam(self, param, replace):
        self.commandLine = self.commandLine.replace(param, replace) 
    
    
    def AppendToAlignerOpts(self, newOption, newArg=None):
        if newArg:
            newOption += (" " + newArg)
        print "self.paramsDict: ", self.paramsDict
        print "\naligner-opts: ", self.paramsDict['--aligner-opts']
        print " "
        self.commandLine += " --aligner-opts \"-Y\""
        
        
    def AutoBuildCommandLine( self, linesInFile, searchTerm="alignmentQC.pl", positionInLine=0 ):
        
        for l in linesInFile:
            if l.find(searchTerm) == positionInLine:
                #found alignmentQC.pl command time to split it up
                self.commandLine = l.rstrip()
                
        
            
        
        
        
    
