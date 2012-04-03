#!/usr/bin/python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

import os
import json

def merge(blockDirs, resultsDir):
    '''mergeBaseCallerJson.merge - Combine BaseCaller.json metrics from multiple blocks'''
    
    combinedJson = {'Phasing' : {'CF' : 0,
                                 'IE' : 0,
                                 'DR' : 0},
                    "BeadSummary" : {"lib" : {"badKey"      : 0,
                                              "highPPF"     : 0,
                                              "highRes"     : 0,
                                              "key"         : "TCAG",
                                              "polyclonal"  : 0,
                                              "short"       : 0,
                                              "valid"       : 0,
                                              "zero"        : 0},
                                     "tf" :  {"badKey"      : 0,
                                              "highPPF"     : 0,
                                              "highRes"     : 0,
                                              "key"         : "ATCG",
                                              "polyclonal"  : 0,
                                              "short"       : 0,
                                              "valid"       : 0,
                                              "zero"        : 0 }}}
    numBlocks = 0.0
    
    for dir in blockDirs:
        try:
            file = open(os.path.join(dir,'BaseCaller.json'), 'r')
            blockJson = json.load(file)
            file.close()
            
            blockCF             = blockJson['Phasing']['CF']
            blockIE             = blockJson['Phasing']['IE']
            blockDR             = blockJson['Phasing']['DR']
            
            blockLibbadKey      = blockJson['BeadSummary']['lib']['badKey']
            blockLibhighPPF     = blockJson['BeadSummary']['lib']['highPPF']
            blockLibhighRes     = blockJson['BeadSummary']['lib']['highRes']
            blockLibkey         = blockJson['BeadSummary']['lib']['key']
            blockLibpolyclonal  = blockJson['BeadSummary']['lib']['polyclonal']
            blockLibshort       = blockJson['BeadSummary']['lib']['short']
            blockLibvalid       = blockJson['BeadSummary']['lib']['valid']
            blockLibzero        = blockJson['BeadSummary']['lib']['zero']
            
            blockTFbadKey      = blockJson['BeadSummary']['tf']['badKey']
            blockTFhighPPF     = blockJson['BeadSummary']['tf']['highPPF']
            blockTFhighRes     = blockJson['BeadSummary']['tf']['highRes']
            blockTFkey         = blockJson['BeadSummary']['tf']['key']
            blockTFpolyclonal  = blockJson['BeadSummary']['tf']['polyclonal']
            blockTFshort       = blockJson['BeadSummary']['tf']['short']
            blockTFvalid       = blockJson['BeadSummary']['tf']['valid']
            blockTFzero        = blockJson['BeadSummary']['tf']['zero']

            combinedJson['Phasing']['CF'] += blockCF
            combinedJson['Phasing']['IE'] += blockIE
            combinedJson['Phasing']['DR'] += blockDR
            numBlocks += 1.0
            
            combinedJson['BeadSummary']['lib']['badKey']        += blockLibbadKey
            combinedJson['BeadSummary']['lib']['highPPF']       += blockLibhighPPF
            combinedJson['BeadSummary']['lib']['highRes']       += blockLibhighRes
            combinedJson['BeadSummary']['lib']['key']           = blockLibkey
            combinedJson['BeadSummary']['lib']['polyclonal']    += blockLibpolyclonal
            combinedJson['BeadSummary']['lib']['short']         += blockLibshort
            combinedJson['BeadSummary']['lib']['valid']         += blockLibvalid
            combinedJson['BeadSummary']['lib']['zero']          += blockLibzero

            combinedJson['BeadSummary']['tf']['badKey']        += blockTFbadKey
            combinedJson['BeadSummary']['tf']['highPPF']       += blockTFhighPPF
            combinedJson['BeadSummary']['tf']['highRes']       += blockTFhighRes
            combinedJson['BeadSummary']['tf']['key']           = blockTFkey
            combinedJson['BeadSummary']['tf']['polyclonal']    += blockTFpolyclonal
            combinedJson['BeadSummary']['tf']['short']         += blockTFshort
            combinedJson['BeadSummary']['tf']['valid']         += blockTFvalid
            combinedJson['BeadSummary']['tf']['zero']          += blockTFzero

        except:
            print 'mergeBaseCallerJson: Pass block ' + dir

    if numBlocks > 0:
        combinedJson['Phasing']['CF'] /= numBlocks
        combinedJson['Phasing']['IE'] /= numBlocks
        combinedJson['Phasing']['DR'] /= numBlocks

    file = open(os.path.join(resultsDir,'BaseCaller.json'), 'w')
    file.write(json.dumps(combinedJson,indent=4))
    file.close()


if __name__=="__main__":
    
    blockDirs = ['a','b']
    resultsDir = '.'
    
    merge(blockDirs,resultsDir)



