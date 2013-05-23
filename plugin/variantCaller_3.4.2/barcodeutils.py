# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
import json_utils


class Barcode(object):
    
    def __init__(self, barcodeLine):
        barcodeLine = barcodeLine.rstrip()
        self.index, self.idString, self.sequence, self.adapterSequence, self.annotation, self.type, self.length, self.flowOrder = barcodeLine.split(',')
        
    

class BarcodeList(object):
    """class to represent the barcodeList.txt file"""
    def __init__(self, barcodeList):
        
        self.barcodes = []
        self.fileId = barcodeList[0]
        self.scoreMode = barcodeList[1]
        self.scoreCutoff = barcodeList[2]
        for line in barcodeList[3:]:
            self.barcodes.append( Barcode(line) )

    def GetBarcodedBams(self):
        """returns a list of absolute paths to all barcoded bam files in the analysis
        directory
        """
        pass





def main(barcodeList="./barcodeList.txt"):
    
    bcList = open(barcodeList, "r").readlines()
    bc = BarcodeList( bcList )
    print bc.fileId
    print bc.scoreMode
    print bc.scoreCutoff
    for barcode in bc.barcodes:
        print barcode.index, barcode.idString, barcode.sequence, barcode.adapterSequence, barcode.annotation, barcode.type, barcode.length, barcode.flowOrder
        
    
if __name__ == '__main__':
    main()
