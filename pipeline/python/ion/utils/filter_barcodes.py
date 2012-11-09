#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import os 
import glob
import csv
from shutil import move
from ion.utils.blockprocessing import printtime

def get_ids(filtfile):
# get barcodeIDs that should be removed
  filterBC = []  
  try:  
    freader = csv.DictReader(open(filtfile, "rU"))
  except:    
    return filterBC  
        
  for frow in freader:
    if frow["Include"] == '0':  filterBC.append(frow["BarcodeId"])
  return filterBC
  
  
def filter_barcodes(DIR_BC_FILES,DIR_BC_FILTERED,filtfile):
  
  if os.path.exists(filtfile):
    printtime ("filter_barcodes: found barcodeFilter.txt")
    filterBC = get_ids(filtfile)      
    
    if len(filterBC) > 0:
      if not os.path.exists(DIR_BC_FILTERED):
            os.mkdir(DIR_BC_FILTERED)
    
      # move all sff and fastq files output by barcodeSplit to filtered dir
      for bc_id in filterBC:
        barcodeFiles = glob.glob(os.path.join(DIR_BC_FILES, "%s_*" % bc_id))
        for bcfile in barcodeFiles:
          if os.path.exists(bcfile):
            printtime ("filter_barcodes: removing %s" % bcfile)
            try:
              move(bcfile, DIR_BC_FILTERED)
            except:
              traceback.print_exc()
        

