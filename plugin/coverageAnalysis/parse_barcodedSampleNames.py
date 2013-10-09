#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
"""
Parse the startplugin.json file to extract and return a simple NVP list of sample names for barcode ids.
"""

import os
import sys
import json
from pprint import pprint

def sampleNames(jsonfile):
  if not os.path.isfile(jsonfile):
    sys.stderr.write('Error: Cannot find json file %s\n'%jsonfile)
  samplenames = ";"
  try:
    JSON_INPUT = json.load( open(jsonfile, "r") )
    BC_SAMPS = JSON_INPUT['plan']['barcodedSamples']
    if isinstance(BC_SAMPS,basestring):
      BC_SAMPS = json.loads(BC_SAMPS)
    for bcname in BC_SAMPS:
      BARCODES = BC_SAMPS[bcname]['barcodes']
      for bc in BARCODES:
        samplenames += bc + '=' + bcname + ';'
  except:
    return None
  return samplenames

if __name__ == '__main__':
  print sampleNames(sys.argv[1])

