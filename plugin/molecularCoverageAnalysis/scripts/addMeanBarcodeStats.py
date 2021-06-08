#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
import subprocess
import json
import traceback

if __name__ == "__main__":
  nargs = len(sys.argv)-1
  if nargs != 2:
    sys.stderr.write( "addMeanBarcodeStats.py: Incorrect number of arguments (%d).\n" % nargs )
    sys.stderr.write( "Usage: addMeanBarcodeStats.py <json file> \"<statistic name>\"\n" )
    sys.exit(1)

  jsonfile = sys.argv[1]
  keystat = sys.argv[2]

  keystat.strip();
  if keystat == "":
    sys.stderr.write( "addMeanBarcodeStats.py: Empty string passed as argument 2 (statistic name).\n" )
    sys.exit(1)

  try:
    with open(jsonfile) as jsonFile:
      jsonParams = json.load(jsonFile)
  except:
    sys.stderr.write( "addMeanBarcodeStats.py: Failed to read json from '%s'.\n" % jsonfile )
    sys.exit(1)

  # look for barcodes and average over stat or just repeat the value or non-bacoded run
  try:
    statsum = 0
    isPercent = False
    numbc = 0
    missing = 0
    if jsonParams['barcoded'] == "true":
      barcodes = jsonParams['barcodes']
      for barcode in barcodes:
        # ignore barcodes with error report
        if 'Error' in barcodes[barcode]:
          continue
        if not keystat in barcodes[barcode]:
          missing += 1
          continue
        val = barcodes[barcode][keystat]
        if "%" in val:
          isPercent = True
          val = val.strip('%%')
        statsum += float(val) if '.' in val else int(val)
        numbc += 1
    # do not have to worry about non-bacoded case as should have the same stat name already
    if numbc > 0:
      statsum /= numbc
      if isPercent:
        jsonParams[keystat] = '%.2f%%' % statsum
      else:
        jsonParams[keystat] = str(statsum)
  except:
    sys.stderr.write( "addMeanBarcodeStats.py: Warning: Could not read key value for key '%s' in '%s'.\n" % (keystat,jsonfile) )
    sys.exit(0)

  if missing > 0:
    sys.stderr.write( "addMeanBarcodeStats.py: Warning: key '%s' was not present for %d barcodes in '%s'.\n" % (keystat,missing,jsonfile) )

  # add the new result to the json and write back out
  if numbc > 0:
    try:
      with open(jsonfile,'w') as jsonFile:
        json.dump(jsonParams,jsonFile,indent=2,sort_keys=True)
    except:
      sys.stderr.write( "addMeanBarcodeStats.py: Error trying to write json file '%s'.\n" % jsonfile )
      traceback.print_exc()
      sys.exit(1)

  sys.exit(0)

