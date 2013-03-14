#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
import os
import sys
import json

if __name__ == '__main__':
  try:
    JSON_INPUT = json.load( open(sys.argv[1], "r") )
    PLAN_INFO = JSON_INPUT['plan']
  except:
    print ";"
    sys.exit(0) 

  if PLAN_INFO:
     runtype = PLAN_INFO['runType']
     regionf = PLAN_INFO['bedfile']
     if ( runtype == 'AMPS' ):
        runtype = "ampliseq"       
     elif ( runtype == 'AMPS_RNA' ):
        runtype = "ampliseq-rna"
     elif ( runtype == 'TARS' ):
        runtype = "targetseq"
     elif ( runtype == 'WGNM' ):
        runtype = "wholegenome"
     else:
        runtype = ""  
     print "%s;%s" % (runtype, regionf)
  else:
     print ";"
