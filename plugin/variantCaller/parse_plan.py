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
    print ";;;"
    sys.exit(0) 

  if PLAN_INFO:
     runtype = PLAN_INFO['runType']
     varfreq = PLAN_INFO['variantfrequency']
     regionf = PLAN_INFO['bedfile']
     hotspot = PLAN_INFO['regionfile']
     if ( runtype == 'AMPS' ):
        runtype = "ampliseq"       
     elif ( runtype == 'TARS' ):
        runtype = "targetseq"
     elif ( runtype == 'WGNM' ):
        runtype = "fullgenome"
     else:
        runtype = ""  
     if ( varfreq == 'Germ Line' ):
        varfreq = "Germ_Line"       
     elif ( varfreq == 'Somatic' ):
        varfreq = "Somatic"
     else:
        varfreq = ""                   
     print "%s;%s;%s;%s" % (runtype, varfreq, regionf, hotspot)
  else:
     print ";;;"
