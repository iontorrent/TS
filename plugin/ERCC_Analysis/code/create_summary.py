# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import os
from time import strftime


def create_summary_block(OUTPUT_DIR,dr,MINIMUM_RSQUARED):
  #in the case where the plugin is run more than once, the summary block will be cached by the browser
  #and not refreshed if it is named simply "summary_block.html".  Therefore, we include an infix of
  #the date and time so that it will be a different filename (still ending in "_block.html" each time,
  #and the browser will therefore not use the cached version.
  for filename in os.listdir(OUTPUT_DIR): #will also catch dir names but they will be screened out by next line
    if (filename[0:7] == 'summary') and (filename.endswith('block.html')):
      os.remove(OUTPUT_DIR+filename)
  infix = strftime("%Y-%m-%d_%H-%M-%S")
  SUMMARY_BLOCK = OUTPUT_DIR + 'summary_' + infix + '_block.html'
  summary_block = open(SUMMARY_BLOCK,'w')
  rsquared = '%.2f' % (dr[5])
  msg_to_user = ''
  if (float(rsquared) < float(MINIMUM_RSQUARED)):
    msg_to_user = '<p style="color:sienna;font-size:300%">R-SQUARED VALUE IS BELOW THE MINIMUM ACCEPTABLE OF '+str(MINIMUM_RSQUARED)+'</p>'
    summary_block.write('<p><img src="/pluginMedia/ERCC_Analysis/img/stoplight-icon-red.JPG" alt="red light" width="14" height="40" />R-squared of '+rsquared+' is below the minimum acceptable of '+str(MINIMUM_RSQUARED)+'.</p><p>'+infix+'</p>')
  else:
    summary_block.write('<p><img src="/pluginMedia/ERCC_Analysis/img/stoplight-icon-green.JPG" alt="green light" width="14" height="40" />R-squared of '+rsquared+' is above the minimum acceptable of '+str(MINIMUM_RSQUARED)+'.</p><p>'+infix+'</p>')
  summary_block.close()
  return msg_to_user, rsquared 
