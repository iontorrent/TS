#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os, sys, string, glob, os.path, json, subprocess
from optparse import OptionParser

sys.path.append('/opt/ion/')
header = None


def reformat_header(header):
  for i in range(len(header)):
    val = list()
    for word in string.split(header[i], "_"):
      if 1 < len(word):
        word = word[0] + string.lower(word[1:])
      val.append(word)
    header[i] = string.join(val, " ")
  return header

def reformat_data(data1):
  for i in range(len(data1)):
    if 0 == len(data1[i]):
      data1[i] = "NA"
  return data1

def print_html(fh, header, data):
  
  s1=map(lambda x: map(lambda y: ("<td>"+y+"</td>\n"),x), data)
  s2=map(lambda x: "<tr>"+string.join(x)+"</tr>\n", s1)
  s3 = string.join(s2)
  
  fh.write("<!DOCTYPE html>\n<html>\n" + \
    "<body>\n" + \
    "<h1>Bam Files with Duplicate Reads Removed</h1>\n" + \
    "<table border=\"1\" cellpadding=\"10\" id=\"markDuplicates\">\n" + \
    "<tr>\n" + \
    "<th>" + \
    string.join(header, "</th>\n<th>") + \
    "</th>\n" + \
    "</tr>\n" + \
    s3 + \
    "</table>\n" + \
    "</body>\n" + \
    "</html>\n")


def markDuplicates( options, data ):
  extra_files = glob.glob(os.path.join(options.analysis_dir,'*rawlib.bam'))
  extra_files.sort()
  for extra_file in extra_files:
    if os.path.exists(extra_file):
      bam_name = os.path.basename(extra_file)
      json_name = ('BamDuplicates.%s.json')%(os.path.normpath(bam_name))
      cmd = "BamDuplicates -i %s -o temp.bam -j %s ; samtools view -F 0x0400 -b temp.bam > %s; rm temp.bam"%(extra_file,json_name,bam_name)
      print "DEBUG: Calling %s"%cmd
      subprocess.call(cmd,shell=True)
      fjson = open(json_name,'r')
      js=json.load(fjson)
      data.append([ "<a href=%s> %s </a>"%(bam_name,bam_name), "%.2g%%"%(100*js[u'fraction_duplicates']),"%.2g%%"%(100*js[u'fraction_with_adaptor'])])
  return data
    
def main(options):
  data = []
  data = markDuplicates(options, data)
  
  #fh = open(options.metrics, 'r')
  #for line in fh:
    #line = line.rstrip('\r\n')
    #if 0 == len(line) or '#' == line[0]:
      #continue
    #if "LIBRARY" == line[:len("LIBRARY")]:
      #header = string.split(line, '\t') 
      #header = reformat_header(header)
    #elif None != header and None == data:
      #data = string.split(line, '\t') 
      #data = reformat_data(data)
  #fh.close()
  # write the html
  header = ['Filtered Bam File','Percent Duplicate Reads Removed','Percent Reads Reaching Adapter']
  data = reformat_data(data)
  fout = open('FilteredBam_block.html','w')
  print_html(fout, header, data)
  fout.close()


if __name__ == '__main__':
  parser = OptionParser()
  parser.add_option('-m', '--metrics-file', help="Duplicate stat file", dest='metrics', default='results.json')
  parser.add_option('-a', '--analysis-dir', help="Analysis Directory", dest='analysis_dir', default='../..')
  parser.add_option('-o', '--output-dir', help="Output Directory", dest='output_dir',default='.')
  
  options, args = parser.parse_args()
  main(options)
