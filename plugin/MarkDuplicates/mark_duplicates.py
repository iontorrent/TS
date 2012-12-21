#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

import os, sys, string
from optparse import OptionParser

def reformat_header(header):
  for i in range(len(header)):
    val = list()
    for word in string.split(header[i], "_"):
      if 1 < len(word):
        word = word[0] + string.lower(word[1:])
      val.append(word)
    header[i] = string.join(val, " ")
  return header

def reformat_data(data):
  for i in range(len(data)):
    if 0 == len(data[i]):
      data[i] = "NA"
  return data

def print_html(fh, header, data):
  fh.write("<html>\n" + \
    "<head>\n" +
    "<!--In the header of your page, paste the following for Kendo UI Web styles-->\n" + \
    "<link href=\"/site_media/resources/kendo/styles/kendo.common.min.css\" rel=\"stylesheet\" />\n" + \
    "<link href=\"/site_media/resources/less/kendo.tb.min.css\" rel=\"stylesheet\" />\n" + \
    "<link type=\"text/css\" rel=\"stylesheet\" href=\"/site_media/resources/styles/tb-layout.css\" />\n" + \
    "<link type=\"text/css\" rel=\"stylesheet\" href=\"/site_media/resources/styles/tb-styles.min.css\" />\n" + \
    "<script type=\"text/javascript\" src=\"/site_media/resources/jquery/jquery-1.7.2.min.js\"></script>\n" + \
    "<script type=\"text/javascript\" src=\"/site_media/resources/scripts/kendo.custom.min.js\"></script>\n" + \
    "<script>\n" + \
    "$(function(){\n" + \
    "$(\"#markDuplicates\").kendoGrid({\n" + \
    "dataSource: {\n" + \
    "pageSize: 10\n" + \
    "},\n" + \
    "height: 'auto',\n" + \
    "groupable: false,\n" + \
    "scrollable: false,\n" + \
    "selectable: false,\n" + \
    "sortable: true,\n" + \
    "pageable: true\n" + \
    "});        \n" + \
    "});\n" + \
    "</script>\n" + \
    "</head>\n" + \
    "<body>\n" + \
    "<div style=\"margin: 15px;\">\n" + \
    "<table id=\"markDuplicates\" >\n" + \
    "<tr>" + \
    "<th>" + \
    string.join(header, "</th><th>") + \
    "</th>" + \
    "</tr>" + \
    "<tr>" + \
    "<td>" + \
    string.join(data, "</td><td>") + \
    "</td>" + \
    "</tr>" + \
    "</table>" + \
    "</div>\n" + \
    "</body>\n" + \
    "</html>\n")


def main(options):
  header = None
  data = None
  fh = open(options.metrics, 'r')
  for line in fh:
    line = line.rstrip('\r\n')
    if 0 == len(line) or '#' == line[0]:
      continue
    if "LIBRARY" == line[:len("LIBRARY")]:
      header = string.split(line, '\t') 
      header = reformat_header(header)
    elif None != header and None == data:
      data = string.split(line, '\t') 
      data = reformat_data(data)
  fh.close()
  # write the html
  print_html(sys.stdout, header, data)

def check_option(parser, value, name):
  if None == value:
    sys.stderr.write('Option ' + name + ' required.\n')
    parser.print_help()
    sys.exit(1)

if __name__ == '__main__':
  parser = OptionParser()
  parser.add_option('-m', '--metrics-file', help="the Picard MarkDuplicates metrics file", dest='metrics')
  if len(sys.argv[1:]) < 1:
    parser.print_help()
  else:
    options, args = parser.parse_args()
    check_option(parser, options.metrics, '-m')
    main(options)
