#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os

def parseLog(logText):
    metrics ={}
    lineDict = {}
    currentKey = None
    ###Get Headings for TF's
    for line in logText:
        if line[0] != "#":
            if 'TF Name' in line:
                name = line.strip().split("=")
                key = name[0].strip()
                value = name[1].strip()
                if key not in metrics:
                    metrics[value] = {}
                    currentKey = value
                    lineDict = {}
            else:        
                met = line.strip().split('=')
                k = met[0].strip()
                v = met[1].strip()
                lineDict[k] = v
                metrics[currentKey]=lineDict
    return metrics


def generateWeb(fileList, title, alignhtml):
    f = open("tf_ionograms_%s.html" % title, "w")
    f.write('<html><body><font size=5, face="Arial"><center>Top %s Ionograms</center></font><font face="Courier">\n' % title)
    for graph,text in zip(fileList,alignhtml):
        f.write("<div id = %s style='float:center'>\n" % title)
        f.write("<center>")
        f.write("<table>")
        f.write("<tr>")
        for i in graph:
            f.write("<td><a href = '%s'><img src = '%s' width=425 height=212 border=0></a></td>\n" % (i,i))
        f.write("</tr>")
        f.write("</table>")
        f.write("</center>")
        f.write("<br/>\n")
        f.write("</div>\n<div style='clear:both;'></div>\n")
        f.write("<center>")
        for i in text:
            f.write("" + "".join(i) + "")
            f.write("<br/>\n")
        f.write("</center>\n")
        f.write("<br/>")
    f.write('</font></body></hmtl>')
    f.close()


def generateMetricsData(fileIn):
    f = open(fileIn, 'r')
    log = f.readlines()
    f.close()
    data = parseLog(log)
    return data
