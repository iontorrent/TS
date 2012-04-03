# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
from operator import itemgetter
from subprocess import *
import numpy
import pylab
from ion.reports.plotters import plotters
from ion.utils import template
#from ion.utils import template
#import os
#import parseRowCol
#import findRegion
from decimal import *
from colorify import *
import xml.dom.minidom
from xml.dom.minidom import Node
import math

def alignmentToIonogram(inputPath, metricKey):
    f = open(inputPath, 'r')
    top = []
    headings = []
    displayNum = 100
    def count_bars(bars):
        index = 0
        for i in str(bars):
            if i == '|':
                index += 1
            else:
                return index

    for count,line in enumerate(f):
        currentline = line.strip().split('\t')
        currentdict = dict([(i,j) for i,j in zip(headings,currentline)])
        if count == 0:
            headings = line.strip().split('\t')
        elif count < displayNum:
            num = count_bars(currentdict.get('match.a'))
            currentdict['perfect_length'] = num
            top.append(currentdict)
            
        else:
            if metricKey != 'perfect_length':
                if top[0].get(metricKey) < currentdict.get(metricKey):
                    top.pop(0)
                    top.append(currentdict)
                    top = sorted(top,key=itemgetter(metricKey))
                
            else: # look for perfect alignments
                num = count_bars(currentdict.get('match.a'))
                if num > top[0].get('perfect_length'):
                    currentdict['perfect_length'] = num
                    top.pop(0)
                    top.append(currentdict)
                    top = sorted(top,key=itemgetter('perfect_length'))

    # resort so longest is first
    stop = sorted(top,key=itemgetter(metricKey), reverse=True)
    return stop

def generate_align_html(alignment):
    ret = []

    inter = []
    for i in alignment.get('qDNA.a'):
        inter.append(colorify(i.upper()))
    ret.append(inter)
    
    inter = []
    for i in alignment.get('match.a'):
        inter.append(colorify(i.upper()))
    ret.append(inter)
    
    inter = []
    for i in alignment.get('tDNA.a'):
        inter.append(i.upper())
    ret.append(inter)
    return ret

def get_align_num(countLen):
    loc_string = []
    modulo = 10 
    for n in range(int(countLen)):
        if n%modulo == 0 and n != 0:
            num = str(n)
            if len(num)>1:
                for i in range(len(num)-1):
                    loc_string.pop()
            loc_string.append(str(n))
        else:
            loc_string.append("&nbsp;")
    return loc_string

def length(string):
    index = 0
    for i in string:
        if i !='-' or i!=' ':
            index+=1
    return index+1

def get_row_col(rc):
    r = rc.split(':')[1:3]
    row = int(r[0])
    col = int(r[1])
    return row,col

def remove_spaces(string):
    ret = []
    for i in string:
        if str(i) not in 'TACG': #if str(i) == '-' or str(i)== ' ' or str(i)=="N":
            continue
        else:
            ret.append(i)
    ret = "".join(ret)
    return ret

def get_ionograms(top, sffIn, outfolder, key, floworder, metricKey):
    nameMap = {'q20Len':'Q20',
               'q17Len':'Q17',
               'q10Len':'Q10',
               'q7Len':'Q7',
               'perfect_length':'Perfect'
               }
    filelist = {}
    alignhtml = {} 
    space_string = {}
    for align in top:
        alignstring = generate_align_html(align)
        row, col = get_row_col(align.get('name'))
        fin = Popen("SFFRead %s -C %s -R %s" % (sffIn,col,row), shell=True, stdout=PIPE)
        ionogram = fin.stdout.read()
        ionogram = ionogram.strip()
        ionogram = ionogram.split(" ")
        ionogram = [float(i) for i in ionogram]
        s = alignstring[1]
        expected = []
        lenOfIon = len(ionogram)   
        l = align.get(metricKey)
        fskey = template.deseqify(key, floworder)
        expected = fskey + list(pylab.zeros(lenOfIon-len(fskey)))
        intexp = []
        for i in expected:
            intexp.append(int(i))
        expected = intexp
        titleString = 'Single-Well Ionogram for (%s,%s) %s Length=%s Bases' % (int(row), int(col), nameMap.get(metricKey), l)
        it = plotters.IonogramJMR(floworder, expected[:len(ionogram)],ionogram,titleString)
        it.render()
        filelist[align.get('name')] = ("Well_(%s,%s).png" % (int(row),int(col)))
        alignhtml[align.get('name')] = alignstring
        countLen = math.floor(len(align.get('qDNA.a')))
        space_string[align.get('name')] = "".join(get_align_num(countLen))
        pylab.savefig(outfolder + "/Well_(%s,%s).png" % (int(row),int(col)))
    generateWeb(alignhtml, filelist, outfolder, top, space_string)#space_string, rowCollist)

def generateWeb(alignhtml, filelist, outfolder, top, space_string):#space_string, rowColList):
    f = open(outfolder + "/alignreport.html", "w")
    f.write('<html><body><font size=5, face="Arial"><center>Library Reads</center></font><font face="Courier"><center>\n')
    index = 0
    for x in top:
        rc = x.get('name')
        f.write('<img src = "%s">\n' % filelist[rc])
        f.write("<br/>\n")
        for i in alignhtml[rc]:
            for j in i:
                f.write("" + j + "")
            f.write("<br/>\n")
        f.write("" + space_string[rc] + "")
        f.write("<br/>\n")
        index += 1
    f.write('</center></font></body></hmtl>')
    f.close()

def CreateIonograms(blastIn, sffIn, outpath, key, floworder, metricKey):
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    top = alignmentToIonogram(blastIn, metricKey)
    get_ionograms(top, sffIn, outpath, key, floworder, metricKey)

if __name__=='__main__':
    import sys
    def check_bool(arg):
        if arg == 'True':
            return True
        else:
            return False
    blastpath = sys.argv[1]
    sffIn = sys.argv[2]
    outpath = sys.argv[3]
    key = sys.argv[4]
    floworder = sys.argv[5]
    metricKey = sys.argv[6]
    CreateIonograms(blastpath, sffIn, outpath, key, floworder, metricKey)


