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

def blastToIonogram(blastPath, displayNum, byLength, key, floworder, getPerfect):
    blast = open(blastPath, "r")
    fullRowCol = {}
    firstIdentity = []
    alignstring = {}
    startVal = {}
    endVal = {}
    currentQuery = None
    for line in blast:
        if "Iteration_query-def>" in line:
            if line not in fullRowCol:
                line = line.strip()
                line = line.strip().split(">")
                line = line[1].split("<")
                line = str(line[0])
                line = line.strip()
                fullRowCol[line] = []
                alignstring[line] = []
                currentQuery = line
        if byLength:
            if "Hsp_qseq" in line:
                fullRowCol[currentQuery].append(line.strip())
        else:
            if "Hsp_bit-score" in line:
                fullRowCol[currentQuery].append(line.strip())
        if "Hsp_query-from" in line:
            alignstring[currentQuery].append(line.strip())
        if "Hsp_query-to" in line:
            alignstring[currentQuery].append(line.strip())
        if "Hsp_align-len" in line:
            alignstring[currentQuery].append(line.strip())
        if "Hsp_positive" in line:
            alignstring[currentQuery].append(line.strip())
        if "Hsp_qseq" in line:
            alignstring[currentQuery].append(line.strip())
        if "Hsp_midline" in line:
            alignstring[currentQuery].append(line.strip())
        if "Hsp_hseq" in line:
            alignstring[currentQuery].append(line.strip())

    #print alignstring
    alignstringfill = {}
    for k,v in alignstring.iteritems():
        curval = None
        curkey = None
        bars = None
        if len(v)>0:
            fr = v[0].split(">")
            fr = fr[1].split("<")
            fr = fr[0]
            #print fr
            to = v[1].split(">")
            to = to[1].split("<")
            to = to[0]
            #print to
            pos = v[2].split(">")
            pos = pos[1].split("<")
            pos = pos[0]
            #print pos
            lengt = v[3].split(">")
            lengt = lengt[1].split("<")
            lengt = lengt[0]
            #print length
            q = v[4].split(">")
            q = q[1].split("<")
            q=q[0]
            #print q
            s = v[5].split(">")
            s = s[1].split("<")
            s = s[0]
            #print s
            bars = v[6].split(">")
            bars = bars[1].split("<")
            bars = bars[0]
            #print bars
            output = [q,bars,s,fr,to,pos,lengt]
            curKey = k.split("|")
            row = curKey[0].split("r")[-1]
            col = curKey[1].split("c")[-1]
            rowcol = (int(row), int(col))
            alignstringfill[rowcol] = output
    #print fullRowCol
    filtered = {}
    for k,v in fullRowCol.iteritems():
        curVal = None
        curKey = None
        if len(v) > 0:
            curVal = v[0].split(">")
            curVal = curVal[1].split("<")
            curVal = curVal[0]
            curKey = k.split("|")
            row = curKey[0].split("r")[-1]
            col = curKey[1].split("c")[-1]
            rowcol = (int(row), int(col))
            if byLength:
                filtered[rowcol] = length(curVal)
            else:
                filtered[rowcol] = float(curVal)
    sortedWells = sorted(filtered.items(), key=itemgetter(1), reverse = True)
    topfive = []
    classData = []
    readsVcycles = []
    perfect_alignments = []
    for rowcol,val in sortedWells:
        dif = abs(int(alignstringfill[rowcol][3])-int(alignstringfill[rowcol][4]))
        to = int(alignstringfill[rowcol][4])
        fro = int(alignstringfill[rowcol][3])
        string = alignstringfill[rowcol][0]
        matched = alignstringfill[rowcol][5]
        lengt = alignstringfill[rowcol][6]
        if 'N' not in string:
            if to<3 or fro<3 and dif>21:
                topfive.append((rowcol,val))
                classData.append((rowcol,val,dif,string))
                if matched == lengt:
                    perfect_alignments.append((rowcol,val))
    if getPerfect:
        topfive = perfect_alignments[:displayNum]
    else:
        topfive = topfive[:displayNum]
    if len(classData)<10000:
        cycles_vs_align_len(classData, blastPath, floworder)
    else:
        cycles_vs_align_len(classData[:10000], blastPath, floworder)
    alignmenthtml = {}
    space_string = {}
    for rowcol,val in topfive:
        inter = []
        index = 0
        for i in alignstringfill[rowcol][:3]:
            if index == 0:
                inter.append(i.upper())
                index += 1
            else:
                inter.append(colorify(i.upper()))
            alignmenthtml[rowcol] = inter
        countLen = math.floor(len(alignstringfill[rowcol][:3][0]))
        #print alignstringfill[rowcol][:3][0]
        space_string[rowcol] = "".join(get_align_num(countLen))
    return (topfive, alignmenthtml, alignstringfill, space_string)

def cycles_vs_align_len(classData, blastPath, floworder):
    pylab.clf()         
    l = []
    c = []
    for rowcol,leng,dif,seq in classData:
        t = template.deseqify(remove_spaces(seq), floworder)
        l.append(dif)
        c.append(len(t)/4)
    pylab.scatter(l,c)
    if "/" in blastPath:
        p = blastPath.strip().split("/")[-1]
        path = p.strip().split(".")[0]
    else:
        path = blastPath.strip().split(",")[0]
    if 'blastn' in path:
        pretitle = 'Relaxed Blast'
    if 'megablast' in path:
        pretitle = 'Strict Blast'
    pylab.title("%s Number of Flows vs. Alignment Length" % pretitle)
    pylab.xlabel("Length")
    pylab.ylabel("Flows")
    pylab.gcf().set_size_inches((8,4))
    pylab.axis([0,max(l)+10,0,max(c)+10])
    #x = [2.2*i for i in range(max(l))]
    #pylab.plot(x)
    pylab.savefig("alignment_vs_cycles_%s.png" % path)
    pylab.clf()

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

def get_ionograms(rowCollist, outfolder, alignmenthtml, sffIn, alignstringfill, byLength, space_string, key, floworder):
    filelist = {}
    alignhtml = {} 
    for i in rowCollist:
        rowcol = i[0]
        row = rowcol[0]
        col = rowcol[1]
        fin = Popen("SFFRead %s -C %s -R %s" % (sffIn,col,row), shell=True, stdout=PIPE)
        ionogram = fin.stdout.read()
        ionogram = ionogram.strip()
        ionogram = ionogram.split(" ")
        ionogram = [float(i) for i in ionogram]
        s = alignstringfill[rowcol][0]
        expected = []
        lenOfIon = len(ionogram)   
        l = length(str(s.strip()))
        alignhtml[rowcol] = check_for_alignment(ionogram, alignstringfill, rowcol, floworder, key)
        fskey = template.deseqify(key, floworder)
        expected = fskey + list(pylab.zeros(lenOfIon-len(fskey)))
        intexp = []
        for i in expected:
            intexp.append(int(i))
        expected = intexp
        titleString = 'Single-Well Ionogram for (%s,%s) Length=%s Bases' % (int(row), int(col), l)
        it = plotters.IonogramJMR(floworder, expected[:len(ionogram)],ionogram,titleString)
        it.render()
        filelist[rowcol] = ("Well_(%s,%s).png" % (int(row),int(col)))
        pylab.savefig(outfolder + "/Well_(%s,%s).png" % (int(row),int(col)))
    generateWeb(alignhtml, filelist, outfolder, space_string, rowCollist)

def gen_colorify(rowCol, alignstringfill):
    align = []
    inter = []
    index = 0
    for i in alignstringfill[rowCol][:3]:
        if index != 2:
            inter.append(colorify(i.upper()))
            index += 1
        else:
            inter.append(i.upper())
    return inter

def check_for_alignment(ionogram, alignstringfill, rowCol, floworder, key):
    baseCall = {}
    flowOrder = floworder
    for ndx,flow in enumerate(ionogram):
        flowIndex = ndx%len(floworder)
        currentFlow=flowOrder[flowIndex]
        flowInt = int(flow)
        baseCall[ndx]=dict({currentFlow: threshold(flow)})
    cString = generateString(baseCall, key)
    called = alignstringfill[rowCol][0]
    calledNoS = remove_spaces(called)
    bars = alignstringfill[rowCol][1]
    align = alignstringfill[rowCol][2]
    corAlign = {}
    last_base = 6
    if calledNoS[0:8] in cString[0:20]:
        corAlign[rowCol] = []
        corAlign[rowCol].append(called)
        corAlign[rowCol].append(bars)
        corAlign[rowCol].append(align)
        return gen_colorify(rowCol,corAlign)

    elif reverse(calledNoS)[0:8] in cString[0:20]:
        corAlign[rowCol] = []
        corAlign[rowCol].append(reverse(called))
        corAlign[rowCol].append(reverse(bars))
        corAlign[rowCol].append(reverse(align))
        return gen_colorify(rowCol, corAlign)

    elif reverse_compliment(calledNoS)[0:8] in cString[0:20]:
        corAlign[rowCol] = []
        corAlign[rowCol].append(reverse_compliment(called))
        corAlign[rowCol].append(reverse(bars))
        corAlign[rowCol].append(reverse_compliment(align))
        return gen_colorify(rowCol, corAlign)
        
    elif compliment(calledNoS)[0:8] in cString[0:20]:
        corAlign[rowCol] = []
        corAlign[rowCol].append(compliment(called))
        corAlign[rowCol].append(bars)
        corAlign[rowCol].append(compliment(align))
        return gen_colorify(rowCol, corAlign)
    # Should the ionogram not pass the other criteria
    # leave it be.  
    else:
        print "shit" 
        corAlign[rowCol] = []
        corAlign[rowCol].append(called)
        corAlign[rowCol].append(bars)
        corAlign[rowCol].append(align)
        return gen_colorify(rowCol,corAlign)

def remove_spaces(string):
    ret = []
    for i in string:
        if str(i) not in 'TACG': #if str(i) == '-' or str(i)== ' ' or str(i)=="N":
            continue
        else:
            ret.append(i)
    ret = "".join(ret)
    return ret

def compliment(string):
    rev = []
    for i in string:
        if i == 'A':
            rev.append('T')
        if i == 'T':
            rev.append('A')
        if i == 'C':
            rev.append('G')
        if i == 'G':
            rev.append('C')
        if i == '-':
            rev.append('-')
    revString = ''.join(rev)
    return revString

def reverse_compliment(string):
    rev = []
    for i in string:
        if i == 'A':
            rev.append('T')
        if i == 'T':
            rev.append('A')
        if i == 'C':
            rev.append('G')
        if i == 'G':
            rev.append('C')
        if i == '-':
            rev.append('-')
    revString = ''.join(rev)
    return reverse(revString)

def reverse(string):
    return string[::-1]
        
def generateString(baseCall, key):
    string = []
    keylen = len(key)
    for k,v in baseCall.iteritems():
        for k,v in v.iteritems():
            if v==0:
                continue
            else:
                string.append(k*v)
    string = "".join(string)
    return string[keylen:] # drop the key for alignment testing

def threshold(i):
    if i>0 and i<=.5 or i==0:
        return 0
    if i>.5 and i<=1.5:
        return 1
    if i>1.5 and i<=2.5:
        return 2
    if i>2.5 and i<=3.5:
        return 3
    if i>3.5 and i<=4.5:
        return 4
    if i>4.5 and i<=5.5:
        return 5
    if i>5.5 and i<=6.5:
        return 6
    if i>6.5 and i<=7.5:
        return 7
    if i>7.5 and i<=8.5:
        return 8
    if i>8.5 and i<=9.5:
        return 9

def generateWeb(alignhtml, filelist, outfolder, space_string, rowColList):
    f = open(outfolder + "/alignreport.html", "w")
    f.write('<html><body><font size=5, face="Arial"><center>Library Reads</center></font><font face="Courier"><center>\n')
    index = 0
    for rc in rowColList:
        rc = rc[0]
        f.write('<img src = "%s">\n' % filelist[rc])
        f.write("<br/>\n")
        for i in alignhtml[rc]:
            f.write("" + i + "")
            f.write("<br/>\n")
        f.write("" + space_string[rc] + "")
        f.write("<br/>\n")
        index += 1
        
    f.write('</center></font></body></hmtl>')
    f.close()

def CreateIonograms(blastIn, pathout, sffIn, displayNum=100, byLength=False, key='TCAG', floworder='TACG', getPefect=False):
    if not os.path.isdir(pathout):
        os.mkdir(pathout)
    (topfive, alignmenthtml, alignstringfill, space_string) = blastToIonogram(blastIn, displayNum, byLength, key, floworder, getPerfect)
    get_ionograms(topfive, pathout, alignmenthtml, sffIn, alignstringfill, byLength, space_string, key, floworder)

if __name__=='__main__':
    import sys
    def check_bool(arg):
        if arg == 'True':
            return True
        else:
            return False
    if len(sys.argv) < 3:
        print "proper usage: BlastPath OutPath sffin dispnum byLength? key floworder getPefect?"
    blastpath = sys.argv[1]
    outpath = sys.argv[2]
    sffIn = sys.argv[3]
    displayNum = int(sys.argv[4])
    byLength = check_bool(sys.argv[5])
    key = sys.argv[6]
    floworder = sys.argv[7]
    getPerfect = check_bool(sys.argv[8])
    CreateIonograms(blastpath, outpath, sffIn, displayNum, byLength, key, floworder, getPerfect)


