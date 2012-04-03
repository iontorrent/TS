# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys, os
import pylab
#from ion.reports.plotters import plotters

class LibGraphs:
    def __init__(self, data, xlabel, ylabel, title, faceColor, saveName):
        self.color = faceColor
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.savepath = saveName
        self.data = data

    def plot(self):
        pylab.clf()
        if len(self.data) > 0:
            xaxis = [i for i in range(len(self.data))]  
            ymax = max(self.data) + 10
            xlen = len(self.data) + 10
            pylab.gcf().set_size_inches((8,4))
            pylab.bar(xaxis, self.data, facecolor = self.color, align = 'center', linewidth=0)
            pylab.xlabel(self.xlabel)
            if self.ylabel != None:
                pylab.ylabel(self.ylabel)
            pylab.title(self.title)
            pylab.axis([0,200,0,ymax])#hardcode x axis to go from 0 - 200
            pylab.savefig(self.savepath)
            pylab.clf()

class ReadLenErr(LibGraphs):
    pass
class ReadLenQscore(LibGraphs):
    pass
class QXXAlign(LibGraphs):
    def plot(self):
        pylab.clf()
        if len(self.data) > 0:
            xaxis = [i for i in range(len(self.data))]
            ymax = max(self.data) + 10
            xlen = len(self.data) + 10
            xmax = len(self.data) - 1
            if xmax < 400:
                xmax = 400
            pylab.gcf().set_size_inches((8,4))
            pylab.bar(xaxis, self.data, facecolor = self.color, align = 'center', linewidth=0, alpha=1.0, width = 1.0)
            pylab.xlabel(self.xlabel)
            if self.ylabel != None:
                pylab.ylabel(self.ylabel)
            pylab.title(self.title)
            pylab.axis([0,xmax,0,ymax])#hardcode x axis to go from 0 - 200
            pylab.savefig(self.savepath)
            pylab.clf()
class ReadLen(LibGraphs):
    pass
class MatchMissMatch(LibGraphs):
    pass

class Q10Align(QXXAlign):
    pass
class Q17Align(QXXAlign):
    pass
class Q20Align(QXXAlign):
    pass
class Q47Align(QXXAlign):
    pass

def parseFile(fileIn):
    f = open(fileIn, 'r')
    data = f.readlines()
    f.close()
    return [float(i.strip().split(' ')[-1]) for i in data]

if __name__=='__main__':
    """
    data = parseFile('/home/jeff/Desktop/Q17.blastn.histo.dat')
    it = Q17Blast(data, "Filtered Relaxed Blast Q17 Read Lengths", 'Count', 
            "Filtered Relaxed Blast Q17 Read Lengths", 'yellow', "Filtered_Blastn_Q17.png")
    it.plot()
    data = parseFile('/home/jeff/Desktop/Q17.megablast.histo.dat')
    it = Q17Blast(data, "Filtered Strict Blast Q17 Read Lengths", 'Count', 
            "Filtered Strict Blast Q17 Read Lengths", 'yellow', "Filtered_MegaBlast_Q17.png")
    it.plot()

    data = parseFile('/home/jeff/Desktop/ReadLenErrorPerc.megablast.histo.dat')
    it = ReadLenErr(data, "Filtered Strict Blast Read Lengths", 'Average Error Percentage', 
            "Filtered Strict Blast Average Error Percentage", 'purple', "Filtered_Strict_Error_Perc.png")
    it.plot()

    data = parseFile('/home/jeff/Desktop/ReadLenErrorPerc.blastn.histo.dat')
    it = ReadLenErr(data, "Filtered Relaxed Blast Read Lengths", 'Average Error Percentage', 
            "Filtered Relaxed Blast Average Error Percentage", 'purple', "Filtered_Relaxed_Error_Perc.png")
    it.plot()

    data = parseFile('/home/jeff/Desktop/ReadLenQScore.megablast.histo.dat')
    it = ReadLenQscore(data, "Filtered Strict Blast Read Lengths", 'Average Q Score', 
            "Filtered Strict Blast Average Q Score", 'orange', "Filtered_Strict_Qscore.png")
    it.plot()

    data = parseFile('/home/jeff/Desktop/ReadLenQScore.blastn.histo.dat')
    it = ReadLenQscore(data, "Filtered Relaxed Blast Read Lengths", 'Average Q Score', 
            "Filtered Relaxed Blast Average Q Score", 'orange', "Filtered_Relaxed_Qscore.png")
    it.plot()
    data = parseFile('/home/jeff/Desktop/Q10.blastn.histo.dat')
    it = ReadLenQscore(data, "Filtered Relaxed Blast Q10 Read Lengths", 'Count', 
            "Filtered Relaxed Blast Average Q10 Read Length", 'red', "Filtered_Blastn_Q10.png")
    it.plot()    
    data = parseFile('/home/jeff/Desktop/Q10.megablast.histo.dat')
    it = ReadLenQscore(data, "Filtered Strict Blast Q10 Read Lengths", 'Count', 
            "Filtered Strict Blast Average Q10 Read Length", 'red', "Filtered_MegaBlast_Q10.png")
    it.plot()
    """
    data = parseFile('/home/jeff/Desktop/ReadLength.blastn.histo.dat')
    it = ReadLen(data,'Alignment Length', 'Filtered Relaxed Blast Alignments',
                 "Filtered Relaxed Blast Alignments", '#CC33CC', "Filtered_Blastn_Align_Len.png")
    it.plot()
    data = parseFile('MatchMismatch.blastn.histo.dat')
    it = ReadLen(data,'Correct Bases - Incorrect Bases', 'Counts',
                 "Match-Mismatch Reads Relaxed Filtered", 'blue', "alignment_histogram.png")
    it.plot()
    data = parseFile('MatchMismatch.megablast.histo.dat')
    it = ReadLen(data,'Correct Bases - Incorrect Bases', 'Counts',
                 "Match-Mismatch Reads Strict Filtered", 'blue', "alignment_histogram_mega.png")
    it.plot()
