# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import pylab
import sys,os
import math
from ion.utils import template

class PerRegionKey:
    def __init__(self,key, floworder, savepath):
        self.key = key
        self.floworder = floworder
        self.savepath = savepath
        self.data = None
    
    def plot(self):
        num = 1
        numGraphs = math.ceil(math.sqrt(self.get_num_graphs()))
        Rows = int(numGraphs)  
        Cols = int(numGraphs)               
        pylab.clf()
        fig = pylab.figure(1, figsize=(Rows, Cols))
        flowspacekey = template.deseqify(self.key[:len(self.key)-1], self.floworder)
        flowspacekey = flowspacekey[:len(flowspacekey)-1]
        for region,data in self.data.iteritems():
            #ATCG 01001011
            trace = {}
            flowstring = [self.floworder[n%len(self.floworder)] for n in range(len(flowspacekey))]
            flowstring = "".join(flowstring)
            flowstring = flowstring
            for n,base in enumerate(list(self.floworder)):
                baseIndexs = self.getindex(base, flowstring)
                min = None
                max = []
                for i in baseIndexs:
                    mer = flowspacekey[i]
                    if mer == 0:
                        if min == None: # only take first zero of any nuc for subtraction
                            min = data[i]
                    if mer == 1:
                        max.append(data[i])
                if len(max) > 0 and len(min) > 0:
                    trace[base] = [float(i-j) for i,j in zip(max[0],min)]
            atrace = []
            for base in list(trace.keys()):
                atrace.append(trace[base])
            ax = fig.add_subplot(Rows,Cols,num)
            pylab.subplots_adjust(wspace=.5,hspace=.5)
            pylab.title("Region %s" % str(region))
            pylab.setp(ax.get_yticklabels(), visible=True)
            pylab.setp(ax.get_xticklabels(), visible=False)
            for label in ax.get_yticklabels():
                label.set_fontsize(8)
            pylab.gcf().set_size_inches((Rows+Cols,Rows+Cols))
            for t in atrace:
                pylab.plot(t)
            num += 1
        pylab.savefig(self.savepath)
    
    def getindex(self, base, flowstring):
        ret = []
        for n,i in enumerate(flowstring):
            if i == base:
                ret.append(n)
        return ret
                
    def parse(self, pathIn):
        f = open(pathIn, 'r')
        data = f.readlines()
        f.close()
        traces = {}
        for line in data:
            l = line.strip().split(' ')
            cycle = l[1]
            region = int(l[0])
            data = [float(i) for i in l[3:]]
            if region not in traces:
                traces[region]=[]
                traces[region].append(data)
            else:
                traces[region].append(data)
        self.data = traces
        return traces   

    def get_num_graphs(self):
        return len(self.data.keys())        

if __name__=="__main__":
    pr = PerRegionKey('ATCGT', 'TACG', 'jeffout.png')
    pr.parse('/home/jeff/Desktop/averagedKeyTraces_TF.txt')
    pr.plot()
