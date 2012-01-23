# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
from os import path
import pylab
import numpy
import math 
import matplotlib

class HeatMap:
    def __init__(self, savePath=None):
        self.height = None
        self.width = None
        self.rawData = None
        self.data = {}
        self.savePath = savePath
        self.title = None
    
    def loadData(self, fileIn):
        """Load data from File"""
        if path.exists(fileIn):
            f = open(fileIn)
            data = f.readlines()
            f.close()
            self.rawData = data
            return data
        else:
            return None

    def parseRegion(self):
        """Parse data from cafieRegions and store data in data structure"""
        imageDat = []
        currentKey = None
        currentTF = None
        if self.rawData == None:
            print "Must load data first:  HeatMap.loadData"
            sys.exit()
        else:
            for line in self.rawData:
                if 'Height' in line:
                    self.height = line.strip().split("=")[-1].strip()
                    continue
                elif 'Width' in line:
                    self.width = line.strip().split("=")[-1].strip()
                    continue
                elif 'TF' in line:
                    currentTF = line.strip().split("=")[-1].strip()
                    self.data[currentTF] = {}
                elif '=' in line:
                    line = line.strip().split("=")
                    key = line[0].strip()
                    self.data[currentTF][key] = []
                    line = [float(i)*100 for i in line[-1].strip().split(" ")]
                    self.data[currentTF][key].append(line)
                    currentKey = key
                else:
                    line = [float(i)*100 for i in line.strip().split(" ")]
                    self.data[currentTF][currentKey].append(line)
                    
    def writeMetricsFile(self):
        '''Writes the lib_cafie.txt file'''
        if self.data == None:
            print "Must load and parse data first"
            sys.exit()
        cafie_out = open('lib_cafie.txt','w')
        for k,v in self.data.iteritems():
            for type, data in v.iteritems():
                flattened = []
                for i in data:
                    for j in i:
                        flattened.append(j)
                flattened = filter(lambda x: x != 0.0, flattened)
                Aver=pylab.average(flattened)
                name = type.replace(" ", "_")
                if 'LIB' in k:
                    if len(flattened)==0:
                        cafie_out.write('%s = %s\n' % (name,0.0))
                        Aver = 0
                    else:
                        cafie_out.write('%s = %s\n' % (name,Aver))
                        Aver=pylab.average(flattened)
        cafie_out.close()
        
    def plot(self):
        """generate the plot formatting"""
        if self.data == None:
            print "Must load and parse data first"
            sys.exit()
            
        for k,v in self.data.iteritems():
            for type, data in v.iteritems():
                pylab.clf()
                height = int(self.height)
                width = int(self.width)
                pylab.figure()
                ax = pylab.gca()
                ax.set_xlabel('<--- Width = %s wells --->' % str(width))
                ax.set_ylabel('<--- Height = %s wells --->' % str(height))
                ax.set_yticks([0,height/10])
                ax.set_xticks([0,width/10])
                ax.set_yticklabels([0,height])
                ax.set_xticklabels([0,width])
                ax.autoscale_view()
                pylab.jet()
            #color = self.makeColorMap()
            #remove zeros for calculation of average
                flattened = []
                for i in data:
                    for j in i:
                        flattened.append(j)
                flattened = filter(lambda x: x != 0.0, flattened)
                Aver=pylab.average(flattened)
                name = type.replace(" ", "_")
                fave = ("%.2f") % Aver
                pylab.title(k.strip().split(" ")[-1] + " Heat Map (Average = "+fave+"%)")
                ticks = None
                vmax = None
                if type == "Region DR":
                    ticks = [0.0,0.2,0.4,0.6,0.8,1.0]
                    vmax = 1.0
                else:
                    ticks = [0.0,0.4,0.8,1.2,1.6,2.0]
                    vmax = 2.0
                    
                pylab.imshow(data, vmin=0, vmax=vmax, origin='lower')
                pylab.colorbar(format='%.2f %%',ticks=ticks)
                pylab.vmin = 0.0
                pylab.vmax = 2.0
            #pylab.colorbar()     

                if self.savePath is None:
                    save = "%s_heat_map_%s.png" % (name,k)
                else:
                    save = path.join(self.savePath,"%s_heat_map_%s.png" % (name,k))
                pylab.savefig(save)
                pylab.clf()
        
if __name__=="__main__":
    cf = HeatMap()
    cf.savePath = "/home/jeff/Desktop"
    cf.loadData(sys.argv[-1])
    cf.parseRegion()
    cf.plot()
    cf.writeMetricsFile()
