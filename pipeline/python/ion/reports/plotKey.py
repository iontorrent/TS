# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import os
from ion.reports.plotters import plotters
from numpy import median


class KeyPlot:

    def __init__(self, key, floworder, title=None):
        self.data = None
        self.key = key
        self.floworder = floworder
        self.title = title
        self.average_peak = None

    def plot(self):
        expected = [1 for i in range(len(self.key)-1)]
        tracePlot = plotters.Iontrace(self.key,
                                      expected,
                                      self.data,
                                      title="Consensus Key 1-Mer - %s Ave. Peak = %s" % (self.title, self.average_peak))
        tracePlot.render()
        tracePlot.save(os.path.join(os.getcwd(), 'iontrace_%s' % self.title.replace(" ", "_")))

    def parse(self, fileIn):
        d = open(fileIn, 'r')
        data = d.readlines()
        d.close()
        trace = {}
        max = None  # max length needed to fill in null values
        for line in data:
            t = line.strip().split(" ")
            fTrace = [float(i) for i in t[1:]]
            trace[t[0]] = fTrace
            if max < len(fTrace) or max == None:
                max = len(fTrace)
        toPlot = []
        for k in self.key:
            if k in trace.keys():
                toPlot.append(trace[k])
            else:
                toPlot.append([0 for i in range(max)])
        self.data = trace
        return toPlot

    def dump_max(self, fileName):
        try:
            with open(fileName, 'a') as f:
                max_array = [max(trace) for k, trace in self.data.iteritems() if k in self.key]
                self.average_peak = int(median(max_array)) if len(max_array) > 0 else 0
                f.write("%s = %s\n" % (self.title, self.average_peak))
        except Exception:
            print("Can't open file")


if __name__ == "__main__":
    libKey = sys.argv[2]
    floworder = sys.argv[3]
    fileIn = sys.argv[1]
    fileOut = sys.argv[4]
    kp = KeyPlot(libKey, floworder, 'Test Fragment')
    kp.parse(fileIn)
    kp.dump_max(fileOut)
    kp.plot()
