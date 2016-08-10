# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys, os
import pylab
import math


class BeadfindHistogram:

    def __init__(self, bins, **kwargs):
        self.bins = bins
        self.rows = kwargs['rows']
        self.cols = kwargs['cols']

    def plot(self, key, graph_data, num, row, col, ax):
        pylab.subplots_adjust(wspace=.5, hspace=.5)
        pylab.title("Region %s" % str(key))
        pylab.setp(ax.get_yticklabels(), visible=False)
        pylab.setp(ax.get_xticklabels(), visible=False)
        pylab.gcf().set_size_inches((row+col, row+col))
        h = pylab.hist(graph_data, bins=self.bins)
        # count, bin, junk = pylab.hist(v, bins=self.bins)


def parseFile(fileIn, rows, cols):
    # one liner for confusion later....
    # JB - this should be cleaned up to parse the file into an
    # array of dicts for folding and transposing
    data = dict([(int(i.strip().split('=')[0].strip()),
                  [float(t) for t in i.strip().split('=')[1].strip().split(' ')
                   if len(i.strip().split('=')[1].strip()) > 0])
                 for i in open(fileIn, 'r')])
    numGraphs = math.ceil(math.sqrt(get_num_graphs(data)))
    redata = []
    tmp = []
    # make it 2-d - remove this loop, see above comments
    for index, (region, dat) in enumerate(data.iteritems()):
        tmp.append(dict([(region, dat)]))
        if region % rows == 0 and index != 0:
            redata.append(tmp)
            tmp = []
    redata = fold(redata, numGraphs)
    redata = transpose(redata)
    return redata, numGraphs


def fold(data, numGraphs):
    for num, line in enumerate(data):
        length = int(len(line))
        last = None
        for begin, end in enumerate(reversed(range(length))):
            if begin >= int(length/2):
                break
            f = line[begin]
            l = line[end]
            data[num][begin] = l
            data[num][end] = f
            last = end
    return data


def transpose(data):
    tmp = []
    for i in range(len(data[0])):
        t = []
        for j in range(len(data)):
            t.append(data[j][i])
        tmp.append(t)
    return tmp


def get_num_graphs(data):
    return len(data.keys())


def makeHist(fileIn, prefix, row, col):
    pylab.clf()
    bg = BeadfindHistogram(bins=50, rows=row, cols=col)
    data, numGraphs = parseFile(fileIn, row, col)
    num = 1
    fig = pylab.figure(1, figsize=(bg.rows, bg.cols))
    for index, line in enumerate(data):
        for ele in line:
            k = ele.keys()[0]
            v = ele.values()[0]
            if len(v) > 0:
                ax = fig.add_subplot(bg.rows, bg.cols, num)
                bg.plot(k, v, num, bg.rows, bg.cols, ax)
                num += 1
    pylab.savefig('%s_beadfind_per_region.png' % prefix)
    # pylab.show()

if __name__ == "__main__":
    # parseFile(sys.argv[1])
    makeHist(sys.argv[1], 'post', 24, 26)
