# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Various plotting classes, used to generate analysis summary plot elements.
"""

import base64
from datetime import datetime, timedelta
import math
import pylab
import random
import re
from scipy import stats, signal
import StringIO
try:
    from PIL import Image
except ImportError:
    HAVE_PIL = False
else:
    HAVE_PIL = True

from ion.utils import template as Template

NUMPY_VERSION = reduce(lambda acc,x: acc*10 + x,
        map(int, pylab.np.version.version.split('.')), 0)

class DataFormatError(Exception):
    def __init__(self, format):
        self.format = format
    def __str__(self):
        return 'Format required was ' + self.format

class Plotter:
    def __init__(self):
        self.colors = ["#ff6666", "#66b366", "#6666ff", "#666666"] + list('y')
        self.numcolors = 4
        self.floworder = 'TACG'
        self.numflows = len(self.floworder)
        self.textcolor = 'k'
        self.figbounds = (8,4)
        self.title = 'Plot'
        self.textposition = (0.5, 0.1)
        self.comments = ''
    def _nuc2ndx(self,ndx):
        # get nuc from ndx
        nuc = self.floworder[ndx % self.numflows]
        # get index of nuc (ok python gods, help me out here, this is really ugly!)
        nucindex = 0
        if nuc == 'T':
            nucindex = 0
        if nuc == 'A':
            nucindex = 1
        if nuc == 'C':
            nucindex = 2
        if nuc == 'G':
            nucindex = 3
        return nucindex
    def _ndx2clr(self,ndx):
        # return color from index
        return self.colors[ndx % self.numcolors]
    def setOptions(self, **kwargs):
        options = {
            'colors':str,
            'floworder':str,
            'textcolor':str,
            'figbounds':tuple,
            'textposition':tuple,
            'numcolors':int,
            'comments':str,
        }
        for k,v in kwargs.iteritems():
            if options.has_key(k) and isinstance(v, options[k]):
                setattr(self, k, v)
    def get_color(self,nuc):
        color_dic = {}
        for n,c in enumerate(self.colors[:self.numcolors]):
            cur_nuc = self.floworder[n]
            color_dic[cur_nuc] = c
        return color_dic[nuc]
    def render(self):
        pass
    def _prepFigure(self):
        pylab.gcf().set_size_inches(self.figbounds)
        #pylab.figure(figsize=(8,4))
        #pylab.legend(loc='upper right')
        #pylab.suptitle(self.title)
    def urlsave(self):
        enc = base64.standard_b64encode(self.binarysave())
        return 'data:image/png;base64,' + enc
    def binarysave(self):
        outbuf = StringIO.StringIO()
        pylab.savefig(outbuf, format='png')
        text = outbuf.getvalue()
        outbuf.close()
        return text
    def filesave(self, fname):
        pylab.savefig(fname)
        return fname
    def _renderArgs(self, kwargs):
        if len(self.comments) > 0:
            pylab.figtext(self.textposition[0], self.textposition[1], self.comments,
                horizontalalignment='center',
                fontproperties=pylab.matplotlib.font_manager.FontProperties(
                    family='monospace', size='x-small'))
        retOptions = {
            'url': lambda fname: self.urlsave(),
            'file': lambda fname: self.filesave(fname),
            'binary': lambda fname: self.binarysave()
        }
        options = (
            ('show', lambda x: x == 'yes' and pylab.show()),
            ('xlabel', lambda x: pylab.xlabel(str(x))),
            ('ylabel', lambda x: pylab.ylabel(str(x))),
            ('ret', lambda x: retOptions.has_key(x) and retOptions[x](str(kwargs.get('fname', None))))
        )
        ret = None
        for k,v in options:
            if kwargs.has_key(k):
                ret = v(kwargs[k])
                pylab.ioff()
        return ret

class Ionogram(Plotter):
    def __init__(self, flowlabels, expected, heights, title=None):
        Plotter.__init__(self)
        self.labels = flowlabels
        self.numflows = len(flowlabels)
        self.expected = expected
        self.heights = heights
        self.title = str(title or 'Ionogram')
        self.scale_y = True
        self.use_text = True
        #self.setOptions(numcolors=self.numflows)
        self._correctLabels()
    def _correctLabels(self):
        if isinstance(self.labels, str):
            arr = [char for char in self.labels]
            self.labels = arr
        lblLen = len(self.labels)
        expLen = len(self.expected)
        hLen = len(self.heights)
        if (hLen == expLen and lblLen < expLen):
            lbls = []
            for i in range(0, expLen/lblLen):
                lbls.extend(self.labels)
            lbls.extend(self.labels[:expLen%lblLen])
            self.labels = lbls
    def getLabelScale(self, lst):
        return float(4*(max(lst) - min(lst)))/35.0
    def _plotPoint(self, ndx, height, clr, alpha, toLabel):
        pylab.bar([ndx], [max(0.015,height)],
                width = 0.3,
                yerr=None,
                ecolor=clr,
                edgecolor=clr,
                color=clr,
                #label=toLabel,
                align="center")
    def render(self, **kwargs):
        def prepSubplot(plotIndex):
            pylab.axhline(0,color='k')
            max_y = max(self.expected) + 1
            if len(self.heights) < plotXLen:
                max_x = min([len(self.heights), plotXLen]) + 0.5
            else:
                max_x = plotXLen*(plotIndex + 1) + 0.5
            xdiff = plotIndex*plotXLen
            axisargs = None
            if self.scale_y:
                for y in pylab.arange(0.5, max_y, 1.0):
                    pylab.axhline(y=y, ls=":", color='k')
                axisargs = [-1.0 + xdiff, max_x, -0.70*yscale, max_y]
                pylab.yticks(pylab.arange(0, max_y + 1))
                pylab.xticks(pylab.arange(0 + xdiff, max_x, 20))
            pylab.xlabel('Flow')
            pylab.ylabel('Bases')
            return axisargs
        pylab.close()
        labelScale = self.getLabelScale(self.heights)
        quads = zip(self.labels, self.expected, self.heights, range(0, len(self.heights)))
        plotXLen = 100
        nplots = max([(len(quads)-1),0])/plotXLen + 1
        yscale = float(max(self.expected))/5.0
        axisargs = None
        for (lbl, exp, height, ndx) in quads:
            if ndx % plotXLen == 0:
                if axisargs is not None:
                    pylab.axis(axisargs)
                plotIndex = ndx/plotXLen
                pylab.subplot(100*nplots + 10 + (plotIndex+1))
                axisargs = prepSubplot(plotIndex)
            cndx = self._nuc2ndx(ndx)
            clr = self._ndx2clr(cndx)
            toLabel = (ndx < self.numflows) and lbl or None
            self._plotPoint(ndx, height, clr, 0.6, toLabel)
            if self.use_text:
                pylab.text(ndx, -0.55*yscale, str(exp),
                        color='#666666',
                        fontsize=8,
                        verticalalignment="center",
                        horizontalalignment="center")
                pylab.text(ndx, -0.20*yscale,  lbl,
                        color=clr,
                        fontsize=8,
                        verticalalignment="center",
                        horizontalalignment="center")
        if axisargs is not None:
            pylab.axis(axisargs)
        pylab.suptitle(self.title)
        #pylab.gca().legend(loc='upper right')
        width,height = self.figbounds
        self.setOptions(figbounds=(width,height*nplots))
        self._prepFigure()
        return self._renderArgs(kwargs)

class Errorgram(Ionogram):
    def __init__(self, flowlabels, expected, heights, errors, title=None):
        Ionogram.__init__(self, flowlabels, expected, heights, title)
        self.title = str(title or 'Errorgram')
        self.errors = errors
        self.barwidth = float(len(expected))/95.0
        self.scale_y = False
        self.use_text = False
    def _plotPoint(self, ndx, height, clr, alpha, toLabel):

        pylab.errorbar([ndx], [height], yerr=self.errors[ndx],
                        fmt="o", c=clr, label=toLabel)
class SignalOverlap(Plotter):
    #KERNEL = signal.blackman(50)/sum(signal.blackman(50))
    KERNEL = signal.blackman(8)/sum(signal.blackman(8))
    def __init__(self, signalgroups, data=None, title=None, custom_labels=None):
        Plotter.__init__(self)
        self.data = data
        if (not self._checkSignalGroups(signalgroups)):
            raise DataFormatError('Dictionary of int->iterable')
        if (hasattr(signalgroups, 'items')
            and callable(getattr(signalgroups, 'items'))):
            self.signalgroups = signalgroups.items()
        else:
            self.signalgroups = signalgroups
        self.title = str(title or 'Signal Resolution')
        stdcolors = 'rygbkcm'
        self.setOptions(colors=stdcolors, numcolors=len(stdcolors))
        if max(map(lambda (a,b): len(b),self.signalgroups)) < 10:
            self.numbins = 15
        else:
            self.numbins = 100
        self.custom_labels = custom_labels
    @classmethod
    def _checkSignalGroups(cls,groups):
        if isinstance(groups, dict):
            items = groups.iteritems()
        else:
            items = groups
        try:
            ret = reduce(lambda acc, (k,v): acc and isinstance(k,int) and iter(v) and acc, items, True)
        except:
            raise
            return False
        return ret
    def remove_outliers(self, vals, iqr_factor=2.0):
        nvals = len(vals)
        if nvals < 5:
            return vals
        vals = list(vals)
        vals.sort()
        nq1 = nvals/4
        nq3 = (3*nvals)/4
        q1 = vals[nq1]
        q3 = vals[nq3]
        iqr = q3 - q1
        lower = q1 - iqr_factor*iqr
        upper = q3 + iqr_factor*iqr
        ret = []
        for v in vals[:nq1]:
            if v > lower:
                ret.append(v)
        ret.extend(vals[nq1:nq3])
        for v in vals[nq3:]:
            if v < upper:
                ret.append(v)
        return ret
    def remove_outliers2(self, vals, binval):
        ret = []
        for v in vals:
            if (v <= (binval+2)):
                ret.append(v)
        return ret

    def render(self, **kwargs):
        pylab.close()
        #pylab.figure(figsize=(8,4))
        extremum = lambda func, choose: reduce(lambda prev, curr: (func(curr,prev) and curr) or prev,
            [choose(lst) for k,lst in self.signalgroups if lst])
        minVal = extremum(lambda a,b: a < b, min)
        maxVal = extremum(lambda a,b: a >= b, max)
        bins = pylab.arange(minVal, maxVal, float(maxVal - minVal)/float(self.numbins))
        pylab.gca().yaxis.set_ticklabels([])
        max_plotted = []
        for group_ndx,(expected,vals) in enumerate(self.signalgroups):
            clr = self._ndx2clr(group_ndx)
            hist_kwargs = {
                'bins':self.numbins,
                'normed':True
            }
            if NUMPY_VERSION <= 121:
                hist_kwargs['new'] = True
            vals = self.remove_outliers2(vals, group_ndx)
            bins,edges = pylab.histogram(vals, **hist_kwargs)
            edges = edges[:-1]
            if self.custom_labels is not None:
                label = self.custom_labels[group_ndx]
            else:
                label = '%d-mers' % expected
            bins = signal.convolve(bins, self.KERNEL, mode='same')
            bins[0] = bins[-1] = 0.0
            max_plotted.append(max(bins))
            pylab.fill(edges, bins, alpha=.5, ec='none', fc=clr,
                        label=label) #width=width,
        self._prepFigure()
        pylab.legend(loc='upper right')
        pylab.suptitle(self.title)
        pylab.xlabel("Homopolymer Length")
        upper_limit = max(max_plotted)
        upper_limit *= 1.10
        pylab.ylim(0, upper_limit)
        maxBin = max(list(bins))
        return self._renderArgs(kwargs)

class Iontrace(Ionogram):
    def __init__(self, labels, expected, traces, timing=None, title=None):
        Plotter.__init__(self)
        self.labels = labels
        self.keylen = len(self.labels)
        self.expected = expected
        self.traces = traces
        self.timing = timing
        self.heights = [None for i in range(0, len(traces))]
        if not self.timing:
            self.timing = pylab.arange(0,
                reduce( lambda acc, curr: acc + curr,
                        [len(t) for k,t in self.traces.iteritems() if k in self.labels],
                        0.0),
                1.0)
        self.offsets = []
        curr = 0
        for k,tr in self.traces.iteritems():
            if k in self.labels:
                self.offsets.append(curr)
                curr += len(tr)
        self.title = title or 'Iontrace'
        self._correctLabels()
        self.numflows = len(traces)
        #self.setOptions(numcolor=self.numflows)

    def render(self, **kwargs):
        def prep_subplot(ndx,tot,ntraces):
            pylab.subplot(100*tot + 10 + ndx + 1)
            axes = pylab.gca()
            interval = (max([int(glob_max),100])/100)*10
            pylab.yticks(pylab.arange(0.0, glob_max*1.075, interval))
            axisframes = plotXLen*trLen
            rframe = axisframes*(ndx+1)
            lframe = axisframes*ndx
            lbound = self.timing[lframe]
            if rframe >= len(self.timing) and ndx > 0:
                missing = float(rframe - len(self.timing))
                rbound = self.timing[-1] + missing/float(self.timing[-1] - self.timing[lframe])
                trsLeft = ntraces - ndx*plotXLen
                rbound = lbound + plotXLen*int((rbound - lbound)/trsLeft)
            else:
                rbound = self.timing[min([rframe, len(self.timing) - 1])]
            axisargs = [lbound, rbound, -5.80*yspace, glob_max*1.075]
            xt_incr = (axisframes/4)*(float((rframe-lframe))/float(rbound-lbound))
            pylab.xticks(pylab.arange(lbound,rbound+xt_incr-1.0, xt_incr),
                    pylab.arange(ndx*plotXLen, (ndx+1)*plotXLen+1, 5))
            pylab.axhline(0, c='k')
            pylab.ylabel("Counts")
            pylab.xlabel("Flows")
            return axisargs
        pylab.close()
        if len(self.traces) != 0:
            quints = zip(self.labels, self.expected, self.offsets, range(0, self.numflows))
            fontsize = 9
            plotXLen = 20
            nplots = max([(len(quints)-1),0])/plotXLen + 1
            axisargs = None
            glob_max = None
            extrema = [max(tr) for k,tr in self.traces.iteritems()] + [min(tr) for k,tr in self.traces.iteritems()]    
            glob_max = float(max(extrema))
            yspace = glob_max/48.0
            for (lbl, exp, offset, ndx) in quints:
                nuc = self.labels[ndx]
                trace = self.traces[nuc]
                trLen = len(trace)
                toLabel = (ndx < self.keylen and lbl) or None
                if ndx%plotXLen == 0:
                    if axisargs is not None:
                        pylab.axis(axisargs)
                    plotIndex = ndx/plotXLen
                    axisargs = prep_subplot(plotIndex,nplots,len(quints))
                pylab.plot(pylab.array(self.timing[offset:offset+trLen]),
                        trace, color=self.get_color(nuc))
                xoff = (float(ndx) + 0.5)*trLen
                pylab.text(xoff, -2.0*yspace,
                        exp, fontsize=fontsize, color="#666666",
                        verticalalignment="center",
                        horizontalalignment="center")
                pylab.text(xoff, -4.70*yspace,
                        lbl, fontsize=fontsize, color=self.get_color(nuc),
                        verticalalignment="center",
                        horizontalalignment="center")
            if axisargs is not None:
                pylab.axis(axisargs)
            pylab.suptitle(self.title)
            width,height = self.figbounds
            self.setOptions(figbounds=(width,nplots*height))
            self._prepFigure()
            #pylab.legend(loc='upper right')
        return self._renderArgs(kwargs)

class SparseTracePlot(Plotter):
    NUC_CYCLE_RE = re.compile(r'(?P<nuc>\w+)(?P<cycle>\d+)$')
    def __init__(self, traceDict, expDict, floworder, timing=None, title=None, **kwargs):
        Plotter.__init__(self)
        self.title = title or "Trace Plot"
        if "tuples" in kwargs:
            tuples = kwargs.pop("tuples")
            try:
                for tr,nuc,cycle,ex in tuples:
                    pass
            except:
                raise ValueError, "Tuples keyword argument had invalid format."
            self.toplot = tuples
        else:
            #we have to acceptable formats for keys: "<nuc><cycle>" or <flow>
            toplot = []
            for k,v in traceDict.iteritems():
                toappend = self.parseNucCycleFmt(k,v,expDict,floworder)
                if toappend is None:
                    toplot = []
            toplot = self.parseFmt(traceDict,expDict,floworder,self.parseNucCycleFmt)
            if toplot is None:
                toplot = self.parseFmt(traceDict,expDict,floworder,self.parseFlowFmt)
            if toplot is None:
                raise ValueError, "Invalid trace dictionary key format."
            self.toplot = toplot
        if len(self.toplot) < 1 or len(self.toplot[0]) < 1:
            raise ValueError, "Must provide at least one non-zero length trace."
        self.tracelen = len(self.toplot[0][0])
        if not timing:
            self.timing = range(self.tracelen)
            self.xlabel = "Frames"
        else:
            self.timing = timing
            self.xlabel = "Seconds"
        if len(self.timing) != self.tracelen:
            raise ValueError, "Traces and timing must have identical length."
        #get some kwargs
        self.use_legend = bool(kwargs.pop('use_legend', True))
        self.ylabel = str(kwargs.pop('ylabel', 'Counts'))
    @classmethod
    def parseFmt(cls, traceDict, expDict, floworder, fmtFunc):
        ret = []
        for k,v in traceDict.iteritems():
            toappend = fmtFunc(k,v,expDict,floworder)
            if toappend is None:
                return None
            else:
                ret.append(toappend)
        return ret
    @classmethod
    def parseNucCycleFmt(cls, k, v, expDict, floworder):
        match = cls.NUC_CYCLE_RE.match(str(k))
        if match is None: return None
        else:
            d = match.groupdict()
            nuc = d['nuc']
            cycle = int(d['cycle'])
            expected = expDict[k]
            return (v, nuc, cycle, expected)
    @classmethod
    def parseFlowFmt(cls, k, v, expDict, floworder):
        if not isinstance(k,int):
            try:
                k = int(k)
            except:
                return None
        nflows = len(floworder)
        cycle = k/nflows
        nuc = floworder[k%nflows]
        try:
            expected = expDict[k]
        except:
            try:
                expected = expDict[str(k)]
            except:
                return None
        return (v, nuc, cycle, expected)
    def render(self, **kwargs):
        pylab.close()
        for tr,nuc,cycle,ex in self.toplot:
            pylab.plot(self.timing,tr, label="%s%d(%d)" % (nuc,cycle,ex))
        if self.use_legend:
            nflows = len(self.toplot)
            if nflows >= 15:
                fsize = 'xx-small'
            elif nflows >= 10:
                fsize = 'x-small'
            elif nflows >= 5:
                fsize = 'small'
            else:
                fsize = 'medium'
            font = pylab.mpl.font_manager.FontProperties(size=fsize)
            pylab.legend(prop=font)
        pylab.xlabel(self.xlabel)
        pylab.ylabel(self.ylabel)
        pylab.title(self.title)
        self._prepFigure()
        return self._renderArgs(kwargs)

class TracePlot(SparseTracePlot):
    def __init__(self, traces, expected, floworder, timing=None,title=None, **kwargs):
        self.title = title or "Trace Plot"
        self.floworder = floworder
        if len(expected) != len(traces):
            raise ValueError, "The number of traces provided must match the number of"\
                " expected values provided."
        if len(traces) < 1:
            raise ValueError, "The list of traces provided was empty."
        self.expected = expected
        expDict = {}
        traceDict = {}
        for tr,ex,ndx in zip(traces,expected,range(len(expected))):
            expDict[ndx] = ex
            traceDict[ndx] = tr
        SparseTracePlot.__init__(self, traceDict, expDict, floworder, timing, title, **kwargs)

class TransferPlot(SignalOverlap):
    def __init__(self, signalgroups, data=None,title=None, custom_labels=None):
        SignalOverlap.__init__(self, signalgroups, data,custom_labels=custom_labels)
        self.title = title or 'Average Signal'
        del self.numbins
    def render(self, **kwargs):
        pylab.close()
        pylab.figure(figsize=(8,4))
        try:
            sgi = self.signalgroups.items
            vals = self.signalgroups.values()
            keys = self.signalgroups.keys()
        except:
            sgi = self.signalgroups
            vals = map(lambda (a,b): b, self.signalgroups)
            keys = map(lambda (a,b): a, self.signalgroups)
        #toplot = [(pylab.average(v), pylab.std(v)) for v in vals]
        toplot = self.data
        for (mean,err),ndx,expected in zip(toplot, range(0, len(toplot)), keys):
            clr = self._ndx2clr(ndx)
            if self.custom_labels is not None:
                label = self.custom_labels[ndx]
            else:
                label = '%d-mer' % expected
            pylab.errorbar([expected], [mean], yerr=err, fmt="o"+clr,
                    label=label)
        minx = min(keys)
        maxx = max(keys)
        miny = min([m - d for (m,d) in toplot])
        maxy = max([m + d for (m,d) in toplot])
        pylab.legend(loc="lower right")
        pylab.legend(loc="lower right")
        pylab.title(self.title)
        pylab.xlabel("Homopolymer Length")
        pylab.ylabel("Normalized Signal")
        pylab.axis([minx - 0.5, maxx + .5, miny - abs(.125*miny), maxy + abs(.125*maxy)])
        return self._renderArgs(kwargs)

class MatchedBasesPlot(Plotter):
    def __init__(self, alignments, title=None):
        Plotter.__init__(self)
        self.title = title or 'Matched Bases'
        for ndx,ele in enumerate(alignments):
            if not isinstance(ele, int):
                raise TypeError, "Alignments must be integers."
            if ele < 0:
                alignments[ndx] = 0
        maxalignment = max(alignments)
        if maxalignment > 1500:
            raise ValueError, ("Implausibly large alignment score "
                    "(%d). Is your input correct?" % maxalignment)
        self.alignments = alignments
        self.xlabel = "Correct Bases Minus Incorrect Bases"
        self.alignments.sort()
    def render(self, **kwargs):
        pylab.close()
        if len(self.alignments) > 1:
            bins = [int(self.alignments[0] == self.alignments[1]) - 1]
            self.alignments.insert(0,0)
            prev = self.alignments[0]
            curr = self.alignments[0]
            for one,two in zip(self.alignments[:-1], self.alignments[1:]):
                for i in range(two-one):
                    bins.append(0)
                bins[-1] += 1
        else:
            bins = []
        if len(bins) > 100:
            ec = 'none'
        else:
            ec = None
        pylab.bar(map(lambda a: a - 0.5, range(len(bins))), bins, width=1, fc='b', ec=ec, alpha='0.8')
        pylab.axis([-1, len(bins), 0, max(bins) + 2])
        pylab.title(self.title)
        pylab.xlabel(self.xlabel)
        self._prepFigure()
        return self._renderArgs(kwargs)

class QPlot(Plotter):
    QCOLORS = {
        10: 'r',
        20: 'g',
        17: 'y',
    }
    def __init__(self, alignments, q, title=None, format='lengths',
                 expected=None, floworder=None):
        q = int(q)
        self.q = q
        if format == 'lengths':
            self.alignments = map(int,alignments)
        elif format == 'twoline':
            self.alignments = []
            for a in alignments:
                converted = self.convert_two_line(a)
                rl = int(self.read_length(converted, self.accuracy(q)))
                self.alignments.append(rl)
        else:
            raise ValueError, "Unknown format: %s." % str(format)
        for ndx,a in enumerate(self.alignments):
            if a < 0:
                self.alignments[ndx] = 0
        Plotter.__init__(self)
        self.title = title or "Q%d Read Lengths" % q
        self.floworder = floworder
        if expected is not None:
            try:
                testint = expected[0] + 1
            except:
                expected = Template.deseqify(expected, self.floworder)
            hporder = []
            nflows = len(self.floworder)
            for ndx,n in enumerate(expected):
                nuc = self.floworder[ndx%nflows]
                if n > 0:
                    hporder.append((n,nuc))
            self.hporder = hporder
            self.maxlen = sum(expected)
        else:
            self.hporder = None
            self.maxlen = max(self.alignments)
    def convert_two_line(self,arrs):
        mistakes = []
        ndx = 0
        for top,bot in zip(arrs[0],arrs[2]):
            if top == '-':
                mistakes.append(ndx)
            elif bot == '-':
                mistakes.append(ndx)
                ndx += 1
            else:
                ndx += 1
        ret = [len(arrs[0]) - arrs[0].count('-')] + mistakes
        return ret
    def accuracy(self, q=10):
        return (1.0/(10.0**(float(q)/10.0)))
    def render(self, **kwargs):
        pylab.close()
        small_font = pylab.mpl.font_manager.FontProperties(size='xx-small')
        nbins = max(self.alignments) + 1
        if nbins < 100:
            ec = None
        else:
            ec = 'none'
        fc = self.QCOLORS.get(self.q, 'b')
        if self.hporder is not None:
            a2 = []
            for a in self.alignments:
                if a > self.maxlen:
                    a = self.maxlen
                a2.append(a)
            self.alignments = a2
        lengths = [0 for i in range(self.maxlen + 1)]
        for a in self.alignments:
            lengths[a] += 1
        pylab.bar(left=range(self.maxlen + 1), height=lengths, fc=fc,ec=ec,
                alpha='0.8', width=1.0)
        if self.hporder is not None:
            nucs_in_key = set()
            ymin,ymax = pylab.ylim()
            xmax = self.maxlen + max([float(self.maxlen)/20.0, 2.0])
            diff = float(ymax - ymin)
            unit = diff/20.0
            pylab.axhline(c='k')
            left = 1
            bary = unit/2
            bottom = bary
            for n,nuc in self.hporder:
                if nuc not in nucs_in_key:
                    label = nuc
                    nucs_in_key.add(nuc)
                else:
                    label = None
                nuc_ndx = self.floworder.index(nuc)
                c = self._ndx2clr(nuc_ndx)
                pylab.bar(left=left, bottom=-bary, height=unit, width=n,
                        orientation='horizontal', align='center',
                        ec='none', fc=c, label=label)
                for i in range(n):
                    pylab.text(i + left + 0.5, -bary, nuc,
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontproperties=small_font)
                left += n
            pylab.bar(left=0, bottom=-bary, height=unit, width=1,
                    orientation='horizontal', align='center', ec='none',
                    fc='k')
            pylab.bar(left=left, bottom=-bary, height=unit, width=xmax - left,
                    orientation='horizontal', align='center', ec='none',
                    fc='k')
            pylab.plot([self.maxlen + 1, self.maxlen + 1], [0, ymax], 'k--')
            #pylab.legend(loc='upper right', prop=small_font)
            pylab.ylim(ymin=-unit)
            pylab.xlim(xmin=0, xmax=xmax)
            xincr = max([1, int(self.maxlen/5.0)])
            xincr += 5 - (xincr % 5)
            pylab.xticks(pylab.arange(0.5, self.maxlen + 0.6, xincr),
                         range(0, self.maxlen + 1, xincr))
        pylab.title(self.title)
        pylab.xlabel("Read Length")
        self._prepFigure()
        return self._renderArgs(kwargs)

class ChipPlot(Plotter):
    def __init__(self, mask, title=None, **kwargs):
        Plotter.__init__(self)
        self.setOptions(figbounds=(7,7))
        nrows = len(mask)
        ncols = None
        for r in mask:
            if ncols is None:
                ncols = len(r)
            if len(r) != ncols:
                raise ValueError, "Mask has rows of differing lengths."
        self.mask = mask
        self.nrows = nrows
        self.ncols = ncols
        self.title = title or "Chip Heatmap"
        self.r0 = int(kwargs.pop('r0',0))
        self.c0 = int(kwargs.pop('c0',0))
        self.no_cbar = bool(kwargs.pop('disallow_cbar', False))
    def generic_render(self, im, kwargs, cbar=False, interp='nearest'):
        pylab.close()
        if self.ncols > 200:
            interp=None
        pylab.imshow(im, origin='lower',
                extent=(self.c0,
                        self.c0 + len(self.mask[0]),
                        self.r0,
                        self.r0 + len(self.mask)),
                interpolation=interp,
                )
        pylab.title(self.title)
        if cbar and not self.no_cbar:
            pylab.colorbar()
        return self._renderArgs(kwargs)
    def check_pil(self):
        if not HAVE_PIL:
            raise ImportError, ("PIL (Python Image Library) not installed. To render images in"
                "4-color mode, please install this package.")
    def render_4color(self, **kwargs):
        self.check_pil()
        colors = (
            (0,0,0), #black
            (255,0,0), #red
            (0,255,0), #green
            (0,0,255), #blue
            (255,255,0)) #yellow
        buf = pylab.zeros((self.nrows, self.ncols, 3), dtype='uint8')
        for rndx,r in enumerate(self.mask):
            for cndx,c in enumerate(r):
                buf[rndx,cndx] = colors[int(c)]
        interp='nearest'
        return self.generic_render(buf, kwargs, cbar=False, interp=interp)
    def render_rgb(self, **kwargs):
        self.check_pil()
        buf = pylab.zeros((self.ncols, self.nrows, 3), dtype='uint8')
        for rndx,r in enumerate(self.mask):
            for cndx,c in enumerate(r):
                buf[cndx,rndx,:] = c
        return self.generic_render(buf, kwargs)
    def render_fp(self, **kwargs):
        return self.generic_render(self.mask, kwargs, True)
    def select_mode(self):
        preds = (
            (self.render_rgb, lambda ele: (hasattr(ele, '__len__') and len(ele) == 3
                and reduce(lambda acc,x: acc and x >= 0 and x <= 255, ele, True))),
            (self.render_4color, lambda ele: (isinstance(ele,float) or isinstance(ele,int))
                    and ele == int(ele) and ele >= 0 and ele <= 4),
            (self.render_fp, float),
            (self.render_fp, int),
        )
        def checkpreds(ele):
            def make_pred(typ):
                def ret(x):
                    try:
                        test = typ(x)
                        truth = True
                    except:
                        truth = False
                    return True
                return ret
            currmode = None
            ret = []
            for fn,pred in preds:
                if isinstance(pred,type):
                    pred = make_pred(pred)
                if pred(ele):
                    ret.append((fn,pred))
            return ret
        modes = None
        mode = None
        pred = None
        def raise_pred_err(c):
            raise ValueError, "No predicate satisfied by %s of type '%s'."\
                    % (repr(c), str(type(c)))
        for r in self.mask:
            for c in r:
                if modes is None:
                    modes = checkpreds(c)
                    if not modes:
                        raise_pred_err(c)
                    mode,pred = modes[0]
                while not pred(c):
                    if not modes:
                        raise_pred_err(c)
                    mode,pred = modes.pop(0)
        return mode
    def render(self, **kwargs):
        return self.select_mode()(**kwargs)

class BeadfindHist(Plotter):
    def __init__(self, empty,weak,bead,clusters=None,title=None):
        def list_nocopy(l):
            if not isinstance(l,list):
                return list(l)
            return l
        Plotter.__init__(self)
        self.title = title or "Beadfind Histogram"
        self.empty = list_nocopy(empty)
        self.weak = list_nocopy(weak)
        self.bead = list_nocopy(bead)
        self.clusters = clusters
    def render(self, **kwargs):
        pylab.close()
        all = self.empty + self.weak + self.bead
        hist_kwargs = {}
        if NUMPY_VERSION <= 121:
            hist_kwargs['new'] = True
        bins,edges = pylab.histogram(all, len(all)/50, **hist_kwargs)

        types = [self.empty, self.weak, self.bead]
        lengths = map(len, types)
        indices = []
        for arr in types:
            arr.sort()
            indices.append(0)
        vals = []
        switches = []
        prevcurrtype = None
        maxlen = sum(map(len, types))
        maxval = max(map(max, types))
        getval = lambda a,n: (n < len(a) and a[n]) or maxval + 1
        while sum(indices) < maxlen:
            currvals = [getval(arr,ndx) for arr,ndx in zip(types,indices)]
            currtype = pylab.argmin(currvals)
            currval = currvals[currtype]
            if prevcurrtype != currtype:
                switches.append((currtype,currval))
            prevcurrtype = currtype
            currindex = indices[currtype]
            if currindex < len(types[currtype]):
                vals.append(types[currtype][currindex])
            indices[currtype] += 1
        names = ["Empty", "Ignored", "Bead"]
        switch_indices = []
        switch_index = 0
        for ndx,e in enumerate(edges):
            typ,val = switches[switch_index]
            #print "VAL:",val," E:", e
            if e >= val:
                switch_indices.append((ndx,typ))
                if switch_index < len(switches) - 1:
                    switch_index += 1
        labels_seen = set()
        make_label = lambda ndx: "%s (%d)" % (names[ndx], lengths[ndx])
        width = math.ceil(float(max(edges) - min(edges)) / float(len(edges)))
        width *= 1.15
        for one,two in zip(switch_indices[:-1], switch_indices[1:]):
            left,ltyp = one
            right,rtyp = two
            clr = self._ndx2clr(ltyp)
            lbl = make_label(ltyp)
            if lbl in labels_seen:
                lbl = None
            else:
                labels_seen.add(lbl)
            pylab.bar(edges[left:right], bins[left:right],
                    label=lbl, ec='none',fc=clr, width=width)
        if right < len(edges):
            lbl = make_label(rtyp)
            if lbl in labels_seen:
                lbl = None
            clr = self._ndx2clr(rtyp)
            pylab.bar(edges[right:len(bins)], bins[right:len(edges)], label=lbl,
                    ec='none', fc=clr, width=width)
        maxy = max(bins)
        maxy += maxy/10.0
        xspread = max(bins) - min(bins)
        yunit = maxy/15.0
        small_font = pylab.mpl.font_manager.FontProperties(size='x-small')
        if self.clusters is not None:
            pylab.axhline(c='k')
            for name,val in self.clusters:
                pylab.plot([float(val),float(val)], [0,float(maxy)], c='k', linestyle=':')
                pylab.text(val, -yunit/2.0, name,
                           horizontalalignment='center',
                           verticalalignment='center',
                           fontproperties=small_font)
            pylab.ylim(ymin=-yunit, ymax=maxy)
        pylab.legend(prop=small_font)
        pylab.title(self.title)
        pylab.xlabel("Count-Seconds")
        self._prepFigure()
        return self._renderArgs(kwargs)

class PerFlowErrorPlot(Plotter):
    def __init__(self, top, bottom, hits, spec, **kwargs):
        self.top = pylab.array(top)
        self.bottom = pylab.array(bottom)
        self.hits = hits
        self.spec = spec
        # self.spec = float(kwargs.get("spec", 0.98))
        self.floworder = kwargs.get("floworder", "TACG")

        colordict = {'T':"#ff6666", 'A':"#66b366", 'C':"#6666ff", 'G':"#666666"}
        self.colors = []
        for flow in list(self.floworder): self.colors.append(colordict[flow])

        # truncate the longer one to the length of the shorter list
        self.top = self.top[:len(self.bottom)]
        self.bottom = self.bottom[:len(self.top)]

        self.simple = bool(kwargs.get("simple", False))

        Plotter.__init__(self)

    def render(self, **kwargs):
        X = len(self.top)
        x = range(X)
        revx = list(x)
        revx.reverse()
        colors = self.colors
        hits = self.hits
        top = self.top
        bottom = self.bottom

        pylab.close()

        if self.simple: # simple line plot of error rates
            pylab.plot(self.top-self.bottom, "b", alpha=0.75, lw=2, label="Correct")
            pylab.plot(-1.0*self.top+1.0, 'y', alpha=0.4, lw=2, label="Overcalls")
            pylab.plot(self.bottom, 'c', lw=2, alpha=0.4, label="Undercalls")
            pylab.axhline(self.spec, lw=1.5, ls=":", alpha=0.8, color="k", label="Spec")
            pylab.axhline(0.0, alpha=0.15, color="k")

            for x in xrange(0, X, 4):
                pylab.axvline(x, ymin=0.13, ymax=1, alpha=0.1, color="k")

            for index in xrange(X):
                color = colors[index % len(self.floworder)]
                pylab.plot([index], [(self.top-self.bottom)[index]], 'o', color=color, ms=3)
                # pylab.plot([index], [(-1.0*self.top+1.0)[index]], 'o', color=color, ms=3)
                # pylab.plot([index], [self.bottom[index]], 'o', color=color, ms=3)


        else: # nutty shaded stacked line graph of error rates
            # pylab.plot((self.bottom+self.top+self.spec)/2.0, 'k:', alpha=0.8,lw=1.5,label="Spec")
            # pylab.plot((self.bottom+self.top-self.spec)/2.0, 'k:', alpha=0.8,lw=1.5)

            fails = True
            meets = True

            for xx in xrange(X):
                label = None
                if top[xx]-bottom[xx] > self.spec:
                    color = colors[1]
                    if meets:
                        label = "Meets spec"
                        meets = False
                else:
                    color = colors[0]
                    if fails:
                        label = "Fails spec"
                        fails = False

                # left side
                if xx > 0:
                    l1 = pylab.average(bottom[xx-1:xx+1])
                    r1 = bottom[xx]
                    l2 = pylab.average(top[xx-1:xx+1]) # top should be reversed here
                    r2 = top[xx]
                    pylab.plot([xx-0.5,xx],[l1, r1], color=colors[xx%4], lw=2)
                    pylab.plot([xx-0.5,xx],[l2, r2], color=colors[xx%4], lw=2)
                    pylab.fill([xx-0.5,xx,xx,xx-0.5], [l1, r1, r2, l2], fc=color, ec=color, lw=1, label=label)
                    label = None

                # right side
                if xx < X-1:
                    l1 = bottom[xx]
                    r1 = pylab.average(bottom[xx:xx+2])
                    l2 = top[xx]
                    r2 = pylab.average(top[xx:xx+2]) # top should be reversed here?
                    pylab.plot([xx,xx+0.5],[l1, r1], color=colors[xx%4], lw=2)
                    pylab.plot([xx,xx+0.5],[l2, r2], color=colors[xx%4], lw=2)
                    pylab.fill([xx,xx+0.5,xx+0.5,xx], [l1,r1,r2,l2], fc=color, ec=color, lw=1, label=label)

                for y in xrange(hits[xx]): pylab.plot([xx],[0.02*y],'o',ms=3,color=colors[xx%4])

            revtop = list(top)
            revtop.reverse()
            pylab.fill(x + revx, [1]*X + revtop, 'y', alpha=0.2, label="Overcalls") # top should be reversed here
            pylab.fill(x + revx, list(bottom) + [0]*X, 'c', alpha=0.2, label="Undercalls")

        for xx in xrange(X):
            pylab.text(xx-0.4, -0.05, "TACG"[xx%4], color=colors[xx%4], fontsize=7)
            pylab.text(xx-0.4, -0.1, str(self.hits[xx]), color='k', alpha=0.8, fontsize=7)

        pylab.axis([-0.5,X-0.5,-0.15,1])
        pylab.xticks(pylab.arange(0,X,16))
        pylab.xlabel("Flow")
        pylab.ylabel("Fraction")
        pylab.title("Per-Flow Error Rates")
        pylab.legend(loc="center left", prop=pylab.mpl.font_manager.FontProperties(size="xx-small"))

        self._prepFigure()
        return self._renderArgs(kwargs)

class HomopolymerErrorPlot(Plotter):
    def __init__(self, correct, counts=None, specs=None, title=None):
        Plotter.__init__(self)
        for k,v in correct.iteritems():
            if k not in counts:
                raise KeyError, "No count found for HP length %d." % int(k)
        self.correct = correct
        self.counts = counts
        if specs is None:
            self.specs = {
                0: 0.98,
                1: 0.98,
                2: 0.95,
                3: 0.90,
                4: 0.85,
                5: 0.80,
                6: 0.72,
                7: 0.6,
                8: 0.45
            }
            self.default_spec = 0.45
        else:
            self.specs = specs
        self.title = title or "HP Error Plot"
    def render(self, **kwargs):
        pylab.close()
        items = self.correct.items()
        items.sort()
        lefts = [a for a,b in items]
        heights = [float(b)/float(self.counts[a]) for a,b in items]
        small_font = pylab.mpl.font_manager.FontProperties(size='x-small')

            #if i + 1 < len(lefts):
            #    pylab.plot((endx, endx), (spec, self.specs[i+1]), c=color,
            #            ls=style, alpha=alpha)

        for l,h in zip(lefts,heights):
            pylab.bar(l, h, 0.35, color='r', align='center', label=None,
                    ec='none')
            text_height = -0.055

        xmax = max([max(lefts) + 1])
        pylab.axhline(c='k')
        pylab.axis([-1, xmax, -0.1, 1.075])
        pylab.yticks(pylab.arange(0,1.01,0.2),
                ["%d%%" % int(100*i) for i in pylab.arange(0,1.01,0.2)])
        pylab.xticks(pylab.arange(0, xmax, 1.0),
                ["%d-mers" % int(i) for i in pylab.arange(0,xmax,1.0)],
                fontproperties=small_font)
        pylab.ylabel("% Correct")
        pylab.xlabel("Homopolymer Lengths")
        pylab.title(self.title)
        self._prepFigure()
        return self._renderArgs(kwargs)

class RminPlot(Plotter):
    def __init__(self, dateValuePairs, windowSize, title=None, allowOutliers=True):
        Plotter.__init__(self)
        self.allowout = bool(allowOutliers)
        if windowSize < 4:
            raise ValueError, 'RminPlotter parameter "windowSize" must be greater than 4.'
        if (not isinstance(dateValuePairs, dict)):
            dateValuePairs = dict(dateValuePairs)
        if not self.checkDvPairs(dateValuePairs):
            errstr = 'DateValuePairs must be a list or dict of (datetime.datetime,float) pairs.'
            raise DataFormatError(errstr)
        self.dvpairs = dateValuePairs
        if windowSize >= len(dateValuePairs):
            errstr = 'a windowSize less than the number of date value pairs.'
            raise DataFormatError(errstr)
        self.window = windowSize
        if title: self.title = title
        else: self.title = 'Metric Tracking'
    @classmethod
    def getTAlpha(cls, probability, sampleSize):
        lowstart = -5
        highstart = 5
        tolerance = 1e-6
        maxiters = 100
        getMid = lambda: float(highstart + lowstart)/2.0
        test = lambda: probability - stats.t.cdf(mid, sampleSize)
        for i in range(0,maxiters):
            mid = getMid()
            diff = test()
            #print mid, lowstart, highstart, diff
            if (abs(diff) < tolerance):
                return mid
            if (diff >= 0.0):
                lowstart = mid
            else:
                highstart = mid
        raise RuntimeError, "Control should not reach here!"
    @classmethod
    def checkDvPairs(cls, pairs):
        for d,v in pairs.iteritems():
            floatable = True
            try: testfloat = float(v)
            except: floatable = False
            if not isinstance(d,datetime) or not floatable:
                return False
        return True

    def render(self, **kwargs):
        pylab.close()
        self._prepFigure()
        talpha = self.getTAlpha(.95, self.window)
        halfSize = int(self.window / 2)
        toAdd = self.window % 2
        dvpairs = self.dvpairs.items()
        dvpairs.sort()
        areNormal = []
        outwindow = self.window + (((self.window % 2) + 1) % 2)
        for i in range(0, outwindow/2):
            areNormal.append(True)
        for ndx in range(outwindow, len(dvpairs)):
            arr = dvpairs[ndx-outwindow:ndx]
            vals = pylab.array([ele[1] for ele in arr])
            y = vals[outwindow/2]
            vals.sort()
            size = len(vals)
            q1,q3 = vals[size/4], vals[3*size/4]
            iqr = q3-q1
            lowerlimit, upperlimit = q1 - 1.5*iqr, q3 + 1.5*iqr
            areNormal.append(bool(y <= upperlimit and y >= lowerlimit))
        for i in range(0, outwindow/2):
            areNormal.append(True)

        haveOutliers = not reduce(lambda truth,x: x and truth, areNormal, True)
        timing = []
        rmins = []
        rmaxs = []
        noOutrmins = []
        noOutrmaxs = []
        noOutTiming = []
        def addWithOutliers(vals, timeslice):
            mean = pylab.mean(vals)
            rdiff = talpha*pylab.std(vals)
            rmins.append(mean - rdiff)
            rmaxs.append(mean + rdiff)
            timing.append(timeslice[len(timeslice)/2])

        def addWithoutOutliers(vals, timeslice, norms):
            vs = []
            ts = []
            time = timeslice[len(timeslice)/2]
            for val,t,n in zip(vals, timeslice, norms):
                if n:
                    vs.append(val)
                    ts.append(t)
            if len(vs) > 0:
                y = pylab.mean(vs)
                rdiff = talpha*pylab.std(vs)
                noOutrmins.append(y - rdiff)
                noOutrmaxs.append(y + rdiff)
                noOutTiming.append(time)
        for ndx in range(self.window, len(dvpairs)):
            arr = dvpairs[ndx-self.window:ndx]
            norms = areNormal[ndx-self.window:ndx]
            vals = pylab.array([ele[1] for ele in arr])
            timeslice = pylab.array([ele[0] for ele in arr])
            if haveOutliers:
                addWithOutliers(vals,timeslice)
            addWithoutOutliers(vals,timeslice,norms)
        normx,normy,outx,outy = [],[],[],[]
        for truth,(date,val) in zip(areNormal, dvpairs):
            if truth:
                normx.append(date)
                normy.append(val)
            else:
                outx.append(date)
                outy.append(val)

        pylab.plot(normx, normy, '+', c=self._ndx2clr(0), ms=10,
                label=('Normal Values' if haveOutliers else 'Values'))
        if haveOutliers and self.allowout:
            pylab.plot(outx, outy, 'o', c=self._ndx2clr(1), ms=5, label='Outliers')
            pylab.plot(timing, rmins, self._ndx2clr(2), alpha='.33')
            pylab.plot(timing, rmaxs, self._ndx2clr(3), alpha='.33')
        pylab.plot(noOutTiming, noOutrmins, self._ndx2clr(2))
        pylab.plot(noOutTiming, noOutrmaxs, self._ndx2clr(3))
        pylab.legend(loc="upper left")
        ax = pylab.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
        pylab.xlabel('Date')
        pylab.ylabel('Value of Metric')
        pylab.title(self.title)
        return self._renderArgs(kwargs)
class LiveDudHist(Plotter):
    def __init__(self, live,dud,clusters=None,title=None):
        def list_nocopy(l):
            if not isinstance(l,list):
                return list(l)
            return l
        Plotter.__init__(self)
        self.title = title or "Live_Dud Histogram"
        self.dud = list_nocopy(dud)
        self.live = list_nocopy(live)
        self.clusters = clusters
    def render(self, **kwargs):
        pylab.close()
        all = self.live + self.dud
        hist_kwargs = {}
        if NUMPY_VERSION <= 121:
            hist_kwargs['new'] = True
        bins,edges = pylab.histogram(all, len(all)/50, **hist_kwargs)

        types = [self.dud, self.live]
        lengths = map(len, types)
        indices = []
        for arr in types:
            arr.sort()
            indices.append(0)
        vals = []
        switches = []
        prevcurrtype = None
        maxlen = sum(map(len, types))
        maxval = max(map(max, types))
        getval = lambda a,n: (n < len(a) and a[n]) or maxval + 1
        while sum(indices) < maxlen:
            currvals = [getval(arr,ndx) for arr,ndx in zip(types,indices)]
            currtype = pylab.argmin(currvals)
            currval = currvals[currtype]
            if prevcurrtype != currtype:
                switches.append((currtype,currval))
            prevcurrtype = currtype
            currindex = indices[currtype]
            if currindex < len(types[currtype]):
                vals.append(types[currtype][currindex])
            indices[currtype] += 1
        names = ["Dud", "Live"]
        switch_indices = []
        switch_index = 0
        for ndx,e in enumerate(edges):
            typ,val = switches[switch_index]
            #print "VAL:",val," E:", e
            if e >= val:
                switch_indices.append((ndx,typ))
                if switch_index < len(switches) - 1:
                    switch_index += 1
        labels_seen = set()
        make_label = lambda ndx: "%s (%d)" % (names[ndx], lengths[ndx])
        width = math.ceil(float(max(edges) - min(edges)) / float(len(edges)))
        width *= 1.15
        for one,two in zip(switch_indices[:-1], switch_indices[1:]):
            left,ltyp = one
            right,rtyp = two
            clr = self._ndx2clr(ltyp)
            lbl = make_label(ltyp)
            if lbl in labels_seen:
                lbl = None
            else:
                labels_seen.add(lbl)
            pylab.bar(edges[left:right], bins[left:right],
                    label=lbl, ec='none',fc=clr, width=width)
        if right < len(edges):
            lbl = make_label(rtyp)
            if lbl in labels_seen:
                lbl = None
                clr = self._ndx2clr(rtyp)
            pylab.bar(edges[right:len(bins)], bins[right:len(edges)], label=lbl,
                          ec='none', fc=clr, width=width)
        maxy = max(bins)
        maxy += maxy/10.0
        xspread = max(bins) - min(bins)
        yunit = maxy/15.0
        small_font = pylab.mpl.font_manager.FontProperties(size='x-small')
        if self.clusters is not None:
            pylab.axhline(c='k')
            for name,val in self.clusters:
                pylab.plot([float(val),float(val)], [0,float(maxy)], c='k', linestyle=':')
                pylab.text(val, -yunit/2.0, name,
                           horizontalalignment='center',
                           verticalalignment='center',
                           fontproperties=small_font)
            pylab.ylim(ymin=-yunit, ymax=maxy)
        pylab.legend(prop=small_font)
        pylab.title(self.title)
        pylab.xlabel("Count-Seconds")
        self._prepFigure()
        return self._renderArgs(kwargs)

class IonogramJMR(Plotter):
    def __init__(self, flowlabels, expected, heights, title=None):
        Plotter.__init__(self)
        self.labels = flowlabels
        self.floworder = flowlabels
        self.numflows = len(flowlabels)
        self.expected = expected
        self.heights = heights
        self.title = str(title or 'Ionogram')
        self.scale_y = True
        self.use_text = True
        #self.setOptions(numcolors=self.numflows)
        self._correctLabels()
    def _correctLabels(self):
        if isinstance(self.labels, str):
            arr = [char for char in self.labels]
            self.labels = arr
        lblLen = len(self.labels)
        expLen = len(self.expected)
        hLen = len(self.heights)
        if (hLen == expLen and lblLen < expLen):
            lbls = []
            for i in range(0, expLen/lblLen):
                lbls.extend(self.labels)
            lbls.extend(self.labels[:expLen%lblLen])
            self.labels = lbls
    def getLabelScale(self, lst):
        return float(4*(max(lst) - min(lst)))/35.0
    def _plotPoint(self, ndx, height, clr, alpha, toLabel):
        pylab.bar([ndx], [max(0.015,height)],
                width = 0.3,
                yerr=None,
                ecolor=clr,
                edgecolor=clr,
                color=clr,
                label=toLabel,
                align="center")
        #pylab.legend(lbls)
    def render(self, **kwargs):
        def prepSubplot(plotIndex):
            pylab.axhline(0,color='k')
            max_y = max(self.expected) + 3
            if len(self.heights) < plotXLen:
                max_x = min([len(self.heights), plotXLen]) + 0.5
            else:
                max_x = plotXLen*(plotIndex + 1) + 0.5
            xdiff = plotIndex*plotXLen
            axisargs = None
            if self.scale_y:
                for y in pylab.arange(0.5, max_y, 1.0):
                    pylab.axhline(y=y, ls=":", color='k')
                axisargs = [-1.0 + xdiff, max_x, -0.70*yscale, max_y]
                pylab.yticks(pylab.arange(0, max_y + 4))
                pylab.xticks(pylab.arange(0 + xdiff, max_x, 20))
            pylab.xlabel('Flow')
            pylab.ylabel('Bases')
            return axisargs
        pylab.close()
        labelScale = self.getLabelScale(self.heights)
        quads = zip(self.labels, self.expected, self.heights, range(0, len(self.heights)))
        plotXLen = 100
        nplots = max([(len(quads)-1),0])/plotXLen + 1
        yscale = float(max(self.expected))/5.0
        axisargs = None
        for (lbl, exp, height, ndx) in quads:
            if ndx % plotXLen == 0:
                if axisargs is not None:
                    pylab.axis(axisargs)
                plotIndex = ndx/plotXLen
                pylab.subplot(100*nplots + 10 + (plotIndex+1))
                axisargs = prepSubplot(plotIndex)
            cndx = self._nuc2ndx(ndx)
            clr = self._ndx2clr(cndx)
            toLabel = (ndx < self.numflows) and lbl or None
            self._plotPoint(ndx, height, clr, 0.6, toLabel)
            if self.use_text:
                #pylab.text(ndx, -0.55*yscale, str(exp),
                #       color='#666666',
                #        fontsize=8,
                #        verticalalignment="center",
                #        horizontalalignment="center")
                pylab.text(ndx, -0.5*yscale,  lbl,
                        color=clr,
                        fontsize=8,
                        verticalalignment="center",
                        horizontalalignment="center")
        if axisargs is not None:
            pylab.axis(axisargs)
        pylab.suptitle(self.title)
        #pylab.legend(toLabel)
        width,height = self.figbounds
        self.setOptions(figbounds=(width,height*nplots))
        self._prepFigure()
        return self._renderArgs(kwargs)

class IonogramPretty(Plotter):
    def __init__(self, flowlabels, expected, heights, title=None):
        Plotter.__init__(self)
        self.labels = flowlabels
        self.floworder = flowlabels
        self.numflows = len(flowlabels)
        self.expected = expected
        self.heights = heights
        self.title = str(title or 'Ionogram')
        self.scale_y = True
        self.use_text = True
        #self.setOptions(numcolors=self.numflows)
        self._correctLabels()
    def _correctLabels(self):
        if isinstance(self.labels, str):
            arr = [char for char in self.labels]
            self.labels = arr
        lblLen = len(self.labels)
        expLen = len(self.expected)
        hLen = len(self.heights)
        if (hLen == expLen and lblLen < expLen):
            lbls = []
            for i in range(0, expLen/lblLen):
                lbls.extend(self.labels)
            lbls.extend(self.labels[:expLen%lblLen])
            self.labels = lbls
    def getLabelScale(self, lst):
        return float(4*(max(lst) - min(lst)))/35.0
    def _plotPoint(self, ndx, height, clr, alpha, toLabel):
        pylab.bar([ndx], [max(0.015,height)],
                width = 0.3,
                yerr=None,
                ecolor=clr,
                edgecolor=clr,
                color=clr,
                label=toLabel,
                align="center")
        #pylab.legend(lbls)
    def render(self, **kwargs):
        def prepSubplot(plotIndex):
            pylab.axhline(0,color='k')
            max_y = max(self.expected) + 1
            if len(self.heights) < plotXLen:
                max_x = min([len(self.heights), plotXLen]) + 0.5
            else:
                max_x = plotXLen*(plotIndex + 1) + 0.5
            xdiff = plotIndex*plotXLen
            axisargs = None
            if self.scale_y:
                for y in pylab.arange(1.0, max_y, 1.0):
                    pylab.axhline(y=y, ls=":", color='k')
                axisargs = [-1.0 + xdiff, max_x, -0.70*yscale, max_y]
                pylab.yticks(pylab.arange(0, max_y + 4))
                pylab.xticks(pylab.arange(0 + xdiff, max_x, 20))
            pylab.xlabel('Flow')
            pylab.ylabel('Bases')
            return axisargs
        pylab.close()
        labelScale = self.getLabelScale(self.heights)
        quads = zip(self.labels, self.expected, self.heights, range(0, len(self.heights)))
        plotXLen = 200
        nplots = max([(len(quads)-1),0])/plotXLen + 1
        yscale = float(max(self.expected))/5.0
        axisargs = None
        for (lbl, exp, height, ndx) in quads:
            if ndx % plotXLen == 0:
                if axisargs is not None:
                    pylab.axis(axisargs)
                plotIndex = ndx/plotXLen
                pylab.subplot(100*nplots + 10 + (plotIndex+1))
                axisargs = prepSubplot(plotIndex)
            cndx = self._nuc2ndx(ndx)
            clr = self._ndx2clr(cndx)
            toLabel = (ndx < self.numflows) and lbl or None
            self._plotPoint(ndx, height, clr, 0.6, toLabel)
            if self.use_text:
                #pylab.text(ndx, -0.55*yscale, str(exp),
                #       color='#666666',
                #        fontsize=8,
                #        verticalalignment="center",
                #        horizontalalignment="center")
                pylab.text(ndx, -0.3*yscale,  lbl,
                        color=clr,
                        fontsize=6,
                        verticalalignment="center",
                        horizontalalignment="center")
        if axisargs is not None:
            pylab.axis(axisargs)
        pylab.suptitle(self.title)
        #pylab.legend(toLabel)
        width,height = self.figbounds
        width = 12
        self.setOptions(figbounds=(width,height*nplots))
        self._prepFigure()
        return self._renderArgs(kwargs)

class QPlot2(Plotter):
    QCOLORS = {
        10: 'r',
        20: 'g',
        17: 'y',
    }
    def __init__(self, heights, q, title=None, expected=None, floworder='TACG'):
        q = int(q)
        self.q = q
        self.heights = heights
        Plotter.__init__(self)
        self.title = title or "AQ%d Read Lengths" % q
        self.floworder = floworder
        if expected is not None:
            try:
                testint = expected[0] + 1
            except:
                expected = Template.deseqify(expected, self.floworder)
            hporder = []
            nflows = len(self.floworder)
            for ndx,n in enumerate(expected):
                nuc = self.floworder[ndx%nflows]
                if n > 0:
                    hporder.append((n,nuc))
            self.hporder = hporder
        self.maxlen = len(self.heights)
    def render(self, **kwargs):
        pylab.close()
        small_font = pylab.mpl.font_manager.FontProperties(size='xx-small')
        fc = self.QCOLORS.get(self.q, 'b')
        pylab.bar(left=range(len(self.heights)), height=self.heights, fc=fc, alpha='0.8', width=1.0)
        if self.hporder is not None:
            nucs_in_key = set()
            ymin,ymax = pylab.ylim()
            xmax = self.maxlen + max([float(self.maxlen)/20.0, 2.0])
            diff = float(ymax - ymin)
            unit = diff/20.0
            pylab.axhline(c='k')
            left = 1
            bary = unit/2
            bottom = bary
            for n,nuc in self.hporder:
                if nuc not in nucs_in_key:
                    label = nuc
                    nucs_in_key.add(nuc)
                else:
                    label = None
                nuc_ndx = self.floworder.index(nuc)
                c = self._ndx2clr(nuc_ndx)
                pylab.bar(left=left, bottom=-bary, height=unit, width=n,
                        orientation='horizontal', align='center',
                        ec='none', fc=c, label=label)
                for i in range(n):
                    pylab.text(i + left + 0.5, -bary, nuc,
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontproperties=small_font)
                left += n
            pylab.bar(left=0, bottom=-bary, height=unit, width=1,
                    orientation='horizontal', align='center', ec='none',
                    fc='k')
            pylab.bar(left=left, bottom=-bary, height=unit, width=xmax - left,
                    orientation='horizontal', align='center', ec='none',
                    fc='k')
            pylab.plot([self.maxlen + 1, self.maxlen + 1], [0, ymax], 'k--')
            #pylab.legend(loc='upper right', prop=small_font)
            pylab.ylim(ymin=-unit)
            pylab.xlim(xmin=0, xmax=xmax)
            xincr = max([1, int(self.maxlen/5.0)])
            xincr += 5 - (xincr % 5)
            pylab.xticks(pylab.arange(0.5, self.maxlen + 0.6, xincr),
                         range(0, self.maxlen + 1, xincr))
        pylab.title(self.title)
        pylab.xlabel("Read Length")
        self._prepFigure()
        return self._renderArgs(kwargs)

class JBSignalOverlap(Plotter):
    KERNEL = signal.blackman(16)/sum(signal.blackman(16))
    def __init__(self, signalgroups, data, title=None, custom_labels=None):
        Plotter.__init__(self)
        self.data = data
        if (not self._checkSignalGroups(signalgroups)):
            raise DataFormatError('Dictionary of int->iterable')
        if (hasattr(signalgroups, 'items')
            and callable(getattr(signalgroups, 'items'))):
            self.signalgroups = signalgroups.items()
        else:
            self.signalgroups = signalgroups
        self.title = str(title or 'Signal Overlap')
        stdcolors = 'rygbkcm'
        self.setOptions(colors=stdcolors, numcolors=len(stdcolors))
        if max(map(lambda (a,b): len(b),self.signalgroups)) < 10:
            self.numbins = 15
        else:
            self.numbins = 100
        self.custom_labels = custom_labels
    @classmethod
    def _checkSignalGroups(cls,groups):
        if isinstance(groups, dict):
            items = groups.iteritems()
        else:
            items = groups
        try:
            ret = reduce(lambda acc, (k,v): acc and isinstance(k,int) and iter(v) and acc, items, True)
        except:
            raise
            return False
        return ret
    def remove_outliers(self, vals, iqr_factor=2.0):
        nvals = len(vals)
        if nvals < 5:
            return vals
        vals = list(vals)
        vals.sort()
        nq1 = nvals/4
        nq3 = (3*nvals)/4
        q1 = vals[nq1]
        q3 = vals[nq3]
        iqr = q3 - q1
        lower = q1 - iqr_factor*iqr
        upper = q3 + iqr_factor*iqr
        ret = []
        for v in vals[:nq1]:
            if v > lower:
                ret.append(v)
        ret.extend(vals[nq1:nq3])
        for v in vals[nq3:]:
            if v < upper:
                ret.append(v)
        return ret
    def render(self, **kwargs):
        pylab.close()
        #extremum = lambda func, choose: reduce(lambda prev, curr: (func(curr,prev) and curr) or prev,
            #[choose(lst) for k,lst in self.signalgroups if lst])
        #minVal = extremum(lambda a,b: a < b, min)
        #maxVal = extremum(lambda a,b: a >= b, max)
        minVal = 0
        maxVal = 1
        #bins = pylab.arange(minVal, maxVal, float(maxVal - minVal)/float(self.numbins))
        pylab.gca().yaxis.set_ticklabels([])
        max_plotted = []
        for d in self.data:
            max_plotted.append(max(d))
            xval = [i for i in range(len(d))]
            pylab.bar(xval, d)
        self._prepFigure()
        pylab.legend(loc='upper right')
        pylab.suptitle(self.title)
        pylab.xlabel("Count-Seconds")
        upper_limit = max(max_plotted)
        upper_limit *= 1.10
        pylab.ylim(0, upper_limit)
        #maxBin = max(list(bins))
        return self._renderArgs(kwargs)
