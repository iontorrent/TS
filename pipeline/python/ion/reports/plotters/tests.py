# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import datetime
from os import path
import os
import unittest

from plotters import *

from ion.analysis import sigproc

IMAGES = []
N_TESTS = 0
TEST_IMAGE_DIR = "./test_images"

def chkdir():
    if not path.isdir(TEST_IMAGE_DIR):
        os.mkdir(TEST_IMAGE_DIR)

def makes_gallery(fn):
    def ret(*args, **kwargs):
        chkdir()
        plots = fn(*args,**kwargs)
        pairs = []
        for ndx,p in enumerate(plots):
            fname = path.join(TEST_IMAGE_DIR, fn.im_func.func_name + ('_%d.png' % ndx))
            p.render(ret='file', fname=fname)
            pairs.append(fname)
        IMAGES.append((fn.im_func.func_name,pairs))
    return ret
    
def write_gallery():
    chkdir()
    tojoin = []
    for name,imgs in IMAGES:
        tojoin.append("<br/><hr/><div><h3>%s</h3>" % name)
        for fname in imgs:
            tojoin.append("<img src=%s alt='A Test Image'/><br/>" % (fname))
        tojoin.append("</div>")
    text = ("<html><head><title>Plotter Test Gallery</title></head><body style='font-family:tahoma;'>" +
            "<div style='text-align:center;'><h2>Gallery Of Plotter Test Output</h2><br/>" +
            ("<p>Ran %d tests.</p><br/><p>%s</p></div>" % (N_TESTS, str(datetime.now()))) +
            ("%s</body></html>" % '\n'.join(tojoin)))
    outfile = open("gallery.html", 'w')
    outfile.write(text)
    outfile.close()

def gallerify(cls):
    attrs = dir(cls)
    to_modify = []
    for attr in attrs:
        val = getattr(cls,attr)
        if callable(val) and attr[:4] == 'test':
            to_modify.append((attr,val))
    for k,v in to_modify:
        fn = makes_gallery(v)
        fn.func_name = k
        setattr(cls,k,fn)
    return cls
  
  
class PlotterTestFixture(unittest.TestCase):
    def setUp(self):
        global N_TESTS
        N_TESTS += 1
        pylab.clf()
    def tearDown(self):
        pass
    def test_Iontrace(self):
        traceLen = 100
        numTraces = 45
        traces = [
            [float(i + 3)*math.exp(-(((j - traceLen/2)/float(traceLen/4))**2)) for j in range(0, traceLen)] \
            for i in range(0, numTraces)        
        ]
        labels = 'TACG'
        expected = [i for i in range(0, numTraces)]
        it = Iontrace(labels, expected, traces, None, 'Custom Iontrace')
        return [it]
    def test_TransferPlot(self):
        numpoints = 10000
        numgroups = 6
        height = 25.0
        spread = 25.0
        stdDev = 25.0
        groups = {}
        ret = []
        for i in range(0, numgroups):
            groups[i] = [random.gauss(float(i)*spread, 25.0) 
                    for j in range(0, int(numpoints*random.random()))]
        so = SignalOverlap(groups, "Custom Overlap")
        ret.append(so)
        tf = TransferPlot(groups, "Custom Transfer Function")
        ret.append(tf)
        return ret
    def test_signal_overlap_custom_labels(self):
        labels = ['One', 'Two', 'Three', 'Four']
        groups = [
            (j, [random.gauss(float(j), 1.0)
                    for i in range(int(1000*random.random()))])
                for j in range(len(labels))
        ]
        so = SignalOverlap(groups, "Custom Labeled Signal Overlap",
                custom_labels=labels)
        tf = TransferPlot(groups, "Custom Labeled Transfer Function",
                custom_labels=labels)
        return [so,tf]
    def test_Ionogram(self):
        labels = 'TACG'
        maxExpected = 5
        numflows = 217
        scale = 1.0
        expected = [random.randint(0, maxExpected) for i in range(0, numflows)]
        heights = [scale*(exp + random.gauss(0, scale/10.0)) for exp in expected]
        io = Ionogram(labels,expected,heights,"Custom Ionogram")
        return [io]
    def test_RminPlot(self):
        numpairs = 100
        dates = [datetime.today() - timedelta(ndx) for ndx in range(0,numpairs)]
        dates.reverse()
        makeRandom = lambda ndx: (ndx**1.25)*math.sin(ndx/float(numpairs/10)) \
            + float(int(10.0*random.random())/9)*(800.0*random.random() - 400.0)
        values = [makeRandom(ndx) for ndx in range(0,numpairs)]
        dvps = zip(dates,values)
        rp = RminPlot(dvps, 8, 'Plot Title', True)
        return [rp]
    def test_Errorgram(self):
        expected = [int(random.random()*10) for i in range(0, 16)]
        observed = [float(ele)/10.0 * random.random()*30.0 + 70.0 for ele in expected]
        errors = [random.random()*3.0 + 1.0 for i in expected]
        e = Errorgram('TACG', expected, observed, errors, title='Custom Errorgram')
        return [e]
    def test_TracePlot(self):
        def maketrace(n, expected):
            ret = []
            expected = float(expected)
            slope = 0.0
            maxchange = float(n)/1000.0
            curr = 0.0
            for i in range(n):
                ret.append(curr)
                curr += (slope*expected)
                slope += maxchange*(2.0*(random.random() - 0.5))
            return ret
        LEN = 125
        N = 12
        exts = [int(10*random.random()) for i in range(N)]
        tp = TracePlot([maketrace(LEN,i) for i in exts[:4]], exts[:4],
                'TACG', None, "Test Title")
        longtp = TracePlot([maketrace(LEN/5,i) for i in exts], exts,
                'TACG', None, "Long Test")
        return [tp, longtp]
    def test_MatchedBasesPlot(self):
        mbp = MatchedBasesPlot(
            [int(random.gauss(300,10)) for i in range(1000)],
            'MBP')
        return [mbp]
    def test_QPlot(self):
        floworder = 'TACG'
        alignments = []
        naligns = 500
        readlen = 40
        quality = 17
        seq = ''.join([('TACG'[i%4])*random.choice(map(int, '000112345'))
                for i in range(readlen)])
        flows = Template.deseqify(seq, floworder)
        toalign = []
        for i in range(naligns):
            #nerrors = max([0, int(random.gauss(
            #        readlen * 10.0**(-0.75*quality/10), 3))])
            #error_indices = random.sample(range(len(flows)), nerrors)
            flows2 = []#list(flows)
            '''
            for ndx in error_indices:
                val = flows[ndx]
                if val == 0:
                    val = 1
                else:
                    val += random.choice([-1, 1])
                flows2[ndx] = val
            '''
            for f in flows:
                have_err = (random.random() / (f + 1)**2.8) < 0.0025
                if have_err:
                    flows2.append(f + 1)
                else:
                    flows2.append(f)
            toalign.append(Template.seqify(flows2, floworder))
        aresults = sigproc.align_all(toalign, seq, False)
        lengths = [ar.getQ17() for ar in aresults]
        qp = QPlot(lengths,quality,expected=seq,floworder=floworder,
                format='lengths', title='Test QPlot')
        return [qp]
    def test_BeadfindHist(self):
        GSIZE = 10000
        clusters = zip(["Empty", "Bead", "Ignored"], [0, 25.0, 50.0])
        all = sum([[random.gauss(cntr, 4.0) for i in range(GSIZE)] for name,cntr in clusters], [])
        all.sort()
        #random.shuffle(all)
        empty = all[:GSIZE]
        bead = all[GSIZE:2*GSIZE]
        weak = all[2*GSIZE:]
        bfh = BeadfindHist(empty,weak,bead, clusters, "Test Beadfind Histogram")
        return [bfh]
    def test_HPErrorPlot(self):
        correct_fractions = {
            0: .994,
            1: .96,
            2: .95,
            3: .85,
            4: .84,
            5: .72,
            6: .59,
            7: .61
        }
        correct, counts = {}, {}
        for k,v in correct_fractions.iteritems():
            N = random.randint(0,1000)
            correct[k] = N*v
            counts[k] = N
        hpep = HomopolymerErrorPlot(correct, counts, title="Test HP Error Plot")
        return [hpep]
    def test_ChipPlot(self):
        nrows = 50
        ncols = 50
        choices = [0,1,2,3,4]
        fourc = pylab.zeros((nrows,ncols), dtype='int32')
        for r in range(nrows):
            for c in range(ncols):
                fourc[r,c] = random.randint(1,4) if ((c+r) % 2) else 0
        cm = ChipPlot(fourc, "Test 4-color Chip Map",
                    r0=random.randint(200,300), c0=random.randint(300,500))
        floats = ChipPlot(1000.0*random.random() * fourc, "Test Float Map",
                    r0 = random.randint(200,300), c0=random.randint(1000,2000))
        rgb = pylab.zeros((nrows,ncols,3), dtype='uint8')
        for r in range(nrows):
            for c in range(ncols):
                rgb[r,c] = tuple([random.randint(0,255) for i in range(3)])
        rgbmap = ChipPlot(rgb, "Test RGB Map", r0=random.randint(100,200),
                          c0=random.randint(-200, -100))
        return [cm,floats,rgbmap]
    def test_PerFlowErrorPlot(self):
        ret = []
        for X in [20, 48, 100]:
            top = list(pylab.array([0.995**x for x in xrange(X)]) + pylab.random((X))/10.0 - 0.1)
            bottom = 1 - pylab.array(top) + pylab.random((X))/10.0
            hits = map(int, pylab.random((X))*5)
            ret.append(PerFlowErrorPlot(top, bottom, hits, simple=True, spec=0.7))
            ret.append(PerFlowErrorPlot(top, bottom, hits, simple=False, spec=0.7))
        return ret
 
PlotterTestFixture = gallerify(PlotterTestFixture)

if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
    write_gallery()
