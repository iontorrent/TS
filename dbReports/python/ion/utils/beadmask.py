# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
This module contains classes for loading and working with bead masks.
"""

import csv
import numpy
import pylab
import StringIO


class BeadMask(object):
    """
    Parent class for bead mask manipulation and loading.
    """
    def load(self, infile, **kwargs):
        infile, cansave = self.preload(infile)
        self._load(infile, **kwargs)
        if cansave:
            infile.close()
        self._check_load()
    def _load(self, infile, **kwargs):
        raise AttributeError, "Beadmask.load must be overriden."
    @classmethod
    def preload(cls, infile):
        if hasattr(infile, 'read'):
            return infile, False
        else:
            infile = open(infile, 'r')
            return infile, True
    def shape(self):
        return (self.r1 - self.r0, self.c1 - self.c0)
    def freeze(self):
        table = []
        for r in self.rows:
            table.append(frozenset(r))
        self.rows = table
    def as_grid(self):
        ret = numpy.zeros(self.shape(), dtype='uint8')
        for rndx,r in enumerate(self.rows):
            for ele in r:
                ret[rndx,ele - self.c0] = 1
        return ret
    def as_pairs(self,rel=False):
        ret = []
        for rndx,r in enumerate(self.rows):
            for c in r:
                if rel:
                    pair = (rndx, c - self.c0)
                else:
                    pair = (rndx + self.r0, c) 
                ret.append(pair)
        return ret
    def _check_load(self):
        must_have = ['r0', 'r1', 'c0', 'c1', 'rows', 'count']
        for mh in must_have:
            if not hasattr(self,mh):
                raise AttributeError, "Required attribute '%s' not found." % mh
    
class CsvBeadMask(BeadMask):
    """
    @TODO: jhoon
    """
    def __init__(self, infile, threshold=0):
        self.load(infile, threshold=threshold)
    def _load(self, infile, threshold):
        rdr = csv.reader(infile)
        self.count, self.rows, coords = self.load_from_csv(rdr, threshold)
        self.r0 = coords['r0']
        self.r1 = coords['r1']
        self.c0 = coords['c0']
        self.c1 = coords['c1']
    @classmethod
    def load_from_csv(cls, rdr, threshold=0):
        raw_rows = [r for r in rdr]
        rrlen = len(raw_rows)
        if rrlen == 0:
            raise ValueError, "No data found in file."
        for row in raw_rows[:2]:
            for ele in row[2:]:
                if len(ele) > 0:
                    raise ValueError, (("Unexpected value '%s' " % ele)
                            + "in first two rows outside first two columns.")
        x,y = [int(float(ele)) for ele in raw_rows[0][:2]]
        width,height = [int(float(ele)) for ele in raw_rows[1][:2]]
        expected_height = height + 2
        if rrlen != expected_height:
            raise ValueError, ("Height mismatch: Expected %d but got %d"
                    % (rrlen, expected_height))
        table = [set() for i in range(height)]
        count = 0
        for row,rowarr in zip(raw_rows[2:], table):
            if width != len(row):
                raise ValueError, ("Width mismatch: Expected %d but got %d"
                        % (width, len(row)))
            for ndx,col in enumerate(row):
                col = int(col)
                if col > threshold:
                    rowarr.add(ndx + x)
                    count += 1
        coords = {
            'r0':y,
            'r1':y + height,
            'c0':x,
            'c1':x + width
        }
        return count, table, coords
    def __contains__(self, *args):
        if len(args) == 1:
            r,c = args[0]
        elif len(args) == 2:
            r,c = args
        else:
            raise ValueError, "Invalid number of arguments: %d > 2" % len(args)
        if r >= r1 or r < r0 or c >= c1 or c < c0:
            return False
        r -= self.r0
        c -= self.c0
        return c in self.rows[r]
    def __getitem__(self, *args):
        return int(self.__contains__(*args))
    
class PairsBeadMask(BeadMask):
    """
    @TODO: jhoon
    """
    def __init__(self, infile, r0, r1, c0, c1):
        self.r0 = r0
        self.r1 = r1
        self.c0 = c0
        self.c1 = c1
        self.load(infile, r0=r0, r1=r1, c0=c0, c1=c1)
    def _load(self, infile, r0, r1, c0, c1):
        self.count, self.rows = self.load_from_pylab(infile, r0, r1)
    @classmethod
    def load_from_pylab(cls, pl, r0, r1):
        import pylab
        raw_beads = pylab.load(pl)
        span = r1 - r0
        table = [set() for i in range(span)]
        for r,c in raw_beads:
            table[int(r)-r0].add(int(c))
        return len(raw_beads), table
        
