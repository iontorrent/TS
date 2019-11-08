# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import torrentPyLib


def bamReader(fname):
    bam = torrentPyLib.BamReader()
    bam.Open(fname)
    # nrec = bam.GetNumRecords()
    hdr = bam.ReadBamHeader()
    dat = bam.ReadBam()
    return (hdr, dat)


def wellsReader(fname):
    w = torrentPyLib.WellsReader()
    w.Open(fname)
    a = w.LoadWells(0, 0, 60000, 60000)
    return a


def bfMaskReader(path, mask):
    bf = torrentPyLib.LoadBfMaskByType(path, mask)
    return bf


def rawDat(path):
    i = torrentPyLib.RawDatReader()
    i.Open(path)
    a = i.LoadSlice(0, 0, 60000, 60000)
    return a
