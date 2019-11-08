# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Utilities for working with time series data (traces).
"""

import numpy

_RISE_TIME = "Rise Time"
_SLOPE = "Slope"
_RMS_AMPLITUDE = "RMS Amplitude"

TRACE_STAT_NAMES = [_RISE_TIME, _SLOPE, _RMS_AMPLITUDE]


def trace_stats(trace, startfrac, endfrac, lbound=None, rbound=None, as_dict=False):
    """
    @TODO: jhoon
    """
    if lbound is not None:
        trace = trace[lbound:]
    if rbound is not None:
        trace = trace[:rbound]
    trace = numpy.array(trace)
    peak = max(trace)
    peakIndex = numpy.argmax(trace)
    upper = endfrac * peak
    lower = startfrac * peak
    beginIndex = 0
    trlen = len(trace)
    while trace[beginIndex] < lower and beginIndex < trlen:
        beginIndex += 1
    if lbound is not None:
        beginIndex += lbound
    endIndex = peakIndex
    while trace[endIndex] > upper and endIndex >= 0:
        endIndex -= 1
    if lbound is not None:
        endIndex += lbound
    if endIndex > beginIndex:
        slope = numpy.polyfit(
            numpy.arange(beginIndex, endIndex), trace[beginIndex:endIndex], 1
        )[0]
        riseTime = endIndex - beginIndex
    else:
        slope = None
        riseTime = None
    rmsAmpl = numpy.sqrt(numpy.average(numpy.square(1000.0 * trace / sum(trace))))
    if as_dict:
        return {_RISE_TIME: riseTime, _SLOPE: slope, _RMS_AMPLITUDE: rmsAmpl}
    else:
        return riseTime, slope, rmsAmpl
