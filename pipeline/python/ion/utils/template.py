# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Utilities for working with DNA template strings.
"""

# from ion.analysis import cafie


def seqify(flowgram, floworder):
    """
    Turns the flowgram into a string of bases.
    @param flowgram: an iterable container of integer flow values
    @param floworder: the flow order
    @return: a string representing the call
    """
    nflows = len(floworder)
    ret = []
    for ndx, ext in enumerate(flowgram):
        ret.append(floworder[ndx % nflows]*ext)
    return ''.join(ret)


def deseqify(seq, floworder):
    """
    Turns the string of bases into a flowgram.
    @param seq: the call
    @param floworder: the flow order
    @return: the flowgram as a list of integers
    """
    nflows = len(floworder)
    if not nflows:
        raise ValueError, "Empty floworder."
    ret = []
    nucIndex = 0
    nuc = floworder[nucIndex]
    seen = 0
    for base in seq:
        if base == nuc:
            seen += 1
        else:
            ret.append(seen)
            for i in range(nflows):
                nucIndex = (nucIndex + 1) % nflows
                nuc = floworder[nucIndex]
                if nuc == base:
                    break
                else:
                    ret.append(0)
            if nuc != base:
                raise ValueError, ("Invalid base: %s in template '%s'."
                                   % (base, seq))
            seen = 1
    ret.append(seen)
    while len(ret) % nflows:
        ret.append(0)
    return ret


def scale_fit(expected, observed, **kwargs):
    """
    @TODO: jhoon
    """
    if len(expected) != len(observed):
        raise ValueError, "Length of expected not equal to length of observed."
    if len(expected) == 0:
        raise ValueError, "Cannot scale 0-length lists."
    if sum(expected) == 0:
        return 0, 0.0
    phi = kwargs.pop('phi', None)
    sq = lambda x: x*x
    top = 0.0
    bot = 0.0
    for pair in zip(expected, observed):
        pair = map(float, pair)
        if phi is not None:
            e, o = map(phi, pair)
        else:
            e, o = pair
        top += 2.0*e*o
        bot += 2.0*e*e
    s = top/bot
    residuals = map(lambda (a, b): a - s*b, zip(observed, expected))
    r2 = 1.0 - reduce(lambda acc, x: acc + sq(x), residuals, 0.0)/sum(map(sq, observed))
    return s, r2


def match_flowgram_to_template(fgram, templates, floworder, allow_lib=False):
    """
    @TODO: jhoon
    """
    targets = map(lambda tmpl: deseqify(tmpl, floworder), templates)
    max_length = max(map(len, targets + [fgram]))
    adjusted_targets = []
    make_extension = lambda t: map(int, '0'*(max_length - len(t)))
    for t in targets:
        t.extend(make_extension(t))
        adjusted_targets.append(t)
    fgram.extend(make_extension(fgram))
    return match_flowgram_to_flowgram(fgram, adjusted_targets, allow_lib)


def match_flowgram_to_flowgram(fgram, targets, allow_lib=False, lib_cut=0.70):
    """
    @TODO: jhoon
    """
    best = None
    bestFit = 0.0
    lenfg = len(fgram)
    for ndx, targ in enumerate(targets):
        scale, fit = scale_fit(targ[:len(fgram)]
                               + [0 for i in range(max([0, lenfg - len(targ)]))],
                               fgram)
        if best is None or fit > bestFit:
            best = ndx
            bestFit = fit
    if allow_lib and bestFit < lib_cut:
        return None
    return best


def key_nmer_flows(nmer, expected, floworder,
                   nkeyflows=None, nuc=None, as_mask=False):
    """
    @TODO: jhoon
    """
    ret = []
    forder_len = len(floworder)
    if nkeyflows is not None:
        key_expected = expected[:nkeyflows]
    else:
        key_expected = expected
    for ndx, exp in enumerate(key_expected):
        current_nuc = floworder[ndx % forder_len]
        if exp == nmer and (nuc is None or nuc == current_nuc):
            ret.append(ndx)
    if as_mask:
        mask = [int(i in ret) for i in range(len(expected))]
        return mask
    return ret


def key_nmer_flows_from_seq(nmer, seq, floworder, **kwargs):
    """
    @TODO: jhoon
    """
    expected = deseqify(seq, floworder)
    return key_nmer_flows(nmer, expected, floworder, **kwargs)


def match_to_keys(raw_fgram, keys, floworder):
    """
    @TODO: jhoon
    """
    return match_to_key_fgrams(raw_fgram, [deseqify(k, floworder) for k in keys])

import random


def match_to_key_fgrams(raw_fgram, key_fgrams):
    """
    @TODO: jhoon
    """
    from scipy import stats
    pvals = []
    for ndx, kf in enumerate(key_fgrams):
        populations = [[], []]
        for val, expected in zip(raw_fgram, kf):
            populations[int(expected)].append(val)
        t, pval2side = stats.ttest_ind(*populations)
        pvals.append((pval2side/2, ndx))
    pvals.sort()
    return pvals[0][-1]


def match_to_key_fgrams_paired(raw_fgram, key_fgrams, floworder):
    """
    @TODO: jhoon
    """
    from scipy import stats
    pvals = []
    key_pairings = []
    n_nucs = len(floworder)
    for keyindex, kf in enumerate(key_fgrams):
        pairs = {}
        for nuc in floworder:
            pairs[nuc] = [None, None]
        for ndx, expected in enumerate(kf):
            nuc = floworder[ndx % n_nucs]
            pairs[nuc][int(expected)] = ndx
        topop = [k for k, v in pairs.iteritems() if None in v]
        for k in topop:
            pairs.pop(k)
        diffs = []
        for k, v in pairs.iteritems():
            # random.shuffle(v)
            diffs.append(raw_fgram[v[1]] - raw_fgram[v[0]])
        t, pval2side = stats.ttest_1samp(diffs, 0.0)
        pval2side /= 2
        pvals.append((pval2side, keyindex))
    pvals.sort()
    return pvals[0][-1]


def grid_estimate_cafie(expected, observed, floworder, onemersOnly=False, ie_range=None,
                        cf_range=None, droop_range=None):
    """
    Estimates CAFIE parameters from a given flowgram.

    @param expected: the expected sequence, as bases or a flowgram
    @param observed: the observed flowgram
    @param floworder: the flow order
    @param ie_range: a list of IE values to search
    @param cf_range: a list of CF values to search
    @param droop_range: a list of droop values to search

    @return: a tuple of the best-fit parameters
    """
    from scipy import optimize
    import numpy, time
    start = time.time()
    try:
        # test to see if we got a string
        teststr = expected[0] + 'a'
    except TypeError:
        # we probably got a flowgram, so we convert to string
        expected = seqify(expected, floworder)
    observed = numpy.array(observed)
    expectedFlowgram = numpy.array(deseqify(expected, floworder)[:len(observed)])

    if onemersOnly: observed *= (expectedFlowgram <= 1)

    if ie_range is None:
        ie_range = numpy.arange(0.0, 0.03, 0.001)
    if cf_range is None:
        cf_range = numpy.arange(0.0, 0.05, 0.002)
    if droop_range is None:
        droop_range = numpy.arange(0.0, 0.01, 0.0002)
    bestError = None
    bestCf = None
    bestIe = None
    bestDroop = None
    scale_l2 = lambda x: observed - (x*predicted)
    for droop in droop_range:
        droopFactors = [(1.0 - droop)**c for c in xrange(len(observed))]
        for cf in cf_range:
            for ie in ie_range:
                # predicted = numpy.array(cafie.simulate(expected, len(observed),
                #                                      floworder, cf, ie))
                if onemersOnly: predicted *= (expectedFlowgram <= 1)
                predicted *= droopFactors
                scale_factor, success = optimize.leastsq(scale_l2, 1.0,
                                                         warning=False)
                error = numpy.sum(numpy.square(scale_l2(scale_factor)))
                if bestError is None or error < bestError:
                    bestError = error
                    bestCf = cf
                    bestIe = ie
                    bestDroop = droop
    print "\t(cafie gridsearch took %.2f sec)" % (time.time() - start)
    print "best cf,ie,dr:", bestCf, bestIe, bestDroop
    return bestCf, bestIe, bestDroop


def shorten_to_n_cycles(seq, floworder, n):
    nflows = len(floworder)
    return seqify(deseqify(seq, floworder)[:nflows*n], floworder)
