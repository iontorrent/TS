#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import os
import json
import logging
import ConfigParser

from matplotlib import use
use("Agg", warn=False)
from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter, LinearLocator
from matplotlib import transforms

logger = logging.getLogger(__name__)


def load_ini(report, filename, namespace="global"):
    parse = ConfigParser.ConfigParser()
    path = os.path.join(report, filename)
    # TODO preseve the case
    try:
        parse.read(path)
        parse = parse._sections.copy()
        return parse[namespace]
    except Exception as err:
        logger.error("Wells Beadogram generation failed parsing %s: %s" %
                    (path, str(err)))
        raise


def load_json(report, filename):
    """shortcut to load the json"""
    path = os.path.join(report, filename)
    try:
        with open(path) as f:
            return json.loads(f.read())
    except Exception as err:
        logger.error("Wells Beadogram generation failed parsing %s: %s" %
                    (path, str(err)))
        raise


def generate_wells_beadogram2(basecaller, sigproc, beadogram_path=None):
    beadogram_path = beadogram_path or os.path.join(basecaller, "wells_beadogram.png")
    basecaller = load_json(basecaller, "BaseCaller.json")
    beadfind = load_ini(sigproc, "analysis.bfmask.stats")

    isp_labels, isp_counts = zip(*[
        ('Have ISPs', int(beadfind["bead wells"])),
        ('Live ISPs', int(beadfind["live beads"])),
        ('Library ISPs', int(beadfind["library beads"]))
    ])

    library_labels, library_counts = zip(*[
                                         ('Polyclonal',
                                          int(basecaller["Filtering"]["LibraryReport"]["filtered_polyclonal"])),
                                        ('Low Quality',
                                         int(basecaller["Filtering"]["LibraryReport"]["filtered_low_quality"])),
                                        ('Primer Dimer',
                                         int(basecaller["Filtering"]["LibraryReport"]["filtered_primer_dimer"])),
                                        ('Final Library',
                                         int(basecaller["Filtering"]["LibraryReport"]["final_library_reads"]))
                                         ])

    fig = pyplot.figure()
    wells_ax = fig.add_subplot(121)
    lib_ax = fig.add_subplot(122)

    if "adjusted addressable wells" in beadfind:
        available_wells = int(beadfind["adjusted addressable wells"])
    else:
        available_wells = int(beadfind["total wells"]) - int(beadfind["excluded wells"])

    suffixes = ('k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')

    def formatter(major, minor):
        base = 1000

        if major < base:
            return '%d' % major

        for i, s in enumerate(suffixes):
            unit = base ** (i+2)
            if major < unit:
                return '%.1f %s' % ((base * major / unit), s)

        return '%.1f %s' % ((base * major / unit), s)
    mils_format = FuncFormatter(formatter)

    wells_ax.bar(range(len(isp_counts)), isp_counts, width=0.5)
    wells_ax.set_ylim(0, available_wells)
    wells_ax.set_xticks(range(len(isp_counts)))
    wells_ax.set_xticklabels(isp_labels, rotation=20)
    wells_ax.yaxis.set_major_locator(LinearLocator(5))
    wells_ax.yaxis.set_major_formatter(mils_format)

    lib_ax.bar(range(len(library_counts)), library_counts,
               color=('r', 'r', 'r', 'g'), width=0.5)
    lib_ax.set_ylim(0, int(beadfind["library beads"]))
    lib_ax.set_xticks(range(len(library_counts)))
    lib_ax.set_xticklabels(library_labels, rotation=20)
    lib_ax.yaxis.set_major_locator(LinearLocator(5))
    lib_ax.yaxis.set_major_formatter(mils_format)

    wells_ax.set_ylabel("Number of ISPs")
    fig.suptitle("Ion Sphere Particle Summary")
    fig.subplots_adjust(wspace=0.3)
    fig.patch.set_alpha(0.0)
    pyplot.savefig(beadogram_path)


def generate_wells_beadogram(basecaller, sigproc, beadogram_path=None):
    beadogram_path = beadogram_path or os.path.join(basecaller, "wells_beadogram.png")
    generate_wells_beadogram_all_or_basic(basecaller, sigproc, beadogram_path, True)
    

def generate_wells_beadogram_all_or_basic(basecaller, sigproc, beadogram_path, is_full_details = True):
    basecaller = load_json(basecaller, "BaseCaller.json")
    beadfind = load_ini(sigproc, "analysis.bfmask.stats")

    def intWithCommas(x):
        if type(x) not in [type(0), type(0L)]:
            raise TypeError("Parameter must be an integer.")
        if x < 0:
            return '-' + intWithCommas(-x)
        result = ''
        while x >= 1000:
            x, r = divmod(x, 1000)
            result = ",%03d%s" % (r, result)
        return "%d%s" % (x, result)

    if "adjusted addressable wells" in beadfind:
        available_wells = int(beadfind["adjusted addressable wells"])
    else:
        available_wells = int(beadfind["total wells"]) - int(beadfind["excluded wells"])

    # Row 1: Loading
    loaded_wells = int(beadfind["bead wells"])
    empty_wells = available_wells - loaded_wells
    if available_wells > 0:
        loaded_percent = int(round(100.0 * float(loaded_wells) / float(available_wells)))
        empty_percent = 100 - loaded_percent
    else:
        loaded_percent = 0.0
        empty_percent = 0.0

    # Row 2: Enrichment
    enriched_wells = int(beadfind["live beads"])
    unenriched_wells = loaded_wells - enriched_wells
    if loaded_wells > 0:
        enriched_percent = int(round(100.0 * float(enriched_wells) / float(loaded_wells)))
        unenriched_percent = 100 - enriched_percent
    else:
        enriched_percent = 0.0
        unenriched_percent = 0.0

    # Row 3: Clonality
    polyclonal_wells = int(basecaller["Filtering"]["LibraryReport"]["filtered_polyclonal"])
    clonal_wells = enriched_wells - polyclonal_wells

    if enriched_wells > 0:
        clonal_percent = int(round(100.0 * float(clonal_wells) / float(enriched_wells)))
        polyclonal_percent = 100 - clonal_percent
    else:
        clonal_percent = 0.0
        polyclonal_percent = 0.0

    # Row 4: Filtering
    final_library_wells = int(basecaller["Filtering"]["LibraryReport"]["final_library_reads"])
    final_tf_wells = int(basecaller["Filtering"]["ReadDetails"]["tf"]["valid"])
    dimer_wells = int(basecaller["Filtering"]["LibraryReport"]["filtered_primer_dimer"])
    low_quality_wells = clonal_wells - final_library_wells - final_tf_wells - dimer_wells

    if not is_full_details:
        low_quality_wells += polyclonal_wells
        
    if clonal_wells > 0:
        final_library_percent = int(round(100.0 * float(final_library_wells) / float(clonal_wells)))
        final_tf_percent = int(round(100.0 * float(final_tf_wells) / float(clonal_wells)))
        dimer_percent = int(round(100.0 * float(dimer_wells) / float(clonal_wells)))
        low_quality_percent = 100 - final_library_percent - final_tf_percent - dimer_percent

        if not is_full_details:
            low_quality_percent = int(round(100.0 * float(low_quality_wells) / float(clonal_wells)))
    else:
        final_library_percent = 0.0
        final_tf_percent = 0.0
        dimer_percent = 0.0
        low_quality_percent = 0.0

    color_blue = "#2D4782"
    color_gray = "#808080"

    fontsize_big = 22
    fontsize_small = 12
    fontsize_medium = 16

    fig = pyplot.figure(figsize=(6, 4), dpi=100)

    #  "111" means "1x1 grid, first subplot"
    ax = fig.add_subplot(111, frame_on=False, xticks=[], yticks=[], position=[0, 0, 1, 1])


    # horizontal bar plot
    # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.barh
    # matplotlib.pyplot.barh(bottom, width, height=0.8, left=None, hold=None, **kwargs)
    # bottom : scalar or array-like; the y coordinate(s) of the bars
    # width : scalar or array-like; the width(s) of the bars
    # height : sequence of scalars, optional, default: 0.8; the heights of the bars
    # left : sequence of scalars; the x coordinates of the left sides of the bars
    
    if is_full_details:
        ax.barh(
            bottom=[3, 2, 1, 0],
            left=[0, 0, 0, 0],
            width=[loaded_wells/float(available_wells), enriched_wells/float(available_wells), clonal_wells/float(available_wells), final_library_wells/float(available_wells)],
            height=0.8, color=color_blue, linewidth=0, zorder=1)

        ax.barh(
            bottom=[3, 2, 1, 0],
            left=[loaded_wells/float(available_wells), enriched_wells/float(available_wells), clonal_wells/float(available_wells), final_library_wells/float(available_wells)],
            width=[empty_wells/float(available_wells), unenriched_wells/float(available_wells), polyclonal_wells/float(available_wells),
                   (final_tf_wells+dimer_wells+low_quality_wells)/float(available_wells)],
            height=0.8, color=color_gray, linewidth=0, zorder=1)
    else:  
        ax.barh(
            bottom=[3, 2, 1],
            left=[0, 0, 0],
            width=[loaded_wells/float(available_wells), enriched_wells/float(available_wells), final_library_wells/float(available_wells)],
            height=0.8, color=color_blue, linewidth=0, zorder=1)
    
        ax.barh(
            bottom=[3, 2, 1],
            left=[loaded_wells/float(available_wells), enriched_wells/float(available_wells), final_library_wells/float(available_wells)],
            width=[empty_wells/float(available_wells), unenriched_wells/float(available_wells), 
                   (final_tf_wells+dimer_wells+low_quality_wells)/float(available_wells)],
            height=0.8, color=color_gray, linewidth=0, zorder=1)
        

    # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.text
    # matplotlib.pyplot.text(x, y, s, fontdict=None, withdash=False, **kwargs)
    # x, y : scalars; data coordinates
    # s : string; text
    # fontdict : dictionary, optional, default: None; A dictionary to override the default text properties. If fontdict is None, the defaults are determined by your rc parameters.
    # withdash : boolean, optional, default: False; Creates a TextWithDash instance instead of a Text instance.
    
    ax.text(-0.21, 3.1, 'Loading', horizontalalignment='center', verticalalignment='center', fontsize=fontsize_small, zorder=2, color=color_blue, weight='bold', stretch='condensed')
    ax.text(-0.21, 3.4, ' %d%%' % loaded_percent, horizontalalignment='center', verticalalignment='center', fontsize=fontsize_big, zorder=2, color=color_blue, weight='bold', stretch='condensed')
    ax.text(1.21, 3.1, 'Empty Wells', horizontalalignment='center', verticalalignment='center',
            fontsize=fontsize_small, zorder=2, color=color_gray, weight='bold', stretch='condensed')
    ax.text(1.21, 3.4, ' %d%%' % empty_percent, horizontalalignment='center', verticalalignment='center', fontsize=fontsize_big, zorder=2, color=color_gray, weight='bold', stretch='condensed')
    ax.text(0.04, 3.4, intWithCommas(loaded_wells), horizontalalignment='left', verticalalignment='center',
            fontsize=fontsize_medium, zorder=2, color='white', weight='bold', stretch='condensed', alpha=0.7)

    ax.text(-0.21, 2.1, 'Enrichment', horizontalalignment='center', verticalalignment='center', fontsize=fontsize_small, zorder=2, color=color_blue, weight='bold', stretch='condensed')
    ax.text(-0.21, 2.4, ' %d%%' % enriched_percent, horizontalalignment='center', verticalalignment='center', fontsize=fontsize_big, zorder=2, color=color_blue, weight='bold', stretch='condensed')
    ax.text(max(0.6, 0.21+loaded_wells/float(available_wells)), 2.1,
            'No Template', horizontalalignment='center', verticalalignment='center', fontsize=fontsize_small, zorder=2, color=color_gray, weight='bold', stretch='condensed')
    ax.text(max(0.6, 0.21+loaded_wells/float(available_wells)), 2.4,
            ' %d%%' % unenriched_percent, horizontalalignment='center', verticalalignment='center', fontsize=fontsize_big, zorder=2, color=color_gray, weight='bold', stretch='condensed')
    ax.text(0.04, 2.4, intWithCommas(enriched_wells), horizontalalignment='left', verticalalignment='center',
            fontsize=fontsize_medium, zorder=2, color='white', weight='bold', stretch='condensed', alpha=0.7)
    ax.text(0.04, 2.4, intWithCommas(enriched_wells), horizontalalignment='left', verticalalignment='center',
            fontsize=fontsize_medium, zorder=0.5, color='black', weight='bold', stretch='condensed')

    bottom_bar_y1 = 0.1
    bottom_bar_y2 = 0.4
    bottom_text_y1 = 0.65
    bottom_text_y2 = 0.4
    bottom_text_y3 = 0.15
            
    if is_full_details:
        ax.text(-0.21, 1.1, 'Clonal', horizontalalignment='center', verticalalignment='center', fontsize=fontsize_small, zorder=2, color=color_blue, weight='bold', stretch='condensed')
        ax.text(-0.21, 1.4, ' %d%%' % clonal_percent, horizontalalignment='center', verticalalignment='center', fontsize=fontsize_big, zorder=2, color=color_blue, weight='bold', stretch='condensed')
        ax.text(max(0.6, 0.21+enriched_wells/float(available_wells)), 1.1,
                'Polyclonal', horizontalalignment='center', verticalalignment='center', fontsize=fontsize_small, zorder=2, color=color_gray, weight='bold', stretch='condensed')
        ax.text(max(0.6, 0.21+enriched_wells/float(available_wells)), 1.4,
                ' %d%%' % polyclonal_percent, horizontalalignment='center', verticalalignment='center', fontsize=fontsize_big, zorder=2, color=color_gray, weight='bold', stretch='condensed')

        ax.text(0.04, 1.4, intWithCommas(clonal_wells), horizontalalignment='left', verticalalignment='center',
                fontsize=fontsize_medium, zorder=2, color='white', weight='bold', stretch='condensed', alpha=0.7)
        ax.text(0.04, 1.4, intWithCommas(clonal_wells), horizontalalignment='left', verticalalignment='center',
                fontsize=fontsize_medium, zorder=0.5, color='black', weight='bold', stretch='condensed')

    else:
        bottom_bar_y1 = 1.1
        bottom_bar_y2 = 1.4
        bottom_text_y1 = 1.65
        bottom_text_y2 = 1.4
        bottom_text_y3 = 1.15
                
    ax.text(-0.21, bottom_bar_y1, 'Final Library', horizontalalignment='center', verticalalignment='center', fontsize=fontsize_small, zorder=2, color=color_blue, weight='bold', stretch='condensed')
    ax.text(-0.21, bottom_bar_y2, ' %d%%' % final_library_percent, horizontalalignment='center', verticalalignment='center', fontsize=fontsize_big, zorder=2, color=color_blue, weight='bold', stretch='condensed')

    ax.text(max(0.90, 0.05+clonal_wells/float(available_wells)), bottom_text_y1,
            '% 2d%% Test Fragments' % final_tf_percent,
            horizontalalignment='left', verticalalignment='center', fontsize=fontsize_small, zorder=2, color=color_gray, weight='bold', stretch='condensed')

    ax.text(max(0.90, 0.05+clonal_wells/float(available_wells)), bottom_text_y2,
            '% 2d%% Adapter Dimer' % dimer_percent,
            horizontalalignment='left', verticalalignment='center', fontsize=fontsize_small, zorder=2, color=color_gray, weight='bold', stretch='condensed')

    ax.text(max(0.90, 0.05+clonal_wells/float(available_wells)), bottom_text_y3,
            '% 2d%% Low Quality' % low_quality_percent,
            horizontalalignment='left', verticalalignment='center', fontsize=fontsize_small, zorder=2, color=color_gray, weight='bold', stretch='condensed')

    ax.text(0.04, bottom_bar_y2, intWithCommas(final_library_wells), horizontalalignment='left', verticalalignment='center',
            fontsize=fontsize_medium, zorder=2, color='white', weight='bold', stretch='condensed', alpha=0.7)
    ax.text(0.04, bottom_bar_y2, intWithCommas(final_library_wells), horizontalalignment='left', verticalalignment='center',
            fontsize=fontsize_medium, zorder=0.5, color="#000000", weight='black', stretch='condensed')

    ax.set_xlim(-0.42, 1.42)

    fig.patch.set_alpha(0.0)
    pyplot.savefig(beadogram_path)


def generate_wells_beadogram_basic(basecaller, sigproc, beadogram_path=None):
    beadogram_path = beadogram_path or os.path.join(basecaller, "wells_beadogram_basic.png")
    generate_wells_beadogram_all_or_basic(basecaller, sigproc, beadogram_path, is_full_details = False)



if __name__ == "__main__":
    import sys
    basecaller = os.path.join(os.getcwd(), "basecaller_results") if len(sys.argv) <= 1 else sys.argv[1]
    sigproc = os.path.join(os.getcwd(), "sigproc_results") if len(sys.argv) <= 2 else sys.argv[2]
    generate_wells_beadogram(basecaller, sigproc)
