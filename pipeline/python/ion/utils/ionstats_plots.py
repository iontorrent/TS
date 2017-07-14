#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import os
import json

from matplotlib import use
use("Agg", warn=False)
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import traceback
import math
from ion.utils.blockprocessing import printtime

color_front = "#2D4782"
color_back = "#822e71"


def group(n):

    SYMBOLS = ('K', 'M', 'G', 'T', 'P',)
    PREFIX = {'G': 1000000000, 'K': 1000, 'M': 1000000, 'P': 1000000000000000, 'T': 1000000000000, }

    for s in reversed(SYMBOLS):
        if n >= PREFIX[s]:
            value = float(n) / PREFIX[s]
            return '%.1f%s' % (value, s)

    return commas(int(n))


def commas(number):
    s = '%d' % number
    groups = []
    while s and s[-1].isdigit():
        groups.append(s[-3:])
        s = s[:-3]
    val = s + ','.join(reversed(groups))
    return val


''' Generate a small 300x30 read length histogram "sparkline" for display in barcode table '''


def read_length_sparkline(ionstats_basecaller_filename, output_png_filename, max_length):

    try:
        printtime("DEBUG: Generating plot %s" % output_png_filename)

        f = open(ionstats_basecaller_filename, 'r')
        ionstats_basecaller = json.load(f);
        f.close()

        histogram_x = range(0, max_length, 5)
        num_bins = len(histogram_x)
        histogram_y = [0] * num_bins

        for read_length, frequency in enumerate(ionstats_basecaller['full']['read_length_histogram']):
            current_bin = min(read_length/5, num_bins-1)
            histogram_y[current_bin] += frequency

        max_y = max(histogram_y)
        max_y = max(max_y, 1)

        fig = plt.figure(figsize=(3, 0.3), dpi=100)
        ax = fig.add_subplot(111, frame_on=False, xticks=[], yticks=[], position=[0, 0, 1, 1])
        ax.bar(histogram_x, histogram_y, width=6.5, color="#2D4782", linewidth=0, zorder=2)

        vline_step = 50 if (max_length < 650) else 100
        for idx in range(0, max_length, vline_step):
            label_bottom = str(idx)
            ax.text(idx, max_y*0.70, label_bottom, horizontalalignment='center', verticalalignment='center',
                    fontsize=8, zorder=1)
            ax.axvline(x=idx, color='#D0D0D0', ymax=0.5, zorder=0)
            ax.axvline(x=idx, color='#D0D0D0', ymin=0.9, zorder=0)

        ax.set_ylim(0, max_y)
        ax.set_xlim(-10, max_length)
        fig.patch.set_alpha(0.0)
        fig.savefig(output_png_filename)
        plt.close()  # Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory.

    except:
        printtime('Unable to generate plot %s' % output_png_filename)
        traceback.print_exc()
        raise


''' Generate read length histogram for the report page '''


def read_length_histogram(ionstats_basecaller_filename, output_png_filename, max_length):

    try:
        printtime("DEBUG: Generating plot %s" % output_png_filename)

        f = open(ionstats_basecaller_filename, 'r')
        ionstats_basecaller = json.load(f);
        f.close()

        histogram_x = range(0, max_length, 1)
        num_bins = len(histogram_x)
        histogram_y = [0] * num_bins

        for read_length, frequency in enumerate(ionstats_basecaller['full']['read_length_histogram']):
            current_bin = min(read_length, num_bins-1)
            if read_length < num_bins:
                histogram_y[current_bin] += frequency

        max_y = max(histogram_y)
        max_y = max(max_y, 1)

        fig = plt.figure(figsize=(4, 3.5), dpi=100)
        ax = fig.add_subplot(111, frame_on=False, yticks=[], position=[0, 0.15, 1, 0.88])
        ax.bar(histogram_x, histogram_y, width=2.5, color="#2D4782", linewidth=0, zorder=2)

        ax.set_ylim(0, 1.2*max_y)
        ax.set_xlim(-5, max_length+15)
        ax.set_xlabel("Read Length")
        fig.patch.set_alpha(0.0)
        fig.savefig(output_png_filename)
        plt.close()

    except:
        printtime('Unable to generate plot %s' % output_png_filename)
        traceback.print_exc()


''' Generate old-style read length histogram for display in classic report and dialog box in new report '''


def old_read_length_histogram(ionstats_basecaller_filename, output_png_filename, max_length):

    try:
        printtime("DEBUG: Generating plot %s" % output_png_filename)

        f = open(ionstats_basecaller_filename, 'r')
        ionstats_basecaller = json.load(f);
        f.close()

        histogram_x = range(0, max_length, 1)
        num_bins = len(histogram_x)
        histogram_y = [0] * num_bins

        for read_length, frequency in enumerate(ionstats_basecaller['full']['read_length_histogram']):
            current_bin = min(read_length, num_bins-1)
            if read_length < num_bins:
                histogram_y[current_bin] += frequency

        fig = plt.figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.bar(histogram_x, histogram_y, width=2, color="#2D4782", linewidth=0)
        ax.set_xlim(0, max_length)
        ax.set_title('Read Length Histogram')
        ax.set_xlabel('Read Length')
        ax.set_ylabel('Count')
        fig.savefig(output_png_filename)
        plt.close()

    except:
        printtime('Unable to generate plot %s' % output_png_filename)
        traceback.print_exc()
        raise


''' Generate cumulative read length plot of called vs aligned read length '''


def alignment_rate_plot(alignStats, ionstats_basecaller_filename, output_png_filename, graph_max_x, y_ticks=None):

    if not os.path.exists(alignStats):
        printtime("ERROR: %s does not exist" % alignStats)
        return

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

    try:
        f = open(alignStats, 'r')
        alignStats_json = json.load(f);
        f.close()
        read_length = alignStats_json["read_length"]
        # nread = alignStats_json["nread"]
        aligned = alignStats_json["aligned"]
    except:
        printtime("ERROR: problem parsing %s" % alignStats)
        traceback.print_exc()
        return

    try:
        f = open(ionstats_basecaller_filename, 'r')
        ionstats_basecaller = json.load(f);
        f.close()
    except:
        printtime('Failed to load %s' % (ionstats_basecaller_filename))
        traceback.print_exc()
        return

    histogram_x = range(0, graph_max_x)
    histogram_y = [0] * graph_max_x

    try:
        for read_length_bin, frequency in enumerate(ionstats_basecaller['full']['read_length_histogram']):
            current_bin = min(read_length_bin, graph_max_x-1)
            histogram_y[current_bin] += frequency
    except:
        printtime('Problem parsing %s' % (ionstats_basecaller_filename))
        traceback.print_exc()
        return

    for idx in range(graph_max_x-1, 0, -1):
        histogram_y[idx-1] += histogram_y[idx]

    nread = histogram_y[1:]

    if not y_ticks:
        fig = plt.figure(figsize=(4, 3), dpi=100)
    else:
        fig = plt.figure(figsize=(7, 5), dpi=100)

    max_x = 1
    if len(read_length) > 0:
        max_x = max(read_length)
    max_x = min(max_x, graph_max_x)

    max_y = max(nread)

    if not y_ticks:
        ax = fig.add_subplot(111, frame_on=False, yticks=[], position=[0.1, 0.15, 0.8, 0.89])
    else:

        import matplotlib.ticker as mticker

        def square_braces(tick_val, tick_pos):
            """Put square braces around the given tick_val """
            return group(tick_val)

        d = len(str(max_y)) - 1
        yticks = range(10**d, max_y, 10**d)
        yticks.append(max_y)

        ax = fig.add_subplot(111, frame_on=False, yticks=yticks, position=[0.14, 0.15, 0.8, 0.89])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(square_braces))

    ax.fill_between(histogram_x[1:], nread, color=color_back, zorder=1)
    ax.fill_between(read_length, aligned, color=color_front, zorder=2)

    ax.set_xlabel('Position in Read')
    ax.set_ylabel('Reads')

    if sum(nread) > 0:
        map_percent = int(round(100.0 * float(alignStats_json["total_mapped_target_bases"])
                                / float(sum(nread))))
        unmap_percent = 100 - map_percent
    else:
        map_percent = 0.0
        unmap_percent = 0.0

    fontsize_big = 15
    fontsize_small = 10
    fontsize_medium = 8

    ax.text(0.8*max_x, 0.95*max_y, 'Aligned Bases', horizontalalignment='center', verticalalignment='center', fontsize=fontsize_small, zorder=4, color=color_front, weight='bold', stretch='condensed')
    ax.text(0.8*max_x, 1.05*max_y, ' %d%%' % map_percent, horizontalalignment='center', verticalalignment='center', fontsize=fontsize_big, zorder=4, color=color_front, weight='bold', stretch='condensed')
    ax.text(0.8*max_x, 0.7*max_y, 'Unaligned', horizontalalignment='center', verticalalignment='center',
            fontsize=fontsize_small, zorder=4, color=color_back, weight='bold', stretch='condensed')
    ax.text(0.8*max_x, 0.8*max_y, ' %d%%' % unmap_percent, horizontalalignment='center', verticalalignment='center', fontsize=fontsize_big, zorder=4, color=color_back, weight='bold', stretch='condensed')
    if not y_ticks:

        ax.text(-0.06*max_x, 1.02*max_y, intWithCommas(max_y), horizontalalignment='left', verticalalignment='bottom',  zorder=4, color="black")

    ax.set_xlim(0, max_x)
    ax.set_ylim(0, 1.2*max_y)
    fig.patch.set_alpha(0.0)
    fig.savefig(output_png_filename)
    plt.close()

    if not y_ticks:
        alignment_rate_plot(alignStats, ionstats_basecaller_filename, "large_" + output_png_filename, graph_max_x, y_ticks=True)


def alignment_rate_plot2(ionstats_alignment_filename, output_png_filename, graph_max_x, y_ticks=None):
    """ Generate cumulative read length plot of called vs aligned read length """

    try:
        printtime("DEBUG: Generating plot %s" % output_png_filename)

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

        f = open(ionstats_alignment_filename, 'r')
        ionstats_alignment = json.load(f);
        f.close()

        histogram_range = range(graph_max_x)
        histogram_full = [0] * graph_max_x
        histogram_aligned = [0] * graph_max_x

        for read_length_bin, frequency in enumerate(ionstats_alignment['full']['read_length_histogram']):
            current_bin = min(read_length_bin, graph_max_x-1)
            histogram_full[current_bin] += frequency

        for read_length_bin, frequency in enumerate(ionstats_alignment['aligned']['read_length_histogram']):
            current_bin = min(read_length_bin, graph_max_x-1)
            histogram_aligned[current_bin] += frequency

        for idx in range(graph_max_x-1, 1, -1):
            histogram_full[idx-1] += histogram_full[idx]
            histogram_aligned[idx-1] += histogram_aligned[idx]

        if not y_ticks:
            fig = plt.figure(figsize=(4, 3), dpi=100)
        else:
            fig = plt.figure(figsize=(7, 5), dpi=100)

        max_x = min(graph_max_x, len(ionstats_alignment['full']['read_length_histogram']))
        max_y_true = max(histogram_full)
        max_y = max(histogram_full)+1

        if not y_ticks:
            ax = fig.add_subplot(111, frame_on=False, yticks=[], position=[0.1, 0.15, 0.8, 0.89])
        else:
            import matplotlib.ticker as mticker

            def square_braces(tick_val, tick_pos):
                """Put square braces around the given tick_val """
                return group(tick_val)

            d = len(str(max_y)) - 1
            yticks = range(10**d, max_y, 10**d)
            if yticks:
                yticks.pop()
            yticks.append(max_y)

            ax = fig.add_subplot(111, frame_on=False, yticks=yticks, position=[0.14, 0.15, 0.8, 0.89])
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(square_braces))

        ax.fill_between(histogram_range, histogram_full, color=color_back, zorder=1)
        ax.fill_between(histogram_range, histogram_aligned, color=color_front, zorder=2)

        ax.set_xlabel('Position in Read')
        ax.set_ylabel('Reads')

        if sum(histogram_full) > 0 and float(ionstats_alignment['full']['num_bases']) > 0:
            map_percent = int(round(100.0 * float(ionstats_alignment['aligned']['num_bases'])
                                    / float(ionstats_alignment['full']['num_bases'])))
            unmap_percent = 100 - map_percent
        else:
            map_percent = 0.0
            unmap_percent = 0.0

        fontsize_big = 15
        fontsize_small = 10
        fontsize_medium = 8

        ax.text(0.8*max_x, 0.95*max_y, 'Aligned Bases', horizontalalignment='center', verticalalignment='center', fontsize=fontsize_small, zorder=4, color=color_front, weight='bold', stretch='condensed')
        ax.text(0.8*max_x, 1.05*max_y, ' %d%%' % map_percent, horizontalalignment='center', verticalalignment='center', fontsize=fontsize_big, zorder=4, color=color_front, weight='bold', stretch='condensed')
        ax.text(0.8*max_x, 0.7*max_y, 'Unaligned', horizontalalignment='center', verticalalignment='center',
                fontsize=fontsize_small, zorder=4, color=color_back, weight='bold', stretch='condensed')
        ax.text(0.8*max_x, 0.8*max_y, ' %d%%' % unmap_percent, horizontalalignment='center', verticalalignment='center', fontsize=fontsize_big, zorder=4, color=color_back, weight='bold', stretch='condensed')
        if not y_ticks:
            ax.text(-0.06*max_x, 1.02*max_y, intWithCommas(max_y_true), horizontalalignment='left', verticalalignment='bottom',  zorder=4, color="black")

        ax.set_xlim(0, max_x)
        ax.set_ylim(0, 1.2*max_y)
        fig.patch.set_alpha(0.0)
        fig.savefig(output_png_filename)
        plt.close()

        if not y_ticks:
            alignment_rate_plot2(ionstats_alignment_filename, "large_" + output_png_filename, graph_max_x, y_ticks=True)

    except:
        printtime('Unable to generate plot %s' % output_png_filename)
        traceback.print_exc()


''' Generate plot with error rate vs base position '''


def old_base_error_plot(ionstats_alignment_filename, output_png_filename, graph_max_x):

    try:
        printtime("DEBUG: Generating plot %s" % output_png_filename)

        f = open(ionstats_alignment_filename, 'r')
        ionstats_alignment = json.load(f);
        f.close()

        histogram_range = range(graph_max_x)
        histogram_aligned = [0] * graph_max_x

        for read_length_bin, frequency in enumerate(ionstats_alignment['aligned']['read_length_histogram']):
            current_bin = min(read_length_bin, graph_max_x-1)
            histogram_aligned[current_bin] += frequency

        for idx in range(graph_max_x-1, 1, -1):
            histogram_aligned[idx-1] += histogram_aligned[idx]

        n_err_at_position = ionstats_alignment["error_by_position"]

        accuracy = []
        reads = []

        for i in range(graph_max_x):
            if histogram_aligned[i] > 1000:
                accuracy.append(100 * (1 - float(n_err_at_position[i]) / float(histogram_aligned[i])))
                reads.append(i+1)

        fig = plt.figure(figsize=(4, 4), dpi=100)
        ax = fig.add_subplot(111, frame_on=False, position=[0.2, 0.1, 0.7, 0.8])

        max_x = min(graph_max_x, len(ionstats_alignment['aligned']['read_length_histogram']))

        ax.plot(reads, accuracy, linewidth=3.0, color="#2D4782")
        ax.set_xlim(0, max_x)
        ax.set_ylim(90, 100.9)
        ax.set_xlabel('Position in Read')
        ax.set_ylabel("Accuracy at Position")

        # ax.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
        fig.patch.set_alpha(0.0)
        fig.savefig(output_png_filename)
        plt.close()

    except:
        printtime('Unable to generate plot %s' % output_png_filename)
        traceback.print_exc()


def cumulative_sum(values):
    """
    Returns the cumulative sum of a collection of numbers
    """
    cum_sum = []
    y = 0
    for value in values:
        y += value
        cum_sum.append(y)

    return cum_sum


def _get_per_base_err(ins_values, del_values, sub_values, max_base):
    """
    Returns an array of errors for each base
    """
    min_array_entries = min(min(len(ins_values), len(del_values)), len(sub_values))
    max_count = min(min_array_entries, max_base)
    # printtime("DEBUG: _get_per_base_err() min_array_entries=%d; max_base=%d" %(min_array_entries, max_base))

    per_base_err_values = [0] * max_count

    for i in range(0, max_count):
        per_base_err_values[i] = ins_values[i] + del_values[i] + sub_values[i]
        # printtime("DEBUG: _get_per_base_err() i=%d; ins=%d; del=%d; sub=%d; per_base_err=%d" %(i, ins_values[i], del_values[i], sub_values[i], per_base_err_values[i]))

    return per_base_err_values


def _which_depth_indices(depth_values, total_depth, threshold):
    """
    Gives the TRUE indices of a logical object, allowing for array indices.
    """
    # printtime("DEBUG: _which_depth_indices() total_depth=%0.2f; depth_throughput_threshold=%0.3f" %(total_depth, threshold))

    count = len(depth_values)
    # true_indices = []
    true_indices = [0] * count
    true_index = 0
    collection_index = 0

    for depth in depth_values:
        if float(depth) / float(total_depth) <= threshold:
            # printtime("DEBUG: _which_depth_indices() true_index=%d; collection_index=%d; depth=%0.2f; total_depth=%0.2f; depth_ratio=%0.4f" %(true_index, collection_index, depth, total_depth, float(depth) / float(total_depth)))

            true_indices[true_index] = collection_index
            true_index += 1
#         else:
#             printtime("DEBUG: SKIPPED!!!! _which_depth_indices() BEYOND THRESHOLD!! collection_index=%d; depth=%d; total_depth=%d" %(collection_index, depth, total_depth))

        collection_index += 1

    return true_indices


def _get_per_base_accuracy(error_values, depth_values):
    """
    Returns the accuracy per base
    """
    error_count = len(error_values)
    depth_count = len(depth_values)
    max_count = min(error_count, depth_count)

    per_base_accuracy = [0] * max_count
    # printtime("DEBUG: _get_per_base_accuracy() error_count=%d; depth_count=%d; max_count=%d" %(error_count, depth_count, max_count))

    for i in range(0, max_count):
        if depth_values[i] != 0:
            per_base_accuracy[i] = 100 * (1 - float(error_values[i]) / float(depth_values[i]))

            # printtime("_get_per_base_accuracy() i=%d; error_value=%d; depth_value=%d; per_base_accuracy=%0.2f" %(i, error_values[i], depth_values[i], per_base_accuracy[i]))

        else:
            printtime("DEBUG: BAD should not happen!!! _get_per_base_accuracy() depth_values[i=%d] is zero!!" % (i))
            per_base_accuracy[i] = 0

    return per_base_accuracy


def base_error_plot(ionstats_alignment_filename, output_png_filename, graph_max_x):
    """
    Generates plot with error rate vs base position
    """
    try:
        printtime("DEBUG: Generating NEW base error (AKA per-base accuracy plot %s" % output_png_filename)

        f = open(ionstats_alignment_filename, 'r')
        ionstats_alignment = json.load(f);
        f.close()

        # Info about this new code vs the old_base_error_plot function...
        #
        # In runs where 5' soft-clipping is disabled (which it is by default),
        # data["by_base"]["depth"] and data["aligned"]["read_length_histogram"] contain mathematically equivalent information.
        # The former is more useful as it remains valid for using with the accuracy plot when 5' soft clipping is used,
        # the latter does not.

        # data["error_by_position"] is equal to the sum of the other three data["by_base"] entries.
        # Relying on the latter three is preferred because they are more flexible
        #(we will have the option of partitioning error rate into sub/ins/del)
        # and because it will allow us to then retire data["error_by_position"] which is redundant.

        depth_throughput_fraction = 0.995

        # Keep only the by_base section from the json file
        by_base_data = ionstats_alignment['by_base']

        # determine the max base that will be plotted
        total_depth = sum(by_base_data['depth'])

        cumulative_depth = cumulative_sum(by_base_data['depth'])

        # printtime("cumulative_depth=%s" %(cumulative_depth))

        my_max_base = max(_which_depth_indices(cumulative_depth, total_depth, depth_throughput_fraction))
        max_base = min(graph_max_x, my_max_base)

        # max_base = min(graph_max_x, len(by_base_data['depth']))

        printtime("DEBUG: >>>>> depth_throughput_fraction=%0.4f; total_depth=%d; my_max_base=%d; max_base=%d" % (depth_throughput_fraction, total_depth, my_max_base, max_base))

        # get per-base error and depth
        per_base_err = _get_per_base_err(by_base_data['ins'], by_base_data['del'], by_base_data['sub'], max_base)

#         for value in per_base_err:
#             printtime("per_base_err=%d" %(value))

        per_base_depth = by_base_data['depth'][0:max_base]

#         for value in per_base_depth:
#             printtime("per_base_depth=%d" %(value))

        per_base_accuracy = _get_per_base_accuracy(per_base_err, per_base_depth)

        overall_accuracy = 100 * (1 - float(sum(per_base_err)) / float(sum(per_base_depth)))

        fig = plt.figure(figsize=(4, 4), dpi=100)
        ax = fig.add_subplot(111, frame_on=False, position=[0.2, 0.1, 0.7, 0.8])
        ax.plot(range(0, max_base), per_base_accuracy, linewidth=3.0, color="#2D4782")

        x_axis_max = min(graph_max_x, len(by_base_data['depth']))
        print("DEBUG: >>>>>  graph_max_x=%d; len(by_base_data[depth]=%d; x_axis_max=%d" % (graph_max_x, len(by_base_data['depth']), x_axis_max))

        ax.set_xlim(0, x_axis_max)

        ax.set_ylim(90, 100.9)
        ax.set_xlabel('Position')
        ax.set_ylabel("Accuracy")

        # ax.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
        fig.patch.set_alpha(0.0)
        fig.savefig(output_png_filename)
        plt.close()

    except:
        printtime('Unable to generate plot %s' % output_png_filename)
        traceback.print_exc()


''' Generate old AQ-x read length histogram '''


def old_aq_length_histogram(ionstats_alignment_filename, output_png_filename, aq_string, color):

    try:
        printtime("DEBUG: Generating plot %s" % output_png_filename)

        f = open(ionstats_alignment_filename, 'r')
        ionstats_alignment = json.load(f);
        f.close()

        data = ionstats_alignment[aq_string]['read_length_histogram']

        xaxis = range(len(data))
        ymax = max(data) + 10
        xlen = len(data) + 10
        xmax = len(data) - 1
        if xmax < 400:
            xmax = 400

        fig = plt.figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)

        ax.bar(xaxis, data, facecolor=color, align='center', linewidth=0, alpha=1.0, width=1.0)
        ax.set_xlabel('Filtered %s Read Length' % aq_string)
        ax.set_ylabel('Count')
        ax.set_title('Filtered %s Read Length' % aq_string)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        fig.savefig(output_png_filename)
        plt.close()

    except:
        printtime('Unable to generate plot %s' % output_png_filename)
        traceback.print_exc()


def quality_histogram(ionstats_basecaller_filename, output_png_filename):

    try:
        printtime("DEBUG: Generating plot %s" % output_png_filename)

        f = open(ionstats_basecaller_filename, 'r')
        ionstats_basecaller = json.load(f);
        f.close()

        qv_histogram = ionstats_basecaller["qv_histogram"]

        sum_total = float(sum(qv_histogram))
        if sum_total > 0:
            percent_0_5 = 100.0 * sum(qv_histogram[0:5]) / sum_total
            percent_5_10 = 100.0 * sum(qv_histogram[5:10]) / sum_total
            percent_10_15 = 100.0 * sum(qv_histogram[10:15]) / sum_total
            percent_15_20 = 100.0 * sum(qv_histogram[15:20]) / sum_total
            percent_20 = 100.0 * sum(qv_histogram[20:]) / sum_total
        else:
            percent_0_5 = 0.0
            percent_5_10 = 0.0
            percent_10_15 = 0.0
            percent_15_20 = 0.0
            percent_20 = 0.0

        graph_x = [0, 5, 10, 15, 20]
        graph_y = [percent_0_5, percent_5_10, percent_10_15, percent_15_20, percent_20]

        max_y = max(graph_y)

        ticklabels = ['0-4', '5-9', '10-14', '15-19', '20+']

        fig = plt.figure(figsize=(4, 4), dpi=100)
        ax = fig.add_subplot(111, frame_on=False, xticks=[], yticks=[], position=[.1, 0.1, 1, 0.9])
        ax.bar(graph_x, graph_y, width=4.8, color="#2D4782", linewidth=0)

        for idx in range(5):
            label_bottom = ticklabels[idx]
            label_top = '%1.0f%%' % graph_y[idx]
            ax.text(idx*5 + 2.5, -max_y*0.04, label_bottom, horizontalalignment='center', verticalalignment='top',
                    fontsize=12)
            ax.text(idx*5 + 2.5, max_y*0.06+graph_y[idx], label_top, horizontalalignment='center', verticalalignment='bottom',
                    fontsize=12)

        ax.set_xlabel("Base Quality")

        ax.set_xlim(0, 34.8)
        ax.set_ylim(-0.1*max_y, 1.2*max_y)
        fig.patch.set_alpha(0.0)
        fig.savefig(output_png_filename)
        plt.close()

    except:
        printtime('Unable to generate plot %s' % output_png_filename)
        traceback.print_exc()


''' Generate old AQ10, old AQ17, and new AQ17 plots for all test fragments '''


def tf_length_histograms(ionstats_tf_filename, output_dir):

    try:
        printtime("DEBUG: Generating TF plots:")

        f = open(ionstats_tf_filename, 'r')
        ionstats_tf = json.load(f);
        f.close()

        for tf_name, tf_data in ionstats_tf['results_by_tf'].iteritems():

            Q10_hist = tf_data['AQ10']['read_length_histogram']
            Q17_hist = tf_data['AQ17']['read_length_histogram']
            sequence = tf_data['sequence']

            # Step 1: New AQ17

            output_png_filename = os.path.join(output_dir, 'new_Q17_%s.png' % (tf_name))
            printtime("DEBUG: Generating plot %s" % output_png_filename)

            num_bases_q = len(Q17_hist)
            num_bases_s = len(sequence)
            num_bases = min(num_bases_q, num_bases_s)
            nuc_color = {'A': "#4DAF4A", 'C': "#275EB8", 'T': "#E41A1C", 'G': "#202020"}
            text_offset = -max(Q17_hist) * 0.1

            fig = plt.figure(figsize=(8, 1), dpi=100)
            ax = fig.add_subplot(111, frame_on=False, xticks=[], yticks=[], position=[0, 0.3, 1, 0.7])
            ax.bar(range(num_bases_q), Q17_hist, linewidth=0, width=1, color="#2D4782")
            for idx in range(num_bases):
                nuc = sequence[idx]
                ax.text(idx+1, text_offset, nuc, horizontalalignment='center', verticalalignment='center', fontsize=8, family='sans-serif', weight='bold', color=nuc_color[nuc])
                if (idx % 10) == 0:
                    ax.text(idx+0.5, 3*text_offset, str(idx), horizontalalignment='center', verticalalignment='center', fontsize=8, family='sans-serif', weight='bold')

            ax.set_xlim(0, num_bases+2)
            fig.patch.set_alpha(0.0)
            fig.savefig(output_png_filename)
            plt.close()

            # Step 2: New AQ10

            output_png_filename = os.path.join(output_dir, 'new_Q10_%s.png' % (tf_name))
            printtime("DEBUG: Generating plot %s" % output_png_filename)

            num_bases_q = len(Q10_hist)
            num_bases_s = len(sequence)
            num_bases = min(num_bases_q, num_bases_s)
            nuc_color = {'A': "#4DAF4A", 'C': "#275EB8", 'T': "#E41A1C", 'G': "#202020"}
            text_offset = -max(Q10_hist) * 0.1

            fig = plt.figure(figsize=(8, 1), dpi=100)
            ax = fig.add_subplot(111, frame_on=False, xticks=[], yticks=[], position=[0, 0.3, 1, 0.7])
            ax.bar(range(num_bases_q), Q10_hist, linewidth=0, width=1, color="#2D4782")
            for idx in range(num_bases):
                nuc = sequence[idx]
                ax.text(idx+1, text_offset, nuc, horizontalalignment='center', verticalalignment='center', fontsize=8, family='sans-serif', weight='bold', color=nuc_color[nuc])
                if (idx % 10) == 0:
                    ax.text(idx+0.5, 3*text_offset, str(idx), horizontalalignment='center', verticalalignment='center', fontsize=8, family='sans-serif', weight='bold')

            ax.set_xlim(0, num_bases+2)
            fig.patch.set_alpha(0.0)
            fig.savefig(output_png_filename)
            plt.close()

    except:
        printtime('Unable to generate TF plots')
        traceback.print_exc()


if __name__ == "__main__":

    import os

    ionstats_file = os.path.join('basecaller_results', 'ionstats_basecaller.json')

    # Make alignment_rate_plot.png
    stats = json.load(open(ionstats_file))
    l = stats['full']['max_read_length']
    graph_max_x = int(round(l + 49, -2))

    a = alignment_rate_plot2(
        'ionstats_alignment.json',
        'alignment_rate_plot.png',
        int(graph_max_x)
    )

    print "made graph"
    print a
