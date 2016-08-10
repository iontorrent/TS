#!/usr/bin/python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

import os
import json
import traceback
import copy

# BeadSummary section will be eventually obsoleted


def merge_bead_summary(block_dirs):
    print 'mergeBaseCallerJson.merge_bead_summary on %s blocks' % len(block_dirs)

    bs_lib = {'badKey': 0, 'highPPF': 0, 'highRes': 0, 'polyclonal': 0, 'short': 0, 'valid': 0, 'zero': 0, 'key': 'TCAG'}
    bs_tf = {'badKey': 0, 'highPPF': 0, 'highRes': 0, 'polyclonal': 0, 'short': 0, 'valid': 0, 'zero': 0, 'key': 'ATCG'}
    for dir in block_dirs:
        try:
            file = open(os.path.join(dir, 'BaseCaller.json'), 'r')
            block_json = json.load(file)
            file.close()

            bs_lib['badKey'] += block_json['BeadSummary']['lib']['badKey']
            bs_lib['highPPF'] += block_json['BeadSummary']['lib']['highPPF']
            bs_lib['highRes'] += block_json['BeadSummary']['lib']['highRes']
            bs_lib['polyclonal'] += block_json['BeadSummary']['lib']['polyclonal']
            bs_lib['short'] += block_json['BeadSummary']['lib']['short']
            bs_lib['valid'] += block_json['BeadSummary']['lib']['valid']
            bs_lib['zero'] += block_json['BeadSummary']['lib']['zero']
            # bs_lib['key']         = block_json['BeadSummary']['lib']['key']

            bs_tf['badKey'] += block_json['BeadSummary']['tf']['badKey']
            bs_tf['highPPF'] += block_json['BeadSummary']['tf']['highPPF']
            bs_tf['highRes'] += block_json['BeadSummary']['tf']['highRes']
            bs_tf['polyclonal'] += block_json['BeadSummary']['tf']['polyclonal']
            bs_tf['short'] += block_json['BeadSummary']['tf']['short']
            bs_tf['valid'] += block_json['BeadSummary']['tf']['valid']
            bs_tf['zero'] += block_json['BeadSummary']['tf']['zero']
            # bs_tf['key']          = block_json['BeadSummary']['tf']['key']
        except:
            traceback.print_exc()
            print 'mergeBaseCallerJson.merge_bead_summary: skipping block ' + dir

    return {"lib": bs_lib, "tf": bs_tf}


def merge_filtering(block_dirs):

    # BaseDetails
    bd = {'adapter_trim': 0, 'extra_trim': 0, 'failed_keypass': 0, 'final': 0, 'high_residual': 0,
          'initial': 0, 'quality_filter': 0, 'quality_trim': 0, 'short': 0, 'tag_trim': 0}
    # LibraryReport
    lr = {"filtered_low_quality": 0, "filtered_polyclonal": 0, "filtered_primer_dimer": 0, "final_library_reads": 0}
    # ReadDetails/lib
    rd_lib = {"adapter_trim": 0, "bkgmodel_high_ppf": 0, "bkgmodel_keypass": 0, "bkgmodel_polyclonal": 0,
              "extra_trim": 0, "failed_keypass": 0, "high_ppf": 0, "high_residual": 0,
              "key": "ATCG", "polyclonal": 0, "quality_filter": 0, "quality_trim": 0,
              "short": 0, "tag_trim": 0, "valid": 0, "zero": 0}
    # ReadDetails/tf
    rd_tf = {"adapter_trim": 0, "bkgmodel_high_ppf": 0, "bkgmodel_keypass": 0, "bkgmodel_polyclonal": 0,
             "extra_trim": 0, "failed_keypass": 0, "high_ppf": 0, "high_residual": 0,
             "key": "ATCG", "polyclonal": 0, "quality_filter": 0, "quality_trim": 0,
             "short": 0, "tag_trim": 0, "valid": 0, "zero": 0}
    # Bead Adapters
    adapters = {}

    qv_hist = [0] * 50

    for dir in block_dirs:
        try:
            file = open(os.path.join(dir, 'BaseCaller.json'), 'r')
            block_json = json.load(file)
            file.close()

            # Merging adapter classification part of Basecaller.json
            adapter_idx = 0
            while ('Adapter_'+str(adapter_idx)) in block_json['Filtering'].get('BeadAdapters', {}):
                if ('Adapter_'+str(adapter_idx)) in adapters:
                    adapters['Adapter_'+str(adapter_idx)]['read_count'] += block_json['Filtering']['BeadAdapters']['Adapter_'+str(adapter_idx)].get('read_count', 0)
                    adapters['Adapter_'+str(adapter_idx)]['num_decisions'] += block_json['Filtering']['BeadAdapters']['Adapter_'+str(adapter_idx)].get('num_decisions', 0)
                    adapters['Adapter_'+str(adapter_idx)]['average_metric'] += block_json['Filtering']['BeadAdapters']['Adapter_'+str(adapter_idx)].get('read_count', 0) * block_json['Filtering']['BeadAdapters']['Adapter_'+str(adapter_idx)].get('average_metric', 0)
                    adapters['Adapter_'+str(adapter_idx)]['average_separation'] += block_json['Filtering']['BeadAdapters']['Adapter_'+str(adapter_idx)].get('num_decisions', 0) * block_json['Filtering']['BeadAdapters']['Adapter_'+str(adapter_idx)].get('average_separation', 0)
                else:
                    adapters['Adapter_'+str(adapter_idx)] = copy.deepcopy(block_json['Filtering']['BeadAdapters']['Adapter_'+str(adapter_idx)])
                    adapters['Adapter_'+str(adapter_idx)]['average_metric'] *= block_json['Filtering']['BeadAdapters']['Adapter_'+str(adapter_idx)]['read_count']
                    adapters['Adapter_'+str(adapter_idx)]['average_separation'] *= block_json['Filtering']['BeadAdapters']['Adapter_'+str(adapter_idx)]['num_decisions']
                adapter_idx += 1

            bd['adapter_trim'] += block_json['Filtering'].get('BaseDetails', {}).get('adapter_trim', 0)
            bd['extra_trim'] += block_json['Filtering'].get('BaseDetails', {}).get('extra_trim', 0)
            bd['failed_keypass'] += block_json['Filtering'].get('BaseDetails', {}).get('failed_keypass', 0)
            bd['final'] += block_json['Filtering'].get('BaseDetails', {}).get('final', 0)
            bd['high_residual'] += block_json['Filtering'].get('BaseDetails', {}).get('high_residual', 0)
            bd['initial'] += block_json['Filtering'].get('BaseDetails', {}).get('initial', 0)
            bd['quality_filter'] += block_json['Filtering'].get('BaseDetails', {}).get('quality_filter', 0)
            bd['quality_trim'] += block_json['Filtering'].get('BaseDetails', {}).get('quality_trim', 0)
            bd['short'] += block_json['Filtering'].get('BaseDetails', {}).get('short', 0)
            bd['tag_trim'] += block_json['Filtering'].get('BaseDetails', {}).get('tag_trim', 0)

            lr['filtered_low_quality'] += block_json['Filtering'].get('LibraryReport', {}).get('filtered_low_quality', 0)
            lr['filtered_polyclonal'] += block_json['Filtering'].get('LibraryReport', {}).get('filtered_polyclonal', 0)
            lr['filtered_primer_dimer'] += block_json['Filtering'].get('LibraryReport', {}).get('filtered_primer_dimer', 0)
            lr['final_library_reads'] += block_json['Filtering'].get('LibraryReport', {}).get('final_library_reads', 0)

            rd_lib['adapter_trim'] += block_json['Filtering'].get('ReadDetails', {}).get('lib', {}).get('adapter_trim', 0)
            rd_lib['bkgmodel_high_ppf'] += block_json['Filtering'].get('ReadDetails', {}).get('lib', {}).get('bkgmodel_high_ppf', 0)
            rd_lib['bkgmodel_keypass'] += block_json['Filtering'].get('ReadDetails', {}).get('lib', {}).get('bkgmodel_keypass', 0)
            rd_lib['bkgmodel_polyclonal'] += block_json['Filtering'].get('ReadDetails', {}).get('lib', {}).get('bkgmodel_polyclonal', 0)
            rd_lib['extra_trim'] += block_json['Filtering'].get('ReadDetails', {}).get('lib', {}).get('extra_trim', 0)
            rd_lib['failed_keypass'] += block_json['Filtering'].get('ReadDetails', {}).get('lib', {}).get('failed_keypass', 0)
            rd_lib['high_ppf'] += block_json['Filtering'].get('ReadDetails', {}).get('lib', {}).get('high_ppf', 0)
            rd_lib['high_residual'] += block_json['Filtering'].get('ReadDetails', {}).get('lib', {}).get('high_residual', 0)
            rd_lib['polyclonal'] += block_json['Filtering'].get('ReadDetails', {}).get('lib', {}).get('polyclonal', 0)
            rd_lib['quality_filter'] += block_json['Filtering'].get('ReadDetails', {}).get('lib', {}).get('quality_filter', 0)
            rd_lib['quality_trim'] += block_json['Filtering'].get('ReadDetails', {}).get('lib', {}).get('quality_trim', 0)
            rd_lib['short'] += block_json['Filtering'].get('ReadDetails', {}).get('lib', {}).get('short', 0)
            rd_lib['tag_trim'] += block_json['Filtering'].get('ReadDetails', {}).get('lib', {}).get('tag_trim', 0)
            rd_lib['valid'] += block_json['Filtering'].get('ReadDetails', {}).get('lib', {}).get('valid', 0)
            rd_lib['zero'] += block_json['Filtering'].get('ReadDetails', {}).get('lib', {}).get('zero', 0)

            rd_tf['adapter_trim'] += block_json['Filtering'].get('ReadDetails', {}).get('tf', {}).get('adapter_trim', 0)
            rd_tf['bkgmodel_high_ppf'] += block_json['Filtering'].get('ReadDetails', {}).get('tf', {}).get('bkgmodel_high_ppf', 0)
            rd_tf['bkgmodel_keypass'] += block_json['Filtering'].get('ReadDetails', {}).get('tf', {}).get('bkgmodel_keypass', 0)
            rd_tf['bkgmodel_polyclonal'] += block_json['Filtering'].get('ReadDetails', {}).get('tf', {}).get('bkgmodel_polyclonal', 0)
            rd_tf['extra_trim'] += block_json['Filtering'].get('ReadDetails', {}).get('tf', {}).get('extra_trim', 0)
            rd_tf['failed_keypass'] += block_json['Filtering'].get('ReadDetails', {}).get('tf', {}).get('failed_keypass', 0)
            rd_tf['high_ppf'] += block_json['Filtering'].get('ReadDetails', {}).get('tf', {}).get('high_ppf', 0)
            rd_tf['high_residual'] += block_json['Filtering'].get('ReadDetails', {}).get('tf', {}).get('high_residual', 0)
            rd_tf['polyclonal'] += block_json['Filtering'].get('ReadDetails', {}).get('tf', {}).get('polyclonal', 0)
            rd_tf['quality_filter'] += block_json['Filtering'].get('ReadDetails', {}).get('tf', {}).get('quality_filter', 0)
            rd_tf['quality_trim'] += block_json['Filtering'].get('ReadDetails', {}).get('tf', {}).get('quality_trim', 0)
            rd_tf['short'] += block_json['Filtering'].get('ReadDetails', {}).get('tf', {}).get('short', 0)
            rd_tf['tag_trim'] += block_json['Filtering'].get('ReadDetails', {}).get('tf', {}).get('tag_trim', 0)
            rd_tf['valid'] += block_json['Filtering'].get('ReadDetails', {}).get('tf', {}).get('valid', 0)
            rd_tf['zero'] += block_json['Filtering'].get('ReadDetails', {}).get('tf', {}).get('zero', 0)

            rd_lib['key'] = block_json['Filtering'].get('ReadDetails', {}).get('lib', {}).get('key', rd_lib['key'])
            rd_tf['key'] = block_json['Filtering'].get('ReadDetails', {}).get('tf', {}).get('key', rd_tf['key'])

            for idx in range(50):
                qv_hist[idx] += block_json['Filtering']['qv_histogram'][idx]

        except:
            print 'mergeBaseCallerJson.merge_filtering: skipping block ' + dir

    # Now looping through the adapters again to turn averages back into averages
    adapter_idx = 0
    while ('Adapter_'+str(adapter_idx)) in adapters:
        adapters['Adapter_'+str(adapter_idx)]['average_metric'] /= max(adapters['Adapter_'+str(adapter_idx)]['read_count'], 1)
        adapters['Adapter_'+str(adapter_idx)]['average_separation'] /= max(adapters['Adapter_'+str(adapter_idx)]['num_decisions'], 1)
        adapter_idx += 1

    return {"BaseDetails": bd, "BeadAdapters": adapters, "LibraryReport": lr, 'ReadDetails': {'lib': rd_lib, 'tf': rd_tf}, 'qv_histogram': qv_hist}


def merge_phasing(block_dirs):

    # Phasing
    ph = {'CF': 0, 'IE': 0, 'DR': 0, 'CFbyRegion': 0, 'IEbyRegion': 0, 'DRbyRegion': 0, 'RegionRows': 1, 'RegionCols': 1}

    # 1. Determine grid size and translate directory names into coordinates

    try:

        coord_x = [-1] * len(block_dirs)
        coord_y = [-1] * len(block_dirs)

        for idx, dir in enumerate(block_dirs):
            parts = dir.split('_')
            if not parts[0] == 'block':
                continue
            coord_x[idx] = int(parts[1][1:])
            coord_y[idx] = int(parts[2][1:])

        coord_x_to_idx = dict((val, idx) for (idx, val) in enumerate(sorted(list(set(coord_x)))))
        coord_y_to_idx = dict((val, idx) for (idx, val) in enumerate(sorted(list(set(coord_y)))))

        region_cols = len(coord_x_to_idx)
        region_rows = len(coord_y_to_idx)

        if region_cols == 0 or region_rows == 0:
            return ph

        ph['RegionRows'] = region_rows
        ph['RegionCols'] = region_cols
        ph['CFbyRegion'] = [0.0] * (region_rows*region_cols)
        ph['IEbyRegion'] = [0.0] * (region_rows*region_cols)
        ph['DRbyRegion'] = [0.0] * (region_rows*region_cols)

        # 2. Populate phasing by region

        for idx, dir in enumerate(block_dirs):
            try:
                file = open(os.path.join(dir, 'BaseCaller.json'), 'r')
                block_json = json.load(file)
                file.close()

                my_x = coord_x_to_idx.get(coord_x[idx], -1)
                my_y = coord_y_to_idx.get(coord_y[idx], -1)
                my_idx = my_y + my_x * region_rows
                if my_x < 0 or my_y < 0 or my_idx >= (region_rows*region_cols):
                    continue

                ph['CFbyRegion'][my_idx] = block_json['Phasing']['CF']
                ph['IEbyRegion'][my_idx] = block_json['Phasing']['IE']
                ph['DRbyRegion'][my_idx] = block_json['Phasing']['DR']

            except:
                print 'mergeBaseCallerJson.merge_phasing: skipping block ' + dir

        # 3. Compute average phasing

        cf = [v for v in ph['CFbyRegion'] if v > 0.0]
        ie = [v for v in ph['IEbyRegion'] if v > 0.0]
        dr = [v for v in ph['DRbyRegion'] if v > 0.0]
        ph['CF'] = sum(cf, 0.0) / len(cf)
        ph['IE'] = sum(ie, 0.0) / len(ie)
        ph['DR'] = sum(dr, 0.0) / len(dr)

    except:
        pass

    return ph


def merge(block_dirs, results_dir):
    '''mergeBaseCallerJson.merge - Combine BaseCaller.json metrics from multiple blocks'''

    combined_json = {'BeadSummary': merge_bead_summary(block_dirs),
                     'Filtering': merge_filtering(block_dirs),
                     'Phasing': merge_phasing(block_dirs)}

    file = open(os.path.join(results_dir, 'BaseCaller.json'), 'w')
    file.write(json.dumps(combined_json, indent=4))
    file.close()


if __name__ == "__main__":

    blockDirs = [name for name in os.listdir('.') if os.path.isdir(name) and name.startswith('block_')]
    resultsDir = '.'

    merge(blockDirs, resultsDir)
