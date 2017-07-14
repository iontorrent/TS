#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved

import os
import traceback
from iondb.bin import djangoinit
from iondb.rundb.models import Plugin, Results, ReferenceGenome
from iondb.rundb.barcodedata import BarcodeSampleInfo
from iondb.rundb.plan.views_helper import dict_bed_hotspot
from ion.plugin.loader import cache
from ion.plugin.barcodetable_columns import get_column

import logging
logger = logging.getLogger(__name__)

'''
    Implementaion to generate Plugin barcodes table UI
'''

def get_columns(selection, isBarcoded=True):
    columns = []
    bedfiles = dict_bed_hotspot()

    for field_options in selection:
        column = get_column(field_options)
        field = column['field']

        if not isBarcoded and field =='barcode_name':
            continue

        if field == 'reference':
            for reference in ReferenceGenome.objects.filter(enabled=True):
                column["options"].append({
                    'value': reference.short_name,
                    'display':'%s (%s)' % (reference.short_name, reference.name),
                })

        if field == 'target_region_filepath':
            for bed in bedfiles.get('bedFiles', []):
                column["options"].append({
                    'value': bed.file,
                    'display': os.path.basename(bed.file),
                    'reference': bed.meta['reference']
                })

        if field == 'hotspot_filepath':
            for bed in bedfiles.get('hotspotFiles', []):
                column["options"].append({
                    'value': bed.file,
                    'display': os.path.basename(bed.file),
                    'reference': bed.meta['reference']
                })

        columns.append(column)

    return columns


def get_initial_data(result, columns):
    # generates data structure based on barcodes.json
    try:
        barcodesjson = BarcodeSampleInfo(result.pk, result).data()
    except:
        logger.error(traceback.format_exc())
        return []

    data = barcodesjson.values()

    for row in data:
        # select all barcodes that had specified sample name
        row['selected'] =  True if row['sample'] else False

        # add any fields not in barcodesjson
        for column in columns:
            field = column['field'] 
            if field not in row:
                row[field] = False if (column['type'] == 'checkbox') else ''

    if len(data) > 1:
        data.sort(key=lambda k: k['barcode_index'])

    return data


def get_plugin_instance(plugin):
    try:
        source_path = os.path.join(plugin.path, plugin.script)
        cache.load_module(plugin.name, source_path)
        instance = cache.get_plugin(plugin.name)()
        return instance
    except:
        logger.error(traceback.format_exc())
        raise Exception("Error loading plugin module from %s" % source_path)


def barcodes_table_for_plugin(result, plugin):
    '''
    Top function called by the view. Columns and initial data for the barcodes table are
        generated jointly by the framework and the plugin
    1) get list of columns to show from plugin
    2) combine the list with column schema defined by the framework
    3) generate barcodes.json - this contains the initial data to populate barcodes table
    4) send barcodesjson to plugin so it has a chance to update and modify if desired
    '''
    table_columns = []
    table_data = []

    if not plugin.script.endswith('.py'):
        return table_columns, table_data
    
    instance = get_plugin_instance(plugin)
    selected_columns = instance.barcodetable_columns()

    if selected_columns:
        isBarcoded = True if result.eas.barcodeKitName else False
        table_columns = get_columns(selected_columns, isBarcoded)

        # get initial data for the table then pass to plugin so it can modify if desired
        data = get_initial_data(result, table_columns)
        if data:
            planconfig = result.eas.selectedPlugins.get(plugin.name, {}).get('userInput', {})
            globalconfig = plugin.config

            data = instance.barcodetable_data(data, planconfig, globalconfig)
            if not data:
                raise Exception("Plugin did not return any table data")

            # remove extra fields
            fields = [v['field'] for v in table_columns]
            for d in data:
                table_data.append(dict((key,val) for key,val in d.items() if key in fields))

    return table_columns, table_data
