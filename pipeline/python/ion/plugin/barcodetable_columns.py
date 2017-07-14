# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved

import copy
import json

'''
This file contains definitions for plugin barcodes table UI
'''

# dict of available columns for Kendo grid
COLUMNS = {
    "selected": {
        "field": "selected",
        "title": "Selected",
        "type":  "checkbox",
        "editable": True,
        "width": '60px',
        "description": "checkbox to select barcode row"
    },
    "barcode_name": {
        "field": "barcode_name",
        "title": "Barcode",
        "type":  "string",
        "editable": False,
        "width": '100px',
        "description": "barcode name, e.g IonXpress_001"
    },
    "sample": {
        "field": "sample",
        "title": "Sample",
        "type":  "string",
        "editable": False,
        "width": '150px',
        "description": "sample name, do not edit"
    },
    "files_BAM": {
        "field": "files_BAM",
        "title": "Select BAM",
        "type":  "checkbox",
        "editable": True,
        "width": '50px',
        "description": "checkbox to select BAM files"
    },
    "files_VCF": {
        "field": "files_VCF",
        "title": "Select VCF",
        "type":  "checkbox",
        "editable": True,
        "width": '50px',
        "description": "checkbox to select VCF files"
    },
    "reference": {
        "field": "reference",
        "title": "Reference",
        "type":  "dropdown",
        "editable": True,
        "width": '200px',
        "options": [],
        "description": "reference dropdown, if editable lists all available references"
    },
    "target_region_filepath": {
        "field": "target_region_filepath",
        "title": "Target Regions BED File",
        "type":  "dropdown",
        "editable": True,
        "width": '200px',
        "options": [],
        "description": "target BED file dropdown, if editable lists all available target files for selected reference"
    },
    "hotspot_filepath": {
        "field": "hotspot_filepath",
        "title": "Hotspot Regions BED File",
        "type":  "dropdown",
        "editable": True,
        "width": '200px',
        "options": [],
        "description": "hotspot BED file dropdown, if editable lists all available target files for selected reference"
    },
    "nucleotide_type": {
        "field": "nucleotide_type",
        "title": "DNA/RNA",
        "type":  "dropdown",
        "editable": True,
        "width": '80px',
        "options": [
            {
                "display": "DNA",
                "value": "DNA"
            },
            {
                "display": "RNA",
                "value": "RNA"
            }
        ],
        "description": "nucleotide type dropdown"
    }
}


CUSTOM_COLUMNS = {
    "custom_checkbox": {
        "field": "my_checkbox",
        "title": "My Checkbox",
        "type":  "custom_checkbox",
        "editable": True,
        "width": '80px',
        "description": "custom checkbox defined by plugin"
    },
    "custom_input": {
        "field": "my_input",
        "title": "My Checkbox",
        "type":  "custom_input",
        "editable": True,
        "width": '150px',
        "description": "custom input defined by plugin"
    },
    "custom_dropdown": {
        "field": "my_dropdown",
        "title": "My Dropdown",
        "type":  "custom_dropdown",
        "editable": True,
        "width": '150px',
        "options": [
            {
                "display": "My option",
                "value": "myoption"
            }
        ],
        "description": "custom dropdown defined by plugin, must specify options to show"
    }
}


def get_column(field_options):
    field = field_options['field']
    _type = field_options.get('type')

    if field in COLUMNS:
        column = copy.deepcopy(COLUMNS[field])
        # plugins can change editable from True to False but not vice versa
        if column['editable'] and 'editable' in field_options:
            column['editable'] = field_options['editable']
        return column

    elif _type and _type in custom_types():
        custom_column = copy.deepcopy(CUSTOM_COLUMNS[_type])
        custom_column['field'] = field
        custom_column['title'] = field_options.get('title', field)
        custom_column['editable'] = field_options.get('editable', True)
        custom_column['width'] = field_options.get('width', custom_column['width'])
        custom_column['options'] = field_options.get('options', None)
        custom_column['type'] = custom_column['type'].split('custom_')[1]
        return custom_column

    else:
        raise Exception('Specified field "%s" is not supported' % field)


def custom_types():
    return CUSTOM_COLUMNS.keys()


def available_columns():
    # returns list available columns with descriptions
    ret = []
    for column in COLUMNS.values():
        ret.append(dict((k,column[k]) for k in column if (k in column) and (k in ["field", "description", "editable"])))

    ret.sort(key= lambda v:v['field'])
    ret.extend(CUSTOM_COLUMNS.values())
    return ret

