#!/usr/bin/python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import json
import subprocess
import xml.etree.ElementTree as ET


index_filename = 'builtin_parameter_sets.json'
with open(index_filename,'r') as f:
    input_files = json.load(f)


output_json = []

output_filename = 'parameter_sets.json'
if len(sys.argv) > 1:
    output_filename = sys.argv[1]

for set_info in input_files:
    entry_filename = set_info["file"]
    print "Processing " + entry_filename
    with open(entry_filename,'r') as f:
        entry_content = json.load(f)

    entry_content["meta"]["based_on"] = entry_filename
    if "name" in set_info:
        entry_content["meta"]["name"] = set_info["name"]
    if "tooltip" in set_info:
        entry_content["meta"]["tooltip"] = set_info["tooltip"]
    if "configuration" in set_info:
        entry_content["meta"]["configuration"] = set_info["configuration"]
    if "replaces" in set_info:
        entry_content["meta"]["replaces"] = set_info["replaces"]
    if "compatibility" in set_info:
        entry_content["meta"]["compatibility"] = set_info["compatibility"]



    '''
    svn_status_xml = subprocess.Popen(['svn', 'status', '-v', '--xml', entry_filename],stdout=subprocess.PIPE).communicate()[0]
    svn_status = ET.fromstring(svn_status_xml)
    svn_details = svn_status.findall('target/entry/wc-status')[0].attrib

    revision_string = svn_details['revision']
    if svn_details['item'] != 'normal':
        revision_string += '/' + svn_details['item']

    if 'meta' not in entry_content:
        entry_content['meta'] = {}

    entry_content['meta']['revision'] = revision_string
    '''
    output_json.append(entry_content)


class PrettyFloat(float):
    def __repr__(self):
        return '%.15g' % self

def pretty_floats(obj):
    if isinstance(obj, float):
        return PrettyFloat(obj)
    elif isinstance(obj, dict):
        return dict((k, pretty_floats(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return map(pretty_floats, obj)
    return obj

print "Saving to " + output_filename
with open(output_filename,'w') as f:
    json.dump(pretty_floats(output_json),f,indent=4)










