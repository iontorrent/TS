#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os
import argparse
import simplejson as json   # simplesjon has object_pairs_hook
import subprocess
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
from datetime import datetime

ERROR = 'error'
WARN = 'warning'
GOOD = 'good'


def get_raid_status(raidinfojson):
    '''Wrapper'''
    # Ignore the passed in values for now and use test json file
    # f = open("/tmp/raidinfo_juicebox.json", "r")
    # drive_status = get_raid_status_json(f.read())
    # return drive_status

    # Uncomment this for final version
    if raidinfojson:
        return get_raid_status_json(raidinfojson)
    else:
        return None


def load_raid_status_json(filename='/var/spool/ion/raidstatus.json'):
    '''debug function'''
    contents = {}
    try:
        with open(filename) as fhandle:
            contents = json.load(fhandle)
            contents['date'] = datetime.strptime(contents['date'], "%Y-%m-%d %H:%M:%S.%f")
    except:
        pass
    return contents


def get_raid_status_json(raidinfojson):
    '''
    This function replaces existing functionality which displays the onboard primary
    storage RAID status only.  But uses the output of the new ion_raidinfo_json
    script.
    '''
    #==========================================================================
    #==========================================================================
    def status_rules(key, value, system_name="", enclosure_name=""):
        '''Determine status from key, value pair'''
        alert_map = {
            'Foreign State': 'None',
            'Port0 status': 'Active',
            'Port1 status': 'Active'
        }
        warn_map = {
            'Media Error Count': '0',
            'Needs EKM Attention': 'No',
            'Other Error Count': '0',
            'Drive has flagged a S.M.A.R.T alert': 'No',
            'Predictive Failure Count': '0',
        }
        drive_temperature_rules = [
            # system name       enclosure     warn threshold  error threshold 
            ("default",         "default",      46.0,         55.0),
            ("PowerEdge T630",  "PERC H730",    56.0,         60.0),
        ]

        if alert_map.get(key):
            return ERROR if value != alert_map[key] else GOOD
        elif warn_map.get(key):
            return WARN if value != warn_map[key] else GOOD
        elif key == 'Firmware state':
            if 'Online' in value:
                return WARN if value != 'Online,SpunUp' else GOOD
            elif 'Hotspare' in value:
                return WARN if value != 'Hotspare,SpunUp' else GOOD
            elif 'Unconfigured' in value:
                return WARN if value != 'Unconfigured(good),SpunUp' else GOOD
            elif 'copyback' in str(value).lower():
                return WARN
            else:
                return ERROR
        elif key == 'Drive Temperature':
            match = [v for v in drive_temperature_rules if v[0] == system_name and v[1] == enclosure_name]
            if match:
                _,_,t_warn,t_err = match[0]
            else:
                _,_,t_warn,t_err = drive_temperature_rules[0]

            try:
                value = float(value.split('C')[0])
                if value > t_err:
                    return ERROR
                elif value > t_warn:
                    return WARN
                else:
                    return GOOD
            except:
                return ''    # Expect value to be "N/A" in this case
        elif key == "Slot Status":
            # This keyword exists only when the slot is empty
            return GOOD
        elif key == 'lv_status':
            if value == 'Optimal':
                return GOOD
            elif value == 'Degraded':
                return WARN
            else:
                return ERROR
        else:
            return ''

    #==========================================================================
    # Main
    #==========================================================================
    raidjson = json.loads(raidinfojson, object_pairs_hook=OrderedDict)

    # Example:
    # Return drive info for primary storage on T620: supporting existing functionality
    raid_status = []
    system_name = raidjson.get('system_name','').strip()
    for adapter in raidjson['adapters']:
        for enclosure in adapter['enclosures']:
            drive_status = []
            logical_drive_status = []
            enclosure_status = GOOD

            for drive in enclosure['drives']:
                status = GOOD
                info = []
                for key, value in drive.iteritems():
                    param_status = status_rules(key, value, system_name, adapter['id'])
                    info.append((key, value, param_status))
                    if status != ERROR and param_status and param_status != GOOD:
                        status = param_status
                # build drives array
                drive_status.append(
                    {
                        'name':             drive.get('Inquiry Data', 'NONE'),
                        'firmware_state':   drive.get('Firmware state', 'Empty'),
                        'slot':             drive.get('Slot', ''),
                        'status':           status,
                        'info':             info,
                    }
                )
                if enclosure_status != ERROR and status != GOOD:
                    enclosure_status = status

            for drive in enclosure.get('logical_drives', []):
                logical_drive_status.append(
                    {
                        'name':     drive.get('lv_name', 'unknown'),
                        'status':   drive.get('lv_status', 'unknown'),
                        'size':     drive.get('lv_size', 0),
                    }
                )

            # status array for an adapter/enclosure pair
            status_summary = \
                {
                    "adapter_id":       adapter['id'],
                    "enclosure_id":     enclosure['id'],
                    "status":           enclosure_status,
                    "drives":           drive_status,
                    "logical_drives":   logical_drive_status
                }

            # list primary storage first, so it shows up in display first
            if filter(adapter.get('id').startswith, ["PERC H710", "PERC 6/i"]):
                raid_status.insert(0, status_summary)
            else:
                raid_status.append(status_summary)

    # Dump drive_status to disk file: /var/spool/ion/raidstatus.json
    try:
        blob = {
            'date': "%s" % datetime.now(),
            'raid_status': raid_status,
        }
        with open(os.path.join('/', 'var', 'spool', 'ion', 'raidstatus.json'), 'w') as fhandle:
            json.dump(blob, fhandle, indent=4)
    except:
        pass

    return raid_status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Parse raid info from ion_raidinfo_json script''')
    parser.add_argument(dest='filename',
                        help='specify input file')
    args = parser.parse_args()

    f = open(args.filename, "r")
    result = get_raid_status_json(f.read())
    print "Number of Enclosures: %d" % len(result)
    needful_keys = [
        'adapter_id',
        'enclosure_id',
        'status',
        'logical_drives',
        'drives',
        'missing_test'
    ]
    for fight in result:
        for thiskey in needful_keys:
            found = fight.get(thiskey, "* * * MISSING * * *")
            if isinstance(found, list):
                for i, item in enumerate(found):
                    print "%s [%d]: %s" % (thiskey, i, item)
            else:
                print "%s: %s" % (thiskey, found)
