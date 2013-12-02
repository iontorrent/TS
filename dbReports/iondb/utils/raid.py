#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os
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
    # Ignore the passed in values for now and use test json file
    #f = open("/tmp/raidinfo_juicebox.json", "r")
    #drive_status = get_raid_status_json(f.read())
    #return drive_status

    # Uncomment this for final version
    if raidinfojson:
        return get_raid_status_json(raidinfojson)
    else:
        return None


def get_raid_status_json(raidinfojson):
    '''
    This function replaces existing functionality which displays the onboard primary
    storage RAID status only.  But uses the output of the new ion_raidinfo_json
    script.
    '''
    #==========================================================================
    #==========================================================================
    def status_rules(key, value):
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
            else:
                return ERROR
        elif key == 'Drive Temperature':
            try:
                value = float(value.split('C')[0])
                if value > 55.0:
                    return ERROR
                elif value > 46.0:
                    return WARN
                else:
                    return GOOD
            except:
                return ''    # Expect value to be "N/A" in this case
        else:
            return ''

    #==========================================================================
    # Main
    #==========================================================================
    raidjson = json.loads(raidinfojson, object_pairs_hook=OrderedDict)

    # Example:
    # Return drive info for primary storage on T620: supporting existing functionality
    raid_status = []
    for adapter in raidjson['adapters']:
        for enclosure in adapter['enclosures']:
            drive_status = []
            enclosure_status = GOOD
            for drive in enclosure['drives']:
                name = "NONE"
                status = GOOD
                info = []
                firmware_state = ''
                slot = ''
                for key, value in drive.iteritems():
                    #print ("%40s%40s" % (key,value))
                    if key == "Inquiry Data":
                        name = value
                    if key == "Firmware state":
                        firmware_state = value
                    if 'Slot' in key:
                        slot = value
                    param_status = status_rules(key, value)
                    info.append((key, value, param_status))
                    if status != ERROR and param_status and param_status != GOOD:
                        status = param_status
                # build drives array
                drive_status.append({
                    'name': name,
                    'status': status,
                    'info': info,
                    'firmware_state': firmware_state,
                    'slot': slot
                })
                if enclosure_status != ERROR and status != GOOD:
                    enclosure_status = status

            # status array for an adapter/enclosure pair
            d = {
                "adapter_id": adapter['id'],
                "enclosure_id": enclosure['id'],
                "status": enclosure_status,
                "drives": drive_status
            }
            if adapter['id'].startswith("PERC H710"):
                # show primary storage on top
                raid_status.insert(0, d)
            else:
                raid_status.append(d)

    # Dump drive_status to disk file: /var/spool/ion/raidstatus.json
    try:
        blob = {
            'date': "%s" % datetime.now(),
            'raid_status': raid_status,
        }
        with open(os.path.join('/', 'var', 'spool', 'ion', 'raidstatus.json'), 'w') as f:
            json.dump(blob, f, indent=4)
    except:
        pass

    return raid_status


def get_raid_stats_json():
    raidCMD = ["/usr/bin/ion_raidinfo_json"]
    q = subprocess.Popen(raidCMD, shell=True, stdout=subprocess.PIPE)
    stdout, stderr = q.communicate()
    if q.returncode == 0:
        raid_stats = stdout
    else:
        raid_stats = None
        print('There was an error executing %s' % raidCMD[0])
    return raid_stats


if __name__ == '__main__':
    f = open("/tmp/raidinfo_juicebox.json", "r")
    foo = get_raid_status_json(f.read())
    print len(foo)
    for fight in foo:
        for key, value in fight.iteritems():
            if key == "info":
                for item in value:
                    print item
            else:
                print ("%40s%40s" % (key, value))
