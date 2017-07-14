#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved

from __future__ import print_function
import urllib2 as u
import traceback
import json
import os

NEXENTACRED = "/etc/torrentserver/nms_access"
#NEXENTACRED = "./nms_access"

class Nexentanms(object):
    'Class for access to Nexenta Storage Devices'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Basic %s' % ''.encode('base64')[:-1]
    }

    def __init__(self, ipadd, _username, _password):
        self.url = 'http://' + ipadd + ':8457/rest/nms'
        self.headers['Authorization'] = 'Basic %s' % (_username + ':' + _password).encode('base64')[:-1]
        self.debug = False  # Print to stdout

    def get_something(self, data, verbose=True, debug=False):
        'Generic information fetching function'
        robj = u.Request(self.url, data, self.headers)
        resp = u.urlopen(robj)
        respstr = resp.read()
        result = json.loads(respstr).get('result')
        error = json.loads(respstr).get('error')

        if debug:
            print(result)

        if result and verbose:
            if isinstance(result, dict):
                for key, value in result.items():
                    print(("%30s %s" % (key, value)))
            elif isinstance(result, list):
                for item in result:
                    print(item)
            else:
                print(result)
        if error:
            print(error.get('message'))
        return result, error
    #
    # def show_license(self):
    #     'Shows license information'
    #     _data_obj = json.dumps({'object': 'appliance', 'method': 'get_license_info', 'params': []})
    #     result, error = self.get_something(_data_obj, verbose=False)
    #     if error:
    #         print error.get('message')
    #     for key in ['license_type', 'license_key', 'license_message', 'license_days_left']:
    #         print ("%30s   %s" % (key, result.get(key)))
    #

    def show_properties(self, _object='folder', _child='pool1'):
        _data_obj = json.dumps({'object': _object, 'method': 'get_child_props', 'params': [_child, '']})
        result, error = self.get_something(_data_obj, verbose=self.debug)
        if error:
            print(error.get('message'))
        if result:
            if 'config' in result:
                if self.debug:
                    for item in result.get('config'):
                        print(item)
            return result

    def get_volume_size(self, label):
        'Returns volume space information'
        props = self.show_properties('volume', label)
        return {u'available': props.get('available'),
                u'allocated': props.get('allocated'),
                u'capacity': props.get('capacity'),
                u'size': props.get('size'),
                u'free': props.get('free'),
                }

    def get_volumes(self):
        data = json.dumps({
            'object': 'volume',
            'method': 'get_all_names',
            'params': [""],
            })
        result, error = self.get_something(data, verbose=self.debug)
        if error:
            print(error.get('message'))
        return result

    def get_volume_status(self, _volume):
        'Return the status of the volume'
        _data_obj = json.dumps({'object': 'volume', 'method': 'get_status', 'params': [_volume]})
        result, error = self.get_something(_data_obj, verbose=self.debug)
        if error:
            print(error.get('message'))
        if result:
            if self.debug:
                if 'config' in result:
                    for item in result.get('config'):
                        print(item)
            return result

    def state_complete(self, ipaddress):
        'Return status of appliance'
        allvols_status = dict()
        volumes = self.get_volumes()
        if volumes:
            allvols_status.update({u'ipaddress': ipaddress})
            allvols_status.update({u'volumes': volumes})
            for volume in volumes:
                volume_dict = self.get_volume_status(volume)
                volume_dict.update(self.get_volume_size(volume))
                allvols_status.update({volume: volume_dict})
        return allvols_status

    # def blink_start(self, disk_id):
    #     'Blink the LED'
    #     data = json.dumps({'object': 'racadm',
    #                        'method': 'blink_on',
    #                        'params': ["Disk.Bay.0:Enclosure.Internal.0-1:RAID.Slot.8-1"]})
    #     _, error = self.get_something(data, debug=True)
    #     if error:
    #         print(error.get('message'))
    #
    # def blink_stop(self, disk_id):
    #     '''Stop LED blink'''
    #     data = json.dumps({'object': 'racadm',
    #                        'method': 'blink_off',
    #                        'params': ["Disk.Bay.0:Enclosure.Internal.0-1:RAID.Slot.8-1"]})
    #     _, error = self.get_something(data, debug=True)
    #     if error:
    #         print(error.get('message'))

#------------------------------------------------------------------------------
# End of class definition
#------------------------------------------------------------------------------


def get_all_torrentnas_data():
    'Returns array with each element representing a TorrentNAS unit'
    tnazzes = None
    data = list()
    errors = list()
    try:
        with open(NEXENTACRED, 'r') as fh:
            tnazzes = json.load(fh)
    except IOError as error:
        errors.append(str(error))
    else:
        for tnas in tnazzes.get('appliances'):
            torrentnas = Nexentanms(tnas.get('ipaddress'),
                                     tnas.get('username'),
                                     tnas.get('password'))
            try:
                result = torrentnas.state_complete(tnas.get('ipaddress'))
                data.append(result)
            except u.URLError as err:
                errors.append("Nexentanms, %s: %s" % (tnas.get('ipaddress'), err))
            except:
                errors.append(traceback.format_exc())
    return data, errors


def has_nexenta_cred():
    # check whether nms_access file exists
    return os.path.isfile(NEXENTACRED)


def this_is_nexenta(ipaddress):
    'Returns True if appliance is Nexenta'
    import requests
    # Nexenta appliances have management port.  If we connect, its Nexenta
    try:
        r = requests.get("http://%s:8457" % (ipaddress))
        return r.status_code == requests.codes.get('ok')
    except requests.exceptions.ConnectionError:
        return False
    except:
        return False


# def blink_the_drive(appliance_id, disk_id, disable=False):
#     '''Blink a specified drive.  Set disable=True to stop blink'''
#     tnazzes = None
#     try:
#         with open(NEXENTACRED, 'r') as fh:
#             tnazzes = json.load(fh)
#     except (OSError, ValueError) as error:
#         print(error)
#     except:
#         print(traceback.format_exc())
#         return
#     for tnas in tnazzes.get('appliances'):
#         if tnas.get('ipaddress') == appliance_id:
#             torrentnas = Nexentanms(tnas.get('ipaddress'),
#                                      tnas.get('username'),
#                                      tnas.get('password'))
#             break
#     if torrentnas:
#         if disable:
#             torrentnas.blink_stop(disk_id)
#             print("stopped blink on %s" % disk_id)
#         else:
#             torrentnas.blink_start(disk_id)
#             print("started blink on %s" % disk_id)
#     else:
#         print("Not found: %s" % appliance_id)
#     return


if __name__ == '__main__':
    import pprint
    seethis = get_all_torrentnas_data()
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(seethis)
    #import time
    #blink_the_drive('10.25.2.128', u'c0t7d1')
    #time.sleep(20)
    #blink_the_drive('10.25.2.128', u'c0t7d1', disable=True)
