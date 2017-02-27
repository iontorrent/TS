#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved
"""Query Nexenta TorrentNAS system for data to send to Axeda"""
import hashlib
import json
import os
import sys
import urllib2 as u


def get_headers(nas):
    """Build HTTP headers from credentials"""
    try:
        user = nas['user']
        password = nas['pass']
        auth = user + ':' + password
        headers = {'Content-Type': 'application/json',
                   'Authorization': 'Basic %s' % auth.encode('base64')[:-1]
                   }
    except KeyError:
        print 'did not find user or password information in nms_access file'
        headers = {'Content-Type': 'application/json'}
    return headers


def get_something(nas, data, verbose=False, debug=False):
    """General information fetching function"""
    result = {}
    try:
        req = u.Request(nas['url'], data, get_headers(nas))
        resp = u.urlopen(req)
        respstr = resp.read()
        result = json.loads(respstr).get('result')
        error = json.loads(respstr).get('error')
        if error:
            print error.get('message')

        if debug:
            print result

        if result and verbose:
            if isinstance(result, dict):
                for key, value in result.iteritems():
                    print '%30s %s' % (key, value)
            elif isinstance(result, list):
                for item in result:
                    print item
            else:
                print result
    except KeyError:
        print 'did not find url key in nas information'
    except u.URLError:
        print 'failed to open url ' + nas['url']
    except IOError:
        print 'failed to read from url ' + nas['url']

    return result


def get_license_info(nas):
    """Get Nexenta license info"""
    data = json.dumps({
        'object': 'appliance',
        'method': 'get_license_info',
        'params': [],
        })
    return get_something(nas, data)


def get_uuid(nas):
    """Get Nexenta UUID"""
    data = json.dumps({
        'object': 'appliance',
        'method': 'get_prop',
        'params': ['uuid'],
        })
    return get_something(nas, data)


def get_properties(nas, _object='folder', _child='pool1'):
    """Retrieve the list of properties for a Nexenta object"""
    data = json.dumps({
        'object': _object,
        'method': 'get_child_props',
        'params': [_child, ''],
        })
    return get_something(nas, data)


def get_diskusage(nas, _object='folder', _child='pool1'):
    """Get data on Nexenta disk volumes"""
    data = json.dumps({
        'object': _object,
        'method': 'get_child_props',
        'params': [_child, ''],
        })
    return get_something(nas, data)


def get_volumes(nas):
    """Get list of Nexenta volumes"""
    data = json.dumps({
        'object': 'volume',
        'method': 'get_all_names',
        'params': [''],
        })
    return get_something(nas, data)


def get_disks_for_volume(nas, _input='pool1'):
    """Get list of disks in a Nexenta volume"""
    data = json.dumps({
        'object': 'volume',
        'method': 'get_luns',
        'params': [_input],
        })
    return get_something(nas, data)


def write_disk_info(nas, ofd, indices, disks, disk):
    """Write out one disk record"""
    try:
        fl1 = 'health'
        val1 = (disks.get(disk))[0]
        dat = get_properties(nas, 'lun', disk)
        fl2 = 'vendor'
        fl3 = 'product'
        fl4 = 'serial'
        fl5 = 'size'
        fmt = 'TS.Nexenta{}_vol{}_d{}: {} {}={} {}={} {}={} {}={} {}={}\n'
        ofd.write(fmt.format(nas['index'], indices['vol'],
                             indices['dsk'], disk,
                             fl1, val1,
                             fl2, dat.get(fl2),
                             fl3, dat.get(fl3),
                             fl4, dat.get(fl4),
                             fl5, dat.get(fl5)))
    except IOError:
        pass
    except IndexError:
        pass
    except KeyError:
        pass


def get_volume_info(nas, vol):
    """Retrieve information on the named volume"""
    dat = get_diskusage(nas, 'volume', vol)
    fl1 = 'health'
    volinfo = {}
    volinfo['name'] = vol
    volinfo[fl1] = dat.get(fl1)

    dat = get_diskusage(nas, 'folder', vol)
    fl2 = 'used'
    fl3 = 'available'
    volinfo[fl2] = dat.get(fl2)
    volinfo[fl3] = dat.get(fl3)

    return volinfo


def write_volume_info(nas, ofd, volinfo, indices):
    """Write out the collected information on a particular volume"""
    fl1 = 'health'
    fl2 = 'used'
    fl3 = 'available'
    fmt = 'TS.Nexenta{}_vol{}: {} {}={} {}={} {}={}\n'
    ofd.write(fmt.format(nas['index'], indices['vol'], volinfo['name'],
                         fl1, volinfo.get(fl1),
                         fl2, volinfo.get(fl2),
                         fl3, volinfo.get(fl3)))


def write_axeda_data(nas, ofd):
    """Retrieve, format, and write out Nexenta data to send to Axeda"""
    indices = {}
    dat = get_license_info(nas)

    fmt = 'TS.Nexenta{}_lic_status: {}\n'
    ofd.write(fmt.format(nas['index'], dat.get('license_message')))

    fmt = 'TS.Nexenta{}_lic_days_left: {}\n'
    ofd.write(fmt.format(nas['index'], dat.get('license_days_left')))

    fmt = 'TS.Nexenta{}_machine_sig: {}\n'
    ofd.write(fmt.format(nas['index'], dat.get('machine_sig')))

    fmt = 'TS.Nexenta{}_UUID: {}\n'
    ofd.write(fmt.format(nas['index'], get_uuid(nas)))

    vols = get_volumes(nas)
    indices['vol'] = 0
    for vol in vols:
        volinfo = get_volume_info(nas, vol)
        write_volume_info(nas, ofd, volinfo, indices)
        if vol == 'syspool':
            continue
        disks = get_disks_for_volume(nas, vol)
        indices['dsk'] = 0
        for disk in disks:
            write_disk_info(nas, ofd, indices, disks, disk)
            indices['dsk'] = indices['dsk'] + 1
        indices['vol'] = indices['vol'] + 1


def get_nexenta_credentials():
    """Read the list of Nexenta credentials from a file"""
    nas_list = []
    nexenta_credentials_file = '/etc/torrentserver/nms_access'
    try:
        with open(nexenta_credentials_file, 'r') as ifd:
            credentials = json.loads(ifd.read())
            index = 0
            appliances = credentials['appliances']
            for unit in appliances:
                try:
                    nas = {}

                    http = 'http://'
                    url = ':8457/rest/nms'
                    nas['url'] = http + unit['ipaddress'] + url
                    nas['index'] = index
                    nas['user'] = unit['username']
                    nas['pass'] = unit['password']

                    nas_list.append(nas)
                    index = index + 1
                except KeyError:
                    pass

    except IOError:
        # No credentials file, so no TorrentNAS. Nothing to do, exit now.
        sys.exit(1)

    return nas_list


def write_summary_for_axeda(axedafile):
    """Get Nexenta credentials, then get Nexenta data to send to Axeda"""
    nas_list = get_nexenta_credentials()
    try:
        with open(axedafile, 'w') as ofd:
            for nas in nas_list:
                try:
                    write_axeda_data(nas, ofd)
                except u.URLError:
                    # if we can't reach an appliance, go on to the next one
                    pass
    except IOError:
        print 'Failed to open ' + axedafile
        return False

    return True


def write_summary_checksum(axedafile, csum_file):
    """
    Compute checksum of file.

    We have to recreate the axedafile every time we query the Nexenta
    NAS, so we cannot use its creation or modification time to tell if there
    is new data to send to Axeda.  The RSMAgent reads the checksum of the file
    and compares it to the checksum it saved from the last axedafile.
    If they differ, there is new data to send.
    """

    # axedafile is about 1kB, so we can read it all at once
    try:
        md5sum = hashlib.md5(open(axedafile, 'rb').read()).hexdigest()
    except IOError:
        print 'Failed to open ' + axedafile
        return

    try:
        with open(csum_file, 'w') as ifd:
            ifd.write('{}\t{}\n'.format(axedafile, md5sum))
    except IOError:
        print 'Failed to open ' + csum_file


def catch_offline_items(prev_file, work_file, axedafile):
    """Open prev and work files"""
    try:
        with open(prev_file, 'r') as prevf:
            try:
                with open(work_file, 'r') as workf:
                    find_offline_items(prevf, workf, work_file, axedafile)
            except IOError:
                # received no data, report null for all previous keys
                workd = {}
                for line in prevf:
                    key, value = get_key_and_value(line, ':')
                    if value != 'Not present':
                        workd[key] = 'Not present'
                write_axeda_file(axedafile, workd)
                return
    except IOError:
        print 'no previous nexenta data to consider'
        try:
            os.rename(work_file, axedafile)
        except OSError:
            print 'no new nexenta data to consider'


def write_axeda_file(axedafile, datadict):
    """Write the file that Axeda will read"""
    with open(axedafile, 'w') as outf:
        for key in sorted(datadict):
            outf.write('{}: {}\n'.format(key, datadict[key]))


def find_offline_items(prevf, workf, work_file, axedafile):
    """Find items in prev file not in work file, and add them"""
    # load the previous and current data files into dictionaries
    prevd = {}
    workd = {}
    for line in prevf:
        key, value = get_key_and_value(line, ':')
        prevd[key] = value
    for line in workf:
        key, value = get_key_and_value(line, ':')
        workd[key] = value

    # for every key that is present in prevd but not work, and is not null,
    # add it to workd with a null value.  This way we can tell in Axeda when
    # equipment goes offline.
    changes_made = False
    for key in prevd:
        if key not in workd and prevd[key] != 'Not present':
            workd[key] = 'Not present'
            changes_made = True

    if changes_made:
        write_axeda_file(axedafile, workd)
    else:
        try:
            os.rename(work_file, axedafile)
        except OSError:
            # if we got into this function, we opened prev & work successfully.
            # if we get an error now, it's beyond our control.
            print 'failed to find work file even though we opened it before'


def get_key_and_value(line, delim):
    """Split a line into two pieces at the specified delimiter"""
    line = line.strip()
    key = line
    value = ''

    index = line.find(delim)
    if index != -1:
        try:
            key = line[:index]
            value = line[index+1:]
        except IndexError:
            pass

    return key.strip(), value.strip()


def main():
    """Get data from Nexenta NAS devices and write it to file with checksum """
    axedafile = '/var/spool/ion/nexenta_data.txt'
    work_file = '/var/spool/ion/nexenta_work.txt'
    prev_file = '/var/spool/ion/nexenta_prev.txt'
    csum_file = '/var/spool/ion/nexenta_md5sum.txt'

    try:
        # Save existing data file for comparison to new data file
        os.rename(axedafile, prev_file)
    except OSError:
        pass

    success = write_summary_for_axeda(work_file)
    if success:
        catch_offline_items(prev_file, work_file, axedafile)
        write_summary_checksum(axedafile, csum_file)


if __name__ == '__main__':
    main()
