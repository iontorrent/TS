#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved
"""look up S5 serial number from tsvm"""

import urllib2


def get_instrument_serial():
    """From tsvm, get S5 instrument serial number"""

    # ignore any proxy setting
    proxy_handler = urllib2.ProxyHandler({})
    opener = urllib2.build_opener(proxy_handler)
    urllib2.install_opener(opener)

    url = 'http://192.168.122.1/instrument/software/config/DataCollect.config'
    response = urllib2.urlopen(url)
    html = response.read()
    lines = html.split('\n')
    for line in lines:
        if 'Serial:' in line:
            return line[7:]
    return None


def main():
    """Look up S5 serial and print it out"""
    serial = get_instrument_serial()

    output_file = "/var/spool/ion/serial_number_from_s5"
    try:
        with open(output_file, "wb") as out_fp:
            out_fp.write(serial + "_tsvm")
    except IOError:
        print "Failed to open " + output_file


if __name__ == '__main__':
    main()
