#!/bin/sh
# Copyright (C) 2013,2014 Ion Torrent Systems, Inc. All Rights Reserved

# get my public IP address
wget -q -O - checkip.dyndns.org|sed -e 's/.*Current IP Address: //' -e 's/<.*$//' > /var/spool/ion/ip.txt

# get my location and save into colon-delimited file
whois `cat /var/spool/ion/ip.txt` | grep network | sed "s/network://" > /var/spool/ion/loc.txt

