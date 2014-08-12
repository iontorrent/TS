#!/bin/sh
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
cd /home/ionadmin
python MultiSiteSummary.py
python Send_html_as_xml.py --htmlfile top-`date +%F`.html

