#!/bin/sh
cd /home/ionadmin
python MultiSiteSummary.py
python Send_html_as_xml.py --htmlfile top-`date +%F`.html

