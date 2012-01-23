#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
This will write out a simple Django template for PHP to use for new style reports

NOTE: The template files created here need to be manually removed from the filesystem via the
prerm.in deb package control script.

"""

from djangoinit import *

from django import template
from django.template import loader
from iondb.rundb import models


def make_base():
    try:
        name = models.GlobalConfig.objects.get(pk=1).site_name
    except models.GlobalConfig.DoesNotExist:
        name = ""
    TEMPLATE_NAME = "rundb/ion_blank.html"
    tmpl = loader.get_template(TEMPLATE_NAME)
    c = template.Context({'tab':"reports", "base_site_name": name})
    html = tmpl.render(c)
    outfile = open('/opt/ion/iondb/templates/rundb/php_base.html', 'w')
    outfile.write(html)
    outfile.close()
    print "Wrote template for PHP for main report"


def make_plugin():
    TEMPLATE_NAME = "rundb/ion_blank_plugin.html"
    tmpl = loader.get_template(TEMPLATE_NAME)
    c = template.Context({'tab':"reports"})
    html = tmpl.render(c)
    outfile = open('/opt/ion/iondb/templates/rundb/php_base_plugin.html', 'w')
    outfile.write(html)
    outfile.close()
    print "Wrote template for PHP for plugins"

    
if __name__ == '__main__':
    make_base()
    make_plugin()
