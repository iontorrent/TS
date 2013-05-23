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
from django import shortcuts
from django.db import transaction


def make_22_legacy_report_template():
    # This path must remain, it is used by all default reports until 3.0
    generate = '/opt/ion/iondb/templates/rundb/php_base.html'
    try:
        gc = models.GlobalConfig.objects.values_list("site_name").get(pk=1)
        name = gc[0]
    except:
        name = "Torrent Server"
    tmpl = loader.get_template("rundb/reports/22_legacy_default_report.html")
    html = tmpl.render(template.Context({
        "base_site_name":name
    }))
    with open(generate, 'w') as outfile:
        outfile.write(html.encode('utf8'))
    print("Wrote base template for the PHP default report, versions 2.2 and earlier: " + generate)


def make_30_report_template():
    generate = '/opt/ion/iondb/templates/rundb/reports/generated/30_php_base.html'
    try:
        gc = models.GlobalConfig.objects.values_list("site_name").get(pk=1)
        name = gc[0]
    except:
        name = "Torrent Server"
    tmpl = loader.get_template("rundb/reports/30_default_report.html")
    html = tmpl.render(template.Context({
        "tab": "reports",
        "base_site_name": name,
        "global_messages": "[]"
    }))
    with open(generate, 'w') as outfile:
        outfile.write(html.encode('utf8'))
    print("Wrote base template for the PHP default report, versions 3.0 and later: " + generate)


def make_plugin():
    TEMPLATE_NAME = "rundb/ion_blank_plugin.html"
    tmpl = loader.get_template(TEMPLATE_NAME)
    c = template.Context({'tab':"reports"})
    html = tmpl.render(c)
    outfile = open('/opt/ion/iondb/templates/rundb/php_base_plugin.html', 'w')
    outfile.write(html.encode('utf8'))
    outfile.close()
    print "Wrote template for PHP for plugins"


if __name__ == '__main__':
    make_22_legacy_report_template()
    make_30_report_template()
    make_plugin()
