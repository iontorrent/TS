# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

try:
    from django.conf.urls import patterns, url, include
except ImportError:
    # Compat Django 1.4
    from django.conf.urls.defaults import patterns, url, include

urlpatterns = patterns(
    'iondb.rundb.report',
    url(r'^(\d+)/$', 'views.report_display', name='report'),
    url(r'^(?P<pk>\d+)/metal/(?P<path>.*)', 'views.metal', name='report_metal'),
    url(r'^(?P<pk>\d+)/log$', 'views.report_log', name='report_log'),
    url(r'^(?P<pk>\d+)/CSA.zip$', 'views.getCSA', name='report_csa'),
    url(r'^getPDF/(\d+).pdf/$', 'views.get_summary_pdf'),
    url(r'^getPlugins/(\d+).pdf/$', 'views.get_plugin_pdf'),
    url(r'^latex/(\d+).pdf/$', 'views.get_report_pdf'),
    url(r'^analyze/(?P<exp_pk>\d+)/(?P<report_pk>\d+)/$', 'views.analyze', name="report_analyze"),

    url(r'^(?P<pk>\d+)/getZip/$', 'views.getZip', name="getZip"),
    url(r'^(?P<pk>\d+)/getVCF/$', 'views.getVCF', name="getVCF"),
    
    url(r'^(?P<pk>\d+)/plugin/(?P<plugin_pk>\d+)/plugin_barcodes_table$', 'views.plugin_barcodes_table', name='plugin_barcodes_table'),
)

