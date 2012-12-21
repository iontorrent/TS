# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from django.conf.urls.defaults import patterns, url

urlpatterns = patterns(
    'iondb.rundb.report',
    url(r'^(\d+)/$', 'views.report_display', name='report'),
    url(r'^(?P<pk>\d+)/metal/(?P<path>.*)', 'views.metal', name='report_metal'),
    url(r'^(?P<pk>\d+)/log$', 'views.report_log', name='report_log'),
    url(r'^(?P<pk>\d+)/CSA.zip$', 'views.getCSA', name='report_csa'),
    url(r'^getPDF/(\d+).pdf/$', 'views.getPDF'),
    url(r'^getPlugins/(\d+).pdf/$', 'views.getPlugins'),
    url(r'^latex/(\d+).pdf/$', 'views.getlatex'),
    url(r'^analyze/(?P<exp_pk>\d+)/(?P<report_pk>\d+)/$', 'views.analyze', name="report_analyze"),

    url(r'^action/(?P<pk>\d+)/(?P<action>[\w.,/_\-]+)$', 'views.report_action', name='report_action'),
    )
