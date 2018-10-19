# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# -*- coding: utf-8 -*-

from django.conf.urls import patterns, url

# Dashboard
urlpatterns = patterns(
    'iondb.rundb.home',
    url(r'^$', 'views.dashboard', name='dashboard'),
    url(r'^news/$', 'views.news', name="news"),
    url(r'^fragments$', 'views.dashboard_fragments', name='dashboard-fragments'),
)
