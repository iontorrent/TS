# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from django.conf.urls.defaults import patterns, url

urlpatterns = patterns(
    'iondb.rundb.monitor',
    url(r'^$', 'views.monitor', name="monitor")
    )
