# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.conf.urls import patterns, url


urlpatterns = patterns(
    "iondb.rundb.monitor",
    url(r"^$", "views.monitor", name="monitor"),
    url(r"^instruments/$", "views.instruments", name="instruments"),
    url(r"^getSoftware/$", "views.getSoftware", name="getSoftware"),
    url(r"^chef/$", "views.chef", name="monitor_chef"),
)
