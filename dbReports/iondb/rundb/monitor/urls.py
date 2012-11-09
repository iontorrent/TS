# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from django.conf import settings
from django.conf.urls.defaults import *
from django.http import HttpResponsePermanentRedirect

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()
#admin.site.login_template = "rundb/login.html"

urlpatterns = patterns(
    'iondb.rundb.monitor',
    url(r'^$', 'views.monitor', name="monitor")
    )
