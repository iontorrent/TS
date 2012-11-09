# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from django.conf import settings
from django.conf.urls.defaults import *
from django.http import HttpResponsePermanentRedirect

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()
#admin.site.login_template = "rundb/login.html"

urlpatterns = patterns(
    'iondb.rundb.report',
    url(r'^(\d+)/$', 'views.report_display', name='report'),
    url(r'^(?P<pk>\d+)/log$', 'views.report_log', name='report_log'),
    url(r'^getPDF/(\d+).pdf/$', 'views.getPDF'),
    url(r'^analyze/(?P<exp_pk>\d+)/(?P<report_pk>\d+)/$', 'views.analyze', name="report_analyze"),
    )
