# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from django.conf import settings
from django.conf.urls.defaults import *
from django.http import HttpResponsePermanentRedirect

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()
#admin.site.login_template = "rundb/login.html"

urlpatterns = patterns(
    'iondb.rundb.data',
    url(r'^$', 'views.data', name="data"),
    url(r'^table/$', 'views.data_table', name="data_table"),
    url(r'^getCSV.csv$', 'views.getCSV'),
    url(r'^projects/$', 'views.projects', name="projects"),
    url(r'^project/(\d+)/results/$', 'views.project_results', name="project_results"),
    url(r'^project/(\d+)/results/([\d,]+)/$', 'views.results_from_project', name="results_from_project"),
    url(r'^results/(?P<results_pks>[\d,]+)/project/$', 'views.results_to_project', name="results_to_project"),
    url(r'^results/(?P<results_pks>[\d,]+)/combine/project/(?P<project_pk>\d+)/$', 'views.results_to_combine', name="results_to_combine"),
#    url(r'^results/([\d,]+)/combine/sendto/project/(\d+)/(\w+)/$', 'views.combine_results_sendto_project', name="combine_results_sendto_project"),
    url(r'^project/(\d+)/$', 'views.project_view', name="project_view"),
    url(r'^project/add/$', 'views.project_add', name="project_add"),
    url(r'^project/(\d+)/edit/', 'views.project_edit', name="project_edit"),
    url(r'^project/(\d+)/delete/', 'views.project_delete', name="project_delete"),
    url(r'^project/(\d+)/log/', 'views.project_log', name="project_log"),
    url(r'^experiment/(?P<pk>\d+)/', 'views.experiment_edit', name="experiment_edit"),
    )
