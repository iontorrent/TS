# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
'''
Created on Aug 13, 2012

@author: ionadmin
'''
from django.conf.urls.defaults import patterns, url

urlpatterns = patterns('',
    url(r'^login/$', 'django.contrib.auth.views.login', {'template_name': 'rundb/login/index.html'}, name='login'),
    url(r'^login/ajax/$', 'iondb.rundb.login.views.login_ajax', {'template_name': 'rundb/login/login.html'}, name='login_ajax'),
    url(r'^logout/?$', 'iondb.rundb.login.views.logout_basic', name='logout'),
)
