# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
'''
Created on Aug 13, 2012

@author: ionadmin
'''
from django.conf.urls.defaults import patterns, url
from django.views.generic.simple import direct_to_template

urlpatterns = patterns('',
    url(r'^login/$', 'iondb.rundb.login.views.remember_me_login', {'template_name': 'rundb/login/index.html'}, name='login'),
    url(r'^login/ajax/$', 'iondb.rundb.login.views.login_ajax', {'template_name': 'rundb/login/login.html'}, name='login_ajax'),
    url(r'^logout/?$', 'iondb.rundb.login.views.logout_view', name='logout'),
    url(r'^login/signup/?$', 'iondb.rundb.login.views.registration', name='signup'),
    url(r'^login/pending/?$', direct_to_template, {'template': 'rundb/login/pending.html'}, name='signup_pending')
)
