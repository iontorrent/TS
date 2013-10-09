# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
'''
Created on Aug 13, 2012

@author: ionadmin
'''
try:
    from django.conf.urls import patterns, url
except ImportError:
    # Compat Django 1.4
    from django.conf.urls.defaults import patterns, url

from django.views.generic import TemplateView

urlpatterns = patterns('',
    url(r'^login/$', 'iondb.rundb.login.views.remember_me_login', {'template_name': 'rundb/login/index.html'}, name='login'),
    url(r'^login/ajax/$', 'iondb.rundb.login.views.login_ajax', {'template_name': 'rundb/login/login.html'}, name='login_ajax'),
    url(r'^logout/?$', 'iondb.rundb.login.views.logout_view', name='logout'),
    url(r'^login/signup/?$', 'iondb.rundb.login.views.registration', name='signup'),
    url(r'^login/pending/?$', TemplateView.as_view(template_name='rundb/login/pending.html'), name='signup_pending')
)
