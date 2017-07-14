# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
from django.conf.urls import patterns, include
from tastypie.api import Api
from iondb.security import api

v1_api = Api(api_name='v1')
v1_api.register(api.SecureStringResource())

urlpatterns = patterns('iondb.security', (r'^api/', include(v1_api.urls)))