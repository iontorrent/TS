# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
from django.conf.urls import patterns, url, include
from tastypie.api import Api

from iondb.product_integration import api

v1_api = Api(api_name='v1')
v1_api.register(api.DeepLaserResponseResource())

urlpatterns = patterns(
    'iondb.product_integration',
    url(r'^tfc/configure/$', 'views.configure', name="tfc_configure"),
    url(r'^tfc/configure/(?P<pk>\d+)/delete$', 'views.delete', name="tfc_delete"),
    url(r'^api/', include(v1_api.urls)),
)
