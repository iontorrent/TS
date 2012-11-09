# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from django.conf import settings
from django.conf.urls.defaults import *
from django.http import HttpResponsePermanentRedirect

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()
#admin.site.login_template = "rundb/login.html"

urlpatterns = patterns(
    'iondb.rundb.plan',
    url(r'^$', 'views.plans', name="plans"),
    url(r'^planned/$', 'views.planned', name="planned"),
    url(r'^template/add/(\d+)/$', 'views.add_plan_template', name='add_plan_template'),
    url(r'^addplan/(\d+)/$', 'views.add_plan_no_template', name='add_plan_no_template'),
    url(r'^template/(\d+)/edit/$', 'views.edit_plan_template', name='edit_plan_template'),
    url(r'^template/(\d+)/copy/$', 'views.copy_plan_template', name='copy_plan_template'),             
    url(r'^template/(?P<pks>[\d,]+)/delete/$', 'views.delete_plan_template', name='delete_plan_template'),             
    url(r'^createplanrunfromtemplate/(\d+)/$', 'views.create_planRunFromTemplate', name='create_plan_from_template'),
    url(r'^reviewtemplate/(\d+)/$', 'views.review_plan_template', name='review_plan_template'), 
    
    url(r'^reviewplan/(\d+)/$', 'views.review_plan_run', name='review_plan_run'),
    url(r'^updateeditedplannedexperiment/(\d+)/$', 'views.save_edited_plan_or_template', name='save_edited_plan_or_template'),
    url(r'^planned/(\d+)/edit/$', 'views.edit_plan_run', name='edit_plan_run'),
            
    url(r'^planned/add/$', 'views.save_plan_or_template', name='save_plan_or_template'),
    url(r'^planned/(\d+)/edit/$', 'views.edit_plan_run', name="edit_plan_run"),       
    url(r'^template/presets/$', 'views.get_application_product_presets', name="get_application_product_presets"),       
    )
