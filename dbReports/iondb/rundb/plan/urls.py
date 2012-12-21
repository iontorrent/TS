# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from django.conf.urls.defaults import patterns, url

urlpatterns = patterns(
    'iondb.rundb.plan',
    url(r'^$', 'views.plans', name="plans"),
    url(r'^planned/$', 'views.planned', name="planned"),
    url(r'^template/add/(\d+)/$', 'views.add_plan_template', name='add_plan_template'),
    url(r'^addplan/(\d+)/$', 'views.add_plan_no_template', name='add_plan_no_template'),
    url(r'^template/(\d+)/edit/$', 'views.edit_plan_template', name='edit_plan_template'),
    url(r'^template/(\d+)/copy/$', 'views.copy_plan_template', name='copy_plan_template'),
    url(r'^template/(?P<pks>[\d,]+)/delete/$', 'views.delete_plan_template', name='delete_plan_template'),
    url(r'^createplanrunfromtemplate/(\d+)/$', 'views.create_plan_from_template', name='create_plan_from_template'),
    url(r'^reviewtemplate/(\d+)/$', 'views.review_plan_template', name='review_plan_template'),

    url(r'^reviewplan/(\d+)/$', 'views.review_plan_run', name='review_plan_run'),
    url(r'^planned/(\d+)/edit/$', 'views.edit_plan_run', name='edit_plan_run'),
    url(r'^planned/(\d+)/copy/$', 'views.copy_plan_run', name='copy_plan_run'),
    url(r'^save/(\d+)/$', 'views.save_plan_or_template', name='save_plan_or_template'),

    url(r'^template/presets/$', 'views.get_application_product_presets', name="get_application_product_presets"),
    
    url(r'^batchplanrunsfromtemplate/(\d+)/$', 'views.batch_plans_from_template', name='batch_plans_from_template'),
    url(r'^uploadplansfortemplate/$', 'views.upload_plans_for_template', name='upload_plans_for_template'),
    url(r'^saveuploadedplansfortemplate/$', 'views.save_uploaded_plans_for_template', name='save_uploaded_plans_for_template'),
    url(r'^template/(?P<templateId>\d+)/planCount/(?P<count>\d+)/getcsvforbatchplanning.csv/$', 'views.getCSV_for_batch_planning', name='getCSV_for_batch_planning'),
)
