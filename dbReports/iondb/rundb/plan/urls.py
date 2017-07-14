# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from django.conf.urls import patterns, url
from iondb.rundb.plan.views import PlanDetailView

urlpatterns = patterns(
    'iondb.rundb.plan',
    url(r'^$', 'views.plan_run_home', name="planRuns"),
    # url(r'^plan/$', 'views.plans', name="plans"),
    url(r'^plan_templates/$', 'views.plan_templates', name="plan_templates"),
    url(r'^planned/$', 'views.planned', name="planned"),
    url(r'^reset_page_plan_session/$', 'views.reset_page_plan_session', name="reset_page_plan_session"),
    url(r'^page_plan_new_template/$', 'views.page_plan_new_template', name="page_plan_new_template"),
    url(r'^page_plan_new_template/(\d+)/$', 'views.page_plan_new_template', name="page_plan_new_template"),
    url(r'^page_plan_edit_template/(\d+)/$', 'views.page_plan_edit_template', name="page_plan_edit_template"),
    url(r'^page_plan_copy_template/(\d+)/$', 'views.page_plan_copy_template', name="page_plan_copy_template"),
    url(r'^page_plan_new_plan/(\d+)/$', 'views.page_plan_new_plan', name="page_plan_new_plan"),
    url(r'^get_ir_config/$', 'views.get_ir_config', name='get_ir_config'),
    url(r'^plan_templates/install_files/$', 'views.upload_and_install_files', name="install_files"),

    url(r'^page_plan_new_template_by_sample/([\d,]+)/$', 'views.page_plan_new_template_by_sample', name="page_plan_new_template_by_sample"),
    url(r'^page_plan_new_plan_by_sample/(\d+)/([\d,]+)$', 'views.page_plan_new_plan_by_sample', name="page_plan_new_plan_by_sample"),
    url(r'^page_plan_by_sample_barcode/$', 'views.page_plan_by_sample_barcode', name="page_plan_by_sample_barcode"),
    url(r'^page_plan_by_sample_save_plan/$', 'views.page_plan_by_sample_save_plan', name="page_plan_by_sample_save_plan"),
    url(r'^page_plan_edit_plan_by_sample/(\d+)$', 'views.page_plan_edit_plan_by_sample', name="page_plan_edit_plan_by_sample"),

    url(r'^page_plan_new_plan_from_code/(\d+)/$', 'views.page_plan_new_plan_from_code', name="page_plan_new_plan_from_code"),
    url(r'^page_plan_edit_plan/(\d+)/$', 'views.page_plan_edit_plan', name="page_plan_edit_plan"),
    url(r'^page_plan_edit_run/(\d+)/$', 'views.page_plan_edit_run', name="page_plan_edit_run"),
    url(r'^page_plan_copy_plan/(\d+)/$', 'views.page_plan_copy_plan', name="page_plan_copy_plan"),
    url(r'^page_plan_copy_plan_by_sample/(\d+)/$', 'views.page_plan_copy_plan_by_sample', name='page_plan_copy_plan_by_sample'),
    url(r'^page_plan_export/$', 'views.page_plan_export', name="page_plan_export"),
    url(r'^page_plan_application/$', 'views.page_plan_application', name="page_plan_application"),
    url(r'^page_plan_save_sample/$', 'views.page_plan_save_sample', name="page_plan_save_sample"),
    url(r'^page_plan_kits/$', 'views.page_plan_kits', name="page_plan_kits"),
    url(r'^page_plan_monitoring/$', 'views.page_plan_monitoring', name="page_plan_monitoring"),
    url(r'^page_plan_reference/$', 'views.page_plan_reference', name="page_plan_reference"),
    url(r'^page_plan_plugins/$', 'views.page_plan_plugins', name="page_plan_plugins"),
    url(r'^page_plan_output/$', 'views.page_plan_output', name="page_plan_output"),
    url(r'^page_plan_ionreporter/$', 'views.page_plan_ionreporter', name="page_plan_ionreporter"),
    url(r'^page_plan_save_template/$', 'views.page_plan_save_template', name="page_plan_save_template"),
    url(r'^page_plan_save_template_by_sample/$', 'views.page_plan_save_template_by_sample', name="page_plan_save_template_by_sample"),
    url(r'^page_plan_save_plan/$', 'views.page_plan_save_plan', name="page_plan_save_plan"),
    url(r'^page_plan_save/$', 'views.page_plan_save', name="page_plan_save"),
    url(r'^page_plan_save/(?P<exp_id>\d+)/$', 'views.page_plan_save', name="page_plan_save_and_reanalyze"),
    url(r'^page_plan_samples_table/save/$', 'views.page_plan_save_samples_table', name="page_plan_save_samples_table"),
    url(r'^page_plan_samples_table/load/$', 'views.page_plan_load_samples_table', name="page_plan_load_samples_table"),

    url(r'^template/(?P<pks>[\d,]+)/delete/$', 'views.delete_plan_template', name='delete_plan_template'),
    url(r'^export_template/(\d+)/$', 'views.plan_template_export', name="export_plan_template"),
    url(r'^import_template/$', 'views.plan_template_import', name="import_plan_template"),

    url(r'^reviewplan/(?P<pk>\d+)/$', PlanDetailView.as_view(), name='review_plan'),
    url(r'^reviewplan/(?P<pk>\d+)/(?P<report_pk>\d+)/$', PlanDetailView.as_view(), name='review_plan'),

    url(r'^transfer/(?P<pk>\d+)/(?P<destination>.*)/$', 'views.plan_transfer', name='plan_transfer'),

    url(r'^batchplanrunsfromtemplate/(\d+)/$', 'views.batch_plans_from_template', name='batch_plans_from_template'),
    url(r'^uploadplansfortemplate/$', 'views.upload_plans_for_template', name='upload_plans_for_template'),
    url(r'^saveuploadedplansfortemplate/$', 'views.save_uploaded_plans_for_template', name='save_uploaded_plans_for_template'),
    url(r'^template/(?P<templateId>\d+)/planCount/(?P<count>\d+)/getcsvforbatchplanning.csv/(?P<uploadtype>\w+)/$', 'views.getCSV_for_batch_planning', name='getCSV_for_batch_planning'),

    url(r'^toggle_template_favorite/(\d+)/$', 'views.toggle_template_favorite', name="toggle_template_favorite"),

)
