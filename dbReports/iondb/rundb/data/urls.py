# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.conf.urls import patterns, url
from django.contrib.auth.decorators import login_required

from iondb.rundb.data.views import ExperimentListView, ResultsListView

urlpatterns = patterns(
    "iondb.rundb.data",
    url(r"^$", "views.data", name="data"),
    url(r"^runs/$", login_required(ExperimentListView.as_view()), name="run_list"),
    url(r"^reports/$", login_required(ResultsListView.as_view()), name="results_list"),
    url(r"^table/$", "views.data_table", name="data_table"),
    url(r"^getCSV.csv$", "views.getCSV"),
    url(r"^projects/$", "views.projects", name="projects"),
    url(r"^project/(\d+)/results/$", "views.project_results", name="project_results"),
    url(
        r"^project/(\d+)/results/([\d,]+)/$",
        "views.results_from_project",
        name="results_from_project",
    ),
    url(
        r"^results/(?P<results_pks>[\d,]+)/project/$",
        "views.results_to_project",
        name="results_to_project",
    ),
    url(
        r"^results/(?P<results_pks>[\d,]+)/combine/project/(?P<project_pk>\d+)/$",
        "views.results_to_combine",
        name="results_to_combine",
    ),
    url(
        r"^project/(?P<project_pk>\d+)/results/(?P<result_pks>[\d,]+)/getSelectedCSV.csv$",
        "views.get_project_CSV",
        name="get_project_CSV",
    ),
    # url(r'^results/([\d,]+)/combine/sendto/project/(\d+)/(\w+)/$',
    # 'views.combine_results_sendto_project',
    # name="combine_results_sendto_project"),
    url(r"^project/(\d+)/$", "views.project_view", name="project_view"),
    url(r"^project/add/$", "views.project_add", name="project_add"),
    url(r"^project/(\d+)/edit/", "views.project_edit", name="project_edit"),
    url(r"^project/(\d+)/delete/", "views.project_delete", name="project_delete"),
    url(r"^project/(\d+)/log/", "views.project_log", name="project_log"),
    url(
        r"^project/(\d+)/compare/pdf",
        "views.project_compare_pdf",
        name="project_compare_pdf",
    ),
    url(
        r"^project/(\d+)/compare/csv",
        "views.project_compare_csv",
        name="project_compare_csv",
    ),
    url(r"^project/(\d+)/compare/$", "views.project_compare", name="project_compare"),
    url(
        r"^experiment/(?P<pk>\d+)/", "views.experiment_edit", name="experiment_edit"
    ),  # TODO: SUSPECTED UNUSED DEAD CODE
    url(r"^datamanagement/$", "views.datamanagement", name="datamanagement"),
    url(
        r"^datamanagement/dm_actions/(?P<results_pks>[\d,]+)/$",
        "views.dm_actions",
        name="dm_actions",
    ),
    url(
        r"^datamanagement/dm_actions/(?P<results_pks>[\d,]+)/(?P<action>\w+)/$",
        "views.dm_action_selected",
        name="dm_action_selected",
    ),
    url(
        r"^datamanagement/dm_list_files/(?P<resultPK>[\d,]+)/(?P<action>\w+)/$",
        "views.dm_list_files",
    ),
    url(
        r"^datamanagement/preserve_data/$", "views.preserve_data", name="preserve_data"
    ),
    url(r"^datamanagement/ack/$", "views.delete_ack"),
    url(
        r"^datamanagement/dm_configuration/$",
        "views.dm_configuration",
        name="dm_configuration",
    ),
    url(r"^datamanagement/log/(\d+)/$", "views.dm_log", name="dm_log"),
    url(
        r"^datamanagement/dmconfig_log/$",
        "views.dm_configuration_log",
        name="dm_configuration_log",
    ),
    url(r"^datamanagement/dm_history/$", "views.dm_history", name="dm_history"),
    url(r"^datamanagement/(.+)/(.+)/file_server_status.png/$", "graphs.fs_statusbar"),
    url(r"^datamanagement/archive_drivespace.png/$", "graphs.archive_drivespace_bar"),
    url(r"^datamanagement/residence_time.png$", "graphs.residence_time"),
    url(
        r"^datamanagement/dmactions_jobs/$",
        "views.dmactions_jobs",
        name="dmactions_jobs",
    ),
    url(
        r"^datamanagement/dmactions_jobs/cancel/(?P<pk>\d+)/",
        "views.cancel_pending_dmaction",
        name="cancel_pending_dmaction",
    ),
    url(
        r"^datamanagement/browse_backup_dirs/(?P<path>.*)",
        "views.browse_backup_dirs",
        name="browse_backup_dirs",
    ),
    url(r"^datamanagement/import_data/", "views.import_data", name="import_data"),
    url(
        r"^datamanagement/import_data_find/(?P<path>.*)",
        "views.import_data_find",
        name="import_data_find",
    ),
    url(
        r"^datamanagement/import_data_log/(?P<path>.*)",
        "views.import_data_log",
        name="import_data_log",
    ),
    url(
        r"^exports/(?P<tag>\w*)",
        "data_export.list_export_uploads",
        name="list_export_uploads",
    ),
    url(
        r"^export/s3/", "data_export.export_upload_report", name="export_upload_report"
    ),
    url(
        r"^export/support/",
        "data_export.report_support_upload",
        name="report_support_upload",
    ),
)
