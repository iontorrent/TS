# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from django.conf.urls.defaults import patterns, url

urlpatterns = patterns(
    'iondb.rundb.configure',
    url(r'^$', 'views.configure', name="configure"),

    url(r'^about/$', 'views.configure_about', name="configure_about"),

    url(r'^account/$', 'views.configure_account', name="configure_account"),

    url(r'^plugins/$', 'views.configure_plugins', name="configure_plugins"),
    url(r'^plugins/plugin/install/$', 'views.configure_plugins_plugin_install', name="configure_plugins_plugin_install"),
    url(r'^plugins/plugin/(?P<pk>\d+)/configure/(?P<action>\w+)/$', 'views.configure_plugins_plugin_configure', name="configure_plugins_plugin_configure"),
    url(r'^plugins/plugin/(?P<pk>\d+)/uninstall/$', 'views.configure_plugins_plugin_uninstall', name="configure_plugins_plugin_uninstall"),
    url(r'^plugins/plugin/(?P<pk>\d+)/refresh/$', 'views.configure_plugins_plugin_refresh', name="configure_plugins_plugin_refresh"),
    url(r'^plugins/plugin/upload/zip/$', 'views.configure_plugins_plugin_zip_upload', name="configure_plugins_plugin_zip_upload"),
    url(r'^plugins/plugin/enable/(\d+)/(\d)$', 'views.configure_plugins_plugin_enable', name='configure_plugins_plugin_enable'),
    url(r'^plugins/plugin/autorun/(\d+)$', 'views.configure_plugins_plugin_autorun', name='configure_plugins_plugin_autorun'),
    
    url(r'^reportSettings/$', 'views.configure_report_data_mgmt', name="configure_report_data_mgmt"),
    url(r'^reportSettings/prunegroups/$', 'views.configure_report_data_mgmt_prunegroups', name="configure_report_data_mgmt_prunegroups"),
    url(r'^reportSettings/pruneoptions/$', 'views.configure_report_data_mgmt_pruneEdit', name="configure_report_data_mgmt_prune_detail"),
    url(r'^reportSettings/editPruneGroups/$', 'views.configure_report_data_mgmt_editPruneGroups', name='configure_report_data_mgmt_edit_prune_groups'),
    url(r'^reportSettings/removePruneGroup/(\d+)/$', 'views.configure_report_data_mgmt_remove_pruneGroup', name='configure_report_data_mgmt_remove_prune_group'),

    url(r'^references/$', 'views.configure_references', name="configure_references"),
    url(r'^references/tf/$', 'views.references_TF_edit', name='references_TF_edit'),
    url(r'^references/tf/(\d+)$', 'views.references_TF_edit', name='references_TF_edit'),
    url(r'^references/tf/(\d+)/delete/$', 'views.references_TF_delete', name='references_TF_delete'),

    url(r'^references/genome/add/$', 'genomes.new_genome', name='references_genome_add'),
    url(r'^references/genome/edit/(\d+)$', 'genomes.edit_genome', name='references_genome_edit'),
    url(r'^references/genome/delete/(\d+)$', 'genomes.delete_genome', name='references_genome_delete'),
    url(r'^references/genome/rebuild/(?P<reference_id>\w+)$', 'genomes.start_index_rebuild', name='references_genome_start_index_rebuild'),
    url(r'^references/genome/status/(\d+)$', 'genomes.genome_status', name='references_genome_status'),
    url(r'^references/genome/upload/$', 'genomes.file_upload', name='references_genome_file_upload'),

    url(r'^references/barcodeset/$', 'views.references_barcodeset_add', name="references_barcodeset_add"),
    url(r'^references/barcodeset/(?P<barCodeSetId>\d+)/$', 'views.references_barcodeset', name="references_barcodeset"),
    url(r'^references/barcodeset/(?P<barCodeSetId>\d+)/delete/$', 'views.references_barcodeset_delete', name="references_barcodeset_delete"),
    url(r'^references/barcodeset/(?P<barCodeSetId>\d+)/barcode/add/$', 'views.references_barcode_add', name="references_barcode_add"),
    url(r'^references/barcodeset/(?P<barCodeSetId>\d+)/barcode/(?P<pk>\d+)/$', 'views.references_barcode_edit', name="references_barcode_edit"),
    url(r'^references/barcodeset/(?P<barCodeSetId>\d+)/barcode/(?P<pks>[\d,]+)/delete/$', 'views.references_barcode_delete', name="references_barcode_delete"),

    url(r'^services/$', 'views.configure_services', name="configure_services"),

    url(r'^configure/$', 'views.configure_configure', name="configure_configure"),
    url(r'^configure/editemail/(\d+)?$', 'views.edit_email', name="edit_email"),

    (r'^services/arch_gone.png$', 'graphs.archive_graph_bar'),
    (r'^services/(.+)/file_server_status.png/$', 'graphs.fs_statusbar'),
    (r'^services/archive_drivespace.png$', 'graphs.archive_drivespace_bar'),
    (r'^services/residence_time.png$', 'graphs.residence_time'),
    url(r'^services/editarchive/(\d+)$', 'views.edit_backup', name="edit_archive"),
    (r'^services/expack/$', 'views.exp_ack'),
    (r'^services/storage/(\d+)/([\w.,/_\-]+)$', 'views.change_storage'),
    (r'^services/enablearchive/(\d+)/(\d)$', 'views.enableArchive'),
    url(r'^services/controljob/(\d+)/((?:term)|(?:stop)|(?:cont))$', 'views.control_job', name='control_job'),
    (r'^chips/$', 'chips.showpage'),
    url(r'^info/$', 'views.configure_system_stats', name="configure_system_stats"),
    url(r'^info/data$', 'views.configure_system_stats_data', name="configure_system_stats_data"),
    url(r'^info/SSA.zip', 'views.system_support_archive', name='configure_support_archive'),
    url(r'^services/queueStat/$', 'views.queueStatus'),
    url(r'^services/jobStat/(\d+)/$', 'views.jobStatus'),
    url(r'^services/sgejob/(\d+)/$', 'views.jobDetails'),
    (r'^getZip/(.+)$', 'chips.getChipZip'),
    (r'^getChipLog/(.+)$', 'chips.getChipLog'),
    (r'^getChipPdf/(.+)$', 'chips.getChipPdf'),
)
