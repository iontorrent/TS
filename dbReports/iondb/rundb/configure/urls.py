# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

try:
    from django.conf.urls import patterns, url, include
except ImportError:
    # Compat Django 1.4
    from django.conf.urls.defaults import patterns, url, include

urlpatterns = patterns(
    'iondb.rundb.configure',
    url(r'^$', 'views.configure', name="configure"),

    url(r'^about/$', 'views.configure_about', name="configure_about"),
    url(r'^ionreporter/$', 'views.configure_ionreporter', name="configure_ionreporter"),

    url(r'^account/$', 'views.configure_account', name="configure_account"),

    url(r'^ampliseq/downloads/$', 'views.configure_ampliseq_download', name='configure_ampliseq_download'),
    url(r'^ampliseq/logout/$', 'views.configure_ampliseq_logout', name='configure_ampliseq_logout'),
    url(r'^ampliseq/(?P<pipeline>\w*)/$', 'views.configure_ampliseq', name='configure_ampliseq'),
    url(r'^ampliseq/$', 'views.configure_ampliseq', name='configure_ampliseq'),

    url(r'^plugins/$', 'views.configure_plugins', name="configure_plugins"),
    url(r'^mesh/$', 'views.configure_mesh', name="configure_mesh"),
    url(r'^plugins/plugin/install/$', 'views.configure_plugins_plugin_install', name="configure_plugins_plugin_install"),
    url(r'^plugins/plugin/(?P<pk>\d+)/configure/(?P<action>\w+)/$', 'views.configure_plugins_plugin_configure', name="configure_plugins_plugin_configure"),
    url(r'^plugins/plugin/(?P<pk>\d+)/configure/(?P<action>\w+)/pluginMedia/(?P<path>.*)$', 'views.configure_plugins_pluginmedia', name="configure_plugins_pluginmedia"),
    url(r'^plugins/plugin/(?P<pk>\d+)/uninstall/$', 'views.configure_plugins_plugin_uninstall', name="configure_plugins_plugin_uninstall"),
    url(r'^plugins/plugin/(?P<pk>\d+)/upgrade/$', 'views.configure_plugins_plugin_upgrade', name="configure_plugins_plugin_upgrade"),
    url(r'^plugins/plugin/(?P<pk>\d+)/(?P<version>[^/]+)/install_to_version/$', 'views.configure_plugins_plugin_install_to_version', name="configure_plugins_plugin_install_to_version"),
    url(r'^plugins/plugin/(?P<pk>\d+)/usage/$', 'views.configure_plugins_plugin_usage', name="configure_plugins_plugin_usage"),
    url(r'^plugins/plugin/upload/zip/$', 'views.configure_plugins_plugin_zip_upload', name="configure_plugins_plugin_zip_upload"),
    url(r'^plugins/plugin/enable/(\d+)/(\d)$', 'views.configure_plugins_plugin_enable', name='configure_plugins_plugin_enable'),
    url(r'^plugins/plugin/defaultSelected/(\d+)/(\d)$', 'views.configure_plugins_plugin_default_selected', name='configure_plugins_plugin_default_selected'),

    url(r'^plugins/publisher/install/$', 'views.configure_publisher_install', name="configure_publisher_install"),

    url(r'^references/$', 'views.configure_references', name="configure_references"),
    url(r'^references/tf/$', 'views.references_TF_edit', name='references_TF_edit'),
    url(r'^references/tf/(\d+)$', 'views.references_TF_edit', name='references_TF_edit'),
    url(r'^references/tf/(\d+)/delete/$', 'views.references_TF_delete', name='references_TF_delete'),

    url(r'^references/genome/download/$', 'genomes.download_genome', name='references_genome_download'),
    url(r'^references/genome/add/$', 'genomes.add_custom_genome', name='add_custom_genome'),
    url(r'^references/genome/upload/$', 'genomes.file_upload', name='references_genome_file_upload'),

    url(r'^references/genome/edit/(\w+)$', 'genomes.edit_genome', name='references_genome_edit'),
    url(r'^references/genome/delete/(\d+)$', 'genomes.delete_genome', name='references_genome_delete'),
    url(r'^references/genome/rebuild/(?P<reference_id>\w+)$', 'genomes.start_index_rebuild', name='references_genome_start_index_rebuild'),
    url(r'^references/genome/status/(\d+)$', 'genomes.genome_status', name='references_genome_status'),

    url(r'^references/barcodeset/$', 'views.references_barcodeset_add', name="references_barcodeset_add"),
    url(r'^references/barcodeset/(?P<barcodesetid>\d+)/$', 'views.references_barcodeset', name="references_barcodeset"),
    url(r'^references/barcodeset/(?P<barcodesetid>\d+)/delete/$', 'views.references_barcodeset_delete', name="references_barcodeset_delete"),
    url(r'^references/barcodeset/(?P<barcodesetid>\d+)/csv/$', 'views.reference_barcodeset_csv', name="references_barcodeset_csv"),
    url(r'^references/barcodeset/(?P<barcodesetid>\d+)/barcode/add/$', 'views.references_barcode_add', name="references_barcode_add"),
    url(r'^references/barcodeset/(?P<barcodesetid>\d+)/barcode/(?P<pk>\d+)/$', 'views.references_barcode_edit', name="references_barcode_edit"),
    url(r'^references/barcodeset/(?P<barcodesetid>\d+)/barcode/(?P<pks>[\d,]+)/delete/$', 'views.references_barcode_delete', name="references_barcode_delete"),

    url(r'^services/$', 'views.configure_services', name="configure_services"),
    url(r'^services/cache/$', 'views.cache_status', name='cache_status'),

    url(r'^configure/$', 'views.configure_configure', name="configure_configure"),
    url(r'^configure/editemail/(\d+)?$', 'views.edit_email', name="edit_email"),
    url(r'^configure/deleteemail/(\d+)?$', 'views.delete_email', name="delete_email"),
    url(r'^configure/auto_detect_timezone/$', 'views.auto_detect_timezone', name="auto_detect_timezone"),
    url(r'^configure/get_all_cities/(?P<zone>\w+)/$', 'views.get_all_cities', name="get_all_cities"),
    url(r'^configure/get_avail_mnts/(?P<ip>.*)/$', 'views.get_avail_mnts'),
    url(r'^configure/get_avail_mnts/([0-2]?\d{0,2}\.){3}([0-2]?\d{0,2})$/', 'views.get_avail_mnts'),
    url(r'^configure/add_nas_storage/', 'views.add_nas_storage'),
    url(r'^configure/get_current_mnts/', 'views.get_current_mnts'),
    url(r'^configure/remove_nas_storage/', 'views.remove_nas_storage'),
    url(r'^configure/get_nas_devices/', 'views.get_nas_devices'),
    url(r'^configure/check_nas_perms/', 'views.check_nas_perms'),

    url(r'^newupdates/$', 'views.offcycle_updates', name = 'offcycle_updates'),
    url(r'^newupdates/product/(.+)/(.+)/$', 'views.offcycle_updates_install_product', name = 'update_product'),
    url(r'^newupdates/package/(.+)/(.+)/$', 'views.offcycle_updates_install_package', name = 'update_package'),

    url(r'^services/controljob/(\d+)/((?:term)|(?:stop)|(?:cont))$', 'views.control_job', name='control_job'),
    (r'^chips/$', 'chips.showpage'),
    url(r'^info/$', 'views.configure_system_stats', name="configure_system_stats"),
    url(r'^info/data$', 'views.configure_system_stats_data', name="configure_system_stats_data"),
    url(r'^info/SSA.zip', 'views.system_support_archive', name='configure_support_archive'),
    url(r'^raid_info/(\d+)/$', 'views.raid_info'),
    url(r'^services/raid_info/refresh/$', 'views.raid_info_refresh', name='raid_info_refresh'),
    url(r'^timezone/$', 'util.readTimezone'),
    url(r'^services/queueStat/$', 'views.queueStatus'),
    url(r'^services/jobStat/(\d+)/$', 'views.jobStatus'),
    url(r'^services/sgejob/(\d+)/$', 'views.jobDetails'),
    url(r'^services/cluster_info/refresh/$', 'views.cluster_info_refresh', name='cluster_info_refresh'),
    url(r'^services/cluster_info/log/(?P<pk>\d+)/$', 'views.cluster_info_log', name='cluster_info_log'),
    url(r'^services/cluster_info/history/$', 'views.cluster_info_history', name='cluster_info_history'),
    url(r'^services/cluster_ctrl/(?P<name>\w+)/(?P<action>\w+)/$', 'views.cluster_ctrl', name='cluster_ctrl'),
    url(r'^services/torrent_nas_section/$', 'views.torrent_nas_section', name="torrent_nas_section"),

    (r'^getZip/(.+)$', 'chips.getChipZip'),
    (r'^getChipLog/(.+)$', 'chips.getChipLog'),
    (r'^getChipPdf/(.+)$', 'chips.getChipPdf'),
    (r'^getProtonDiags/(.+)$', 'chips.getProtonDiags'),

    url(r'^analysisargs/$', 'views.configure_analysisargs', name="configure_analysisargs"),
    url(r'^analysisargs/(?P<pk>\d+)/(?P<action>\w+)$', 'views.configure_analysisargs_action', name="configure_analysisargs_action"),
)
