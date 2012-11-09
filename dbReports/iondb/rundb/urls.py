# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
from django.conf.urls.defaults import *
from django.conf import settings

from tastypie.api import Api

from iondb.rundb import api

v1_api = Api(api_name='v1')
v1_api.register(api.GlobalConfigResource())
v1_api.register(api.ExperimentResource())
v1_api.register(api.ResultsResource())
v1_api.register(api.ReferenceGenomeResource())
v1_api.register(api.ObsoleteReferenceGenomeResource())
v1_api.register(api.LocationResource())
v1_api.register(api.RigResource())
v1_api.register(api.PluginResource())
v1_api.register(api.PluginResultResource())
v1_api.register(api.FileServerResource())
v1_api.register(api.TFMetricsResource())
v1_api.register(api.LibMetricsResource())
v1_api.register(api.AnalysisMetricsResource())
v1_api.register(api.QualityMetricsResource())
v1_api.register(api.RunTypeResource())
v1_api.register(api.dnaBarcodeResource())
v1_api.register(api.PlannedExperimentResource())
v1_api.register(api.PublisherResource())
v1_api.register(api.ContentResource())
v1_api.register(api.ContentUploadResource())
v1_api.register(api.UserEventLogResource())

v1_api.register(api.KitInfoResource())
v1_api.register(api.KitPartResource())
v1_api.register(api.SequencingKitInfoResource())
v1_api.register(api.SequencingKitPartResource())
v1_api.register(api.ActiveSequencingKitInfoResource())
v1_api.register(api.ActivePGMSequencingKitInfoResource())
v1_api.register(api.ActiveProtonSequencingKitInfoResource())
v1_api.register(api.LibraryKitInfoResource())
v1_api.register(api.LibraryKitPartResource())
v1_api.register(api.ActiveLibraryKitInfoResource())
v1_api.register(api.ActivePGMLibraryKitInfoResource())
v1_api.register(api.ActiveProtonLibraryKitInfoResource())
v1_api.register(api.LibraryKeyResource())
v1_api.register(api.ThreePrimeadapterResource())
v1_api.register(api.TemplateResource())

v1_api.register(api.MessageResource())

v1_api.register(api.TorrentSuite())
v1_api.register(api.IonReporter())

v1_api.register(api.ProjectResource())
v1_api.register(api.UserResource())

v1_api.register(api.CompositeResultResource())
v1_api.register(api.CompositeExperimentResource())
v1_api.register(api.MonitorExperimentResource())

v1_api.register(api.ApplProductResource())
v1_api.register(api.QCTypeResource())
v1_api.register(api.PEMetricsResource())
v1_api.register(api.PlannedExperimentQCResource())
v1_api.register(api.EventLogResource())
v1_api.register(api.EmailAddressResource())

v1_api.register(api.ChipResource())

urlpatterns = patterns(
    'iondb.rundb',
    url(r'^newanalysis/(\d+)/(\d+)$', 'views.createReport', name='createReport'),
    url(r'^$', 'data.views.rundb_redirect'),
    url(r'^old_runs$', 'views.experiment', name='old_homepage'),
    (r'^reports/$', 'views.reports'),
    (r'^report/(\d+)/([\w.,/_\-]+)$', 'ajax.reportAction'),
    (r'^metaDataLog/(?P<pkR>.*)/$', 'views.viewLog'),
    (r'^getCSA/(\d+).zip/$', 'views.getCSA'),
    (r'^getCSV.csv$', 'views.getCSV'),
    (r'^getPDF/(?P<pkR>.*)/$', 'views.PDFGen'),
    (r'^getOldPDF/(?P<pkR>.*)/$', 'views.PDFGenOld'),
    (r'^jobDetails/$', 'views.jobDetails'),
    (r'^getZip/(.+)$', 'views.getChipZip'),
    (r'^getChipLog/(.+)$', 'views.getChipLog'),
    (r'^getChipPdf/(.+)$', 'views.getChipPdf'),
    #(r'^blank/$', 'views.blank', {'tab': False}),
    (r'^tfcsv/$', 'views.tf_csv'),
    (r'^crawler/$', 'views.crawler_status'),
    (r'^getPDF/(?P<pkR>.*)/$', 'views.PDFGen'),
    (r'^experiment/(\d+)/$', 'views.single_experiment'),
    (r'^islive/(\d+)$', 'ajax.analysis_liveness'),
    url(r'^started/(\d+)$', 'views.report_started', name='report-started'),
    url(r'^jobs/$', 'views.current_jobs', name='ion-jobs'),
    (r'^star/(\d+)/(\d)$', 'ajax.starRun'),
    (r'^progress_bar/(\d+)$', 'ajax.progress_bar'),
    (r'^edittemplate/(\d+)$', 'views.edit_template'),
    (r'^controljob/(\d+)/((?:term)|(?:stop)|(?:cont))$', 'ajax.control_job'),
    (r'^tooltip/(.+)$', 'ajax.tooltip'),
    (r'^api$', 'ajax.apibase'),
    url(r'^archive/$', 'views.db_backup', name='ion-archive'),
    (r'^editarchive/(\d+)$', 'views.edit_backup'),
    (r'^about/$', 'views.about'),
    url(r'^planning/$', 'views.planning', name='planning'),
    (r'^storage/(\d+)/([\w.,/_\-]+)$', 'ajax.change_storage'),
    (r'^changelibrary/(\d+)$', 'ajax.change_library'),
    (r'^autorunplugin/(\d+)$', 'ajax.autorunPlugin'),
    url(r'^config/$', 'views.global_config', name='ion-config'),
    (r'^editemail/(\d+)$', 'views.edit_email'),
    (r'^enableplugin/(\d+)/(\d)$', 'ajax.enablePlugin'),
    (r'^enableemail/(\d+)/(\d)$', 'ajax.enableEmail'),
    (r'^enabletestfrag/(\d+)/(\d)$', 'ajax.enableTestFrag'),
    (r'^bestruns/$', 'views.best_runs'),
    (r'^info/$', 'views.stats_sys'),
    (r'^configure/info/$', 'views.stats_sys'),
    (r'^enablearchive/(\d+)/(\d)$', 'ajax.enableArchive'),
    url(r'^servers/$', 'views.servers', name='ion-daemon'),
    (r'^servers/arch_gone.png$', 'graphs.archive_graph'),
    (r'^servers/file_server_status.png$', 'graphs.file_server_status'),
    (r'^servers/per_file_server_status.png$', 'graphs.per_file_server_status'),
    (r'^servers/archive_drivespace.png$', 'graphs.archive_drivespace'),
    (r'^servers/residence_time.png$', 'graphs.residence_time'),
    (r'^reports/progressbox/(\d+)$', 'ajax.progressbox'),
    url(r'^references/$', 'genomes.references', name='ion-references'),
    (r'^deletegenome/(\d+)$', 'genomes.delete_genome'),
    (r'^editgenome/(\d+)$', 'genomes.edit_genome'),
    (r'^genomestatus/(\d+)$', 'genomes.genome_status'),
    (r'^newgenome/$', 'genomes.new_genome'),
    (r'^rebuild_index/(?P<reference_id>\w+)$', 'genomes.start_index_rebuild'),
    (r'^upload/$', 'genomes.fileUpload'),
    (r'^upload_plugin_zip/$', 'views.pluginZipUpload'),
    (r'^mobile/runs/$', 'mobile_views.runs'),
    (r'^mobile/run/(\d+)$', 'mobile_views.run'),
    (r'^mobile/report/(\d+)$', 'mobile_views.report'),
    (r'^barcode/$', 'views.add_barcode'),
    (r'^savebarcode/$', 'views.save_barcode'),
    (r'^editbarcode/([\w.,/_\-]+)$', 'views.edit_barcode'),
    (r'^deletebarcode/(\d+)$', 'ajax.delete_barcode'),
    (r'^deletebarcodeset/([\w.,/_\-]+)$', 'ajax.delete_barcode_set'),
    (r'^addeditbarcode/([\w.,/_\-]+)$', 'views.add_edit_barcode'),
    url(r'^plugininput/(\d+)/$', 'views.plugin_iframe', name="plugin_iframe"),
    (r'^graphiframe/(\d+)/$', 'views.graph_iframe'),
    (r'^addplan/$', 'views.add_plan'),
    (r'^addplans/$', 'views.add_plans'),
    (r'^editplan/(\d+)/$', 'views.edit_plan'),
    (r'^report/(\d+)$', 'views.displayReport'),

    (r'^editexperiment/(\d+)/$', 'views.edit_experiment'),
    (r'^expack/$', 'views.exp_ack'),
    (r'^publish/frame/(\w+)$', 'publishers.publisher_upload', {"frame": True}),
    (r'^publish/api/(?P<pub_name>\w+)$', 'publishers.publisher_api_upload'),
    (r'^publish/plupload/(?P<pub_name>\w+)/$', 'publishers.write_plupload'),
    (r'^publish/(\w+)$', 'publishers.publisher_upload'),
    (r'^publish/$', 'publishers.upload_view'),
    (r'^published/$', 'publishers.list_content'),
    (r'^uploadstatus/(\d+)/$', 'publishers.upload_status'),
    (r'^uploadstatus/frame/(\d+)/$', 'publishers.upload_status', {"frame": True}),

    url(r'^account/$', 'views.account', name="account"),
    #url(r'^register/', 'views.registration', name="register"),
    (r'^chips/$', 'chips.showpage'),
    )

urlpatterns.extend(patterns(
            '',
            (r'^api/', include(v1_api.urls)),
))
