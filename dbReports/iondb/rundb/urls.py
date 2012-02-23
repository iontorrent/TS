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


v1_api.register(api.TorrentSuite())

urlpatterns = patterns(
    'iondb.rundb',
    url(r'^newanalysis/(\d+)/(\d+)$', 'views.createReport', name='createReport'),
    (r'^$', 'views.experiment'),
    (r'^reports/$', 'views.reports'),
    (r'^blank/$', 'views.blank', {'tab': False}),
    (r'^tfcsv/$', 'views.tf_csv'),
    (r'^crawler/$', 'views.crawler_status'),
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
    (r'^planning/$', 'views.planning'),
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
    (r'^upload/$', 'genomes.fileUpload'),
    (r'^mobile/runs/$', 'mobile_views.runs'),
    (r'^mobile/run/(\d+)$', 'mobile_views.run'),
    (r'^mobile/report/(\d+)$', 'mobile_views.report'),
    (r'^barcode/$', 'views.add_barcode'),
    (r'^savebarcode/$', 'views.save_barcode'),
    (r'^editbarcode/([\w.,/_\-]+)$', 'views.edit_barcode'),
    (r'^deletebarcode/(\d+)$', 'ajax.delete_barcode'),
    (r'^deletebarcodeset/([\w.,/_\-]+)$', 'ajax.delete_barcode_set'),
    (r'^addeditbarcode/([\w.,/_\-]+)$', 'views.add_edit_barcode'),
    (r'^plugininput/(\d+)/$', 'views.plugin_iframe'),
    (r'^graphiframe/(\d+)/$', 'views.graph_iframe'),
    (r'^addplan/$', 'views.add_plan'),
    (r'^addplans/$', 'views.add_plans'),
    (r'^editplan/(\d+)/$', 'views.edit_plan'),
    (r'^editexperiment/(\d+)/$', 'views.edit_experiment'),
    (r'^expack/$', 'views.exp_ack'),
    (r'^test_task/(\w+)$', 'views.test_task'),
    (r'^publish/frame/(\w+)$', 'publishers.publisher_upload', {"frame": True}),
    (r'^publish/(\w+)$', 'publishers.publisher_upload'),
    (r'^publish/$', 'publishers.upload_view'),
    (r'^published/$', 'publishers.list_content'),
    (r'^uploadstatus/(\d+)/$', 'publishers.upload_status'),
    (r'^uploadstatus/frame/(\d+)/$', 'publishers.upload_status', {"frame": True}),
    (r'how_is/(?P<host>[\w\.]+):(?P<port>\d+)/feeling$', 'views.how_are_you'),
    (r'^external_ip/$', 'views.fetch_remote_content', {"url": settings.EXTERNAL_IP_URL}),
    )

urlpatterns.extend(patterns(
            '',
            (r'^api/', include(v1_api.urls)),
))
