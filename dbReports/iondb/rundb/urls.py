# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
from django.conf.urls.defaults import patterns, url, include

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
v1_api.register(api.PlannedExperimentQCResource())
v1_api.register(api.EventLogResource())
v1_api.register(api.EmailAddressResource())

v1_api.register(api.ChipResource())

v1_api.register(api.AccountResource())

urlpatterns = patterns(
    'iondb.rundb',
    url(r'^$', 'data.views.rundb_redirect'),
    url(r'^old_runs$', 'views.experiment', name='old_homepage'),
    (r'^reports/$', 'views.reports'),
    url(r'^metaDataLog/(?P<pkR>.*)/$', 'views.viewLog', name="report_metadata_log"),
    (r'^getCSV.csv$', 'views.getCSV'),
    (r'^getPDF/(?P<pkR>.*)/$', 'views.PDFGen'),
    (r'^getOldPDF/(?P<pkR>.*)/$', 'views.PDFGenOld'),
    #(r'^blank/$', 'views.blank', {'tab': False}),
    (r'^tfcsv/$', 'views.tf_csv'),
    (r'^getPDF/(?P<pkR>.*)/$', 'views.PDFGen'),    
    
    (r'^islive/(\d+)$', 'ajax.analysis_liveness'),
    (r'^star/(\d+)/(\d)$', 'ajax.starRun'),
    (r'^progress_bar/(\d+)$', 'ajax.progress_bar'),
    (r'^api$', 'ajax.apibase'),
    (r'^changelibrary/(\d+)$', 'ajax.change_library'),
    (r'^reports/progressbox/(\d+)$', 'ajax.progressbox'),
    
    (r'^addplans/$', 'views.add_plans'),
    (r'^report/(\d+)$', 'views.displayReport'),
    (r'^graphiframe/(\d+)/$', 'report.classic.graph_iframe'),
     
    (r'^publish/frame/(\w+)$', 'publishers.publisher_upload', {"frame": True}),  #REFACTOR: move to rundb/configure 
    (r'^publish/api/(?P<pub_name>\w+)$', 'publishers.publisher_api_upload'),  #REFACTOR: move to rundb/configure 
    (r'^publish/plupload/(?P<pub_name>\w+)/$', 'publishers.write_plupload'),  #REFACTOR: move to rundb/configure 
    (r'^publish/(\w+)$', 'publishers.publisher_upload'),  #REFACTOR: move to rundb/configure 
    (r'^publish/$', 'publishers.upload_view'),  #REFACTOR: move to rundb/configure 
    (r'^published/$', 'publishers.list_content'),  #REFACTOR: move to rundb/configure 
    (r'^uploadstatus/(\d+)/$', 'publishers.upload_status'),  #REFACTOR: move to rundb/configure 
    (r'^uploadstatus/frame/(\d+)/$', 'publishers.upload_status', {"frame": True}),  #REFACTOR: move to rundb/configure 

    (r'^demo_consumer/?(?P<name>\w+)', 'events.demo_consumer'),

    )

urlpatterns.extend(patterns(
            '',
            (r'^api/', include(v1_api.urls)),
))
