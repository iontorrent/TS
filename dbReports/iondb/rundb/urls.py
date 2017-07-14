# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
try:
    from django.conf.urls import patterns, url, include
except ImportError:
    # Compat Django 1.4
    from django.conf.urls.defaults import patterns, url, include


from tastypie.api import Api

from iondb.rundb import api
from iondb.rundb import mesh_api

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
v1_api.register(api.PluginResultJobResource())
v1_api.register(api.FileServerResource())
v1_api.register(api.TFMetricsResource())
v1_api.register(api.LibMetricsResource())
v1_api.register(api.AnalysisMetricsResource())
v1_api.register(api.QualityMetricsResource())
v1_api.register(api.RunTypeResource())
v1_api.register(api.dnaBarcodeResource())
v1_api.register(api.PlannedExperimentDbResource())
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
v1_api.register(api.MonitorDataResource())

v1_api.register(api.TorrentSuite())
v1_api.register(api.NetworkResource())
v1_api.register(api.IonReporter())

v1_api.register(api.ProjectResource())
v1_api.register(api.ProjectResultsResource())
v1_api.register(api.UserResource())

v1_api.register(api.CompositeResultResource())
v1_api.register(api.CompositeExperimentResource())
v1_api.register(api.MonitorResultResource())

v1_api.register(api.ApplProductResource())
v1_api.register(api.QCTypeResource())
v1_api.register(api.PlannedExperimentQCResource())
v1_api.register(api.EventLogResource())
v1_api.register(api.EmailAddressResource())

v1_api.register(api.ChipResource())

v1_api.register(api.AccountResource())

v1_api.register(api.SampleResource())
v1_api.register(api.ExperimentAnalysisSettingsResource())
v1_api.register(api.CompositeDataManagementResource())
v1_api.register(api.DataManagementHistoryResource())
v1_api.register(api.ClusterInfoHistoryResource())

v1_api.register(api.IonChefPrepKitInfoResource())
v1_api.register(api.ActiveIonChefPrepKitInfoResource())
v1_api.register(api.ActiveIonChefLibraryPrepKitInfoResource())

v1_api.register(api.AvailableIonChefPlannedExperimentResource())
v1_api.register(api.AvailableIonChefPlannedExperimentSummaryResource())
v1_api.register(api.IonChefPlanTemplateResource())
v1_api.register(api.IonChefPlanTemplateSummaryResource())
v1_api.register(api.GetChefScriptInfoResource())

v1_api.register(api.AvailableOneTouchPlannedExperimentResource())
v1_api.register(api.AvailableOneTouchPlannedExperimentSummaryResource())
v1_api.register(api.OneTouchPlanTemplateResource())
v1_api.register(api.OneTouchPlanTemplateSummaryResource())

v1_api.register(api.AvailablePlannedExperimentSummaryResource())
v1_api.register(api.PlanTemplateSummaryResource())
v1_api.register(api.PlanTemplateBasicInfoResource())

v1_api.register(api.ApplicationGroupResource())
v1_api.register(api.SampleGroupType_CVResource())
v1_api.register(api.SampleAnnotation_CVResource())

v1_api.register(api.SampleSetResource())
v1_api.register(api.SampleSetItemResource())
v1_api.register(api.SampleSetItemInfoResource())

v1_api.register(api.SampleAttributeResource())
v1_api.register(api.SampleAttributeDataTypeResource())

v1_api.register(api.SamplePrepDataResource())

v1_api.register(api.AnalysisArgsResource())
v1_api.register(api.FlowOrderResource())
v1_api.register(api.common_CVResource())

v1_api.register(api.FileMonitorResource())
v1_api.register(api.SupportUploadResource())


v1_api.register(api.PrepopulatedPlanningSessionResource())
v1_api.register(api.IonMeshNodeResource())

v1_mesh_api = Api(api_name='v1')
v1_mesh_api.register(mesh_api.MeshCompositeExperimentResource())
v1_mesh_api.register(mesh_api.MeshPrefetchResource())

urlpatterns = patterns(
    'iondb.rundb',
    url(r'^$', 'data.views.rundb_redirect'),
    url(r'^metaDataLog/(?P<pkR>.*)/$', 'views.viewLog', name="report_metadata_log"),
    (r'^getCSV.csv$', 'views.getCSV'),
    (r'^getPDF/(?P<pkR>.*)/$', 'views.PDFGen'),
    (r'^getOldPDF/(?P<pkR>.*)/$', 'views.PDFGenOld'),
    (r'^tfcsv/$', 'views.tf_csv'),
    (r'^getPDF/(?P<pkR>.*)/$', 'views.PDFGen'),

    (r'^islive/(\d+)$', 'ajax.analysis_liveness'),
    (r'^star/(\d+)/(\d)$', 'ajax.starRun'),
    (r'^progress_bar/(\d+)$', 'ajax.progress_bar'),
    (r'^api$', 'ajax.apibase'),
    (r'^changelibrary/(\d+)$', 'ajax.change_library'),
    (r'^reports/progressbox/(\d+)$', 'ajax.progressbox'),

    (r'^report/(\d+)$', 'views.displayReport'),
    (r'^graphiframe/(\d+)/$', 'report.classic.graph_iframe'),

    (r'^publish/api/(?P<pub_name>\w+)$', 'publishers.publisher_api_upload'),  #REFACTOR: move to rundb/configure
    (r'^publish/plupload/(?P<pub_name>\w+)/$', 'publishers.write_plupload'),  #REFACTOR: move to rundb/configure
    (r'^publish/(\w+)$', 'publishers.publisher_upload'),  #REFACTOR: move to rundb/configure
    (r'^published/$', 'publishers.list_content'),  #REFACTOR: move to rundb/configure
    (r'^uploadstatus/(\d+)/$', 'publishers.upload_status'),  #REFACTOR: move to rundb/configure
    (r'^uploadstatus/frame/(\d+)/$', 'publishers.upload_status', {"frame": True}),  #REFACTOR: move to rundb/configure
    (r'^uploadstatus/download/(\d+)/$', 'publishers.upload_download'),  #REFACTOR: move to rundb/configure
    (r'^content/(\d+)/$', 'publishers.content_details'),  #REFACTOR: move to rundb/configure
    (r'^content/download/(\d+)/$', 'publishers.content_download'),  #REFACTOR: move to rundb/configure
    (r'^content/targetregions/add/$', 'publishers.content_add', {'hotspot': False}),  #REFACTOR: move to rundb/configure
    (r'^content/hotspots/add/$', 'publishers.content_add', {'hotspot': True}),  #REFACTOR: move to rundb/configure

    (r'^updateruninfo/$', 'views.updateruninfo'),

    (r'^demo_consumer/?(?P<name>\w+)', 'events.demo_consumer'),

    )

urlpatterns.extend(patterns(
            '',
            (r'^api/', include(v1_api.urls)),
            (r'^api/mesh/', include(v1_mesh_api.urls)),
))
