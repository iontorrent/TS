# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from tastypie.resources import ModelResource, Resource
from tastypie.constants import ALL, ALL_WITH_RELATIONS
from tastypie import fields
import glob

from iondb.rundb import models

from tastypie.serializers import Serializer

from django.core import serializers

import json

#auth
from tastypie.authentication import BasicAuthentication, Authentication
from tastypie.authorization import DjangoAuthorization, Authorization

#tastypie
from django.utils.encoding import force_unicode
from tastypie.bundle import Bundle
from tastypie.fields import ApiField, ToOneField, ToManyField, CharField, ApiFieldError, DictField
from tastypie.exceptions import ImmediateHttpResponse, UnsupportedFormat
from tastypie.exceptions import NotFound, BadRequest, InvalidFilterError, HydrationError, InvalidSortError

from django.contrib.auth.models import User
from tastypie.validation import Validation
from tastypie.resources import ModelResource

from tastypie.http import HttpBadRequest

from tastypie.http import *
from tastypie.utils import dict_strip_unicode_keys, trailing_slash
from tastypie.utils.mime import build_content_type

from iondb.rundb.views import barcodeData
import iondb.rundb.admin

from ion.utils.plugin_json import *
from ion.utils.TSversion import findVersions
from iondb.plugins import runner

#custom query
from django.db.models.sql.constants import QUERY_TERMS, LOOKUP_SEP
from django.db import transaction

import socket
import datetime
import logging

import httplib2

try:
    import lxml
    from lxml.etree import parse as parse_xml
    from lxml.etree import Element, tostring
except ImportError:
    lxml = None
try:
    import yaml
    from django.core.serializers import pyyaml
except ImportError:
    yaml = None

from django.conf import settings
from django.http import HttpResponse, HttpResponseNotFound
from django.conf.urls.defaults import url

import tasks

from iondb.plugins.manager import PluginManager

logger = logging.getLogger(__name__)

def field_dict(field_list):
    """will build a dict with to tell TastyPie to allow filtering by everything"""
    field_dict = {}
    field_dict = field_dict.fromkeys(field_list,ALL_WITH_RELATIONS)
    return field_dict

def JSONconvert(self, value):
    """If a CharFiled is really a Python dict, keep it as a dict """
    if value is None:
        return None
    #if it is a dict don't make it a string
    if isinstance(value, dict):
        return value

    return unicode(value)

#replace the tastypie CharField
CharField.convert = JSONconvert

class GlobalConfigResource(ModelResource):
    class Meta:
        queryset = models.GlobalConfig.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.GlobalConfig._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = Authentication()
        authorization = Authorization()


class TFMetricsResource(ModelResource):
    class Meta:
        queryset = models.TFMetrics.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.TFMetrics._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

class LibMetricsResource(ModelResource):
    class Meta:
        queryset = models.LibMetrics.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.LibMetrics._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

class AnalysisMetricsResource(ModelResource):
    class Meta:
        queryset = models.AnalysisMetrics.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.AnalysisMetrics._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

class QualityMetricsResource(ModelResource):
    class Meta:
        queryset = models.QualityMetrics.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.QualityMetrics._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

class ReportStorageResource(ModelResource):
    class Meta:
        queryset = models.ReportStorage.objects.all()

class BaseMetadataResource(ModelResource):
    """It is wrong for both exps and results to subclass from this"""


    def dispatch_metadata(self, request, **kwargs):
        return self.dispatch('metadata', request, **kwargs)

    def get_metadata(self, request, **kwargs):
        results = self.cached_obj_get(request=request, **self.remove_api_resource_names(kwargs))
        if results is None:
            return HttpGone()

        metadata = results.metaData or { }
        return self.create_response(request, metadata)

    def post_metadata(self, request, **kwargs):
        deserialized = self.deserialize(request, request.raw_post_data, format=request.META.get('CONTENT_TYPE', 'application/json'))
        data=dict_strip_unicode_keys(deserialized)
        results = self.get_object_list(request).get(pk=kwargs["pk"])
        if results is None:
            return HttpGone()

        results.metaData = results.metaData or { }
        if "remove" in data:
            for key in data["remove"]:
                results.metaData.pop(key, None)
        if "metadata" in data:            
            results.metaData.update(data["metadata"])
        results.save()
        return HttpAccepted()

    class Meta:
        metadata_allowed_methods = ['get','post']

class ResultsResource(BaseMetadataResource):

    def override_urls(self):
       urls = [
           url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/plugin%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_plugin'), name="api_dispatch_plugin"),
           url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/pluginresults%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_pluginresults'), name="api_dispatch_pluginresults"),
           url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/pluginstore%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_pluginstore'), name="api_dispatch_pluginstore"),
           url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/barcode%s$" % (self._meta.resource_name, trailing_slash()),
               self.wrap_view('dispatch_barcode'), name="api_dispatch_barcode"),
           url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/metadata%s$" % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('dispatch_metadata'), name="api_dispatch_metadata")
        ]

       return urls

    #TODO: extend the scheme so the docs can be auto generated
    def dispatch_plugin(self, request, **kwargs):
        return self.dispatch('plugin', request, **kwargs)

    # @deprecated - use get_pluginresults instead
    def get_plugin(self, request, **kwargs):
        """
        Will return a list of the html files for all the plugs that a report has
        this is going to be based on the file system output
        """
        results = self.cached_obj_get(request=request, **self.remove_api_resource_names(kwargs))
        if results is None:
            return HttpGone()

        #we want to get the dir name, so we can grab junk from the plugin_out dir
        #do we return a list of all the html?
        #do we have to provide a full url to the page?

        #when the SGE job is done it should it call a call back to let the page know it is done?

        #for now we will return the plugin state
        pluginStatus = {}
        pluginStatus["pluginState"] = results.getPluginState()

        #get the plugin files
        pluginPath = os.path.join(results.get_report_dir(),"plugin_out")

        pluginFiles = {}
        pluginDirs = [ name for name in os.listdir(pluginPath) if os.path.isdir(os.path.join(pluginPath, name)) ]
        for pluginDir in pluginDirs:
            full_path = os.path.join(pluginPath, pluginDir)
            htmlfiles = [pfile for pfile in os.listdir(full_path) if (pfile.endswith(".html") or pfile.endswith(".php"))]
            pluginDir = ''.join(pluginDir.split())[:-4]
            pluginFiles[pluginDir] = htmlfiles

        pluginStatus["pluginFiles"] = pluginFiles

        # Show results for output folders without database records
        for plugin in pluginStatus["pluginFiles"].keys():
            if plugin not in pluginStatus["pluginState"]:
                pluginStatus["pluginState"][plugin] = 'Unknown'

        return self.create_response(request, pluginStatus)

    def post_plugin(self, request, **kwargs):
        """
        Expects dict with a plugin to run, with an optional plugin config dict
        """

        deserialized = self.deserialize(request, request.raw_post_data, format=request.META.get('CONTENT_TYPE', 'application/json'))
        data=dict_strip_unicode_keys(deserialized)
        plugins = data["plugin"]

        config = {}

        if "pluginconfig" in data:
            config = data["pluginconfig"]

        result = self.get_object_list(request).get(pk=kwargs["pk"])
        if result is None:
            return HttpGone()


        #get the hostname try to get the name from global config first
        sge_master = str(socket.getfqdn())
        gc = models.GlobalConfig.objects.all()[0]
        if gc.web_root:
            net_location = gc.web_root
        else:
            #if a hostname was not found in globalconfig.webroot then use what the system reports
            net_location = 'http://%s' % sge_master

        report = str(result.reportLink)
        reportList = report.split("/")[:-1]
        reportUrl = ("/").join(reportList)

        # URL ROOT is a relative path, no hostname,
        # as there is no canonical hostname for users visiting reports
        url_root = reportUrl


        log = logging.getLogger('iondb.rundb.api.ResultsResource')

        for plugin_name in plugins:
            try:
                qs = models.Plugin.objects.filter(name=plugin_name,active=True).exclude(path='')
                if not qs:
                    return HttpGone()
                plugin_orm = None
                ## Get Latest version of plugin
                for p in qs:
                    if (not plugin_orm) or p.versionGreater(plugin_orm):
                        plugin_orm = p
                if not plugin_orm:
                    return HttpGone()
            except:
                return HttpGone()

            plugin_output_dir = os.path.join(result.get_report_dir(),"plugin_out", plugin_orm.name + "_out")

            #get the plan json
            plan_json = models.PlannedExperiment.objects.filter(planShortID=result.planShortID())
            if plan_json:
                plan_json = serializers.serialize("json", plan_json)
                plan_json = json.loads(plan_json)
                plan_json = plan_json[0]["fields"]
            else:
                plan_json = {}

            #this will have to be built, at least in part for all the plugins sent in the list
            sigproc_results = "."
            basecaller_results = "."
            alignment_results = "."

            env={
                'pathToRaw':result.experiment.unique,
                'analysis_dir':result.get_report_dir(),
                'libraryKey':result.experiment.libraryKey,
                'results_dir' : plugin_output_dir,
                'net_location': net_location,
                'master_node': sge_master,
                'testfrag_key':'ATCG',
                'plan': plan_json,
                'tmap_version': settings.TMAP_VERSION
            }

            # if thumbnail
            is_thumbnail = result.metaData.get("thumb",False)
            if is_thumbnail:
                env['pathToRaw'] =  os.path.join(env['pathToRaw'],'thumbnail')

            plugin={
                'name':plugin_orm.name,
                'path':plugin_orm.path
            }

            start_json = make_plugin_json(env,plugin,result.pk,"plugin_out",url_root)

            # Override plugin config with instance config
            start_json["pluginconfig"].update(config)

            # Set Started status before launching to avoid race condition
            # Set here so it appears immediately in refreshed plugin status list
            (pluginresult, created) = result.pluginresult_set.get_or_create(plugin=plugin_orm)
            pluginresult.state = 'Started'
            pluginresult.save()

            launcher = runner.PluginRunner()
            ret = launcher.callPluginXMLRPC(start_json)

            if not ret:
                logger.error('Unable to launch plugin: %s', plugin_orm.name) # See ionPlugin.log for details
                pluginresult.state = 'Error'
                pluginresult.save()

        return HttpAccepted()

    def put_plugin(self, request, **kwargs):
        """
        Set the status of the plugins.
        This allow a way to see if the plugin was ran successfully.
        """

        deserialized = self.deserialize(request, request.raw_post_data, format=request.META.get('CONTENT_TYPE', 'application/json'))
        data=dict_strip_unicode_keys(deserialized)
        results = self.get_object_list(request).get(pk=kwargs["pk"])
        if results is None:
            return HttpGone()
        #we are only doing to use the first value of the dict
        if len(data) != 1:
            return HttpBadRequest()

        try:
            with transaction.commit_on_success():
                for (key, value) in data.items():
                    # Note: If multiple plugins exist, use one with last install date.
                    # FIXME Must uniquely identify plugins
                    plugin = models.Plugin.objects.get(name=key,active=True)
                    (pluginresult, created) = results.pluginresult_set.get_or_create(plugin=plugin)
                    pluginresult.state = value
                    pluginresult.save()
        except:
            logger.exception('Failed plugin state update')
            return HttpBadRequest()

        return HttpAccepted()


    def dispatch_pluginresults(self, request, **kwargs):
        """ Uses pluginresults_allowed_methods,
            Passes GET to get_pluginresults, passes post and put to %_plugin """
        if request.method.lower() in ['get']:
            return self.dispatch('pluginresults', request, **kwargs)
        # else:
        return self.dispatch('plugin', request, **kwargs)

    def get_pluginresults(self, request, **kwargs):

        results = self.cached_obj_get(request=request, **self.remove_api_resource_names(kwargs))
        if results is None:
            return HttpGone()

        # In the order they were generated, newest plugin entry first.
        pluginresults = results.pluginresult_set.all().order_by("-id", '-plugin')
        plugin_out_path = os.path.join(results.get_report_dir(),"plugin_out")

        # Inverse of model.Plugin.versionedName()
        #vtail= re.compile(r'(?P<name>.*?)--v(?P<version>[\d\.\_\-\w]+)_out')
        pluginDirs={}
        for d in os.listdir(plugin_out_path):
            full_path = os.path.join(plugin_out_path, d)
            if not os.path.isdir(full_path):
                logger.warn("Unexpected file in top level of plugin_out: %s", full_path)
                continue
            if d in ('.', '..'):
                continue
            htmlfiles = [pfile for pfile in os.listdir(full_path) if (pfile.endswith(".html") or pfile.endswith(".php"))]

            key = ''.join(d.split())[:-4] # get rid of _out suffix
            name = key ## temporary - parse out version
            ## (name, version, path, files)
            pluginDirs[key] = (name, None, d, htmlfiles)

        # Iterate through DB results first, find matching directories
        pluginArray = []
        for pr in pluginresults:
            logger.debug('Got pluginresult for %s v%s', pr.plugin.name, pr.plugin.version)
            data = {
                'Name': pr.plugin.name,
                'Version': pr.plugin.version,
                'State': pr.state,
                'Path': None,
                'Files': [],
            }
            # Add pr.plugin.versionedName to search..
            candidates = [ "%s--v%s" % (pr.plugin.name, pr.plugin.version),
                           pr.plugin.name,
                         ]
            for plugin_out in candidates:
                if plugin_out in pluginDirs:
                    (path_name, path_version, path, files) = pluginDirs[plugin_out]
                    if path_version and path_version != pr.plugin.version:
                        # ~-~ these aren't the results you are looking for ~-~
                        continue
                    del pluginDirs[plugin_out] # consume
                    data.update({'Files': files, 'Path': path})
                    break # skip else block
            else:
                # Plugin in DB with no data on filesystem!
                logger.info("Plugin %s v%s has no plugin_out folder", pr.plugin.name, pr.plugin.version)
                pass     # FIXME

            pluginArray.append(data)

        # Now - directories found which don't match plugin result entries
        # Sort by name
        for key, plugininfo in iter(sorted(pluginDirs.items())):
            (name, version, path, files) = plugininfo
            # Fixme - check startplugin.json too
            version = version or '?'
            data = {
                'Name': name,
                'Version': version,
                'State': 'Unknown',
                'Files': files,
                'Path': path,
            }
            logger.info("Plugin folder with no db record: %s v%s at '%s'", name, version, path)
            pluginArray.append(data)

        return self.create_response(request, pluginArray)

    #TODO: extend the scheme so the docs can be auto generated
    def dispatch_pluginstore(self, request, **kwargs):
        """ Uses Meta::pluginstore_allowed_methods, and dispatches to <method>_pluginstore """
        return self.dispatch('pluginstore', request, **kwargs)

    def get_pluginstore(self, request, **kwargs):
        """
        Returns pluginStore for all plugins
        """
        results = self.cached_obj_get(request=request, **self.remove_api_resource_names(kwargs))
        if results is None:
            return HttpGone()

        pluginStore = results.getPluginStore()
        return self.create_response(request, pluginStore)

    def put_pluginstore(self, request, **kwargs):
        """
        Set the store of the plugins. - results.json
        """
        deserialized = self.deserialize(request, request.raw_post_data, format=request.META.get('CONTENT_TYPE', 'application/json'))
        data=dict_strip_unicode_keys(deserialized)
        results = self.get_object_list(request).get(pk=kwargs["pk"])
        if results is None:
            return HttpGone()

        try:
            with transaction.commit_on_success():
                ## All or nothing within transaction. Exceptions will rollback changes
                for (key, value) in data.items():
                    plugin = models.Plugin.objects.get(name=key,active=True)
                    (pluginresult, created) = results.pluginresult_set.get_or_create(plugin=plugin)
                    pluginresult.store = value
                    pluginresult.save()
        except:
            logger.exception('Failed plugin state update')
            return HttpBadRequest()

        return HttpAccepted()

    def dispatch_barcode(self, request, **kwargs):
        return self.dispatch('barcode', request, **kwargs)

    def get_barcode(self, request, **kwargs):
        metric = request.GET.get('metric',False)
        result = self.get_object_list(request).get(pk=kwargs["pk"])

        if result is None:
            return HttpGone()

        barcodeSummary = "alignment_barcode_summary.csv"
        barcode = barcodeData(os.path.join(result.get_report_dir(),barcodeSummary),metric)
        if not barcode:
            return HttpBadRequest()

        return self.create_response(request, barcode)

    filesystempath = CharField('get_report_dir')
    bamLink = CharField('bamLink')
    planShortID = CharField('planShortID')
    experimentReference = CharField('experimentReference')
    libmetrics = ToManyField(LibMetricsResource, 'libmetrics_set', full=False)
    tfmetrics = ToManyField(TFMetricsResource, 'tfmetrics_set', full=False)
    analysismetrics = ToManyField(AnalysisMetricsResource, 'analysismetrics_set', full=False)
    qualitymetrics = ToManyField(QualityMetricsResource, 'qualitymetrics_set', full=False)
    reportstorage = fields.ToOneField(ReportStorageResource, 'reportstorage', full=True)

    #parent experiment
    experiment = fields.ToOneField('iondb.rundb.api.ExperimentResource', 'experiment', full=False, null = True)

    # Nested plugin results - replacement for pluginStore pluginState
    pluginresults = ToManyField('iondb.rundb.api.PluginResultResource', 'pluginresult_set', related_name='result', full=False)
    # Add back pluginState/pluginStore for compatibility
    # But using pluginresults should be preferred.
    pluginState = DictField(readonly=True)
    pluginStore = DictField(readonly=True)

    # Only dehydrate State/Store if we are retrieving a full path
    def dehydrate_pluginState(self, bundle):
        return bundle.obj.getPluginState()
    def dehydrate_pluginStore(self, bundle):
        return bundle.obj.getPluginStore()

    # NOTE: dehydrate bundle doesn't have request (which would allow us to suppress data above).
    # So we remove the plugin data prior to sending. bundle has request in post 0.9.11 tastypie
    def alter_detail_data_to_serialize(self, request, bundle):
        if request.POST.get('noplugin', False) or request.GET.get('noplugin', False):
            del bundle.data['pluginStore']
            del bundle.data['pluginState']
        return bundle

    def alter_list_data_to_serialize(self, request, data):
        if request.POST.get('noplugin', False) or request.GET.get('noplugin', False):
            for bundle in data['objects']:
                del bundle.data['pluginStore']
                del bundle.data['pluginState']
        return data

    class Meta:
        queryset = models.Results.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.Results._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        #this should check for admin rights
        authentication = Authentication()
        authorization = Authorization()

        plugin_allowed_methods = ['get', 'post', 'put']
        pluginresults_allowed_methods = ['get']
        pluginstore_allowed_methods = ['get','put']
        barcode_allowed_methods = ['get']
        metadata_allowed_methods = ['get','post']

class ExperimentResource(BaseMetadataResource):

    def override_urls(self):
        urls = [
                url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/metadata%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_metadata'), name="api_dispatch_metadata"),
                url(r"^(?P<resource_name>%s)/projects%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_projects'), name="api_dispatch_projects")
        ]

        return urls
    results = fields.ToManyField(ResultsResource, 'results_set', full=False)
    runtype = CharField('runtype')

    def dispatch_projects(self, request, **kwargs):
        return self.dispatch('projects', request, **kwargs)

    def get_projects(self, request, **kwargs):
        projectList = models.Experiment.objects.values_list('project').distinct()
        status = [project[0] for project in projectList]
        return self.create_response(request, status)


    def hydrate_runtype(self, bundle):

        if bundle.data.get("runtype",False):
            if bundle.data.get("log",False):
                del bundle.data["log"]["runtype"]
            bundle.obj.log["runtype"] = bundle.data["runtype"]
        return bundle

    class Meta:
        queryset = models.Experiment.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.Experiment._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        projects_allowed_methods = ['get']

        authentication = Authentication()
        authorization = Authorization()

class ReferenceGenomeResource(ModelResource):
    class Meta:
        limit = 0
        queryset = models.ReferenceGenome.objects.filter(enabled=True, index_version = settings.TMAP_VERSION)

        #allow ordering and filtering by all fields
        field_list = models.ReferenceGenome._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = Authentication()
        authorization = Authorization()


class LocationResource(ModelResource):
    class Meta:
        queryset = models.Location.objects.all()

        #allow ordering and filtering by all fields
        field_list = ['name','comment']
        ordering = field_list
        filtering = field_dict(field_list)


class FileServerResource(ModelResource):
    class Meta:
        queryset = models.FileServer.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.FileServer._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

class RigResource(ModelResource):

    def override_urls(self):
        urls = [url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/status%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_status'), name="api_dispatch_status"),
                url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/config%s$" % (self._meta.resource_name, trailing_slash()),
                            self.wrap_view('dispatch_config'), name="api_dispatch_config")
        ]

        return urls

    def dispatch_status(self, request, **kwargs):
        return self.dispatch('status', request, **kwargs)

    def get_status(self, request, **kwargs):
        rig = self.cached_obj_get(request=request, **self.remove_api_resource_names(kwargs))
        if rig is None:
            return HttpGone()

        status_fields = ["state", "last_init_date", "last_clean_date", "last_experiment","version","alarms"]
        status = {}

        for field in status_fields:
            status[field] = getattr(rig,field)

        return self.create_response(request, status)

    def put_status(self, request, **kwargs):
        """
        Set the status of the Rigs
        """

        deserialized = self.deserialize(request, request.raw_post_data, format=request.META.get('CONTENT_TYPE', 'application/json'))
        status = dict_strip_unicode_keys(deserialized)
        rig = self.get_object_list(request).get(pk=kwargs["pk"])

        if rig is None:
            return HttpBadRequest()

        status_fields = ["state", "last_init_date", "last_clean_date", "last_experiment","version","alarms"]
        for field in status_fields:
            if field in status:
                setattr(rig,field,status[field])

        rig.save()

        return HttpAccepted()

    def dispatch_config(self, request, **kwargs):
        return self.dispatch('config', request, **kwargs)

    def get_config(self, request, **kwargs):
        rig = self.cached_obj_get(request=request, **self.remove_api_resource_names(kwargs))
        if rig is None:
            return HttpGone()

        config_fields = ["ftppassword", "ftprootdir", "ftpserver", "ftpusername","updateflag"]
        config = {}

        for field in config_fields:
            config[field] = getattr(rig,field)

        return self.create_response(request, config)

    location = fields.ToOneField(LocationResource, 'location', full=True)
    
    class Meta:
        queryset = models.Rig.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.Rig._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        
        authentication = Authentication()
        authorization = Authorization()
        status_allowed_methods = ['get', 'put']
        config_allowed_methods = ['get',]

class PluginResource(ModelResource):
    """Get a list of plugins"""

    def override_urls(self):
        #this is meant for internal use only
        urls = [url(r"^(?P<resource_name>%s)/set/(?P<keys>\w[\w/-;]*)/type%s$" % (self._meta.resource_name,
                    trailing_slash()),self.wrap_view('get_type_set'), name="api_get_type_set"),
                url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/type%s$" % (self._meta.resource_name,
                    trailing_slash()), self.wrap_view('dispatch_type'), name="api_dispatch_type"),
                url(r"^(?P<resource_name>%s)/install%s$" % (self._meta.resource_name,
                    trailing_slash()), self.wrap_view('dispatch_install'), name="api_dispatch_install"),
                url(r"^(?P<resource_name>%s)/uninstall/(?P<pk>\w[\w/-]*)%s$" % (self._meta.resource_name,
                    trailing_slash()), self.wrap_view('dispatch_uninstall'), name="api_dispatch_uninstall")
        ]

        return urls

    def dispatch_type(self, request, **kwargs):
        return self.dispatch('type', request, **kwargs)

    def dispatch_install(self, request, **kwargs):
        return self.dispatch('install', request, **kwargs)

    def dispatch_uninstall(self, request, **kwargs):
        return self.dispatch('uninstall', request, **kwargs)

    def _get_type(self, plugin):
        if plugin is None:
            return HttpGone()

        try:
            files =  os.listdir(plugin.path)
        except:
            return HttpGone()

        type = {}

        type["files"] = files

        #provide a link to load the plugins html
        if "instance.html" in files:
            type["input"] = "/rundb/plugininput/" + str(plugin.pk) + "/"

        return type

    def get_type(self, request, **kwargs):
        plugin = self.cached_obj_get(request=request, **self.remove_api_resource_names(kwargs))
        return self.create_response(request, self._get_type(plugin))

    def get_type_set(self, request, **kwargs):
        """Take a list of ; separated plugin IDs and return the type for each.
        In the event that one of them is missing, raise an error.
        """
        request_method = self.method_check(request, ['get']) # raises MethodNotAllowed
        plugin_set = kwargs['keys'].split(";")
        queryset = self.cached_obj_get_list(request).filter(pk__in=plugin_set)
        types = dict((p.pk, self._get_type(p)) for p in queryset)
        if any(isinstance(t, HttpGone) for t in types.values()):
            return HttpGone()
        else:
            return self.create_response(request, types)

    def post_install(self, request, **kwargs):
        try:
            deserialized = self.deserialize(request, request.raw_post_data, format=request.META.get('CONTENT_TYPE', 'application/json'))
            data=dict_strip_unicode_keys(deserialized)
        except UnsupportedFormat:
            return HttpBadRequest()

        # Assumes plugin_name is the same as last component of url. Questionable.
        url  = data.get("url","")
        filename = data.get("file",False)

        if url:
            plugin_name = os.path.splitext(os.path.basename(url))[0]
        elif filename:
            plugin_name = os.path.splitext(os.path.basename(filename))[0]
        else:
            # Must specify either url or filename
            return HttpBadRequest()

        # Create mock plugin object to pass details to downloadPlugin.
        # Do not save(), as we do not know if this is an install or upgrade yet.
        # And we don't know the version number.
        plugin = models.Plugin(name=plugin_name,
                               version="0",
                               date=datetime.datetime.now(),
                               active=False,
                               url=url,
                               status={'result': 'queued'},
                               )

        #now install the plugin
        tasks.downloadPlugin.delay(url, plugin, filename)

        # Task run async - return success
        return HttpAccepted()

    def delete_uninstall(self, request, **kwargs):
        obj = self.cached_obj_get(request=request, **self.remove_api_resource_names(kwargs))

        pm = PluginManager()
        killed = pm.uninstall(obj)

        status = {"Status": killed}

        return self.create_response(request, status)


    class Meta:
        # Note API only sees active plugins
        queryset = models.Plugin.objects.filter(active=True)

        #allow ordering and filtering by all fields
        field_list = models.Plugin._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = Authentication()
        authorization = Authorization()
        type_allowed_methods = ['get',]
        install_allowed_methods = ['post',]
        uninstall_allowed_methods = ['delete']

class PluginResultResource(ModelResource):
    result = fields.ToOneField(ResultsResource,'result')
    plugin = fields.ToOneField(PluginResource, 'plugin', full=True)

    class Meta:
        queryset = models.PluginResult.objects.all()
        #allow ordering and filtering by all fields
        field_list = models.PluginResult._meta.get_all_field_names()
        field_list.extend(['result','plugin']),
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = Authentication()
        authorization = Authorization()

class RunTypeResource(ModelResource):

    class Meta:
        queryset = models.RunType.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.RunType._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = Authentication()
        authorization = Authorization()

class dnaBarcodeResource(ModelResource):

    class Meta:
        queryset = models.dnaBarcode.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.dnaBarcode._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = Authentication()
        authorization = Authorization()


class PlannedExperimentValidation(Validation):
    def is_valid(self, bundle, request=None):
        if not bundle.data:
            return {'__all__': 'Fatal Error, no bundle!'}

        errors = {}

        for key, value in bundle.data.items():
            if not isinstance(value, basestring):
                continue

            if key == "notes":
                if not set(value).issubset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.- "):
                    errors[key] = "That Report name has invalid characters. The valid values are letters, numbers, underscore and period."

        return errors

class PlannedExperimentResource(ModelResource):

    class Meta:
        queryset = models.PlannedExperiment.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = Authentication()
        authorization = Authorization()
        validation = PlannedExperimentValidation()

class TorrentSuite(ModelResource):
    """Allow system updates via API"""

    def override_urls(self):
        urls = [url(r"^(?P<resource_name>%s)%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_update'), name="api_dispatch_update")
        ]

        return urls

    def dispatch_update(self, request, **kwargs):
        return self.dispatch('update', request, **kwargs)

    def get_update(self, request, **kwargs):

        updateStatus = {}
        versions, meta_version = findVersions()
        updateStatus["versions"] = versions
        updateStatus["meta_version"] = meta_version

        updateStatus["locked"] = iondb.rundb.admin.update_locked()
        updateStatus["logs"] = iondb.rundb.admin.install_log_text()

        return self.create_response(request, updateStatus)

    def put_update(self, request, **kwargs):

        updateStatus = {}
        updateStatus["locked"] = iondb.rundb.admin.run_update()
        updateStatus["logs"] = iondb.rundb.admin.install_log_text()

        return self.create_response(request, updateStatus)

    class Meta:

        authentication = Authentication()
        authorization = Authorization()
        update_allowed_methods = ['get', 'put']


class PublisherResource(ModelResource):
    
    class Meta:
        queryset = models.Publisher.objects.all()

        # allow ordering and filtering by all fields
        field_list = models.Publisher._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = Authentication()
        authorization = Authorization()

    def override_urls(self):
        return [
            url(r"^(?P<resource_name>%s)/(?P<name>[\w\d_.-]+)/$" % \
                self._meta.resource_name,
                self.wrap_view('dispatch_detail'),
                name="api_dispatch_detail")
        ]

    def get_resource_uri(self, bundle_or_obj):
        """Handles generating a resource URI for a single resource.

        Uses the publisher's ``name`` in order to create the URI.
        """
        kwargs = {
            'resource_name': self._meta.resource_name,
        }

        if isinstance(bundle_or_obj, Bundle):
            kwargs['pk'] = bundle_or_obj.obj.name
        else:
            kwargs['pk'] = bundle_or_obj.name

        if self._meta.api_name is not None:
            kwargs['api_name'] = self._meta.api_name

        return self._build_reverse_url("api_dispatch_detail", kwargs=kwargs)


class ContentUploadResource(ModelResource):

    class Meta:
        queryset = models.ContentUpload.objects.all()

        # allow ordering and filtering by all fields
        field_list = models.ContentUpload._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = Authentication()
        authorization = Authorization()

class ContentResource(ModelResource):
    publisher = ToOneField(PublisherResource, 'publisher')
    contentupload = ToOneField(ContentUploadResource, 'contentupload')

    class Meta:
        queryset = models.Content.objects.all()

        # allow ordering and filtering by all fields
        field_list = models.Content._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = Authentication()
        authorization = Authorization()


class UserEventLogResource(ModelResource):
    upload = ToOneField(ContentUploadResource, 'upload')

    class Meta:
        queryset = models.UserEventLog.objects.all()
        # This magic variable defines the api path .../log to refer here
        resource_name = "log"

        # allow ordering and filtering by all fields
        field_list = models.UserEventLog._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = Authentication()
        authorization = Authorization()

#201203 - SequencingKitResource is now obsolete
class SequencingKitResource(ModelResource):
    class Meta:
        queryset = models.SequencingKit.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.SequencingKit._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

#201203 - LibraryKitResource is now obsolete
class LibraryKitResource(ModelResource):
    class Meta:
        queryset = models.LibraryKit.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.LibraryKit._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

class KitInfoResource(ModelResource):
    parts = fields.ToManyField('iondb.rundb.api.KitPartResource', 'kitpart_set', full=True)
        
    class Meta:
        queryset = models.KitInfo.objects.all()
        
        #allow ordering and filtering by all fields
        field_list = models.KitInfo._meta.get_all_field_names()        
        ordering = field_list
        filtering = field_dict(field_list)


class KitPartResource(ModelResource):
    #parent kit
    kit = fields.ToOneField(KitInfoResource, 'kit', full=False)

    class Meta:
        queryset = models.KitPart.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.KitPart._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)


class SequencingKitInfoResource(ModelResource):
    parts = fields.ToManyField('iondb.rundb.api.KitPartResource', 'kitpart_set', full=True)
    class Meta:
        queryset = models.KitInfo.objects.filter(kitType='SequencingKit')
        
        #allow ordering and filtering by all fields
        field_list = models.KitInfo._meta.get_all_field_names()
        ordering = field_list
        filtering = {'kitType' : ['SequencingKit'] }
        
        
class SequencingKitPartResource(ModelResource):
    kit = fields.ToOneField(KitInfoResource, 'kit', full=False)
    
    class Meta:
        kitQuerySet = models.KitInfo.objects.filter(kitType='SequencingKit')
        kitIdList = kitQuerySet.values_list('id')
        
        #matching on the kit id to the sequencingKit's
        queryset = models.KitPart.objects.filter(kit__id__in = kitIdList)
        
        #allow ordering and filtering by all fields
        field_list = models.KitPart._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        

class LibraryKitInfoResource(ModelResource):
    parts = fields.ToManyField('iondb.rundb.api.KitPartResource', 'kitpart_set', full=True)
        
    class Meta:
        queryset = models.KitInfo.objects.filter(kitType='LibraryKit')
        
        #allow ordering and filtering by all fields
        field_list = models.KitInfo._meta.get_all_field_names()
        ordering = field_list
        filtering = {'kitType' : ['LibraryKit'] }


class LibraryKitPartResource(ModelResource):
    kit = fields.ToOneField(KitInfoResource, 'kit', full=False)
    
    class Meta:
        kitQuerySet = models.KitInfo.objects.filter(kitType='LibraryKit')
        kitIdList = kitQuerySet.values_list('id')
        #matching on the kit id to the libraryKit's
        queryset = models.KitPart.objects.filter(kit__id__in = kitIdList)
        
        #allow ordering and filtering by all fields
        field_list = models.KitPart._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)


class ThreePrimeadapterResource(ModelResource):
    class Meta:
        queryset = models.ThreePrimeadapter.objects.all()
        
        #allow ordering and filtering by all fields
        field_list = models.ThreePrimeadapter._meta.get_all_field_names()        
        ordering = field_list
        filtering = field_dict(field_list)
        

class LibraryKeyResource(ModelResource):
    class Meta:
        queryset = models.LibraryKey.objects.all()
        
        #allow ordering and filtering by all fields
        field_list = models.LibraryKey._meta.get_all_field_names()        
        ordering = field_list
        filtering = field_dict(field_list)

  
class MessageResource(ModelResource):

    class Meta:
        queryset = models.Message.objects.all()

        # allow ordering and filtering by all fields
        field_list = models.Message._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = Authentication()
        authorization = Authorization()

class IonReporter(Resource):
    """Allow system updates via API"""

    def override_urls(self):
        urls = [url(r"^(?P<resource_name>%s)%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_update'), name="api_dispatch_update")
        ]

        return urls

    def dispatch_update(self, request, **kwargs):
        return self.dispatch('update', request, **kwargs)

    def get_update(self, request, **kwargs):

        status, workflows = tasks.IonReporterWorkflows(autorun=False)

        if not status:
            raise ImmediateHttpResponse(HttpBadRequest(workflows))


        return self.create_response(request, workflows)

    class Meta:

        authentication = Authentication()
        authorization = Authorization()
        #the name at the start is important
        update_allowed_methods = ['get']
