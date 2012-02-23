# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from tastypie.resources import ModelResource, Resource
from tastypie.constants import ALL, ALL_WITH_RELATIONS
from tastypie import fields
import glob

from iondb.rundb import models

from tastypie.serializers import Serializer

from django.core import serializers

import simplejson
from django.core.serializers import json

#auth
from tastypie.authentication import BasicAuthentication, Authentication
from tastypie.authorization import DjangoAuthorization, Authorization

#tastypie
from django.utils.encoding import force_unicode
from tastypie.bundle import Bundle
from tastypie.fields import ApiField, ToOneField, ToManyField, CharField, ApiFieldError, DictField
from tastypie.exceptions import ImmediateHttpResponse
from tastypie.exceptions import NotFound, BadRequest, InvalidFilterError, HydrationError, InvalidSortError
from tastypie.http import *
from tastypie.utils import dict_strip_unicode_keys, trailing_slash
from tastypie.utils.mime import build_content_type

from iondb.rundb.views import barcodeData, findVersions
import iondb.rundb.admin

from ion.utils.plugin_json import *
from iondb.plugins import runner

#custom query
from django.db.models.sql.constants import QUERY_TERMS, LOOKUP_SEP
from django.db import transaction

import os
import socket
import datetime
import logging

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
        request_method = request.method.lower()
        if request_method not in ("get", "post"):
            raise ImmediateHttpResponse(response=HttpMethodNotAllowed())

        method = getattr(self, "%s_metadata" % request_method, None)

        if method is None:
            raise ImmediateHttpResponse(response=HttpNotImplemented())

        self.is_authenticated(request)
        self.is_authorized(request)
        self.throttle_check(request)

        # All clear. Process the request.                                                                                                         
        response = method(request, **kwargs)

        # Add the throttled request.                                                                                                              
        self.log_throttled_access(request)

        # If what comes back isn't a ``HttpResponse``, assume that the
        # request was accepted and that some action occurred. This also
        # prevents Django from freaking out.
        if not isinstance(response, HttpResponse):
            return HttpAccepted()

        return response

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

class ResultsResource(BaseMetadataResource):

    def override_urls(self):
       urls = [
           url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/plugin%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_plugin'), name="api_dispatch_plugin"),
           url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/pluginresults%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_pluginResults'), name="api_dispatch_pluginResults"),
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
        request_method = request.method.lower()
        if request_method not in ("get", "post", "put"):
            raise ImmediateHttpResponse(response=HttpMethodNotAllowed())

        method = getattr(self, "%s_plugin" % request_method, None)

        if method is None:
            raise ImmediateHttpResponse(response=HttpNotImplemented())

        self.is_authenticated(request)
        self.is_authorized(request)
        self.throttle_check(request)

        # All clear. Process the request.
        response = method(request, **kwargs)

        # Add the throttled request.
        self.log_throttled_access(request)

        # If what comes back isn't a ``HttpResponse``, assume that the
        # request was accepted and that some action occurred. This also
        # prevents Django from freaking out.
        if not isinstance(response, HttpResponse):
            return HttpAccepted()

        return response

    def get_plugin(self, request, **kwargs):
        """
        Will return a list of the html files for all the plugs that a report has
        this is going to be based on the file system output
        """
        results = self.cached_obj_get(request=request, **self.remove_api_resource_names(kwargs))
        if results is None:
            return HttpGone()
        pluginStatus = {}
        pluginStatus["pluginState"] = results.getPluginState()
        pluginPath = os.path.join(results.get_report_dir(),"plugin_out")
        pluginFiles = {}
        pluginDirs = [ name for name in os.listdir(pluginPath) if os.path.isdir(os.path.join(pluginPath, name)) ]
        for pluginDir in pluginDirs:
            htmlFiles = [pfile for pfile in os.listdir(os.path.join(pluginPath, pluginDir)) if (pfile.endswith(".html") or pfile.endswith(".php"))]
            pluginDir = ''.join(pluginDir.split())[:-4]
            pluginFiles[pluginDir] = htmlFiles
        pluginStatus["pluginFiles"] = pluginFiles

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
        gc = models.GlobalConfig.objects.all()[0]
        if gc.web_root:
            hostname = gc.web_root
        else:
            #if a hostname was not found in globalconfig.webroot then use what the system reports
            hostname = "http://" + str(socket.getfqdn())

        report = str(result.reportLink)
        reportList = report.split("/")[:-1]
        reportUrl = ("/").join(reportList)

        url_root = hostname + reportUrl

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

            #this will have to be built, at least in part for all the plugins sent in the list
            sigproc_results = "."
            basecaller_results = "."
            alignment_results = "."

            env={
                'pathToRaw':result.experiment.unique,
                'report_root_dir':result.get_report_dir(),
                'analysis_dir':result.get_report_dir(),
                'sigproc_dir':os.path.join(result.get_report_dir(),sigproc_results),
                'basecaller_dir':os.path.join(result.get_report_dir(),basecaller_results),
                'alignment_dir':os.path.join(result.get_report_dir(),alignment_results),
                'libraryKey':result.experiment.libraryKey,
                'results_dir' : plugin_output_dir,
                'net_location' : hostname,
                'testfrag_key':'ATCG',
            }

            # if thumbnail
            is_thumbnail = result.metaData.get("thumb",False)
            if is_thumbnail:
                env['pathToRaw'] =  os.path.join(env['pathToRaw'],'thumbnail')

            plugin={
                'name':plugin_orm.name,
                'path':plugin_orm.path,
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


    def dispatch_pluginResults(self, request, **kwargs):
        request_method = request.method.lower()
        if request_method not in ("get", "post", "put"):
            raise ImmediateHttpResponse(response=HttpMethodNotAllowed())

        if request_method in ("get"):
            method = getattr(self, "%s_pluginResults" % request_method, None)
        else:
            # Delegate to existing put_plugin and post_plugin methods
            method = getattr(self, "%s_plugin" % request_method, None)

        if method is None:
            raise ImmediateHttpResponse(response=HttpNotImplemented())

        self.is_authenticated(request)
        self.is_authorized(request)
        self.throttle_check(request)

        # All clear. Process the request.
        response = method(request, **kwargs)

        # Add the throttled request.
        self.log_throttled_access(request)

        # If what comes back isn't a ``HttpResponse``, assume that the
        # request was accepted and that some action occurred. This also
        # prevents Django from freaking out.
        if not isinstance(response, HttpResponse):
            return HttpAccepted()

        return response

    def get_pluginResults(self, request, **kwargs):

        results = self.cached_obj_get(request=request, **self.remove_api_resource_names(kwargs))
        if results is None:
            return HttpGone()

        pluginresults = {}
        for pluginresult in results.pluginresult_set.all():
            pname = pluginresult.plugin.name
            # FIXME - handle multiple versions
            if pname in pluginresults:
                logger.warn("Got two plugins with same name: %s", pname)
            pluginresults[pname] = pluginresult

        #pluginStatus = {}
        #pluginStatus["pluginState"] = results.getPluginState() #Nidhi - Get the values from database. Refer models.py

        pluginPath = os.path.join(results.get_report_dir(),"plugin_out")

        pluginArray = []
        pluginDict = {}
        pluginDirs = [ name for name in os.listdir(pluginPath) if os.path.isdir(os.path.join(pluginPath, name)) ]
        for pluginDir in pluginDirs:
            if pluginDirs in ('.', '..'):
                continue
            pluginFiles = {}
            htmlFiles = [pfile for pfile in os.listdir(os.path.join(pluginPath, pluginDir)) if (pfile.endswith(".html") or pfile.endswith(".php"))]
            #htmlFiles = [pfile for pfile in glob(os.path.join(pluginDir, "*.html"))]
            #get rid of _out
            pluginDir = ''.join(pluginDir.split())[:-4]
            #pluginFiles[pluginDir] = htmlFiles
            pluginFiles["Files"] = htmlFiles
            if pluginDir in pluginresults:
                p = pluginresults[pluginDir]
                plugin = p.plugin
                pluginFiles["State"] = p.state
                pluginFiles["Name"] = plugin.name
                pluginFiles["Version"] = plugin.version
                pluginFiles["about"] = "TBD. Scrape from torrent circuit xml"
            else:
                pluginFiles["State"] = 'Unknown'
                #plugin = models.Plugin.objects.get(name=pluginDir) # FIXME handle multiple versions
                pluginFiles["Name"] = pluginDir # plugin.name
                pluginFiles["Version"] = "?"    # plugin.version
                pluginFiles["Description"] = "TBD" # plugin.description. Keeping it for now. Prototyping purposes
                pluginFiles["Type"] = "Ion Plugin" # may or may not be needed. This is to create a prototype for Select Plugins to run page.
            pluginArray.append(pluginFiles)

        #return self.create_response(request, pluginDict)
        return self.create_response(request, pluginArray)


    #TODO: extend the scheme so the docs can be auto generated
    def dispatch_pluginstore(self, request, **kwargs):
        request_method = request.method.lower()
        if request_method not in ("get", "put"):
            raise ImmediateHttpResponse(response=HttpMethodNotAllowed())

        method = getattr(self, "%s_pluginstore" % request_method, None)

        if method is None:
            raise ImmediateHttpResponse(response=HttpNotImplemented())

        self.is_authenticated(request)
        self.is_authorized(request)
        self.throttle_check(request)

        # All clear. Process the request.
        response = method(request, **kwargs)

        # Add the throttled request.
        self.log_throttled_access(request)

        # If what comes back isn't a ``HttpResponse``, assume that the
        # request was accepted and that some action occurred. This also
        # prevents Django from freaking out.
        if not isinstance(response, HttpResponse):
            return HttpAccepted()

        return response

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
        request_method = request.method.lower()
        if request_method not in ("get"):
            raise ImmediateHttpResponse(response=HttpMethodNotAllowed())

        method = getattr(self, "%s_barcode" % request_method, None)

        if method is None:
            raise ImmediateHttpResponse(response=HttpNotImplemented())

        self.is_authenticated(request)
        self.is_authorized(request)
        self.throttle_check(request)

        # All clear. Process the request.
        response = method(request, **kwargs)

        # Add the throttled request.
        self.log_throttled_access(request)

        # If what comes back isn't a ``HttpResponse``, assume that the
        # request was accepted and that some action occurred. This also
        # prevents Django from freaking out.
        if not isinstance(response, HttpResponse):
            return HttpAccepted()

        return response

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

    def dehydrate_pluginState(self, bundle):
        return bundle.obj.getPluginState()
    def dehydrate_pluginStore(self, bundle):
        return bundle.obj.getPluginStore()

    class Meta:
        queryset = models.Results.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.Results._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        #this should check for admin rights
        authentication = Authentication()
        authorization = Authorization()

class ExperimentResource(BaseMetadataResource):

    def override_urls(self):
        urls = [
                url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/metadata%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_metadata'), name="api_dispatch_metadata")
        ]

        return urls
    results = fields.ToManyField(ResultsResource, 'results_set', full=False)

    class Meta:
        queryset = models.Experiment.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.Experiment._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

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
        request_method = request.method.lower()
        if request_method not in ("get", "put"):
            raise ImmediateHttpResponse(response=HttpMethodNotAllowed())

        method = getattr(self, "%s_status" % request_method, None)

        if method is None:
            raise ImmediateHttpResponse(response=HttpNotImplemented())

        self.is_authenticated(request)
        self.is_authorized(request)
        self.throttle_check(request)

        # All clear. Process the request.
        response = method(request, **kwargs)

        # Add the throttled request.
        self.log_throttled_access(request)

        # If what comes back isn't a ``HttpResponse``, assume that the
        # request was accepted and that some action occurred. This also
        # prevents Django from freaking out.
        if not isinstance(response, HttpResponse):
            return HttpAccepted()

        return response

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
        request_method = request.method.lower()
        if request_method not in ("get"):
            raise ImmediateHttpResponse(response=HttpMethodNotAllowed())

        method = getattr(self, "%s_config" % request_method, None)

        if method is None:
            raise ImmediateHttpResponse(response=HttpNotImplemented())

        self.is_authenticated(request)
        self.is_authorized(request)
        self.throttle_check(request)

        # All clear. Process the request.
        response = method(request, **kwargs)

        # Add the throttled request.
        self.log_throttled_access(request)

        # If what comes back isn't a ``HttpResponse``, assume that the
        # request was accepted and that some action occurred. This also
        # prevents Django from freaking out.
        if not isinstance(response, HttpResponse):
            return HttpAccepted()

        return response

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

class PluginResource(ModelResource):
    """Get a list of plugins"""

    def override_urls(self):
        #this is meant for internal use only
        urls = [url(r"^(?P<resource_name>%s)/set/(?P<keys>\w[\w/-;]*)/type%s$" % (self._meta.resource_name, trailing_slash()),self.wrap_view('get_type_set'), name="api_get_type_set"),
                url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/type%s$" % (self._meta.resource_name, trailing_slash()),self.wrap_view('dispatch_type'), name="api_dispatch_type"),
                url(r"^(?P<resource_name>%s)/install%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_install'), name="api_dispatch_install")
        ]

        return urls

    def dispatch_type(self, request, **kwargs):
        request_method = request.method.lower()
        if request_method not in ("get"):
            raise ImmediateHttpResponse(response=HttpMethodNotAllowed())

        method = getattr(self, "%s_type" % request_method, None)

        if method is None:
            raise ImmediateHttpResponse(response=HttpNotImplemented())

        self.is_authenticated(request)
        self.is_authorized(request)
        self.throttle_check(request)

        # All clear. Process the request.
        response = method(request, **kwargs)

        # Add the throttled request.
        self.log_throttled_access(request)

        # If what comes back isn't a ``HttpResponse``, assume that the
        # request was accepted and that some action occurred. This also
        # prevents Django from freaking out.
        if not isinstance(response, HttpResponse):
            return HttpAccepted()

        return response

    def dispatch_install(self, request, **kwargs):
        request_method = request.method.lower()
        if request_method not in ("post"):
            raise ImmediateHttpResponse(response=HttpMethodNotAllowed())

        method = getattr(self, "%s_install" % request_method, None)

        if method is None:
            raise ImmediateHttpResponse(response=HttpNotImplemented())

        self.is_authenticated(request)
        self.is_authorized(request)
        self.throttle_check(request)

        response = method(request, **kwargs)
        self.log_throttled_access(request)

        if not isinstance(response, HttpResponse):
            return HttpAccepted()

        return response

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
        In the event
        """
        plugin_set = kwargs['keys'].split(";")
        queryset = self.cached_obj_get_list(request).filter(pk__in=plugin_set)
        types = dict((p.pk, self._get_type(p)) for p in queryset)
        if any(isinstance(t, HttpGone) for t in types):
            return HttpGone()
        else:
            return self.create_response(request, types)

    def post_install(self, request, **kwargs):
        deserialized = self.deserialize(request, request.raw_post_data, format=request.META.get('CONTENT_TYPE', 'application/json'))
        data=dict_strip_unicode_keys(deserialized)

        try:
            url = data["url"]
        except KeyError:
            return HttpBadRequest()

        plugin_name = os.path.splitext(os.path.basename(url))[0]

        (plugin,created) = models.Plugin.objects.get_or_create(name=plugin_name)
        plugin.date=datetime.datetime.now()
        plugin.path=settings.TEMP_PATH
        plugin.active=True
        plugin.selected=False ## Avoid running while path=TEMP_PATH
        plugin.status["result"] = "queued"
        plugin.save()

        tasks.downloadPlugin.delay(url, plugin)
        return HttpAccepted()


    class Meta:
        # Note API only sees active plugins
        queryset = models.Plugin.objects.filter(active=True)

        #allow ordering and filtering by all fields
        field_list = models.Plugin._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = Authentication()
        authorization = Authorization()

class PluginResultResource(ModelResource):
    result = fields.ToOneField(ResultsResource,'result')
    plugin = fields.ToOneField(PluginResource, 'plugin', full=True)

    class Meta:
        queryset = models.PluginResult.objects.all()
        #allow ordering and filtering by all fields
        field_list = models.Plugin._meta.get_all_field_names()
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

class PlannedExperimentResource(ModelResource):

    class Meta:
        queryset = models.PlannedExperiment.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = Authentication()
        authorization = Authorization()

class TorrentSuite(ModelResource):
    """Allow system updates via API"""

    def override_urls(self):
        urls = [url(r"^(?P<resource_name>%s)%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_update'), name="api_dispatch_update")
        ]

        return urls

    def dispatch_update(self, request, **kwargs):
        request_method = request.method.lower()
        if request_method not in ("get", "put"):
            raise ImmediateHttpResponse(response=HttpMethodNotAllowed())

        method = getattr(self, "%s_update" % request_method, None)

        if method is None:
            raise ImmediateHttpResponse(response=HttpNotImplemented())

        self.is_authenticated(request)
        self.is_authorized(request)
        self.throttle_check(request)

        # All clear. Process the request.
        response = method(request, **kwargs)

        # Add the throttled request.
        self.log_throttled_access(request)

        # If what comes back isn't a ``HttpResponse``, assume that the
        # request was accepted and that some action occurred. This also
        # prevents Django from freaking out.
        if not isinstance(response, HttpResponse):
            return HttpAccepted()

        return response

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


class ContentResource(ModelResource):
    publisher = ToOneField(PublisherResource, 'publisher')
    
    class Meta:
        queryset = models.Content.objects.all()

        # allow ordering and filtering by all fields
        field_list = models.Content._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = Authentication()
        authorization = Authorization()


class ContentUploadResource(ModelResource):

    class Meta:
        queryset = models.ContentUpload.objects.all()

        # allow ordering and filtering by all fields
        field_list = models.ContentUpload._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = Authentication()
        authorization = Authorization()


class UserEventLogResource(ModelResource):

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
