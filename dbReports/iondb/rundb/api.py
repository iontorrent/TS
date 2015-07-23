# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import os
from django.db.models.aggregates import Count, Max
from django.core.cache import cache
from tastypie.api import Api
import StringIO
import csv
import uuid
import json
from tastypie.utils.formatting import format_datetime
from tastypie.resources import ModelResource as _ModelResource, Resource, ModelDeclarativeMetaclass
from tastypie.constants import ALL, ALL_WITH_RELATIONS
from tastypie.cache import SimpleCache
from tastypie import fields
import glob

from django.contrib.auth.models import User
from iondb.rundb import models

from tastypie.serializers import Serializer

from django.core import serializers, urlresolvers

import json
import imp

#auth
from iondb.rundb.authn import IonAuthentication
from tastypie.authorization import DjangoAuthorization

#tastypie
from django.utils import timezone
from django.utils.encoding import force_unicode
from tastypie.bundle import Bundle
from tastypie.fields import ApiField, ToOneField, ToManyField, CharField, ApiFieldError, DictField
from tastypie.exceptions import ImmediateHttpResponse, UnsupportedFormat
from tastypie.exceptions import NotFound, BadRequest, InvalidFilterError, HydrationError, InvalidSortError

from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from django.core.exceptions import ValidationError

from django.utils.decorators import method_decorator
from tastypie.validation import Validation, FormValidation
from tastypie.resources import ModelResource

from tastypie.http import *
from tastypie.utils import dict_strip_unicode_keys, trailing_slash
from tastypie.utils.mime import build_content_type

from iondb.rundb.views import barcodeData
from iondb.rundb.barcodedata import BarcodeSampleInfo
from iondb.rundb import forms
import iondb.rundb.admin
from iondb.rundb.plan import plan_validator
from iondb.rundb.sample import sample_validator
from iondb.rundb.plan.plan_share import prepare_for_copy, transfer_plan, update_transferred_plan

from ion.utils.TSversion import findVersions
#from iondb.plugins import runner
from ion.plugin import remote, Feature
from ion.plugin.constants import RunLevel, RunType, runLevelsList

#custom query
from django.db.models.sql.constants import QUERY_TERMS

from django.db import transaction
from django.db.models import Q

import socket
import datetime
import logging
from operator import itemgetter

import httplib2
from iondb.utils import toBoolean
import urllib
import re
from subprocess import Popen, PIPE
import operator

from django.conf import settings
from django.http import HttpResponse, HttpResponseNotFound, HttpResponseRedirect, Http404

try:
    from django.conf.urls import url
except ImportError:
    # Compat Django 1.4
    from django.conf.urls.defaults import url

from iondb.rundb import tasks
from iondb.rundb.data import dmactions_types

from iondb.plugins.manager import pluginmanager

# shared functions for making startplugin json
from iondb.plugins.launch_utils import get_plugins_dict

import ast
import simplejson
import traceback

from iondb.rundb import json_field
from iondb.utils import toBoolean

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

class CustomSerializer(Serializer):
    formats = settings.TASTYPIE_DEFAULT_FORMATS
    content_types = {
        'json': 'application/json',
        'jsonp': 'text/javascript',
        'xml': 'application/xml',
        'yaml': 'text/yaml',
        'html': 'text/html',
        'plist': 'application/x-plist',
        'csv': 'text/csv',
    }

    def format_datetime(self, data):
        """
        A hook to control how datetimes are formatted.

        DO NOT RETURN NAIVE datetime objects!!
        tastypie.utils.timezone.make_naive() is stripping our datetime object's tzinfo
        because django.utils.timezone.get_default_timezone() is returning a timezone of None.
        Django is using None because DJANGO_SETTINGS_MODULE defines TIME_ZONE=None.
        The timezone is set to None to default to use the OS timezone. This custom serializer
        to format datetime objects overriddes this undesired behavior.
        """
        # data = make_naive(data)
        if self.datetime_formatting == 'rfc-2822':
            return format_datetime(data)
        data = data.replace(microsecond=int(round(data.microsecond / 1000)))
        return data.isoformat()

    def to_csv(self, data, options=None):
        options = options or {}
        data = self.to_simple(data, options)
        raw_data = StringIO.StringIO()
        first = True
        writer = None
        for item in data.get('objects', {}):
            if first:
                writer = csv.DictWriter(raw_data, fieldnames=item.keys(), extrasaction='ignore')
                #writer.writeheader() # python 2.7+
                writer.writerow(dict(zip(writer.fieldnames, writer.fieldnames)))
            writer.writerow(item)
        raw_data.seek(0)
        return raw_data.getvalue()

    def from_csv(self, content):
        raw_data = StringIO.StringIO(content)
        with csv.Sniffer() as sniffer:
            has_header = sniffer.has_header(raw_data.read(1024))
        raw_data.seek(0)
        with csv.DictReader(raw_data) as reader:
            if has_header:
                fieldnames = reader.next()
            for item in reader:
                data.append(item)
        return data


class MyModelDeclarativeMetaclass(ModelDeclarativeMetaclass):
    def __new__(cls, name, bases, attrs):
        meta = attrs.get('Meta')

        if meta and not hasattr(meta, 'serializer'):
            setattr(meta, 'serializer', CustomSerializer())
        if meta and not hasattr(meta, 'default_format'):
            setattr(meta, 'default_format', 'application/json')
        if meta and not hasattr(meta, 'authentication'):
            setattr(meta, 'authentication', IonAuthentication())
        if meta and not hasattr(meta, 'authorization'):
            setattr(meta, 'authorization', DjangoAuthorization())

        new_class = super(MyModelDeclarativeMetaclass, cls).__new__(cls, name, bases, attrs)

        return new_class

class ModelResource(_ModelResource):
    __metaclass__ = MyModelDeclarativeMetaclass


class GlobalConfigResource(ModelResource):
    class Meta:
        queryset = models.GlobalConfig.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.GlobalConfig._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class MonitorDataResource(ModelResource):
    class Meta:
        queryset = models.MonitorData.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.MonitorData._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class TFMetricsResource(ModelResource):
    report = fields.ToOneField("iondb.rundb.api.ResultsResource", 'report', full=False)
    class Meta:
        queryset = models.TFMetrics.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.TFMetrics._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class LibMetricsResource(ModelResource):
    report = fields.ToOneField("iondb.rundb.api.ResultsResource", 'report', full=False)
    class Meta:
        queryset = models.LibMetrics.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.LibMetrics._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class AnalysisMetricsResource(ModelResource):
    report = fields.ToOneField("iondb.rundb.api.ResultsResource", 'report', full=False)
    class Meta:
        queryset = models.AnalysisMetrics.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.AnalysisMetrics._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class QualityMetricsResource(ModelResource):
    report = fields.ToOneField("iondb.rundb.api.ResultsResource", 'report', full=False)
    class Meta:
        queryset = models.QualityMetrics.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.QualityMetrics._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class ReportStorageResource(ModelResource):
    class Meta:
        queryset = models.ReportStorage.objects.all()
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class BaseMetadataResource(ModelResource):
    """It is wrong for both exps and results to subclass from this"""


    def dispatch_metadata(self, request, **kwargs):
        return self.dispatch('metadata', request, **kwargs)

    def get_metadata(self, request, **kwargs):
        bundle = self.build_bundle(request=request)
        results = self.cached_obj_get(bundle, **self.remove_api_resource_names(kwargs))
        if results is None:
            return HttpGone()

        metadata = results.metaData or { }
        return self.create_response(request, metadata)

    def post_metadata(self, request, **kwargs):
        deserialized = self.deserialize(request, request.body, format=request.META.get('CONTENT_TYPE', 'application/json'))
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


## Stub only for embedding in User
class UserProfileResource(ModelResource):
    class Meta:
        queryset = models.UserProfile.objects.all()
        allowed_methods = []
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class UserResource(ModelResource):
    profile = fields.ToOneField(UserProfileResource, 'userprofile', full=True)
    full_name = fields.CharField()

    def dehydrate(self, bundle):
        bundle.data['full_name'] = bundle.obj.get_full_name()
        return bundle
    def prepend_urls(self):
        urls = [
            url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/activate%s$" % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('dispatch_activate'), name="api_dispatch_activate"),
            url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/reject%s$" % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('dispatch_reject'), name="api_dispatch_reject"),
        ]
        return urls

    def dispatch_activate(self, request, **kwargs):
        return self.dispatch('activate', request, **kwargs)

    def dispatch_reject(self, request, **kwargs):
        return self.dispatch('reject', request, **kwargs)

    def post_reject(self, request, **kwargs):
        if not request.user.is_superuser:
            raise HttpUnauthorized()
        bundle = self.build_bundle(request=request)
        user = self.cached_obj_get(bundle, **self.remove_api_resource_names(kwargs))
        if user is None:
            return HttpGone()

        user.delete()
        return HttpAccepted()

    def post_activate(self, request, **kwargs):
        if not request.user.is_superuser:
            raise HttpUnauthorized()
        bundle = self.build_bundle(request=request)
        user = self.cached_obj_get(bundle, **self.remove_api_resource_names(kwargs))
        if user is None:
            return HttpGone()

        if not user.is_active:
            user.is_active=True
            user.save()
        return HttpAccepted()

    class Meta:
        queryset = User.objects.all()
        resource_name = 'user'
        excludes = ['password', 'is_staff', 'is_superuser']
        allowed_methods = ['get']
        activate_allowed_methods = ['post']
        reject_allowed_methods = ['post']

        # allow ordering and filtering by all fields
        field_list = models.User._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class ProjectResource(ModelResource):
    resultsCount = fields.IntegerField(readonly=True)
    creator = fields.ToOneField(UserResource, 'creator', full=False)

    def dehydrate(self, bundle):
        bundle.data['resultsCount'] = bundle.obj.results.count()
        return bundle

# TODO: The creator field should be immutable after record insertion.
#    def hydrate_creator (self, bundle):
#        bundle = super(ProjectResource,self).hydrate_creator(bundle)
#        bundle.data['creator'] = bundle.request.user
#        return bundle

    class Meta:
        queryset = models.Project.objects.all()

        creator_allowed_methods = ['get']
        # allow ordering and filtering by all fields
        field_list = models.Project._meta.get_all_field_names()
        ordering = field_list + ['resultsCount']
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class ProjectResultsResource(ModelResource):
    projects = fields.ToManyField(ProjectResource, 'projects', full=False)

    def apply_filters(self, request, applicable_filters):
        base_object_list = super(ProjectResultsResource, self).apply_filters(request, applicable_filters)
        # include/exclude thumbnail results
        isthumbnail = request.GET.get('isThumbnail',None)
        if isthumbnail:
            if isthumbnail == 'yes':
                base_object_list = base_object_list.filter(metaData__contains='thumb')
            if isthumbnail == 'no':
                base_object_list = base_object_list.exclude(metaData__contains='thumb')

        return base_object_list

    class Meta:
        queryset = models.Results.objects.all()

        field_list = ['id','resultsName','timeStamp','projects','status','reportLink','reportStatus', 'reference']
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class ResultsResource(BaseMetadataResource):


    def prepend_urls(self):
       urls = [
           url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/plugin%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_plugin'), name="api_dispatch_plugin"),
           url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/pluginresults%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_pluginresults'), name="api_dispatch_pluginresults"),
           url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/pluginstore%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_pluginstore'), name="api_dispatch_pluginstore"),
           url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/pluginresult_set%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_pluginresult_set'), name="api_dispatch_pluginresult_set"),
           url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/barcode%s$" % (self._meta.resource_name, trailing_slash()),
               self.wrap_view('dispatch_barcode'), name="api_dispatch_barcode"),
           url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/barcodesampleinfo%s$" % (self._meta.resource_name, trailing_slash()),
               self.wrap_view('dispatch_barcodesampleinfo'), name="api_dispatch_barcodesampleinfo"),
           url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/metadata%s$" % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('dispatch_metadata'), name="api_dispatch_metadata"),
           url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/representative%s$" % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('post_representative'), name="api_post_representative"),
        ]

       return urls

    def post_representative(self, request, **kwargs):
        bundle = self.build_bundle(request=request)
        result = self.cached_obj_get(bundle, **self.remove_api_resource_names(kwargs))
        if result is None:
            return HttpGone()
        deserialized = self.deserialize(request, request.body, format=request.META.get('CONTENT_TYPE', 'application/json'))
        data=dict_strip_unicode_keys(deserialized)
        state = data.get("representative", None)
        experiment = result.experiment
        current_result = experiment.repResult
        post_result = result

        if current_result == post_result:
            if not state:
                post_result.representative = False
                # Because we must have a report for this to even execute, and
                # we get this report's experiment, the experiment must have at least 1 report.
                latest = experiment.results_set.order_by('-timeStamp')[0]
                experiment.repResult = latest
                experiment.resultDate = latest.timeStamp
            else:
                post_result.representative = True
            experiment.pinnedRepResult = not experiment.pinnedRepResult
        else:
            if state:
                experiment.pinnedRepResult = True
                experiment.repResult = post_result
                current_result.reprsentative = False
                post_result.representative = True
                current_result.save()
            else:
                post_result.representative = False
        post_result.save()
        experiment.save()

        logger.info("CompExp obj_update {0} {1} {2}".format(current_result, post_result, experiment))

        return HttpAccepted()

    #TODO: extend the scheme so the docs can be auto generated
    def dispatch_plugin(self, request, **kwargs):
        return self.dispatch('plugin', request, **kwargs)

    def get_plugin(self, request, **kwargs):
        """
        Will return a list of the html files for all the plugs that a report has
        this is going to be based on the file system output

        @deprecated - use get_pluginresults instead
        """
        bundle = self.build_bundle(request=request)
        results = self.cached_obj_get(bundle, **self.remove_api_resource_names(kwargs))
        if results is None:
            return HttpGone()

        #import warnings
        #warnings.warn("/rundb/api/v1/report/%d/plugin/ is deprecated. Use /pluginresults/ instead." % results.pk, DeprecationWarning)

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
        pluginDirs = [ name for name in os.listdir(pluginPath) if os.path.isdir(os.path.join(pluginPath, name)) and (name.find("_out") > 0)]
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

    def _launch_plugin(self, plugin):
        pass


    def post_plugin(self, request, **kwargs):
        """
        Expects dict with a plugin to run, with an optional plugin config dict
        """
        logger = logging.getLogger('iondb.rundb.api.ResultsResource')
        bundle = self.build_bundle(request=request)
        result = self.cached_obj_get(bundle, **self.remove_api_resource_names(kwargs))
        if result is None:
            logger.error("Invalid result: %s", kwargs)
            return HttpGone()

        deserialized = self.deserialize(request, request.body, format=request.META.get('CONTENT_TYPE', 'application/json'))
        data=dict_strip_unicode_keys(deserialized)


        #get the hostname try to get the name from global config first
        sge_master = str(socket.getfqdn())
        gc = models.GlobalConfig.get()
        if gc.web_root:
            net_location = gc.web_root
        else:
            #if a hostname was not found in globalconfig.webroot then use what the system reports
            net_location = 'http://%s' % sge_master

        report_root = result.get_report_dir()

        # Get the user specified in the POST, fallback to request.user (probably ionadmin)
        user = request.user
        if request.user.is_superuser:
            # Admin user can override plugin owner
            try:
                username = data.get('owner', data.get('username', None))
                if username is not None:
                    user = User.objects.get(username=username)
            except User.DoesNotExist:
                logger.exception("Invalid owner specified for plugin launch: %s", username)

        pluginkwargs = data.get("plugin",None)
        if pluginkwargs is None:
            return HttpGone()
        plugin = None

        if isinstance(pluginkwargs, int):
            pluginkwargs = { "id": pluginkwargs }
        if hasattr(pluginkwargs, 'items'):
            # Got a dict of kwargs, use as query
            pluginkwargs.update({'active':True, 'selected':True}) # Cannot run an plugin unless active and enabled
            logger.info("Searching for plugin: '%s'", pluginkwargs)
            try:
                plugin = models.Plugin.objects.get(**pluginkwargs)
            except:
                logger.exception("Failed to fetch plugin: '%s'", pluginkwargs)
                return HttpGone()
        else:
            if isinstance(pluginkwargs, (list, tuple)):
                if len(pluginkwargs) > 1:
                    logger.error("ERROR: Ignoring extra pluginkwargs. Please run one at a time: '%s'", pluginkwargs)
                pluginkwargs = pluginkwargs[0]
            logger.info("Searching for plugin by name: '%s'", pluginkwargs)
            qs = models.Plugin.objects.filter(name=pluginkwargs,active=True,selected=True).order_by('id')
            ## Get Latest version of plugin
            for p in qs:
                if (not plugin) or p.versionGreater(plugin):
                    plugin = p
            if not plugin:
                logger.error("Invalid plugin specified (No active versions): %s", pluginkwargs)
                return HttpGone()

        plugins_dict = get_plugins_dict([plugin], result.eas.selectedPlugins)
        plugins_dict = json.loads(json_field.dumps(plugins_dict))

        params = {
            'run_mode': 'manual', ## Launches via REST API are all manual (instance?), pipeline calls direct to XMLRPC
            'plugins': {
                plugin.name: {
                    'instance_config': data.get("pluginconfig",{}),
                    'pluginresult': data.get('pluginresult',None),
                }
            }
        }

        report_type = RunType.FULLCHIP
        if result.experiment.rawdatastyle != 'single':
            report_type = RunType.THUMB if result.isThumbnail else RunType.COMPOSITE

        if report_type == RunType.COMPOSITE:
            # get analyzed block directories for Proton analysis
            params['block_dirs'] = []
            explog = result.experiment.log
            blockIds = []
            for line in explog.get('blocks', []):
                args = line.strip().replace('BlockStatus:','').split(',')
                if args[0] =='thumbnail' or (args[0] == '0' and args[1] == '0'):
                    continue
                blockId = args[0].strip() + '_' + args[1].strip()
                block_dir = os.path.join(report_root,'block_'+blockId)
                if os.path.isdir(block_dir):
                    blockIds.append(blockId)
                    params['block_dirs'].append(block_dir)

        # launch plugin(s)
        for runlevel in runLevelsList():
            # plugins_dict may change in loop if dependency plugins get added
            if runlevel == RunLevel.BLOCK:
                plugin_runlevels = sum([(p.get('runlevel') or [RunLevel.DEFAULT]) for p in plugins_dict.values()], [])
                if RunLevel.BLOCK in plugin_runlevels and report_type == RunType.COMPOSITE:
                    for blockId in blockIds:
                        params['blockId'] = blockId
                        plugins_dict, msg = remote.call_launchPluginsXMLRPC(result.pk, plugins_dict, net_location, user.username, runlevel, params)
            else:
                plugins_dict, msg = remote.call_launchPluginsXMLRPC(result.pk, plugins_dict, net_location, user.username, runlevel, params)

            if msg:
                if 'ERROR' in msg:
                    logger.error('Unable to launch plugin: %s', plugin.name) # See ionPlugin.log for details
                else:
                    logger.info('Launched plugin runlevel= %s. %s' % (runlevel, msg))

        return HttpAccepted()

    def put_plugin(self, request, **kwargs):
        """
        Set the status of the plugins.
        This allow a way to see if the plugin was ran successfully.
        """
        # DEPRECATED - disabled
        import warnings
        warnings.warn("Updating status by pluginname is deprecated. Use pluginresult by pk instead.", DeprecationWarning)

        deserialized = self.deserialize(request, request.body,
                                        format=request.META.get('CONTENT_TYPE', 'application/json'))
        data=dict_strip_unicode_keys(deserialized)

        logger.error("Invalid plugin status update: '%s'", data)
        return HttpBadRequest()


    def dispatch_pluginresults(self, request, **kwargs):
        return self.dispatch('pluginresults', request, **kwargs)

    def _startplugin_version(self, outputpath):
        startjson_fname = os.path.join(outputpath, 'startplugin.json')
        try:
            with open(startjson_fname) as f:
                startjson = json.load(f)
            return startjson["runinfo"]["plugin"]["version"]
        except (OSError, IOError):
            #logger.exception("Missing startplugin.json? %s", outputpath)
            # This is surprisingly common - plugins delete it. Ignore.
            pass
        except (KeyError, ValueError):
            logger.exception("Malformed startplugin.json? %s", outputpath)
        return None

    def post_pluginresults(self, request, **kwargs):
        try:
            basic_bundle = self.build_bundle(request=request)
            obj = self.cached_obj_get(bundle=basic_bundle, **self.remove_api_resource_names(kwargs))
        except ObjectDoesNotExist:
            return HttpNotFound()
        except MultipleObjectsReturned:
            return HttpMultipleChoices("More than one resource is found at this URI.")

        # Result Bundle...
        bundle = self.build_bundle(obj=obj, request=request)
        bundle = self.full_dehydrate(bundle)

        kwargs.update({
            #"result": bundle, # obj,
            "result": bundle.obj,
            "id" : None, # Force creation of new PluginResult
        })
        return self.pluginresults.to_class().post_list(request, **kwargs)

        #prbundle = prc.build_bundle(data=dict_strip_unicode_keys(deserialized), request=request)
        #updated_bundle = prc.obj_create(prbundle, **self.remove_api_resource_names(kwargs))
        #location = prc.get_resource_uri(updated_bundle)
        #if not prc._meta.always_return_data:
        #    return http.HttpCreated(location=location)
        #else:
        #    updated_bundle = prc.full_dehydrate(updated_bundle)
        #    updated_bundle = prc.alter_detail_data_to_serialize(request, updated_bundle)
        #    return self.create_response(request, updated_bundle, response_class=http.HttpCreated, location=location)

    def get_pluginresults(self, request, **kwargs):
        bundle = self.build_bundle(request=request)
        results = self.cached_obj_get(bundle, **self.remove_api_resource_names(kwargs))
        if results is None:
            return HttpGone()

        #filter for just the major blocks
        major = request.GET.get('major', None)

        # In the order they were generated, newest plugin entry first.
        pluginresults = results.pluginresult_set.all().order_by('-starttime')
        if major is not None:
            major = (major.lower() == "true")
            pluginresults = pluginresults.filter(plugin__majorBlock=major)

        gc = models.GlobalConfig.get()

        # Iterate through DB results first, find matching directories
        pluginArray = []
        seen = {}
        show_plugins = request.session.setdefault("show_plugins", {})
        for pr in pluginresults:
            outputpath = pr.path(create=False)
            base_path = os.path.basename(outputpath)
            #logger.debug("Plugin %s v%s - searching %s", pr.plugin.name, pr.plugin.version, outputpath)
            if not os.path.exists(outputpath):
                logger.info("Plugin %s v%s has no plugin_out folder", pr.plugin.name, pr.plugin.version)
                outputpath = None
                pr.state = pr.state + ' [Missing]'

            if outputpath:
                # Got a matching path, populate list of files
                all_files = [pfile for pfile in os.listdir(outputpath) if (pfile.endswith(".html") or pfile.endswith(".php"))]
                outputfiles = all_files
                seen[base_path] = True
                seen[os.path.basename(outputpath)] = True
                if major:
                    outputfiles = filter(lambda x: "_block" in x, all_files)
                link_files = filter(lambda x: "_block" not in x, all_files)
            else:
                # Missing Output, or unknown version
                outputpath = pr.path() ## assume default path, let link be broken
                seen[base_path] = True
                outputfiles = [] # but mask files
                link_files = []

                logger.info("Plugin %s v%s - unable to find output folder: %s", pr.plugin.name, pr.plugin.version, outputpath)

            # Convert to PluginReportResource...
            data = {
                'Name': pr.plugin.name,
                'Version': pr.plugin.version,
                'State': pr.state,
                'Path': os.path.join(outputpath,''),
                'URL': os.path.join(results.reportLink, gc.plugin_output_folder, base_path,''),
                'Files': outputfiles,
                'Links': link_files,
                'Major' : pr.plugin.majorBlock,
                'Size': pr.size,
                'inodes': pr.inodes,
                'id': pr.id,
                'jobid': pr.jobid,
                'show': show_plugins.get(pr.plugin.name, True)
            }
            pluginArray.append(data)

        # Report base path for all plugins
        if not major:
            plugin_out_path = os.path.join(results.get_report_dir(), gc.plugin_output_folder)
            try:
                plugin_out_listing = ( d for d in os.listdir(plugin_out_path) if os.path.isdir(os.path.join(plugin_out_path,d)) and (d.find("_out") > 0))
            except (OSError, IOError):
                logger.info("Error listing %s", plugin_out_path)
                plugin_out_listing = []

            # Now - directories found which don't match plugin result entries
            import re
            re_clean = re.compile(r'_out(\.\d+)?$')
            for orphan in ( p for p in plugin_out_listing if p not in seen ):
                outputpath = os.path.join(plugin_out_path, orphan)
                version = self._startplugin_version(outputpath)
                if version is None:
                    version = '??'

                name = re_clean.sub('',orphan)
                outputfiles = [pfile for pfile in os.listdir(outputpath) if (pfile.endswith(".html") or pfile.endswith(".php"))]

                data = {
                    'Name': name,
                    'Version': version,
                    'State': 'Unknown',
                    'Files': outputfiles,
                    'Path': os.path.join(outputpath,''),
                    'URL': os.path.join(results.reportLink, gc.plugin_output_folder, orphan, ''),
                    'Major' : False, # NB: Plugins without DB record cannot be major
                    'Size': -1,
                    'inodes': -1,
                    'id': None,
                    'jobid': None,
                    'show': show_plugins.get(name, True)
                }
                logger.info("Plugin folder with no db record: %s v%s at '%s'", name, version, outputpath)
                pluginArray.append(data)

        # Resort by plugin name
        pluginArray = sorted(pluginArray, key=itemgetter('Name'), cmp=(lambda a, b: cmp(a.lower(), b.lower())))

        return self.create_response(request, pluginArray)

    #TODO: extend the scheme so the docs can be auto generated
    def dispatch_pluginstore(self, request, **kwargs):
        """ Uses Meta::pluginstore_allowed_methods, and dispatches to <method>_pluginstore """
        return self.dispatch('pluginstore', request, **kwargs)

    def get_pluginstore(self, request, **kwargs):
        """
        Returns pluginStore for all plugins
        @deprecated
        """
        bundle = self.build_bundle(request=request)
        results = self.cached_obj_get(bundle, **self.remove_api_resource_names(kwargs))
        if results is None:
            return HttpGone()

        pluginStore = results.getPluginStore()
        return self.create_response(request, pluginStore)

    def put_pluginstore(self, request, **kwargs):
        """
        Set the store of the plugins. - results.json
        @deprecated
        """
        deserialized = self.deserialize(request, request.body, format=request.META.get('CONTENT_TYPE', 'application/json'))
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

    def dispatch_barcodesampleinfo(self, request, **kwargs):
        return self.dispatch('barcodesampleinfo', request, **kwargs)

    def get_barcodesampleinfo(self, request, **kwargs):
        result = self.get_object_list(request).get(pk=kwargs["pk"])
        if result is None:
            return HttpGone()
        barcodeSampleInfo = BarcodeSampleInfo(kwargs["pk"],result=result)
        logger.debug("BarcodeSampleInfo: %s", barcodeSampleInfo.to_json())
        return self.create_response(request, barcodeSampleInfo.data())

    def dispatch_pluginresult_set(self, request, **kwargs):
        return self.dispatch('pluginresult_set', request, **kwargs)

    def get_pluginresult_set(self, request, **kwargs):
        # Helper to return full resource for pluginresults, 
        # different from get_pluginresults, which returns custom report resource
        try:
            basic_bundle = self.build_bundle(request=request)
            obj = self.cached_obj_get(basic_bundle, **self.remove_api_resource_names(kwargs))
        except ObjectDoesNotExist:
            return HttpGone()
        except MultipleObjectsReturned:
            return HttpMultipleChoices("More than one resource is found at this URI.")

        bundle = self.build_bundle(obj=obj, request=request)
        bundle = self.full_dehydrate(bundle)

        kwargs.update({
            "result" : bundle.obj,
        })

        pluginresults = self.pluginresults.to_class()
        return pluginresults.get_list(request, **kwargs)

    filesystempath = CharField('get_report_dir')
    bamLink = CharField('bamLink')
    planShortID = CharField('planShortID')
    libmetrics = ToManyField(LibMetricsResource, 'libmetrics_set', full=False)
    tfmetrics = ToManyField(TFMetricsResource, 'tfmetrics_set', full=False)
    analysismetrics = ToManyField(AnalysisMetricsResource, 'analysismetrics_set', full=False)
    qualitymetrics = ToManyField(QualityMetricsResource, 'qualitymetrics_set', full=False)
    reportstorage = fields.ToOneField(ReportStorageResource, 'reportstorage', full=True)

    sffLink = CharField('sffLinkPatch', null=True)
    tfSffLink = CharField('sffTFLinkPatch', null=True)

    #parent experiment
    experiment = fields.ToOneField('iondb.rundb.api.ExperimentResource', 'experiment', full=False, null = True)

    # Nested plugin results - replacement for pluginStore pluginState
    pluginresults = ToManyField('iondb.rundb.api.PluginResultResource', 'pluginresult_set', related_name='result', full=False)
    # Add back pluginState/pluginStore for compatibility
    # But using pluginresults should be preferred.
    pluginState = DictField(readonly=True)
    pluginStore = DictField(readonly=True)

    projects = fields.ToManyField(ProjectResource, 'projects', full=False)

    eas = fields.ToOneField('iondb.rundb.api.ExperimentAnalysisSettingsResource', 'eas', full=False, null=True, blank=True)

    def apply_filters(self, request, applicable_filters):
        base_object_list = super(ResultsResource, self).apply_filters(request, applicable_filters)
        # include/exclude thumbnail results
        isthumbnail = request.GET.get('isThumbnail',None)
        if isthumbnail:
            if isthumbnail == 'yes':
                base_object_list = base_object_list.filter(metaData__contains='thumb')
            if isthumbnail == 'no':
                base_object_list = base_object_list.exclude(metaData__contains='thumb')

        return base_object_list

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
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

        plugin_allowed_methods = ['get', 'post', 'put']
        pluginresults_allowed_methods = ['get','post']
        pluginresult_set_allowed_methods = ['get', ]
        projects_allowed_methods = ['get']
        pluginstore_allowed_methods = ['get','put']
        barcode_allowed_methods = ['get']
        barcodesampleinfo_allowed_methods = ['get']
        metadata_allowed_methods = ['get','post']

class ExperimentResource(BaseMetadataResource):

    def prepend_urls(self):
        urls = [
                url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/metadata%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_metadata'), name="api_dispatch_metadata"),
                url(r"^(?P<resource_name>%s)/projects%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_projects'), name="api_dispatch_projects")
        ]

        return urls
    results = fields.ToManyField(ResultsResource, 'results_set')

    runtype = CharField('runtype')

    plan = fields.ToOneField('iondb.rundb.api.PlannedExperimentResource', 'plan', full=False, null=True, blank=True)
    eas_set = fields.ToManyField('iondb.rundb.api.ExperimentAnalysisSettingsResource', 'eas_set', full=True, null=True, blank=True)
    samples = fields.ToManyField('iondb.rundb.api.SampleResource', 'samples', full=True, null=True, blank=True)

    isProton = fields.CharField(readonly=True, attribute="isProton")

    # Backwards support - single sample field
    sample = fields.CharField(readonly=True, blank=True)

    def dehydrate_sample(self, bundle):
        """Return the first sample name"""
        if bundle.obj.samples.all():
            return bundle.obj.samples.all()[0].name
        else:
            return ""

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

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class SampleResource(ModelResource):
    #associated experiment & sample
    experiments = fields.ToManyField(ExperimentResource, 'experiments', full=False, null=True, blank=True)
    sampleSets = fields.ToManyField('iondb.rundb.api.SampleSetItemResource', 'sampleSets', full=False, null=True, blank=True)

    class Meta:
        queryset = models.Sample.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.Sample._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()
        allowed_methods = ['get','post','put']

    def hydrate_name(self,bundle):
        sampleName = bundle.data.get("name", "")
        if not sampleName:
            sampleName = bundle.data.get("displayedName","")
        if sampleName:
            sampleName = '_'.join(sampleName.split())
        bundle.data['name'] = sampleName

        return bundle

    def hydrate_displayedName(self,bundle):
        sampleDisplayedName = bundle.data.get("displayedName","")
        if not sampleDisplayedName:
            bundle.data['displayedName'] = bundle.data.get("name")
        return bundle

    def _prepare_for_sample_update(self, bundle, **kwargs):
            id = -1
            isNewSample = True
            if bundle.obj and bundle.obj.pk:
                id = bundle.obj.pk
                isNewSample = False
            else:
                id = bundle.data.get("id", "")
                if id:
                    isNewSample = False
                else:
                    id = kwargs.get("pk", "")
                    if id:
                        isNewSample = False
            if not isNewSample:
                input_sampleName          = bundle.data.get("name")
                input_sampleDisplayedName = bundle.data.get("displayedName",'')
                input_sampleExternalId    = bundle.data.get("externalId")
                input_samplestatus    = bundle.data.get("status",'')
                if input_sampleName or input_sampleExternalId:
                        existingSample = models.Sample.objects.filter(id = id)
                        print id
                        if existingSample:
                            existingSample = existingSample[0]
                            if not input_sampleName:
                                bundle.data["name"] = existingSample.name
                            if not input_sampleExternalId:
                                bundle.data["externalId"] = existingSample.externalId
                            if not input_sampleDisplayedName:
                                bundle.data["displayedName"] = existingSample.displayedName
                isValid,error = self.is_valid_sample(bundle)

                if isValid:
                    return isNewSample
                else:
                    logger.error('Failed SampleResource.obj_create()')
                    logger.error(traceback.format_exc())
                    raise ValidationError(json.dumps(error))

    def is_valid_sample(self, bundle, request=None):
        if not bundle.data:
            return {'__all__': 'Fatal Error, no bundle!'}

        sampleDesc = bundle.data.get("description","")
        querydict = {
                     'sampleName' : bundle.data['name'],
                     'sampleDisplayedName' : bundle.data['displayedName'],
                     'sampleExternalId' : bundle.data['externalId'],
                     'sampleDescription' : sampleDesc,
                     'sampleStatus' : bundle.data['status'],
                     }

        errors = sample_validator.validate_sample_data(querydict)

        if errors:
            return errors

    def obj_create( self, bundle, request = None, **kwargs ):
        bundle.obj = self._meta.object_class()

        for key, value in kwargs.items():
            setattr(bundle.obj, key, value)

        bundle = self.full_hydrate(bundle)
        logger.debug("SampleResources.obj_create()...AFTER full_hydrate() bundle.data=%s" %(bundle.data))

        # validate sample bundle
        isValid,error = self.is_valid_sample(bundle)

        if isValid:
            with transaction.commit_on_success():
                self.save_related(bundle)
                bundle.obj.save()
            return bundle
        else:
            logger.error('Failed SampleResource.obj_create()')
            logger.error(traceback.format_exc())
            raise ValidationError(json.dumps(error))

    def obj_update( self, bundle, **kwargs ):
        logger.debug("ENTER SampleResources.obj_update() bundle.data=%s" %(bundle.data))
        logger.debug("ENTER SampleResources.obj_update() kwargs=%s" %(kwargs))

        self._prepare_for_sample_update(bundle, **kwargs)
        bundle = super(SampleResource, self).obj_update(bundle, **kwargs)
        self.save_related(bundle)
        bundle.obj.save()
        return bundle


class SampleSetResource(ModelResource):
    SampleGroupType_CV = fields.ToOneField('iondb.rundb.api.SampleGroupType_CVResource', 'SampleGroupType_CV', full=False, null=True, blank=True)
    samples = fields.ToManyField('iondb.rundb.api.SampleSetItemResource', 'samples', full=False, null=True, blank=True)

    sampleCount = fields.IntegerField(readonly = True)
    sampleGroupTypeName = fields.CharField(readonly = True, attribute="sampleGroupTypeName", null=True, blank=True)

    libraryPrepInstrumentData = fields.ToOneField('iondb.rundb.api.SamplePrepDataResource', 'libraryPrepInstrumentData', full=True, null=True, blank=True)

    libraryPrepTypeDisplayedName = fields.CharField(readonly = True, attribute="libraryPrepTypeDisplayedName", null=True, blank=True)

    def dehydrate(self, bundle):
        sampleSetItems = bundle.obj.samples
        bundle.data['sampleCount'] = sampleSetItems.count() if sampleSetItems else 0

        groupType = bundle.obj.SampleGroupType_CV
        bundle.data['sampleGroupTypeName'] = groupType.displayedName if groupType else ""

        libPrepType_tuple = models.SampleSet.ALLOWED_LIBRARY_PREP_TYPES
        bundle.data['libraryPrepTypeDisplayedName'] = ""
        if bundle.data['libraryPrepType']:
            for i, (internalValue, displayedValue) in enumerate(libPrepType_tuple):
                if internalValue == bundle.data['libraryPrepType']:
                    bundle.data['libraryPrepTypeDisplayedName'] = displayedValue

        return bundle

    def obj_update(self, bundle, **kwargs):
        bundle = super(SampleSetResource, self).obj_update(bundle,**kwargs)

        pcrplates_barcodes_dict = bundle.data.get('sampleBarcodeMapping',None)
        if pcrplates_barcodes_dict:
            sampleSetItems = bundle.obj.samples
            sampleset_items = [samplesetitem.id for samplesetitem in sampleSetItems.all()]
            for pcr_plate_barcode in pcrplates_barcodes_dict:
                row = pcr_plate_barcode["sampleToBarcode"]["sampleRow"]
                col = pcr_plate_barcode["sampleToBarcode"]["sampleColumn"]
                barcodekit = pcr_plate_barcode["sampleToBarcode"]["barcodeKit"]
                barcode = pcr_plate_barcode["sampleToBarcode"]["barcode"]
                try:
                    dna_obj = models.dnaBarcode.objects.get(name=barcodekit,id_str=barcode)
                except:
                    print "Invalid barcodes"
                    logger.debug("Invalid Barcodes SampleSetResources.obj_update() bundle.data=%s" %(bundle.data))
                    next
                for samplesetitem_id in sampleset_items:
                    try:
                        item_to_update = models.SampleSetItem.objects.get(sampleSet__id=bundle.obj.id,pcrPlateRow=row,pcrPlateColumn=col)
                        if item_to_update:
                            item_to_update.dnabarcode = dna_obj
                            item_to_update.save()
                    except:
                        next
            bundle.obj.save()
        return bundle


    class Meta:
#        isSupported = models.GlobalConfig.get().enable_compendia_OCP
#
#        if isSupported:
#            queryset = models.SampleSet.objects.all().select_related(
#                    'SampleGroupType_CV__displayedName'
#                ).prefetch_related(
#                    'samples'
#                ).all()
#        else:
#            queryset = models.SampleSet.objects.all().exclude(SampleGroupType_CV__displayedName = "DNA_RNA").select_related(
#                    'SampleGroupType_CV__displayedName'
#                ).prefetch_related(
#                    'samples'
#                ).all()

        queryset = models.SampleSet.objects.all().select_related(
                'SampleGroupType_CV__displayedName'
            ).prefetch_related(
                'samples'
            ).all()

        resource_name = 'sampleset'

        #allow ordering and filtering by all fields
        field_list = models.SampleSet._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        always_return_data=True
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class SamplePrepDataResource(ModelResource):
    sampleSet = fields.ToOneField('iondb.rundb.api.SampleSetResource', 'sampleSet', full=False, null=True, blank=True)
    
    class Meta:
        queryset = models.SamplePrepData.objects.all()
        
        resource_name = 'sampleprepdata'
        
        #allow ordering and filtering by all fields
        field_list = models.SamplePrepData._meta.get_all_field_names()
        ordering = field_list
        
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class SampleSetItemResource(ModelResource):
    sampleSet = fields.ToOneField('iondb.rundb.api.SampleSetResource', 'sampleSet', full=False, null=True, blank=True)
    sample = fields.ToOneField('iondb.rundb.api.SampleResource', 'sample', full=False, null=True, blank=True)
    dnabarcode = fields.ToOneField('iondb.rundb.api.dnaBarcodeResource', "dnabarcode", full=False, null=True, blank=True)

    class Meta:
        queryset = models.SampleSetItem.objects.all().select_related('sampleSet', 'sample', 'dnabarcode')

        resource_name = 'samplesetitem'

        #allow ordering and filtering by all fields
        field_list = models.SampleSetItem._meta.get_all_field_names()
        field_list = field_list + ['sampleSet']
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class SampleSetItemInfoResource(SampleSetItemResource):
    sampleSetPk = fields.IntegerField(readonly = True, attribute="sampleSetPk", null=True, blank=True)
    sampleSetStatus = fields.CharField(readonly = True, attribute="sampleSetStatus", null=True, blank=True)

    samplePk = fields.IntegerField(readonly = True, attribute="samplePk", null=True, blank=True)
    sampleExternalId = fields.CharField(readonly = True, attribute="sampleExternalId", null=True, blank=True)
    sampleDisplayedName = fields.CharField(readonly = True, attribute="sampleDisplayedName", null=True, blank=True)
    sampleDescription = fields.CharField(readonly = True, attribute="sampleDescription", null=True, blank=True)
    relationshipRole = fields.CharField(readonly = True, attribute="relationshipRole", null=True, blank=True)
    relationshipGroup = fields.IntegerField(readonly = True, attribute="relationshipGroup", null=True, blank=True)
    dnabarcode = fields.ToOneField("iondb.rundb.api.dnaBarcodeResource", readonly=True, attribute="dnabarcode", null=True, blank=True)
    dnabarcodeKit = fields.CharField(readonly = True, attribute="dnaBarcodeKit", null=True, blank=True)


    def dehydrate(self, bundle):
        sampleSet = bundle.obj.sampleSet
        sample = bundle.obj.sample

        bundle.data['sampleSetPk'] = sampleSet.id if sampleSet else 0
        bundle.data['sampleSetStatus'] = sampleSet.status if sampleSet else ""

        bundle.data['samplePk'] = sample.id if sample else 0
        bundle.data['sampleExternalId'] = sample.externalId if sample else ""
        bundle.data['sampleDisplayedName'] = sample.displayedName if sample else ""
        bundle.data['sampleDescription'] = sample.description if sample else ""

        bundle.data['relationshipRole'] = bundle.obj.relationshipRole
        bundle.data['relationshipGroup'] = bundle.obj.relationshipGroup
        bundle.data['dnabarcode'] = bundle.obj.dnabarcode.id_str if bundle.obj.dnabarcode else ""
        bundle.data['dnabarcodeKit'] = bundle.obj.dnabarcode.name if bundle.obj.dnabarcode else ""

        sampleAttribute_list = models.SampleAttribute.objects.filter(isActive = True).order_by('id')

        attribute_dict = {}
        for attribute in sampleAttribute_list:
            try:
                sampleAttributeValue = models.SampleAttributeValue.objects.get(sample_id = sample, sampleAttribute_id = attribute)

            except:
                ##logger.debug("api - sampleAttributeValue NONE OK #1 for sample=%s; attribute=%s" %(str(sample), str(attribute)))

                sampleAttributeValue = None
                bundle.data[attribute.displayedName] = ""

            attr_value =  "attr_value_%s" % attribute.displayedName
            if sampleAttributeValue and sampleAttributeValue.value and sampleAttributeValue.value != 'None':
                bundle.data[attribute.displayedName] = sampleAttributeValue.value
                attribute_dict[attribute.displayedName] = sampleAttributeValue.value

            else:
                ##logger.debug("api - sampleAttributeValue NONE OK #2 for sample=%s; attribute=%s" %(str(sample), str(attribute)))

                bundle.data[attribute.displayedName] = ""
                attribute_dict[attribute.displayedName] = ""

        bundle.data["attribute_dict"] = attribute_dict

        return bundle

    class Meta:
        queryset = models.SampleSetItem.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.SampleSetItem._meta.get_all_field_names()

        fields = field_list + ['sampleSetPk',
                               'sampleSetStatus',
                               'samplePk',
                               'sampleExternalId',
                               'sampleDisplayedName',
                               'sampleDescription',
                               'relationshipRole',
                               'relationshipGroup']
        ordering = fields
        filtering = field_dict(fields)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

        allowed_methods = ['get']


class SampleGroupType_CVResource(ModelResource):
    sampleSets = fields.ToManyField('iondb.rundb.api.SampleSetResource', 'sampleSets', full=False, null=True, blank=True)
    sampleAnnotation_set = fields.ToManyField('iondb.rundb.api.SampleAnnotation_CVResource', 'sampleAnnotation_set', full=False, null=True, blank=True)

    class Meta:
#        isSupported = models.GlobalConfig.get().enable_compendia_OCP
#
#        if isSupported:
#            queryset = models.SampleGroupType_CV.objects.all()
#        else:
#            queryset = models.SampleGroupType_CV.objects.all().exclude(displayedName = "DNA_RNA").all()
        queryset = models.SampleGroupType_CV.objects.all().exclude(displayedName = "DNA_RNA").all()

        #allow ordering and filtering by all fields
        field_list = models.SampleGroupType_CV._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class SampleAnnotation_CVResource(ModelResource):
    sampleGroupType_CV = fields.ToOneField('iondb.rundb.api.SampleGroupType_CVResource', "sampleGroupType_CV", full=False, null=True, blank=True)

    class Meta:
        queryset = models.SampleAnnotation_CV.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.SampleAnnotation_CV._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class SampleAttributeResource(ModelResource):
    dataType = fields.ToOneField('iondb.rundb.api.SampleAttributeDataTypeResource', "dataType", full=False, null=True, blank=True)

    sampleCount = fields.IntegerField(readonly = True)
    dataType_name = fields.CharField(readonly = True, attribute="dataType_name", null=True, blank=True)

    def dehydrate(self, bundle):
        samplesWithAttribute = bundle.obj.samples
        bundle.data['sampleCount'] = samplesWithAttribute.count() if samplesWithAttribute else 0

        dataType = bundle.obj.dataType
        bundle.data['dataType_name'] = dataType.dataType if dataType else ""

        return bundle

    class Meta:
        queryset = models.SampleAttribute.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.SampleAttribute._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class SampleAttributeDataTypeResource(ModelResource):

    class Meta:
        queryset = models.SampleAttributeDataType.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.SampleAttributeDataType._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class ExperimentAnalysisSettingsResource(ModelResource):
    #associated experiment & results
    experiment = fields.ToOneField(ExperimentResource, 'experiment', full=False, null=True, blank=True)
    results = fields.ToManyField('iondb.rundb.api.ResultsResource', 'results_set', full=False, null=True, blank=True)

    class Meta:
        queryset = models.ExperimentAnalysisSettings.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.ExperimentAnalysisSettings._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class ReferenceGenomeResource(ModelResource):
    def apply_filters(self, request, applicable_filters):
        base_object_list = super(ReferenceGenomeResource, self).apply_filters(request, applicable_filters)
        query = request.GET.get('special', None)
        if query:
            qset = (
                Q(status__exact="Rebuilding index") |
                Q(index_version__exact=settings.TMAP_VERSION)
            )
            base_object_list = base_object_list.filter(qset)

        return base_object_list

    class Meta:
        limit = 0
        queryset = models.ReferenceGenome.objects.filter(index_version = settings.TMAP_VERSION)

        #allow ordering and filtering by all fields
        field_list = models.ReferenceGenome._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class ObsoleteReferenceGenomeResource(ModelResource):
    class Meta:
        limit = 0
        queryset = models.ReferenceGenome.objects.filter(enabled=True).exclude(index_version = settings.TMAP_VERSION)

        #allow ordering and filtering by all fields
        field_list = models.ReferenceGenome._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class LocationResource(ModelResource):
    class Meta:
        queryset = models.Location.objects.all()

        #allow ordering and filtering by all fields
        field_list = ['name','comment']
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class FileServerResource(ModelResource):
    class Meta:
        queryset = models.FileServer.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.FileServer._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class EmailAddressResource(ModelResource):
    class Meta:
        queryset = models.EmailAddress.objects.all()
        validation = FormValidation(form_class=forms.EmailAddress)

        field_list = models.EmailAddress._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class RigResource(ModelResource):

    def prepend_urls(self):
        urls = [url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/status%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_status'), name="api_dispatch_status"),
                url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/config%s$" % (self._meta.resource_name, trailing_slash()),
                            self.wrap_view('dispatch_config'), name="api_dispatch_config")
        ]

        return urls

    def dispatch_status(self, request, **kwargs):
        return self.dispatch('status', request, **kwargs)

    def get_status(self, request, **kwargs):
        bundle = self.build_bundle(request=request)
        rig = self.cached_obj_get(bundle, **self.remove_api_resource_names(kwargs))
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

        deserialized = self.deserialize(request, request.body, format=request.META.get('CONTENT_TYPE', 'application/json'))
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
        bundle = self.build_bundle(request=request)
        rig = self.cached_obj_get(bundle, **self.remove_api_resource_names(kwargs))
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

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()
        status_allowed_methods = ['get', 'put']
        config_allowed_methods = ['get']

class PluginResource(ModelResource):
    """Get a list of plugins"""
    isConfig = fields.BooleanField(readonly=True, attribute='isConfig')
    isPlanConfig = fields.BooleanField(readonly=True, attribute='isPlanConfig')
    isInstance = fields.BooleanField(readonly=True, attribute='isInstance')
    input = fields.CharField(readonly=True, attribute='isInstance')
    hasAbout = fields.BooleanField(readonly=True, attribute='hasAbout')
    versionedName = fields.CharField(readonly=True, attribute='versionedName')

    def prepend_urls(self):
        #this is meant for internal use only
        urls = [url(r"^(?P<resource_name>%s)/set/(?P<keys>\w[\w/-;]*)/type%s$" % (self._meta.resource_name,
                    trailing_slash()),self.wrap_view('get_type_set'), name="api_get_type_set"),
                url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w\./-]*)/extend/(?P<extend>\w[\w/-;]*)%s$" % (self._meta.resource_name,
                    trailing_slash()),self.wrap_view('dispatch_extend'), name="api_dispatch_extend"),
                url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/type%s$" % (self._meta.resource_name,
                    trailing_slash()), self.wrap_view('dispatch_type'), name="api_dispatch_type"),
                url(r"^(?P<resource_name>%s)/install%s$" % (self._meta.resource_name,
                    trailing_slash()), self.wrap_view('dispatch_install'), name="api_dispatch_install"),
                url(r"^(?P<resource_name>%s)/uninstall/(?P<pk>\w[\w/-]*)%s$" % (self._meta.resource_name,
                    trailing_slash()), self.wrap_view('dispatch_uninstall'), name="api_dispatch_uninstall_compat"),
                url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/uninstall%s$" % (self._meta.resource_name,
                    trailing_slash()), self.wrap_view('dispatch_uninstall'), name="api_dispatch_uninstall"),
                url(r"^(?P<resource_name>%s)/rescan%s$" % (self._meta.resource_name,
                    trailing_slash()), self.wrap_view('dispatch_rescan'), name="api_dispatch_rescan"),
                url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/info%s$" % (self._meta.resource_name,
                    trailing_slash()), self.wrap_view('dispatch_info'), name="api_dispatch_info"),
                url(r"^(?P<resource_name>%s)/show%s$" % (self._meta.resource_name,
                    trailing_slash()), self.wrap_view('dispatch_show'), name="api_plugins_show")
        ]

        return urls

    def dispatch_type(self, request, **kwargs):
        return self.dispatch('type', request, **kwargs)

    def dispatch_install(self, request, **kwargs):
        return self.dispatch('install', request, **kwargs)

    def dispatch_uninstall(self, request, **kwargs):
        return self.dispatch('uninstall', request, **kwargs)

    def dispatch_rescan(self, request, **kwargs):
        return self.dispatch('rescan', request, **kwargs)

    def dispatch_info(self, request, **kwargs):
        return self.dispatch('info', request, **kwargs)

    def dispatch_show(self, request, **kwargs):
        return self.dispatch('show', request, **kwargs)

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
            type["input"] =  urlresolvers.reverse('configure_plugins_plugin_configure', kwargs = {'pk':plugin.pk, 'action':'report'})

        return type

    def get_type(self, request, **kwargs):
        bundle = self.build_bundle(request=request)
        plugin = self.cached_obj_get(bundle, **self.remove_api_resource_names(kwargs))
        return self.create_response(request, self._get_type(plugin))

    def get_type_set(self, request, **kwargs):
        """Take a list of ; separated plugin IDs and return the type for each.
        In the event that one of them is missing, raise an error.
        """
        request_method = self.method_check(request, ['get']) # raises MethodNotAllowed
        plugin_set = kwargs['keys'].split(";")
        bundle = self.build_bundle(request=request)
        queryset = self.cached_obj_get_list(bundle).filter(pk__in=plugin_set)
        types = dict((p.pk, self._get_type(p)) for p in queryset)
        if any(isinstance(t, HttpGone) for t in types.values()):
            return HttpGone()
        else:
            return self.create_response(request, types)

    def dispatch_extend(self, request, **kwargs):

        bundle = self.build_bundle(request=request)
        extend_function= kwargs['extend']

        #if the PK was a int lookup by PK, if it was the name of a Plugin use the name instead.
        if kwargs["pk"].isdigit():
            plugin = self.cached_obj_get(bundle, pk=kwargs["pk"])
        else:
            plugin = self.cached_obj_get(bundle, name=kwargs["pk"])

        bucket = {}
        bucket["request_get"] = request.GET
        #assume json
        if request.method == "POST":
            bucket["request_post"] = json.loads(request.body)
        bucket["user"] = request.user
        bucket["request_method"] = request.method
        bucket["name"] = plugin.name
        bucket["version"] = plugin.version
        #not sure if we want this or not, keep it for now
        bucket["config"] = plugin.config

        module = imp.load_source(plugin.name, os.path.join(plugin.path,"extend.py"))
        #maybe make these classes instead with a set of methods to use
        func = getattr(module,extend_function)
        result = func(bucket)

        return self.create_response(request, result)

    def post_install(self, request, **kwargs):
        try:
            deserialized = self.deserialize(request, request.body, format=request.META.get('CONTENT_TYPE', 'application/json'))
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
                               date=timezone.now(),
                               active=False,
                               url=url,
                               status={'result': 'queued'},
                               )

        #now install the plugin
        tasks.downloadPlugin.delay(url, plugin, filename)

        # Task run async - return success
        return HttpAccepted()

    def delete_uninstall(self, request, **kwargs):
        bundle = self.build_bundle(request=request)
        obj = self.cached_obj_get(bundle, **self.remove_api_resource_names(kwargs))

        killed = pluginmanager.uninstall(obj)

        status = {"Status": killed}

        return self.create_response(request, status)

    def apply_filters(self, request, applicable_filters):
        """
        An ORM-specific implementation of ``apply_filters``.

        The default simply applies the ``applicable_filters`` as ``**kwargs``,
        but should make it possible to do more advanced things.
        """
        if request.GET.get('isConfig',None):
            value = applicable_filters.pop('isConfig__exact')
            results = self.get_object_list(request).filter(**applicable_filters)
            matches = []
            for r in results:
                print r.isConfig()
                if value and value == bool(r.isConfig()):
                    matches.append(r)
            return matches
        else:
            return self.get_object_list(request).filter(**applicable_filters)

    def get_rescan(self, request, **kwargs):
        pluginmanager.rescan()
        return HttpAccepted()

    def get_info(self, request, **kwargs):
        bundle = self.build_bundle(request=request)
        plugin = self.cached_obj_get(bundle, **self.remove_api_resource_names(kwargs))
        if not plugin:
            return HttpGone()
        info = plugin.info(use_cache=toBoolean(request.GET.get('use_cache', True)))
        if not info:
            return HttpGone()
        return self.create_response(request, info)

    def post_show(self, request, **kwargs):
        plugins = self.deserialize(request, request.body)
        logger.info(plugins)
        show_plugins = request.session.setdefault("show_plugins", {})
        show_plugins.update(plugins)
        request.session.modified = True

    class Meta:
        # Note API only sees active plugins
        queryset = models.Plugin.objects.filter(active=True)

        #allow ordering and filtering by all fields
        field_list = models.Plugin._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list + ['isConfig'])

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()
        type_allowed_methods = ['get',]
        install_allowed_methods = ['post',]
        uninstall_allowed_methods = ['delete']
        rescan_allowed_methods = ['get',]
        info_allowed_methods = ['get',]
        show_allowed_methods = ['post']

class PluginResultResource(ModelResource):
    result = fields.ToOneField(ResultsResource,'result')
    plugin = fields.ToOneField(PluginResource, 'plugin', full=False)
    owner = fields.ToOneField(UserResource, 'owner', full=False)

    # Computed fields
    path = fields.CharField(readonly=True, attribute='path')
    duration = fields.CharField(readonly=True, attribute='duration')

    # Helper methods for display / context
    resultName = fields.CharField(readonly=True, attribute='result__resultsName')
    reportLink = fields.CharField(readonly=True, attribute='result__reportLink')

    pluginName = fields.CharField(readonly=True, attribute='plugin__name')
    pluginVersion = fields.CharField(readonly=True, attribute='plugin__version')

    def prepend_urls(self):
        urls = [url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/rescan%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_rescan'), name="api_dispatch_rescan"),
                url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/control%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_control'), name="api_dispatch_control"),
        ]
        return urls

    def dispatch_rescan(self, request, **kwargs):
        return self.dispatch('rescan', request, **kwargs)

    def get_rescan(self, request, **kwargs):
        bundle = self.build_bundle(request=request)
        obj = self.cached_obj_get(bundle, **self.remove_api_resource_names(kwargs))
        try:
            size, inodes = obj._calc_size
            if size > 0:
                obj.size = size
                obj.inodes = inodes
                obj.save()
        except (OSError, IOError):
            raise Http404()

        return HttpAccepted()

    def dispatch_control(self, request, **kwargs):
        return self.dispatch('control', request, **kwargs)

    def get_control(self, request, **kwargs):
        bundle = self.build_bundle(request=request)
        obj = self.cached_obj_get(bundle, **self.remove_api_resource_names(kwargs))
        if obj.jobid is None:
            logger.error("PluginResult has no SGE jobid, not running?")
            return HttpGone()

        status, ret = remote.call_pluginStatus(obj.jobid)
        if not status:
            logger.error("Failed to get Plugin Status: %s", ret)
            raise Http404()

        return HttpAccepted()

    def post_control(self, request, **kwargs):
        bundle = self.build_bundle(request=request)
        obj = self.cached_obj_get(bundle, **self.remove_api_resource_names(kwargs))
        if obj.jobid is None:
            logger.error("PluginResult has no SGE jobid, not running?")
            raise HttpGone()

        status = remote.call_sgeStop(obj.jobid)
        #if status:
        obj.complete(state="Terminated")
        return HttpAccepted()


    def hydrate_state(self, bundle):
        # Call trigger methods for special states
        # - Started, Complete, Error
        value = bundle.data.get('state', None)
        if value is not None and value != bundle.obj.state:
            # Caller provided state change, trigger special eventss
            if value == 'Pending':
                bundle.obj.prepare(
                    config=bundle.data.get('config',None),
                    jobid=bundle.data.get('jobid',None)
                )
            elif value in ('Completed', 'Error'):
                bundle.obj.complete(state=value)
                # Ignore inodes/size, as they are computed by the model in complete
                bundle.data.pop('inodes')
                bundle.data.pop('size')
            elif value == 'Started':
                bundle.obj.start(bundle.data.get('jobid', None))
                bundle.obj.save()

        return bundle

    def hydrate_owner(self, bundle):
        if bundle.obj.pk:
            # Ignore existing objects here, do not allow overriding
            bundle.data['owner'] = bundle.obj.owner
            return bundle

        # Automatically get owner for new requests
        owner = bundle.request.user
        # Admin users can override.
        if owner.is_superuser and u'owner' in bundle.data:
            # handle resource_uri vs string username
            try:
                username = bundle.data['owner']
                owner = User.objects.get(username=username)
            except User.DoesNotExist:
                logger.warn("Failed to lookup pluginresult owner by name '%s'", username)

            bundle.data['owner'] = owner
            return bundle

        # NB: implicit else clause forces owner to current user
        bundle.data['owner'] = owner
        return bundle

    def hydrate_plugin(self, bundle):
        if bundle.obj.pk:
            bundle.data['plugin'] = bundle.obj.plugin
        else:
            if 'plugin' in bundle.data:
                if hasattr(bundle.data['plugin'], 'items'):
                    # Add implicit filter criteria
                    # Cannot run an plugin unless active and enabled
                    bundle.data['plugin'].update({'active':True, 'selected':True})
        return bundle

    def hydrate_result(self, bundle):
        if bundle.obj.pk:
            # Ignore existing objects here, do not allow overriding
            bundle.data['result'] = bundle.obj.result
        return bundle

    class Meta:
        queryset = models.PluginResult.objects.all()
        #allow ordering and filtering by all fields
        field_list = models.PluginResult._meta.get_all_field_names()
        field_list.extend(['result','plugin', 'path', 'duration', 'id']),
        #excludes = ['apikey',]
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

        list_allowed_methods = ['get', 'post'] # block delete, put
        rescan_allowed_methods = ['get',]
        control_allowed_methods = ['get', 'post', ]

class ApplicationGroupResource(ModelResource):
    applications = fields.ToManyField('iondb.rundb.api.RunTypeResource', 'applications', full=True, null = True)

    class Meta:
        queryset = models.ApplicationGroup.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.ApplicationGroup._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class RunTypeResource(ModelResource):
    applicationGroups = fields.ToManyField('iondb.rundb.api.ApplicationGroupResource', 'applicationGroups', full = False, null = True)

    class Meta:
        queryset = models.RunType.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.RunType._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class dnaBarcodeResource(ModelResource):

    class Meta:
        queryset = models.dnaBarcode.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.dnaBarcode._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

    def apply_filters(self, request, applicable_filters):
        base_object_list = super(dnaBarcodeResource, self).apply_filters(request, applicable_filters)

        query = request.GET.get('distinct', None)
        if query:
            base_object_list = base_object_list.order_by('name').distinct('name')

        return base_object_list


class PlannedExperimentValidation(Validation):
    def is_valid(self, bundle, request=None):
        if not bundle.data:
            return {'__all__': 'Fatal Error, no bundle!'}

        errors = {}
        barcodedSampleWarnings = []
        barcodedWarnings = []

        # validate required parameters
        isNewPlan=bundle.data.get('isNewPlan', False)
        isTemplate = bundle.data.get('isReusable', False)
        
        value = bundle.data.get("planName",'')
        if value or isNewPlan:
            err = plan_validator.validate_plan_name(value)
            if err:
                errors["planName"] = err

        value = bundle.data.get("chipType",'')
        if value or (isNewPlan and not isTemplate):
            err = plan_validator.validate_chip_type(value)
            if err:
                errors["chipType"] = err

        value = bundle.data.get("templatingKitName")
        if value or (isNewPlan and not isTemplate):
            err = plan_validator.validate_plan_templating_kit_name(value)
            if err:
                errors["templatingKitName"] = err

        # validate optional parameters
        for key, value in bundle.data.items():
            err = []
            if key == "sample":
                err = plan_validator.validate_sample_name(value)
            if key == "sampleTubeLabel":
                err = plan_validator.validate_sample_tube_label(value)
            if key == "barcodedSamples" and value:
                barcodedSamples = json.loads(value) if isinstance(value, basestring) else value
                keyErrMsg = "KeyError: Missing %s key"
                for sample in barcodedSamples:
                    barcodedSampleErrors = []
                    barcodedWarnings = []
                    barcodeSampleInfo = None
                    err.extend(plan_validator.validate_sample_name(sample))
                    try:
                        get_barcodedSamples = barcodedSamples[sample]['barcodes']
                    except KeyError, e:
                        if isNewPlan:
                            err.append({sample : keyErrMsg % e})
                        else:
                            barcodedSampleWarnings += [{sample : keyErrMsg % e}]
                        continue
                    get_barcodedSamples = [x.encode('UTF8') for x in get_barcodedSamples]
                    barcodeId = bundle.data.get("barcodeId","")
                    try:
                        barcodeSampleInfo = barcodedSamples[sample]['barcodeSampleInfo']
                    except KeyError, e:
                        if isNewPlan:
                            err.append({sample : keyErrMsg % e})
                        else:
                            barcodedSampleWarnings += [{sample : keyErrMsg % e}]
                            barcodeSampleInfo = None
                    if barcodeSampleInfo:
                        for barcode in get_barcodedSamples:
                            if barcode:
                                reference = None
                                eachBarcodeErr = []
                                warnings = []
                                isValid, errInvalidBarcode, item = sample_validator.validate_barcodekit_and_id_str(barcodeId, barcode)
                                if not isValid:
                                    eachBarcodeErr.append(str(errInvalidBarcode))
                                else:
                                    try:
                                        reference = barcodeSampleInfo[barcode]['reference']
                                        if reference:
                                            err_ref = plan_validator.validate_reference_short_name(reference)
                                            if err_ref:
                                                err_ref = "".join(err_ref)
                                                eachBarcodeErr.append(str(err_ref))
                                        else:
                                            warnings.append("Barcoded Sample Reference is empty")
                                    except KeyError, e:
                                        if isNewPlan:
                                            eachBarcodeErr.append(keyErrMsg % e)
                                        else:
                                            warnings.append(keyErrMsg % e)

                                    #validate targetRegionBedFile - Error if invalid / Warning if empty
                                    try:
                                        targetRegionBedFile = barcodedSamples[sample]['barcodeSampleInfo'][barcode]['targetRegionBedFile']
                                        if targetRegionBedFile:
                                            if reference:
                                                runType = bundle.data.get("runType")
                                                if not isTemplate:
                                                    err_targetRegionBedFile = plan_validator.validate_targetRegionBedFile_for_runType(targetRegionBedFile,runType,reference)
                                                    if err_targetRegionBedFile:
                                                        err_targetRegionBedFile = "".join(err_targetRegionBedFile)
                                                        eachBarcodeErr.append(err_targetRegionBedFile)
                                            else:
                                                eachBarcodeErr.append("Bed file exists but No Reference")
                                        if not targetRegionBedFile and reference:
                                            warnings.append("Barcoded Sample targetRegionBedFile is empty")
                                    except KeyError, e:
                                        if isNewPlan:
                                            eachBarcodeErr.append(keyErrMsg % e)
                                        else:
                                            warnings.append(keyErrMsg)
                                    except Exception, e:
                                        warnings.append(e)
    
                                    #validate hotSpotRegionBedFile - Error if invalid / Warning if empty
                                    try:
                                        hotSpotRegionBedFile = barcodedSamples[sample]['barcodeSampleInfo'][barcode]['hotSpotRegionBedFile']
                                        if hotSpotRegionBedFile:
                                            if targetRegionBedFile and reference:
                                                err_hotSpotRegionBedFile = plan_validator.validate_hotspot_bed(hotSpotRegionBedFile)
                                                if err_hotSpotRegionBedFile:
                                                    err_hotSpotRegionBedFile = "".join(err_hotSpotRegionBedFile)
                                                    eachBarcodeErr.append(err_hotSpotRegionBedFile)
                                            else:
                                                eachBarcodeErr.append("Hot spot exists but No Reference or Bed File")
                                        if not hotSpotRegionBedFile and (targetRegionBedFile or reference):
                                            warnings.append("Barcoded Sample hotSpotRegionBedFile value is empty")
                                    except KeyError, e:
                                        if isNewPlan:
                                            eachBarcodeErr.append(keyErrMsg % e)
                                        else:
                                            warnings.append(keyErrMsg % e)
                                    except Exception, e:
                                        warnings.append(e)

                                    #validate nucleotideType - Error if invalid / Warning if empty
                                    try:
                                        nucleotideType = barcodedSamples[sample]['barcodeSampleInfo'][barcode]['nucleotideType']
                                        if nucleotideType:
                                            applicationGroupName = bundle.data.get("applicationGroupDisplayedName","")
                                            runType = bundle.data.get("runType", "")
                                            err_nucleotideType, sample_nucleotideType = plan_validator.validate_sample_nucleotideType(nucleotideType, runType, applicationGroupName)
                                            if err_nucleotideType:
                                                err_nucleotideType = "".join(err_nucleotideType)
                                                eachBarcodeErr.append(err_nucleotideType)
                                            if not runType or not applicationGroupName:
                                                warnings.append("runType or applicationGroup is empty for nucleotideType (%s)" % nucleotideType)
                                        else:
                                            eachBarcodeErr.append("nucleotideType should not be empty")
                                    except KeyError, e:
                                        if isNewPlan:
                                            eachBarcodeErr.append(keyErrMsg % e)
                                        else:
                                            warnings.append(keyErrMsg % e)
                                    except Exception, e:
                                        warnings.append(e)
                                if eachBarcodeErr:
                                    barcodedSampleErrors += [{barcode : eachBarcodeErr}]
                                if warnings:
                                    barcodedWarnings += [{barcode : warnings}]

                    if barcodedSampleErrors:
                        err.append({sample : barcodedSampleErrors})
                    if barcodedWarnings:
                        barcodedSampleWarnings += [{sample : barcodedWarnings}]
            if key == "notes":
                err = plan_validator.validate_notes(value)
            if key == "flows" and value!="0":
                err = plan_validator.validate_flows(value)
            if key == "project" or key == "projects":
                projectNames = value if isinstance(value, basestring) else ','.join(value)

                project_errors, trimmed_projectNames = plan_validator.validate_projects(projectNames)
                err.extend(project_errors)
            if key == "barcodeId":
                err = plan_validator.validate_barcode_kit_name(value)
            if key == "sequencekitname":
                err = plan_validator.validate_sequencing_kit_name(value)
            if key == "applicationGroupDisplayedName":
                runType = bundle.data.get("runType")
                err = plan_validator.validate_application_group_for_runType(value, runType)
            if key == "sampleGroupingName":
                err = plan_validator.validate_sample_grouping(value)
            if key == "libraryReadLength":
                err = plan_validator.validate_libraryReadLength(value)
            if key == "templatingSize":
                err = plan_validator.validate_templatingSize(value)
            if key == "bedfile":
                runType = bundle.data.get("runType")
                reference = bundle.data.get("library")

                planStatus = bundle.data.get("planStatus")

                applicationGroupDisplayedName = bundle.data.get("applicationGroupDisplayedName", "")
                if planStatus != "run" and (not isTemplate):            
                    err = plan_validator.validate_targetRegionBedFile_for_runType(value, runType, reference, "", applicationGroupDisplayedName)
            if key == "library":
                    err = plan_validator.validate_reference_short_name(value)
                
            if err:
                errors[key] = err

        if errors:
            planName = bundle.data.get("planName",'')
            logger.error("plan validation errors for plan=%s: Errors=%s" %(planName, errors))
            raise ValidationError(json.dumps(errors))
        if barcodedSampleWarnings:
            bundle.data["Warnings"] = barcodedSampleWarnings

        return errors


class PlannedExperimentDbResource(ModelResource):
    # Backwards support - single project field
    project = fields.CharField(readonly=True, blank=True)

    projects = fields.ToManyField('iondb.rundb.api.ProjectResource', 'projects', full=False, null=True, blank=True)
    qcValues = fields.ToManyField('iondb.rundb.api.PlannedExperimentQCResource', 'plannedexperimentqc_set', full=True, null=True, blank=True)
    parentPlan = fields.CharField(blank=True, default=None)
    childPlans = fields.ListField(default=[])
    experiment = fields.ToOneField(ExperimentResource, 'experiment', full=False, null=True, blank=True)

    sampleSet = fields.ToOneField(SampleSetResource, 'sampleSet', full=False, null=True, blank=True)
    applicationGroup = fields.ToOneField(ApplicationGroupResource, 'applicationGroup', full=False, null=True, blank=True)
    sampleGrouping = fields.ToOneField(SampleGroupType_CVResource, 'sampleGrouping', full=False, null=True, blank=True)

    def hydrate_m2m(self, bundle):
        if 'projects' in bundle.data or 'project' in bundle.data:
            # Promote projects from names to something tastypie recognizes in hydrate_m2m
            projects_list = []
            for k in ["projects", "project"]:
                value = bundle.data.get(k, [])

                if isinstance(value, basestring):
                    value = [value]
                for p in value:
                    if p not in projects_list:
                        projects_list.append(p)

            project_objs = []
            if projects_list:
                user = getattr(bundle.request, 'user', None)
                if user is not None and user.is_superuser:
                    username = bundle.data.get('username')
                    if username is not None:
                        try:
                            user = User.objects.get(username=username)
                        except User.DoesNotExist:
                            pass
                project_objs = models.Project.bulk_get_or_create(projects_list, user)
        else:
            project_objs = bundle.obj.projects.all()

        bundle.data['projects'] = project_objs
        bundle.data['project'] = None

        return super(PlannedExperimentDbResource, self).hydrate_m2m(bundle)


    def dehydrate_projects(self, bundle):
        """Return a list of project names rather than any specific objects"""
        #logger.debug("Dehydrating %s with projects %s", bundle.obj, projects_names)
        return [str(project.name) for project in bundle.obj.projects.all()]

    def dehydrate_project(self, bundle):
        """Return the first project name"""
        try:
            firstProject = str(bundle.obj.projects.all()[0].name)
        except IndexError:
            firstProject = ""

        return firstProject

    def build_filters(self, filters=None):
        if filters is None:
            filters = {}

        filter_platform = filters.get("platform") or filters.get("instrument")
        if 'platform' in filters:
            del filters['platform']

        orm_filters = super(PlannedExperimentDbResource, self).build_filters(filters)
        if filter_platform:
            orm_filters.update({
                "custom_platform": (Q(experiment__platform="") | Q(experiment__platform__iexact=filter_platform))
            })

        return orm_filters


    def apply_filters(self, request, filters):
        custom_query = filters.pop("custom_platform", None)
        base_object_list = super(PlannedExperimentDbResource, self).apply_filters(request, filters)

        if custom_query is not None:
            base_object_list = base_object_list.filter(custom_query)

        name_or_id = request.GET.get('name_or_id')
        if name_or_id is not None:
            qset = (
                Q(planName__icontains=name_or_id) |
                Q(planDisplayedName__icontains=name_or_id) |
                Q(planShortID__icontains=name_or_id)
            )
            base_object_list = base_object_list.filter(qset)

        combinedLibraryTubeLabel = request.GET.get('combinedLibraryTubeLabel', None)
        if combinedLibraryTubeLabel is not None:
            combinedLibraryTubeLabel = combinedLibraryTubeLabel.strip()
            qset = (
                Q(sampleSet__combinedLibraryTubeLabel__icontains=combinedLibraryTubeLabel)
            )
            base_object_list = base_object_list.filter(qset)


        return base_object_list


    class Meta:
        queryset = models.PlannedExperiment.objects.select_related(
            'experiment',
            'latestEAS'
        ).prefetch_related(
            'projects',
            'plannedexperimentqc_set',
            'plannedexperimentqc_set__qcType',
            'experiment__samples'
        ).all()

        transfer_allowed_methods = ['get','post']
        always_return_data=True

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()
        field_list = field_list + ['sampleSet']
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()
        validation = PlannedExperimentValidation()


class PlannedExperimentResource(PlannedExperimentDbResource):
    autoAnalyze = fields.BooleanField()
    barcodedSamples = fields.CharField(blank=True, null=True, default='')
    barcodeId = fields.CharField(blank=True, null=True, default='')
    bedfile = fields.CharField(blank=True, default='')
    chipType = fields.CharField(default='')
    flows = fields.IntegerField(default=0)
    forward3primeadapter = fields.CharField(blank=True, null=True, default='')
    library = fields.CharField(blank=True, null=True, default='')
    libraryKey = fields.CharField(blank=True, default='')
    tfKey = fields.CharField(blank=True, default='')
    librarykitname = fields.CharField(blank=True, null=True, default='')
    notes = fields.CharField(blank=True, null=True, default='')
    regionfile = fields.CharField(blank=True, default='')
    reverse3primeadapter = fields.CharField(readonly = True, default='')
    reverselibrarykey = fields.CharField(readonly = True, default='')
    sample = fields.CharField(blank=True, null=True, default='')
    sampleDisplayedName = fields.CharField(blank=True, null=True, default='')
    selectedPlugins = fields.CharField(blank=True, null=True, default='')
    sequencekitname = fields.CharField(blank=True, null=True, default='')
    variantfrequency = fields.CharField(readonly = True, default='')
    isDuplicateReads = fields.BooleanField()
    base_recalibration_mode = fields.CharField(blank=True, null=True, default='')
    realign = fields.BooleanField()
    flowsInOrder = fields.CharField(blank=True, null=True, default='')

    sampleSetDisplayedName = fields.CharField(readonly = True, blank = True, null = True)
    sampleSetGroupType = fields.CharField(readonly = True, blank = True, null = True)

    applicationGroupDisplayedName = fields.CharField(readonly = True, blank=True, null=True)
    sampleGroupingName = fields.CharField(readonly = True, blank=True, null=True)

    libraryPrepType = fields.CharField(readonly = True, blank=True, null=True)
    libraryPrepTypeDisplayedName = fields.CharField(readonly = True, blank=True, null=True)
    
    platform = fields.CharField(blank=True, null=True)

    chefInfo = fields.DictField(default={})

    def prepend_urls(self):
        return [
            url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/transfer%s$" % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('dispatch_transfer'), name="api_dispatch_transfer"),
        ]

    def dispatch_transfer(self, request, **kwargs):
        return self.dispatch('transfer', request, **kwargs)
    
    def get_transfer(self, request, **kwargs):
        # runs on destination TS to update plan-related objects and return any errors
        plan = self.get_object_list(request).get(pk=kwargs["pk"])
        status = update_transferred_plan(plan, request)
        
        return self.create_response(request, status)
    
    def post_transfer(self, request, **kwargs):
        # runs on origin TS to initiate plan transfer
        destination = request.POST.get('destination')
        if not destination:
            return HttpBadRequest('Missing transfer destination')
    
        plan = self.get_object_list(request).get(pk=kwargs["pk"])
        if plan is None:
            return HttpGone()
        
        try:
            # transfer plan to destination
            bundle = self.build_bundle(obj=plan, request=request)
            bundle = prepare_for_copy(self.full_dehydrate(bundle))
            serialized = self.serialize(None, bundle, 'application/json')

            status = transfer_plan(plan, serialized, destination, request.user.username)

        except Exception as err:
            return HttpBadRequest(err)

        return self.create_response(request, status)


    def dehydrate(self, bundle):
        try:
            experiment = bundle.obj.experiment
            bundle.data['autoAnalyze'] = experiment.autoAnalyze
            bundle.data['chipType'] = experiment.chipType
            bundle.data['flows'] = experiment.flows
            bundle.data['flowsInOrder'] = experiment.flowsInOrder
            bundle.data['notes'] = experiment.notes
            bundle.data['sequencekitname'] = experiment.sequencekitname
            bundle.data['platform'] = experiment.platform

            latest_eas = bundle.obj.latestEAS
            if not latest_eas:
                latest_eas = experiment.get_EAS()

            bundle.data['barcodedSamples'] = latest_eas.barcodedSamples if latest_eas else ""
            bundle.data['barcodeId'] = latest_eas.barcodeKitName if latest_eas else ""
            bundle.data['bedfile'] = latest_eas.targetRegionBedFile if latest_eas else ""
            bundle.data['forward3primeadapter'] = latest_eas.threePrimeAdapter if latest_eas else ""
            bundle.data['library'] = latest_eas.reference if latest_eas else ""
            bundle.data['libraryKey'] = latest_eas.libraryKey if latest_eas else ""
            bundle.data['tfKey'] = latest_eas.tfKey if latest_eas else ""
            bundle.data['librarykitname'] = latest_eas.libraryKitName if latest_eas else ""
            bundle.data['regionfile'] = latest_eas.hotSpotRegionBedFile if latest_eas else ""
            bundle.data['isDuplicateReads'] = latest_eas.isDuplicateReads if latest_eas else False
            bundle.data['base_recalibration_mode'] = latest_eas.base_recalibration_mode if latest_eas else "no_recal"
            bundle.data['realign'] = latest_eas.realign if latest_eas else False

            bundle.data['beadfindargs'] = latest_eas.beadfindargs if latest_eas else ""
            bundle.data['thumbnailbeadfindargs'] = latest_eas.thumbnailbeadfindargs if latest_eas else ""
            bundle.data['analysisargs'] = latest_eas.analysisargs if latest_eas else ""
            bundle.data['thumbnailanalysisargs'] = latest_eas.thumbnailanalysisargs if latest_eas else ""
            bundle.data['prebasecallerargs'] = latest_eas.prebasecallerargs if latest_eas else ""
            bundle.data['prethumbnailbasecallerargs'] = latest_eas.prethumbnailbasecallerargs if latest_eas else ""
            bundle.data['calibrateargs'] = latest_eas.calibrateargs if latest_eas else ""
            bundle.data['thumbnailcalibrateargs'] = latest_eas.thumbnailcalibrateargs if latest_eas else ""
            bundle.data['basecallerargs'] = latest_eas.basecallerargs if latest_eas else ""
            bundle.data['thumbnailbasecallerargs'] = latest_eas.thumbnailbasecallerargs if latest_eas else ""
            bundle.data['alignmentargs'] = latest_eas.alignmentargs if latest_eas else ""
            bundle.data['thumbnailalignmentargs'] = latest_eas.thumbnailalignmentargs if latest_eas else ""
            bundle.data['ionstatsargs'] = latest_eas.ionstatsargs if latest_eas else ""
            bundle.data['thumbnailionstatsargs'] = latest_eas.thumbnailionstatsargs if latest_eas else ""

            if latest_eas and latest_eas.barcodeKitName:
                bundle.data['sample'] = ""
                bundle.data['sampleDisplayedName'] = ""
            else:
                if experiment.samples.all():
                    bundle.data['sample'] = experiment.samples.all()[0].name
                    bundle.data['sampleDisplayedName'] = experiment.samples.all()[0].displayedName
                else:
                    bundle.data['sample'] = ""
                    bundle.data['sampleDisplayedName'] = ""

            bundle.data['selectedPlugins'] = latest_eas.selectedPlugins if latest_eas else ""

            if latest_eas:
                selectedPlugins = latest_eas.selectedPlugins

                pluginInfo = selectedPlugins.get("variantCaller", {})

                if pluginInfo:
                    userInput = pluginInfo.get("userInput", {})
                    if userInput:
                        bundle.data['variantfrequency'] = userInput.get("variationtype", "")

        except models.Experiment.DoesNotExist:
            logger.error('Missing experiment for Plan %s(%s)' % (bundle.obj.planName, bundle.obj.pk))

        sampleSet = bundle.obj.sampleSet
        bundle.data['sampleSetDisplayedName'] = sampleSet.displayedName if sampleSet else ""

        bundle.data['libraryPrepType'] = ""
        bundle.data['libraryPrepTypeDisplayedName'] = ""
        if sampleSet:
            bundle.data['sampleSetGroupType'] = sampleSet.SampleGroupType_CV.displayedName if sampleSet.SampleGroupType_CV else ""
            bundle.data['libraryPrepType'] = sampleSet.libraryPrepType
            
            if sampleSet.libraryPrepType:
                allowed_lib_prep_type = sampleSet.ALLOWED_LIBRARY_PREP_TYPES
                for internalValue, displayedValue in allowed_lib_prep_type:
                    if internalValue == sampleSet.libraryPrepType:
                        bundle.data['libraryPrepTypeDisplayedName'] = displayedValue

            bundle.data['combinedLibraryTubeLabel'] = sampleSet.combinedLibraryTubeLabel

        applicationGroup = bundle.obj.applicationGroup
        bundle.data['applicationGroupDisplayedName'] = applicationGroup.description if applicationGroup else ""

        sampleGrouping = bundle.obj.sampleGrouping
        bundle.data['sampleGroupingName'] = sampleGrouping.displayedName if sampleGrouping else ""

        # IonChef parameters from Experiment
        if experiment.chefInstrumentName:
            try:
                chefFields = [f.name for f in experiment._meta.fields if f.name.startswith('chef')]
                for field in chefFields:
                    bundle.data['chefInfo'][field] = getattr(experiment,field)
            except:
                logger.error('Error getting Chef fields for Plan %s(%s)' % (bundle.obj.planName, bundle.obj.pk))

        return bundle


    def hydrate_autoAnalyze(self, bundle):
        if 'autoAnalyze' in bundle.data:
            bundle.data['autoAnalyze'] = toBoolean(bundle.data['autoAnalyze'], True)
        return bundle

    def hydrate_barcodedSamples(self, bundle):
        barcodedSamples = bundle.data.get('barcodedSamples', "")
        # soft validation, will not raise errors
        # valid barcodedSamples format is {'sample_XXX':{'barcodes':['IonXpress_001', ]}, }
        valid = True
        if barcodedSamples:
            #for both string and unicode
            if isinstance(barcodedSamples, basestring):
                #example: "barcodedSamples":"{'s1':{'barcodes': ['IonSet1_01']},'s2': {'barcodes': ['IonSet1_02']},'s3':{'barcodes': ['IonSet1_03']}}"
                barcodedSamples = ast.literal_eval(barcodedSamples)

            try:
                barcoded_bedfiles = {}
                for k,v in barcodedSamples.items():
                    if isinstance(v['barcodes'],list):
                        for bc in v['barcodes']:
                            if not isinstance(bc,basestring):
                                logger.debug("api.PlannedExperiment.hydrate_barcodedSamples() - INVALID bc - NOT an str - bc=%s" %(bc))
                                valid = False
                    else:
                        logger.debug("api.PlannedExperiment.hydrate_barcodedSamples() -  INVALID v[barcodes] - NOT a list!!! v[barcodes]=%s" %(v['barcodes']))
                        valid = False

                    if isinstance(v.get('barcodeSampleInfo'),dict):
                        for barcode in v['barcodeSampleInfo'].values():
                            reference = barcode.get('reference') or bundle.data.get('reference','')
                            for target_or_hotspot in ['targetRegionBedFile', 'hotSpotRegionBedFile']:
                                bedfile = barcode.get(target_or_hotspot)
                                if bedfile:
                                    found_key = '%s__%s' % (os.path.basename(bedfile), reference)
                                    if found_key not in barcoded_bedfiles:
                                        barcoded_bedfiles[found_key] = self._get_bedfile_path(bedfile, reference)
                                    if barcoded_bedfiles[found_key]:
                                        barcode[target_or_hotspot] = barcoded_bedfiles[found_key]

            except:
                logger.error(traceback.format_exc())
                valid = False

            #validate for malformed JSON "barcodedSamples" - raise error if it is New Plan
            bundle.data['barcodedSamples'] = barcodedSamples
        return bundle

    def hydrate_barcodeId(self, bundle):
        if bundle.data.get('barcodeId') and bundle.data['barcodeId'].lower() == "none":
            bundle.data['barcodeId'] = ""
        return bundle

    def hydrate_bedfile(self, bundle):
        bedfile = bundle.data.get('bedfile')
        if bedfile:
            if bedfile.lower() == "none":
                bundle.data['bedfile'] = ""
            else:
                bedfile_path = self._get_bedfile_path(bedfile, bundle.data.get('reference',''))
                if bedfile_path:
                    bundle.data['bedfile'] = bedfile_path

            bundle.data['targetRegionBedFile'] = bundle.data['bedfile']

        return bundle

    def hydrate_chipType(self, bundle):
        if bundle.data.get('chipType') and bundle.data['chipType'].lower() == "none":
            bundle.data['chipType'] = ""
        return bundle

    def hydrate_library(self,bundle):
        if bundle.data.get('library') and bundle.data['library'].lower() == "none":
            bundle.data['library'] = ""
        return bundle

    def hydrate_librarykitname(self, bundle):
        if bundle.data.get('librarykitname') and bundle.data['librarykitname'].lower() == "none":
            bundle.data['librarykitname'] = ""
        return bundle

    def hydrate_regionfile(self, bundle):
        bedfile = bundle.data.get('regionfile')
        if bedfile:
            if bedfile.lower() == "none":
                bundle.data['regionfile'] = ""
            else:
                bedfile_path = self._get_bedfile_path(bedfile, bundle.data.get('reference',''))
                if bedfile_path:
                    bundle.data['regionfile'] = bedfile_path

            bundle.data['hotSpotRegionBedFile'] = bundle.data['regionfile']

        return bundle

    def hydrate_selectedPlugins(self, bundle):
        selectedPlugins = bundle.data.get('selectedPlugins', "")
        # soft validation, will not raise errors
        # valid selectedPlugins format is {'plugin_XXX': {'name': 'plugin_XXX', 'userInput': {} }, }
        valid = True
        if selectedPlugins:
            try:
                for k,v in selectedPlugins.items():
                    name = v['name']
                    userInput = v.get('userInput',{})
                    if not isinstance(userInput, (dict, list, basestring)):
                        valid = False
            except:
                valid = False

            bundle.data['selectedPlugins'] = selectedPlugins if valid else ""
        return bundle

    def hydrate_sequencekitname(self, bundle):
        if bundle.data.get('sequencekitname') and bundle.data['sequencekitname'].lower() == "none":
            bundle.data['sequencekitname'] = ""
        return bundle

    def hydrate_isDuplicateReads(self, bundle):
        if 'isDuplicateReads' in bundle.data:
            bundle.data['isDuplicateReads'] = toBoolean(bundle.data['isDuplicateReads'], False)
        return bundle

    def hydrate_realign(self, bundle):
        if 'realign' in bundle.data:
            bundle.data['realign'] = toBoolean(bundle.data['realign'], False)
        return bundle

    def hydrate_sampleSet(self, bundle):
        sampleSetDisplayedName = bundle.data.get('sampleSetDisplayedName', "")

        logger.debug("api.PlannedExperimentResource.hydrate_sampleSet() sampleSetDisplaydName=%s" %(sampleSetDisplayedName))

        if sampleSetDisplayedName:
            sampleSets = models.SampleSet.objects.filter(displayedName = sampleSetDisplayedName)
            if sampleSets:
                bundle.data['sampleSet'] = sampleSets[0]

        return bundle

    def hydrate_metaData(self, bundle):
        #logger.debug("api.PlannedExperimentResource.hydrate_metaData() metaData=%s" %(bundle.data.get('metaData', "")))        
        plan_metaData = bundle.data.get('metaData', "")
        valid = True
        if plan_metaData:
            try:
                if isinstance(plan_metaData, basestring):
                    plan_metaData_dict = ast.literal_eval(plan_metaData)
                    plan_metaData = simplejson.dumps(plan_metaData_dict)
            except:
                logger.error(traceback.format_exc())
                valid = False
                raise ValidationError("Error: Invalid JSON value for field metaData=%s in plannedExperiment with planName=%s" %(plan_metaData, bundle.data.get("planName", "")))

            bundle.data['metaData'] = plan_metaData if valid else ""
        return bundle

    def hydrate_platform(self, bundle):
        if 'platform' in bundle.data:
            platform = bundle.data['platform'].upper()
            if platform and platform.lower() == "none":
                platform = ""
            bundle.data['platform'] = platform
        return bundle

    def hydrate(self, bundle):
        #boolean handling for API posting
        for key in ['planExecuted', 'isReverseRun', 'isReusable', 'isFavorite', 'isSystem', 'isSystemDefault', 'isPlanGroup']:
            if key in bundle.data:
                bundle.data[key] = toBoolean(bundle.data[key], False)
        for key in ['preAnalysis', 'usePreBeadfind', 'usePostBeadfind']:
            if key in bundle.data:
                bundle.data[key] = toBoolean(bundle.data[key], True)

        #populate renamed fields
        if 'barcodeId' in bundle.data:
            bundle.data['barcodeKitName'] = bundle.data['barcodeId']
        if 'bedfile' in bundle.data:
            bundle.data['targetRegionBedFile'] = bundle.data['bedfile']
        if 'forward3primeadapter' in bundle.data:
            bundle.data['threePrimeAdapter'] = bundle.data['forward3primeadapter']
        if 'library' in bundle.data:
            bundle.data['reference'] = bundle.data['library']
        if 'regionfile' in bundle.data:
            bundle.data['hotSpotRegionBedFile'] = bundle.data['regionfile']
        if 'librarykitname' in bundle.data:
            bundle.data['libraryKitName'] = bundle.data['librarykitname']
        
        applicationGroupDisplayedName = bundle.data.get('applicationGroupDisplayedName', "")

        if applicationGroupDisplayedName:
            applicationGroups = models.ApplicationGroup.objects.filter(description__iexact = applicationGroupDisplayedName.strip())
            if applicationGroups:
                bundle.data['applicationGroup'] = applicationGroups[0]

        sampleGroupingName = bundle.data.get('sampleGroupingName', "")

        if sampleGroupingName:
            sampleGroupings = models.SampleGroupType_CV.objects.filter(displayedName__iexact = sampleGroupingName.strip())
            if sampleGroupings:
                bundle.data['sampleGrouping'] = sampleGroupings[0]

        #strip leading zeros of sampleTubeLabel
        if (bundle.data.get('planStatus', "") != "run" or not bundle.data['planExecuted']):
            sampleTubeLabel = bundle.data.get("sampleTubeLabel", "")
            if sampleTubeLabel:
                ##sampleTubeLabel = sampleTubeLabel.strip().lstrip("0")
                sampleTubeLabel = sampleTubeLabel.strip()
                bundle.data["sampleTubeLabel"] = sampleTubeLabel

        return bundle


    def _get_bedfile_path(self, bedfile, reference):
        bedfile_path = ''
        if os.path.exists(bedfile):
            bedfile_path = bedfile
        else:
            name = os.path.basename(bedfile)
            path = "/%s/unmerged/detail/%s" % (reference, name)
            content = models.Content.objects.filter(publisher__name="BED", path=path)
            if content:
                bedfile_path = content[0].file
        return bedfile_path


    def _isNewPlan(self, bundle, **kwargs):
        isNewPlan = True
        if bundle.obj and bundle.obj.pk:
            isNewPlan = False
        elif bundle.data.get("id") or kwargs.get("pk"):
            isNewPlan = False

        return isNewPlan


    def obj_create( self, bundle, request = None, **kwargs ):
        """
        A ORM-specific implementation of ``obj_create``.
        """
        bundle.obj = self._meta.object_class()

        for key, value in kwargs.items():
            setattr(bundle.obj, key, value)

        bundle = self.full_hydrate(bundle)
        bundle.data['isNewPlan'] = self._isNewPlan(bundle, **kwargs)

        logger.debug("PlannedExperimentResource.obj_create()...bundle.data=%s" % bundle.data)

        # validate plan bundle
        self.is_valid(bundle)
        
        # Save FKs just in case.
        self.save_related(bundle)

        try:
            with transaction.commit_on_success():
                bundle.obj.save()
                bundle.obj.save_plannedExperiment_association(bundle.data.pop('isNewPlan'), **bundle.data)
                bundle.obj.update_plan_qcValues(**bundle.data)
        except:
            logger.error('Failed PlannedExperimentResource.obj_create()')
            logger.error(traceback.format_exc())
            return HttpBadRequest()

        # Now pick up the M2M bits.
        m2m_bundle = self.hydrate_m2m(bundle)
        self.save_m2m(m2m_bundle)
        return bundle


    def obj_update( self, bundle, **kwargs ):
        
        bundle = super(PlannedExperimentResource, self).obj_update(bundle, **kwargs)
        
        logger.debug("PlannedExperimentResource.obj_update() bundle.data=%s" % bundle.data)
        
        bundle.obj.save_plannedExperiment_association(False, **bundle.data)
        return bundle


class AvailableIonChefPlannedExperimentResource(PlannedExperimentResource):

    def build_filters(self, filters=None):
        if filters is None:
            filters = {}
        orm_filters = super(AvailableIonChefPlannedExperimentResource, self).build_filters(filters)

        if not hasattr(self, 'kit_names'):
            self.kit_names = [kit.name for kit in models.KitInfo.objects.filter(kitType = "IonChefPrepKit")]
        orm_filters["templatingKitName__in"] = self.kit_names

        return orm_filters

    class Meta:
        queryset = models.PlannedExperiment.objects.filter(planStatus__in=['pending'], isReusable = False, planExecuted = False)

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()
        ordering = field_list
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class AvailableIonChefPlannedExperimentSummaryResource(ModelResource):

    def build_filters(self, filters=None):
        if filters is None:
            filters = {}
        orm_filters = super(AvailableIonChefPlannedExperimentSummaryResource, self).build_filters(filters)

        if not hasattr(self, 'kit_names'):
            self.kit_names = [kit.name for kit in models.KitInfo.objects.filter(kitType = "IonChefPrepKit")]
        orm_filters["templatingKitName__in"] = self.kit_names

        return orm_filters

    class Meta:
        queryset = models.PlannedExperiment.objects.filter(planStatus__in=['pending'], isReusable = False, planExecuted = False)

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()

        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

        metadata_allowed_methods = ['get',]


class IonChefPlanTemplateResource(PlannedExperimentResource):

    def build_filters(self, filters=None):
        if filters is None:
            filters = {}
        orm_filters = super(IonChefPlanTemplateResource, self).build_filters(filters)

        if not hasattr(self, 'kit_names'):
            self.kit_names = [kit.name for kit in models.KitInfo.objects.filter(kitType = "IonChefPrepKit")]
        orm_filters["templatingKitName__in"] = self.kit_names

        return orm_filters

    class Meta:
        queryset = models.PlannedExperiment.objects.filter(isReusable = True)

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()
        ordering = field_list
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class IonChefPlanTemplateSummaryResource(ModelResource):

    def build_filters(self, filters=None):
        if filters is None:
            filters = {}
        orm_filters = super(IonChefPlanTemplateSummaryResource, self).build_filters(filters)

        if not hasattr(self, 'kit_names'):
            self.kit_names = [kit.name for kit in models.KitInfo.objects.filter(kitType = "IonChefPrepKit")]
        orm_filters["templatingKitName__in"] = self.kit_names

        return orm_filters

    class Meta:
        queryset = models.PlannedExperiment.objects.filter(isReusable = True)

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()
        ordering = field_list
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

        metadata_allowed_methods = ['get',]


class AvailableOneTouchPlannedExperimentResource(PlannedExperimentResource):

    def build_filters(self, filters=None):
        if filters is None:
            filters = {}
        orm_filters = super(AvailableOneTouchPlannedExperimentResource, self).build_filters(filters)

        if not hasattr(self, 'kit_names'):
            self.kit_names = [kit.name for kit in models.KitInfo.objects.filter(kitType = "TemplatingKit")]
        orm_filters["templatingKitName__in"] = self.kit_names

        return orm_filters

    class Meta:
        queryset = models.PlannedExperiment.objects.filter(planStatus__in=['', 'planned'], isReusable = False, planExecuted = False)

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()
        ordering = field_list
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class OneTouchPlanTemplateResource(PlannedExperimentResource):

    def build_filters(self, filters=None):
        if filters is None:
            filters = {}
        orm_filters = super(OneTouchPlanTemplateResource, self).build_filters(filters)

        if not hasattr(self, 'kit_names'):
            self.kit_names = [kit.name for kit in models.KitInfo.objects.filter(kitType = "TemplatingKit")]
        orm_filters["templatingKitName__in"] = self.kit_names

        return orm_filters

    class Meta:
        queryset = models.PlannedExperiment.objects.filter(isReusable = True)

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()
        ordering = field_list
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class AvailableOneTouchPlannedExperimentSummaryResource(ModelResource):

    def build_filters(self, filters=None):
        if filters is None:
            filters = {}
        orm_filters = super(AvailableOneTouchPlannedExperimentSummaryResource, self).build_filters(filters)

        if not hasattr(self, 'kit_names'):
            self.kit_names = [kit.name for kit in models.KitInfo.objects.filter(kitType = "TemplatingKit")]
        orm_filters["templatingKitName__in"] = self.kit_names

        return orm_filters

    class Meta:
        queryset = models.PlannedExperiment.objects.filter(planStatus__in=['', 'planned'], isReusable = False, planExecuted = False)

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()

        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

        metadata_allowed_methods = ['get',]


class OneTouchPlanTemplateSummaryResource(ModelResource):

    def build_filters(self, filters=None):
        if filters is None:
            filters = {}
        orm_filters = super(OneTouchPlanTemplateSummaryResource, self).build_filters(filters)

        if not hasattr(self, 'kit_names'):
            self.kit_names = [kit.name for kit in models.KitInfo.objects.filter(kitType = "TemplatingKit")]
        orm_filters["templatingKitName__in"] = self.kit_names

        return orm_filters

    class Meta:
        queryset = models.PlannedExperiment.objects.filter(isReusable = True)

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()
        ordering = field_list
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

        metadata_allowed_methods = ['get',]


class AvailablePlannedExperimentSummaryResource(ModelResource):
    class Meta:
        queryset = models.PlannedExperiment.objects.filter(planStatus__in=['', 'planned'], isReusable = False, planExecuted = False)

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()

        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

        metadata_allowed_methods = ['get',]


class PlanTemplateSummaryResource(ModelResource):
    class Meta:
        queryset = models.PlannedExperiment.objects.filter(isReusable = True)

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()

        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

        metadata_allowed_methods = ['get',]


class PlanTemplateBasicInfoResource(ModelResource):
    barcodeKitName = fields.CharField(readonly = True, attribute="barcodeKitName", null=True, blank=True)
    templatePrepInstrumentType = fields.CharField(readonly = True, attribute="templatePrepInstrumentType", null=True, blank=True)
    sequencingInstrumentType = fields.CharField(readonly = True, attribute="sequencingInstrumentType", null=True, blank=True)
    notes = fields.CharField(readonly = True, blank=True, null=True, default='')

    reference = fields.CharField(readonly = True, blank=True, null=True, default='')
    #Target Regions BED File: old name is bedfile
    #Hotspot Regions BED File: old name is regionfile
    targetRegionBedFile = fields.CharField(readonly = True, blank=True, null=True, default='')
    hotSpotRegionBedFile = fields.CharField(readonly = True, blank=True, null=True, default='')

    irAccountName = fields.CharField(readonly = True, attribute="irAccountName", null=True, blank=True)

    applicationGroup = fields.ToOneField('iondb.rundb.api.ApplicationGroupResource', 'applicationGroup', null=True, blank=True, full=False)

    applicationGroupDisplayedName = fields.CharField(readonly = True, attribute="applicationGroupDisplayedName", null=True, blank=True)
    sampleGroupName = fields.CharField(readonly = True, attribute="sampleGroupName", null=True, blank=True)

    def dehydrate(self, bundle):
        planTemplate = bundle.obj
        eas = planTemplate.latestEAS

        bundle.data['barcodeKitName'] = eas.barcodeKitName if eas else ''

        bundle.data['templatePrepInstrumentType'] = ""
        templatingKitName = planTemplate.templatingKitName

        if templatingKitName:
            kits = models.KitInfo.objects.filter(name = templatingKitName)
            if kits:
                kit = kits[0]
                if kit.kitType == "TemplatingKit":
                    bundle.data['templatePrepInstrumentType'] = "OneTouch"
                elif kit.kitType == "IonChefPrepKit":
                    bundle.data['templatePrepInstrumentType'] = "IonChef"

        bundle.data['sequencingInstrumentType'] = ""
        chipType = planTemplate.experiment.chipType if planTemplate.experiment else ''

        if chipType:
            chips = models.Chip.objects.filter(name = chipType)
            if chips:
                chip = chips[0]
                bundle.data['sequencingInstrumentType'] = chip.instrumentType.upper()
        else:
            bundle.data['sequencingInstrumentType'] = planTemplate.experiment.platform.upper()

        notes = planTemplate.experiment.notes if planTemplate.experiment else ''
        bundle.data['notes'] = notes

        bundle.data['irAccountName'] = ""

        if eas:
            bundle.data['reference'] = eas.reference
            bundle.data['targetRegionBedFile'] = eas.targetRegionBedFile
            bundle.data['hotSpotRegionBedFile'] = eas.hotSpotRegionBedFile

            templateSelectedPlugins = eas.selectedPlugins
            if "IonReporterUploader" in templateSelectedPlugins:
                templateIRUConfig = templateSelectedPlugins.get("IonReporterUploader", {})
                pluginDict = templateIRUConfig

                templateUserInput = pluginDict.get("userInput", "")
                if templateUserInput:
                    bundle.data['irAccountName'] = templateUserInput.get("accountName", "")

        applicationGroup = bundle.obj.applicationGroup
        sampleGroup = bundle.obj.sampleGrouping
        bundle.data['applicationGroupDisplayedName'] = applicationGroup.description if applicationGroup else ''
        bundle.data['sampleGroupName'] = sampleGroup.displayedName if sampleGroup else ''

        return bundle

    class Meta:
        queryset = models.PlannedExperiment.objects.filter(isReusable = True)

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()

        fields = field_list + ['planTemplatePk',
                               'planDisplayedName',
                               'runType',
                               'runMode',
                               'irworkflow',
                               'date',
                               'username',
                               'isReusable',
                               'isSystem',
                               'isFavorite',
                               'barcodeKitName',
                               'templatePrepInstrumentType',
                               'sequencingInstrumentType',
                               'irAccountName',
                               'applicationGroupDisplayedName',
                               'applicationGroup',
                               'sampleGroupName']
        ordering = fields
        filtering = field_dict(fields)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

        allowed_methods = ['get']



class TorrentSuite(ModelResource):
    """Allow system updates via API"""

    def prepend_urls(self):
        urls = [url(r"^(?P<resource_name>%s)%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_update'), name="api_dispatch_update"),
                url(r"^(?P<resource_name>%s)/version%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_version'), name="api_dispatch_version")
        ]

        return urls

    def dispatch_update(self, request, **kwargs):
        return self.dispatch('update', request, **kwargs)

    def dispatch_version(self, request, **kwargs):
        return self.dispatch('version', request, **kwargs)

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

    def get_version(self, request, **kwargs):
        from ion import version
        ret = {'meta_version':version}
        return self.create_response(request, ret)

    class Meta:

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()
        update_allowed_methods = ['get', 'put']
        version_allowed_methods = ['get']

    def get_schema(self, request, **kwargs):
        schema = {
            "allowed_detail_http_methods": [],
            "allowed_list_http_methods": [
                "get",
                "put",
            ],
            "default_format": "application/json",
            "default_limit": 0,
            "fields": {
                "locked": {
                    "blank": True,
                    "default": "No default provided.",
                    "help_text": "Unicode string data. Ex: \"Hello World\"",
                    "nullable": True,
                    "readonly": True,
                    "type": "boolean",
                    "unique": False
                },
                "logs": {
                    "blank": True,
                    "default": "No default provided.",
                    "help_text": "Unicode string data. Ex: \"Hello World\"",
                    "nullable": True,
                    "readonly": True,
                    "type": "boolean",
                    "unique": False
                },
                "meta_version": {
                    "blank": True,
                    "default": "No default provided.",
                    "help_text": "Unicode string data. Ex: \"Hello World\"",
                    "nullable": True,
                    "readonly": True,
                    "type": "string",
                    "unique": False
                },
                "versions": {
                    "blank": True,
                    "default": "No default provided.",
                    "help_text": "Unicode string data. Ex: \"Hello World\"",
                    "nullable": True,
                    "readonly": True,
                    "type": "string",
                    "unique": False
                },
            },
            "filtering": {},
            "ordering": []
        }
        return HttpResponse(json.dumps(schema), content_type="application/json")

class PublisherResource(ModelResource):

    class Meta:
        queryset = models.Publisher.objects.all()
        detail_uri_name = 'name' # default is pk
        install_allowed_methods = ['post']

        # allow ordering and filtering by all fields
        field_list = models.Publisher._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

    def prepend_urls(self):
        return [
            url(r"^(?P<resource_name>%s)/install%s$" % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('dispatch_install'), name="api_dispatch_install")
        ]

    def dispatch_install(self, request, **kwargs):
        return self.dispatch('install', request, **kwargs)

    def post_install(self, request, **kwargs):
        try:
            deserialized = self.deserialize(request, request.body, format=request.META.get('CONTENT_TYPE', 'application/json'))
            data = dict_strip_unicode_keys(deserialized)
        except UnsupportedFormat:
            return HttpBadRequest()

        uploadurl = data.get("url", None)
        filename = data.get("file", None)

        if (not uploadurl) and (not filename):
            # Must specify either url or filename
            return HttpBadRequest()

        tasks.downloadPublisher.delay(url, filename)

        # Task run async - return success
        return HttpAccepted()

class ContentUploadResource(ModelResource):
    pub = fields.CharField(readonly = True, attribute='publisher__name')

    def dehydrate(self, bundle):
        bundle.data['name'] = os.path.basename(bundle.obj.file_path)
        bundle.data['upload_type'] = bundle.obj.upload_type()
        meta = bundle.data.get('meta')
        if meta:
            bundle.data['upload_date'] = meta.get('upload_date')
        
        return bundle

    class Meta:
        limit = 0
        queryset = models.ContentUpload.objects.select_related('publisher').order_by('-pk')

        # allow ordering and filtering by all fields
        field_list = models.ContentUpload._meta.get_all_field_names() + ['pub']
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class ContentResource(ModelResource):
    publisher = ToOneField(PublisherResource, 'publisher')
    contentupload = ToOneField(ContentUploadResource, 'contentupload')

    def prepend_urls(self):
        return [
            url(r"^(?P<resource_name>%s)/register/(?P<upload_id>\w[\w/-]*)%s$" % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('dispatch_register'), name="api_dispatch_register"),
        ]

    def dispatch_register(self, request, **kwargs):
        return self.dispatch('register', request, **kwargs)

    def post_register(self, request, **kwargs):
        try:
            upload_id = kwargs['upload_id']
            upload = models.ContentUpload.objects.get(pk=upload_id)

            deserialized = self.deserialize(request, request.body, format=request.META.get('CONTENT_TYPE', 'application/json'))
            data=dict_strip_unicode_keys(deserialized)

            file_path = data.get('file','')
            if not os.path.exists(file_path):
                raise Exception("File not found %s" % file_path)

            content_kwargs = {
                'contentupload': upload,
                'publisher': upload.publisher,
                'meta': upload.meta,
                'file': file_path,
                'path': data.get('path') or os.path.basename(file_path)
            }
            content_kwargs['meta'].update(data.get('meta', {}))

            content = models.Content(**content_kwargs)
            content.save()

            return HttpAccepted()

        except Exception as e:
            return HttpBadRequest(str(e))

    class Meta:
        queryset = models.Content.objects.all()

        register_allowed_methods = ['post']

        # allow ordering and filtering by all fields
        field_list = models.Content._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


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

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

#201203 - SequencingKitResource is now obsolete
class SequencingKitResource(ModelResource):
    class Meta:
        queryset = models.SequencingKit.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.SequencingKit._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

#201203 - LibraryKitResource is now obsolete
class LibraryKitResource(ModelResource):
    class Meta:
        queryset = models.LibraryKit.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.LibraryKit._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class KitInfoResource(ModelResource):
    parts = fields.ToManyField('iondb.rundb.api.KitPartResource', 'kitpart_set', full=True)

    class Meta:
        queryset = models.KitInfo.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.KitInfo._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class KitPartResource(ModelResource):
    #parent kit
    kit = fields.ToOneField(KitInfoResource, 'kit', full=False)

    class Meta:
        queryset = models.KitPart.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.KitPart._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class SequencingKitInfoResource(ModelResource):
    parts = fields.ToManyField('iondb.rundb.api.KitPartResource', 'kitpart_set', full=True)
    class Meta:
        queryset = models.KitInfo.objects.filter(kitType='SequencingKit')

        #allow ordering and filtering by all fields
        field_list = models.KitInfo._meta.get_all_field_names()
        ordering = field_list
        filtering = {'kitType' : ['SequencingKit'] }
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


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
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class ActiveSequencingKitInfoResource(ModelResource):
    parts = fields.ToManyField('iondb.rundb.api.KitPartResource', 'kitpart_set', full=True)
    class Meta:
        queryset = models.KitInfo.objects.filter(kitType='SequencingKit', isActive=True)

        #allow ordering and filtering by all fields
        field_list = models.KitInfo._meta.get_all_field_names()
        ordering = field_list
        filtering = {'kitType' : ['SequencingKit'] }
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

#offering API returning pre-filtered set enables instrument to continue using simple code for the REST api
class ActivePGMSequencingKitInfoResource(ModelResource):
    parts = fields.ToManyField('iondb.rundb.api.KitPartResource', 'kitpart_set', full=True)
    class Meta:
        queryset = models.KitInfo.objects.filter(kitType='SequencingKit', isActive=True, instrumentType__in=['', 'pgm'])

        #allow ordering and filtering by all fields
        field_list = models.KitInfo._meta.get_all_field_names()
        ordering = field_list
        filtering = {'kitType' : ['SequencingKit'] }
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

#offering API returning pre-filtered set enables instrument to continue using simple code for the REST api
class ActiveProtonSequencingKitInfoResource(ModelResource):
    parts = fields.ToManyField('iondb.rundb.api.KitPartResource', 'kitpart_set', full=True)
    class Meta:
        queryset = models.KitInfo.objects.filter(kitType='SequencingKit', isActive=True, instrumentType__in=['', 'proton'])

        #allow ordering and filtering by all fields
        field_list = models.KitInfo._meta.get_all_field_names()
        ordering = field_list
        filtering = {'kitType' : ['SequencingKit'] }
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class IonChefPrepKitInfoResource(ModelResource):
    parts = fields.ToManyField('iondb.rundb.api.KitPartResource', 'kitpart_set', full=True)
    class Meta:
        queryset = models.KitInfo.objects.filter(kitType='IonChefPrepKit')

        #allow ordering and filtering by all fields
        field_list = models.KitInfo._meta.get_all_field_names()
        ordering = field_list
        filtering = {'kitType' : ['IonChefPrepKit'] }
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class ActiveIonChefPrepKitInfoResource(ModelResource):
    parts = fields.ToManyField('iondb.rundb.api.KitPartResource', 'kitpart_set', full=True)
    class Meta:
        queryset = models.KitInfo.objects.filter(kitType='IonChefPrepKit', isActive=True)

        #allow ordering and filtering by all fields
        field_list = models.KitInfo._meta.get_all_field_names()
        ordering = field_list
        filtering = {'kitType' : ['IonChefPrepKit'] }
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class ActiveIonChefLibraryPrepKitInfoResource(ModelResource):
    parts = fields.ToManyField('iondb.rundb.api.KitPartResource', 'kitpart_set', full=True)
    class Meta:
        queryset = models.KitInfo.objects.filter(kitType='LibraryPrepKit', samplePrep_instrumentType__icontains="IC", isActive=True)

        #allow ordering and filtering by all fields
        field_list = models.KitInfo._meta.get_all_field_names()
        ordering = field_list
        filtering = {'kitType' : ['LibraryPrepKit'] }
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()
        
class LibraryKitInfoResource(ModelResource):
    parts = fields.ToManyField('iondb.rundb.api.KitPartResource', 'kitpart_set', full=True)

    class Meta:
        queryset = models.KitInfo.objects.filter(kitType='LibraryKit')

        #allow ordering and filtering by all fields
        field_list = models.KitInfo._meta.get_all_field_names()
        ordering = field_list
        filtering = {'kitType' : ['LibraryKit'] }
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


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
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class ActiveLibraryKitInfoResource(ModelResource):
    parts = fields.ToManyField('iondb.rundb.api.KitPartResource', 'kitpart_set', full=True)

    class Meta:
        queryset = models.KitInfo.objects.filter(kitType='LibraryKit', isActive=True)

        #allow ordering and filtering by all fields
        field_list = models.KitInfo._meta.get_all_field_names()
        ordering = field_list
        filtering = {'kitType' : ['LibraryKit'] }
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class ActivePGMLibraryKitInfoResource(ModelResource):
    parts = fields.ToManyField('iondb.rundb.api.KitPartResource', 'kitpart_set', full=True)

    class Meta:
        queryset = models.KitInfo.objects.filter(kitType='LibraryKit', isActive=True, instrumentType__in=['', 'pgm'])

        #allow ordering and filtering by all fields
        field_list = models.KitInfo._meta.get_all_field_names()
        ordering = field_list
        filtering = {'kitType' : ['LibraryKit'] }
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class ActiveProtonLibraryKitInfoResource(ModelResource):
    parts = fields.ToManyField('iondb.rundb.api.KitPartResource', 'kitpart_set', full=True)

    class Meta:
        queryset = models.KitInfo.objects.filter(kitType='LibraryKit', isActive=True, instrumentType__in=['', 'proton'])

        #allow ordering and filtering by all fields
        field_list = models.KitInfo._meta.get_all_field_names()
        ordering = field_list
        filtering = {'kitType' : ['LibraryKit'] }
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class ThreePrimeadapterResource(ModelResource):
    class Meta:
        queryset = models.ThreePrimeadapter.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.ThreePrimeadapter._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class LibraryKeyResource(ModelResource):
    class Meta:
        queryset = models.LibraryKey.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.LibraryKey._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class MessageResource(ModelResource):

    class Meta:
        queryset = models.Message.objects.all()

        # allow ordering and filtering by all fields
        field_list = models.Message._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class IonReporter(Resource):
    """Allow system updates via API"""

    def prepend_urls(self):
        urls = [url(r"^(?P<resource_name>%s)%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_update'), name="api_dispatch_update"),
                url(r"^(?P<resource_name>%s)/version%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_version'), name="api_dispatch_version")
        ]

        return urls

    def dispatch_version(self, request, **kwargs):
        return self.dispatch('version', request, **kwargs)

    def get_version(self, request, **kwargs):
        plugin = request.GET.get('plugin', '')
        status, workflows = tasks.IonReporterVersion(plugin)
        if not status:
            raise ImmediateHttpResponse(HttpBadRequest(workflows))
        return self.create_response(request, workflows)

    def dispatch_update(self, request, **kwargs):
        return self.dispatch('update', request, **kwargs)

    def get_update(self, request, **kwargs):

        status, workflows = tasks.IonReporterWorkflows(autorun=False)

        if not status:
            raise ImmediateHttpResponse(HttpBadRequest(workflows))


        return self.create_response(request, workflows)

    class Meta:

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()
        #the name at the start is important
        update_allowed_methods = ['get']
        version_allowed_methods = ['get']


class CompositeProjectsResource(ModelResource):
    class Meta:
        queryset = models.Project.objects.all()
        field_list = ['id', 'modified', 'name']
        fields = field_list
        ordering = field_list
        filtering = field_dict(field_list)

class CompositeTFMetricsResource(ModelResource):
    class Meta:
        queryset = models.TFMetrics.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.TFMetrics._meta.get_all_field_names()
        fields = field_list
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class CompositeLibMetricsResource(ModelResource):
    class Meta:
        queryset = models.LibMetrics.objects.all()

        #allow ordering and filtering by all fields
        field_list = ['id', 'q20_bases', 'i100Q20_reads',
            'q20_mean_alignment_length', 'aveKeyCounts']
        fields = field_list
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class CompositeAnalysisMetricsResource(ModelResource):

    def dehydrate(self, bundle):
        a = bundle.obj
        bundle.data['total_wells'] = a.bead + a.empty + a.excluded + a.pinned + a.ignored
        return bundle

    class Meta:
        queryset = models.AnalysisMetrics.objects.all()

        #allow ordering and filtering by all fields
        field_list = ['id', 'bead', 'empty', 'excluded', 'pinned', 'ignored',
            'lib', 'libFinal', 'live']
        fields = field_list
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class CompositeQualityMetricsResource(ModelResource):
    class Meta:
        queryset = models.QualityMetrics.objects.all()

        #allow ordering and filtering by all fields
        field_list = ['id', 'q0_bases', 'q20_bases', 'q20_mean_read_length',
        'q0_mean_read_length', 'q0_reads', 'q20_reads']
        fields = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class CompositeExperimentAnalysisSettingsResource(ModelResource):
    class Meta:
        queryset = models.ExperimentAnalysisSettings.objects.all()
        fields = ['reference', 'barcodeKitName']
        ordering = fields
        filtering = field_dict(fields)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class CompositeResultResource(ModelResource):

    projects = fields.ToManyField(CompositeProjectsResource, 'projects', full=True)
    eas = fields.ToOneField(CompositeExperimentAnalysisSettingsResource, 'eas', full=True)
    analysismetrics = fields.ToOneField(CompositeAnalysisMetricsResource, 'analysismetrics', full=True, null=True)
    libmetrics = fields.ToOneField(CompositeLibMetricsResource, 'libmetrics', full=True, null=True)
    qualitymetrics = fields.ToOneField(CompositeQualityMetricsResource, 'qualitymetrics', full=True, null=True)

    def dehydrate(self, bundle):
        bundle.data['analysis_metrics'] = bundle.data['analysismetrics']
        bundle.data['quality_metrics'] = bundle.data['qualitymetrics']
        return bundle

    class Meta:
        queryset = models.Results.objects.all().select_related('eas')

        #allow ordering and filtering by all fields
        field_list = ['id', 'resultsName', 'processedflows', 'timeStamp',
            'projects', 'status', 'reportLink', 'representative', 'eas',
            'reportStatus', 'autoExempt', 'analysismetrics',
            'libmetrics', 'qualitymetrics']
        fields = field_list
        ordering = field_list
        filtering = field_dict(field_list)

        #this should check for admin rights
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


dm_types = [("sigproc", dmactions_types.SIG),
            ("basecall", dmactions_types.BASE),
            ("output", dmactions_types.OUT),
            ("misc", dmactions_types.INTR)]

class CompositeDataManagementResource(ModelResource):
    expDir = fields.CharField(readonly = True, attribute='experiment__expDir')
    expName = fields.CharField(readonly = True, attribute='experiment__expName')
    sigproc_state = fields.CharField(readonly = True, default='Unknown')
    misc_state = fields.CharField(readonly = True, default='Unknown')
    basecall_state = fields.CharField(readonly = True, default='Unknown')
    output_state = fields.CharField(readonly = True, default='Unknown')
    sigproc_keep = fields.CharField(readonly = True, attribute='sigproc_keep', null=True)
    misc_keep = fields.CharField(readonly = True, attribute='misc_keep', null=True)
    basecall_keep = fields.CharField(readonly = True, attribute='basecall_keep', null=True)
    output_keep = fields.CharField(readonly = True, attribute='output_keep', null=True)
    in_process = fields.BooleanField(default=False)

    def dehydrate(self, bundle):
        if bundle.data['diskusage'] is not None and bundle.obj.experiment.diskusage is not None:
            expDir_usage = bundle.obj.experiment.diskusage
            if bundle.obj.isThumbnail:
                # for thumbnails want only the size of file in expDir/thumbnail/, get from dmfilestat or estimate
                sig_filestat = bundle.obj.get_filestat(dmactions_types.SIG)
                bundle.data['diskusage'] += int(sig_filestat.diskspace) if sig_filestat.diskspace else 0.1*expDir_usage
            else:
                bundle.data['diskusage'] += expDir_usage

        for t, dm_type in dm_types:
            try:
                dmfilestat = bundle.obj.get_filestat(dm_type)
                bundle.data['%s_state' % t] = dmfilestat.get_action_state_display()
                if not dmfilestat.isdisposed():
                    bundle.data['%s_keep' % t] = dmfilestat.getpreserved()
                if dmfilestat.in_process():
                    bundle.data['in_process'] = True
                bundle.data['%s_diskspace' % t] = dmfilestat.diskspace
            except:
                pass

        return bundle

    def apply_filters(self, request, applicable_filters):
        base_object_list = super(CompositeDataManagementResource, self).apply_filters(request, applicable_filters)

        name = request.GET.get('search_name', None)
        if name is not None:
            qset = (
                Q(experiment__expName__iregex=name) |
                Q(resultsName__iregex=name)
            )
            base_object_list = base_object_list.filter(qset)

        for t, dm_type in dm_types:
            state_filter = request.GET.get('%s_filter' % t, None)
            if state_filter:
                qset = Q(dmfilestat__dmfileset__type=dm_type)
                if state_filter == "K":
                    if dm_type == dmactions_types.SIG:
                        qset = qset & Q(experiment__storage_options='KI')
                    else:
                        qset = qset & Q(dmfilestat__preserve_data=True)
                    qset = qset & Q(dmfilestat__action_state__in=['L','S','N','A','SE','EG','E'])
                elif state_filter == 'P':
                    qset = qset &  Q(dmfilestat__action_state__in=['AG','DG','EG','SA','SE','S','N','A'])
                else:
                    qset = qset &  Q(dmfilestat__action_state__in=[state_filter])

                base_object_list = base_object_list.filter(qset)

        return base_object_list.distinct()

    class Meta:
        queryset = models.Results.objects.exclude(experiment__expDir="") \
            .select_related('experiment').select_related('dmfilestat_set')

        #allow ordering and filtering by all fields
        field_list = ['id', 'resultsName', 'timeStamp', 'diskusage', 'expDir', 'expName']
        fields = field_list
        ordering = field_list
        filtering = field_dict(field_list)

        #this should check for admin rights
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class DataManagementHistoryResource(ModelResource):
    resultsName = fields.CharField(readonly=True, attribute='content_object__resultsName', null=True)

    def apply_filters(self, request, applicable_filters):
        name = applicable_filters.pop('content_object__resultsName__exact', None)
        base_object_list = super(DataManagementHistoryResource, self).apply_filters(request, applicable_filters)

        if name is not None:
            results = models.Results.objects.filter(resultsName__iregex=name)
            base_object_list = base_object_list.filter(object_pk__in=results)

        return base_object_list.distinct()

    class Meta:
        queryset = models.EventLog.objects.for_model(models.Results)

        field_list = ['created', 'text', 'username', 'resultsName']
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class ClusterInfoHistoryResource(ModelResource):
    name = fields.CharField(readonly=True, attribute='content_object__name', null=True)

    def prepend_urls(self):
        urls = [
            url(r"^(?P<resource_name>%s)/clear%s$" % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('dispatch_clear'), name="api_dispatch_clear"),
        ]
        return urls

    def dispatch_clear(self, request, **kwargs):
        return self.dispatch('clear', request, **kwargs)

    def get_clear(self, request, **kwargs):
        queryset = models.EventLog.objects.for_model(models.Cruncher)
        queryset.delete()
        return HttpAccepted()

    def dehydrate(self, bundle):
        text = bundle.data['text'].split('<br>')
        if len(text)>0:
            bundle.data['state'] = text[0].split()[-1]
            bundle.data['error'] = text[1][7:] if text[1].startswith('Error:') else ''
            for line in text[2:]:
                try:
                    k,v = line.split(':')
                    bundle.data[k.strip()] = v.strip()
                except:
                    pass
        return bundle

    def apply_filters(self, request, applicable_filters):
        name = applicable_filters.pop('content_object__name__exact', None)
        base_object_list = super(ClusterInfoHistoryResource, self).apply_filters(request, applicable_filters)

        if name is not None:
            nodes = models.Cruncher.objects.filter(name=name)
            base_object_list = base_object_list.filter(object_pk__in=nodes)

        return base_object_list.distinct()

    class Meta:
        queryset = models.EventLog.objects.for_model(models.Cruncher)

        clear_allowed_methods = ['get']

        field_list = ['created', 'name']
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class MonitorPlannedExperimentResource(ModelResource):
    class Meta:
        queryset = models.PlannedExperiment.objects.all()

        #allow ordering and filtering by all fields
        field_list = ['id', 'runType']
        fields = field_list
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()
        validation = PlannedExperimentValidation()

def monitor_comparator(bundle):
    return bundle.data['timeStamp']

chips = dict((c.name, c) for c in models.Chip.objects.all())

class MonitorExperimentResource(ModelResource):
    plan = fields.ToOneField(MonitorPlannedExperimentResource, 'plan', full=True, null=True, blank=True)

    def dehydrate(self, bundle):
        chip = chips.get(bundle.obj.chipType, None) or chips.get(bundle.obj.chipType[:3], None)
        if chip:
            bundle.data['chipDescription'] = chip.description
            bundle.data['chipInstrumentType'] = chip.instrumentType
        else:
            bundle.data['chipDescription'] = bundle.obj.chipType
            bundle.data['chipInstrumentType'] = ""

        try:
            qcThresholds = dict((qc.qcType.qcName, qc.threshold) for qc in
                            bundle.obj.plan.plannedexperimentqc_set.all())
        except:
            qcThresholds = {}
        bundle.data['qcThresholds'] = qcThresholds

        return bundle

    class Meta:
        queryset = models.Experiment.objects.all()

        #allow ordering and filtering by all fields
        field_list = ['id', 'expName', 'displayName', 'date', 'library', 'ftpStatus',
            'pgmName', 'storage_options', 'sample', 'flows', 'chipType',
            'notes', 'results', 'runMode', 'barcodeId', 'resultDate', 'star', 'platform'
        ]
        fields = field_list
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()
        validation = PlannedExperimentValidation()

class MonitorResultResource(ModelResource):
    experiment = fields.ToOneField(MonitorExperimentResource, 'experiment', full=True)
    eas = fields.ToOneField(CompositeExperimentAnalysisSettingsResource, 'eas', full=True)
    analysismetrics = fields.ToOneField(CompositeAnalysisMetricsResource, 'analysismetrics', full=True, null=True)
    libmetrics = fields.ToOneField(CompositeLibMetricsResource, 'libmetrics', full=True, null=True)
    qualitymetrics = fields.ToOneField(CompositeQualityMetricsResource, 'qualitymetrics', full=True, null=True)

    barcodeId = fields.CharField(readonly = True, attribute='eas__barcodeKitName', null=True)
    library = fields.CharField(readonly = True, attribute='eas__reference', null=True)

    projects = fields.ToManyField(CompositeProjectsResource, 'projects', full=True)

    def get_object_list(self, request):
        one_week_ago = datetime.datetime.now() - datetime.timedelta(weeks=1)
        two_days_ago = datetime.datetime.now() - datetime.timedelta(days=2)
        return super(MonitorResultResource, self).get_object_list(
            request).filter(
                Q(timeStamp__gte=two_days_ago) |
                (
                    ~(  Q(experiment__ftpStatus="Complete") |
                        Q(experiment__ftpStatus="Completed") |
                        Q(experiment__ftpStatus="User Aborted") |
                        Q(experiment__ftpStatus="Missing File(s)") |
                        Q(experiment__ftpStatus="Lost Chip Connection")
                    ) & Q(experiment__date__gte=one_week_ago)
                )
            )

    class Meta:
        #TODO: CONSIDER MAKING A SOUTH MIGRATION TO UPDATE EXPERIMENT WITH runMode=="" for .exclude(runMode="") using Experiment.objects.filter(runMode="").update(ftpStatus="Missing File(s)")
        queryset = models.Results.objects.select_related(
                'experiment',
                'eas',
                'analysismetrics',
                'libmetrics',
                'qualitymetrics',
                'experiment__plan',
                'experiment__plan__plannedexperimentqc_set',
                'experiment__plan__plannedexperimentqc_set__qcType'
            ).prefetch_related('projects') \
            .exclude(experiment__expName="NONE_ReportOnly_NONE") \
            .exclude(experiment__status="planned") \
            .exclude(experiment__expDir = "") \
            .order_by('-timeStamp')

        field_list = ['id', 'resultsName', 'processedflows', 'timeStamp',
            'projects', 'status', 'reportLink', 'representative', 'eas',
            'reportStatus', 'autoExempt', 'analysismetrics',
            'libmetrics', 'qualitymetrics']
        fields = field_list
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()
        cache = SimpleCache(timeout=9)


class CompositePlannedExperimentResource(ModelResource):
    class Meta:
        queryset = models.PlannedExperiment.objects.all()

        #allow ordering and filtering by all fields
        fields = ['runType', 'id', 'sampleTubeLabel']
        ordering = fields
        filtering = field_dict(fields)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()
        validation = PlannedExperimentValidation()


def results_comparator(bundle):
    return (bundle.data['representative'], bundle.data['timeStamp'])

class CompositeExperimentResource(ModelResource):

    results = fields.ToManyField(CompositeResultResource, 'results_set',
        full=True, related_name='experiment')
    plan = fields.ToOneField(CompositePlannedExperimentResource, 'plan',
        full=True, null=True, blank=True)
    repResult = fields.ToOneField(CompositeResultResource, 'repResult', null=True, blank=True)

    def prepend_urls(self):
        #this is meant for internal use only
        urls = [url(r"^(?P<resource_name>%s)/show%s$" % (self._meta.resource_name,
                trailing_slash()), self.wrap_view('dispatch_show'), name="api_data_show")]
        return urls

    def dispatch_show(self, request, **kwargs):
        return self.dispatch('show', request, **kwargs)

    def dehydrate(self, bundle):
        status = bundle.request.GET.get('result_status', None)
        if status is not None:
            if status == "Completed":
                qfilter = lambda r: r.data['status'] == "Completed"
            elif status == "Progress":
                qfilter = lambda r: r.data['status'] in ("Pending", "Started",
                    "Signal Processing", "Flow Space Recalibration", "Base Calling", "Alignment")
            elif status == "Error":
                qfilter = lambda r: r.data['status'] in (
                    "Failed to contact job server.",
                    "Error",
                    "TERMINATED",
                    "Checksum Error",
                    "Moved",
                    "CoreDump",
                    "ERROR",
                    "Error in alignmentQC.pl",
                    "MissingFile",
                    "Separator Abort",
                    "PGM Operation Error",
                    "Error in Reads sampling with samtools",
                    "No Live Beads",
                    "Error in Analysis",
                    "Error in BaseCaller"
                    )
            bundle.data['results'] = filter(qfilter, bundle.data['results'])
        bundle.data['results'].sort(key=results_comparator)
        bundle.data['results'].reverse()

        chip = chips.get(bundle.obj.chipType, None) or chips.get(bundle.obj.chipType[:3], None)
        if chip:
            bundle.data['chipDescription'] = chip.description
            bundle.data['chipInstrumentType'] = chip.instrumentType
        else:
            bundle.data['chipDescription'] = bundle.obj.chipType
            bundle.data['chipInstrumentType'] = ""

        samples = bundle.obj.samples.all()
        if samples and len(samples) > 1:
            bundle.data["sample"] = bundle.data['sampleDisplayedName'] = "%d Samples ..." % len(samples)
        else:
            bundle.data['sample'] = samples[0].name if samples else ""
            bundle.data['sampleDisplayedName'] = samples[0].displayedName if samples else ""

        if bundle.data['results']:
            eas = bundle.data['results'][0].obj.eas
            if eas:
                bundle.data['barcodeId'] = eas.barcodeKitName
                bundle.data['library'] = eas.reference
                bundle.data['barcodedSamples'] = eas.barcodedSamples
        else:
            bundle.data['barcodedSamples'] = bundle.obj.plan.latestEAS.barcodedSamples if bundle.obj.plan.latestEAS else ''

        try:
            dmfilestat = bundle.obj.results_set.all()[0].get_filestat(dmactions_types.SIG)
            bundle.data['archived'] = False if not dmfilestat.isdisposed() else dmfilestat.get_action_state_display()
        except:
            bundle.data['archived'] = False
        bundle.data['keep'] = bundle.obj.storage_options == 'KI'

        if bundle.obj.plan:
            sampleSet = bundle.obj.plan.sampleSet
            bundle.data['sampleSetName'] = sampleSet.displayedName if sampleSet else ""
        else:
            bundle.data['sampleSetName'] = ""

        return bundle


    def apply_filters(self, request, applicable_filters):
        '''
        !!DEVELOPERS!!: BE SURE TO ALSO UPDATE THE CORRESPONDING rundb.data.views.getCSV() if you ADD, MODIFY this method.
        '''
        base_object_list = super(CompositeExperimentResource, self).apply_filters(request, applicable_filters)

        all_text = request.GET.get('all_text', '')
        qset = []
        for name in all_text.split(" OR "):
            name = name.strip()
            if name:
                qset.extend([
                    Q(expName__icontains=name),
                    Q(results_set__resultsName__icontains=name),
                    Q(notes__icontains=name)
                ])
        if qset:
            or_qset = reduce(operator.or_, qset)
            base_object_list = base_object_list.filter(or_qset)

        samplestube = request.GET.get('plan__sampleTubeLabel', None)
        if samplestube:
            samplestube = samplestube.strip()
            qset = (
                Q(plan__sampleTubeLabel=samplestube)
            )
            base_object_list = base_object_list.filter(qset)

        samples = request.GET.get('samples__name', None)
        if samples is not None:
            qset = (
                Q(samples__name=samples)
            )
            base_object_list = base_object_list.filter(qset)

        date = request.GET.get('all_date', None)
        if date is not None:
            logger.debug("Got all_date='%s'" % str(date))
            reify_date = lambda d: datetime.datetime.strptime(d, '%m/%d/%Y %H:%M').strftime('%Y-%m-%d %H:%M')
            try:
                date_limits = map(reify_date, date.split(','))
            except Exception as err:
                logger.error(err)
            logger.debug("Made all_date='%s'" % str(date_limits))
            qset = (
                Q(date__range=date_limits) |
                Q(results_set__timeStamp__range=date_limits)
            )
            base_object_list = base_object_list.filter(qset)

        status = request.GET.get('result_status', None)
        if status is not None:
            if status == "Completed":
                qset = Q(results_set__status="Completed")
            elif status == "Progress":
                qset = Q(results_set__status__in=("Pending", "Started",
                    "Signal Processing", "Base Calling", "Alignment"))
            elif status == "Error":
                # This list may be incomplete, but a coding standard needs to
                # be established to make these more coherent and migration
                # written to normalize the exisitng entries
                qset = Q(results_set__status__in=(
                    "Failed to contact job server.",
                    "Error",
                    "TERMINATED",
                    "Checksum Error",
                    "Moved",
                    "CoreDump",
                    "ERROR",
                    "Error in alignmentQC.pl",
                    "MissingFile",
                    "Separator Abort",
                    "PGM Operation Error",
                    "Error in Reads sampling with samtools",
                    "No Live Beads",
                    "Error in Analysis",
                    "Error in BaseCaller"
                    ))
            base_object_list = base_object_list.filter(qset)

        return base_object_list.distinct()

    def post_show(self, request, **kwargs):
        state = 'full' if request.body.strip() == 'full' else 'table'
        logger.info("show data state: " + state)
        request.session['show_data_tab'] = state

    class Meta:
        queryset = models.Experiment.objects.select_related(
            'plan',
            'repResult'
        ).prefetch_related(
            'results_set',
            'results_set__analysismetrics',
            'results_set__libmetrics',
            'results_set__qualitymetrics',
            'results_set__eas',
            'results_set__projects',
            'results_set__dmfilestat_set',
            'samples'
        ).exclude(expName="NONE_ReportOnly_NONE").exclude(status = "planned").exclude(expDir = "").order_by('-resultDate')

        field_list = [
                    'barcodeId',    'chipType',     'date',
                    'expName',      'flows',        'ftpStatus',
                    'id',           'library',      'notes',
                    'pgmName',      'resultDate',   'results',
                    'results_set',  'runMode',      'repResult',
                    'star',         'storage_options', 'plan', 'platform'
        ]

        fields = field_list
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()
        # This query is expensive, and used on main data tab.
        # Cache frequent access
        cache = SimpleCache(timeout=17)

        show_allowed_methods = ['post']


class TemplateResource(ModelResource):

    class Meta:
        queryset = models.Template.objects.all()

        # allow ordering and filtering by all fields
        field_list = models.Template._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class ApplProductResource(ModelResource):
    appl = fields.ToOneField(RunTypeResource, 'applType', full=True)
    defaultSeqKit = fields.ToOneField(KitInfoResource, 'defaultSequencingKit', full=True, null=True)
    defaultLibKit = fields.ToOneField(KitInfoResource, 'defaultLibraryKit', full=True, null=True)
    defaultControlSeqKit = fields.ToOneField(KitInfoResource, 'defaultControlSeqKit', full=False, null=True)

    defaultIonChefPrepKit = fields.ToOneField(KitInfoResource, 'defaultIonChefPrepKit', full=False, null=True)
    defaultIonChefSequencingKit = fields.ToOneField(KitInfoResource, 'defaultIonChefSequencingKit', full=False, null=True)

    defaultSamplePrepKit = fields.ToOneField(KitInfoResource, 'defaultSamplePrepKit', full=False, null=True)
    defaultTemplateKit = fields.ToOneField(KitInfoResource, 'defaultTemplateKit', full=False, null=True)

    class Meta:
        queryset = models.ApplProduct.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.ApplProduct._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class QCTypeResource(ModelResource):

    class Meta:
        queryset = models.QCType.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.QCType._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class PlannedExperimentQCResource(ModelResource):

    plannedExperiment = fields.ToOneField(PlannedExperimentResource, 'plannedExperiment', full=False)
    qcType = fields.ToOneField(QCTypeResource, 'qcType', full=True)

    class Meta:
        queryset = models.PlannedExperimentQC.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperimentQC._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class EventLogResource(ModelResource):

    def apply_filters(self, request, applicable_filters):
        base_object_list = super(EventLogResource, self).apply_filters(request, applicable_filters)

        cttype = request.GET.get('content_type_id', None)
        if cttype is not None:
            base_object_list = base_object_list.filter(content_type_id=cttype)

        return base_object_list.distinct()

    class Meta:
        queryset = models.EventLog.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.EventLog._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class ChipResource(ModelResource):

    def dehydrate(self, bundle):
        # backwards compatibility: provide default args from AnalysisArgs table
        args = models.AnalysisArgs.objects.filter(chipType=bundle.obj.name, chip_default=True)
        if args:
            args_dict = args[0].get_args()
        else:
            args_dict = {}

        for key,value in args_dict.items():
            bundle.data[key] = value

        return bundle

    class Meta:
        queryset = models.Chip.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.Chip._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class AccountObject(object):
    def __init__(self, user=None):
        self.uid = None
        self.username = None
        self.admin = False
        self.api_key = None
        if user:
            self.uid = user.pk
            self.username = user.username
            self.admin = user.is_superuser
            if user.is_authenticated():
                self.api_key = user.api_key.key

    def to_dict(self):
        return self.__dict__

class AccountResource(Resource):
    username = fields.CharField(attribute='username', blank=True, null=True)
    uid = fields.CharField(attribute='uid', blank=True, null=True)
    api_key = fields.CharField(attribute='api_key', blank=True, null=True)
    admin = fields.BooleanField(attribute='admin', blank=True)

    class Meta:
        resource_name = 'account'
        object_class = AccountObject
        authentication = IonAuthentication(allow_get=False)
        authorization = DjangoAuthorization()

    def detail_uri_kwargs(self, bundle_or_obj):
        kwargs = {}
        if isinstance(bundle_or_obj, Bundle):
            kwargs['pk'] = bundle_or_obj.obj.uid
        else:
            kwargs['pk'] = bundle_or_obj.uid
        return kwargs

    def get_object_list(self, request):
        # Return a list of one element for current user
        return [ AccountObject(request.user) ]

    def obj_get_list(self, bundle, **kwargs):
        return self.get_object_list(bundle.request)

    def obj_get(self, bundle, **kwargs):
        return AccountObject(bundle.request.user)

class NetworkResource(ModelResource):
    """Get external network info"""

    def prepend_urls(self):
        urls = [url(r"^(?P<resource_name>%s)%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_update'), name="api_dispatch_update")
        ]

        return urls

    def dispatch_update(self, request, **kwargs):
        return self.dispatch('update', request, **kwargs)

    def get_update(self, request, **kwargs):


        def script(script_text, shell_bool = True):
            """run system commands"""
            p = Popen(args=script_text, shell=shell_bool, stdout=PIPE, stdin=PIPE)
            output, errors = p.communicate()
            return output, errors

        result = {
            "eth_device": None,
            "route": None,
            "internal_ip": None,
            }
        try:
            stdout, stderr = script("PORT=$(route|awk '/default/{print $8}') && /sbin/ifconfig $PORT")
            for line in stdout.splitlines():
                m = re.search(r"inet addr:(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", line)
                if m:
                    result["internal_ip"] = m.group(1)
                elif "UP" in line and "MTU" in line:
                    result["eth_device"] = True
            stdout, stderr = script("/bin/netstat -r")
            result["route"] = "default" in stdout
        except Exception as err:
            logger.error("Exception raised during network self exam, '%s'" % err)

        try:
            xremote = urllib.urlopen(settings.EXTERNAL_IP_URL)
            data = xremote.read()
            xremote.close()
        except Exception as complaint:
            logger.warn(complaint)
            data = ""

        result["external_ip"] = data.strip()

        return self.create_response(request, result)

    class Meta:
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()
        update_allowed_methods = ['get']

    def get_schema(self, request, **kwargs):
        schema = {
            "allowed_detail_http_methods": [],
            "allowed_list_http_methods": [
                "get",
            ],
            "default_format": "application/json",
            "default_limit": 0,
            "fields": {
                "eth_device": {
                    "blank": True,
                    "default": "No default provided.",
                    "help_text": "Unicode string data. Ex: \"Hello World\"",
                    "nullable": True,
                    "readonly": True,
                    "type": "boolean",
                    "unique": False
                },
                "external_ip": {
                    "blank": True,
                    "default": "No default provided.",
                    "help_text": "Unicode string data. Ex: \"Hello World\"",
                    "nullable": True,
                    "readonly": True,
                    "type": "string",
                    "unique": False
                },
                "internal_ip": {
                    "blank": True,
                    "default": "No default provided.",
                    "help_text": "Unicode string data. Ex: \"Hello World\"",
                    "nullable": True,
                    "readonly": True,
                    "type": "string",
                    "unique": False
                },
                "route": {
                    "blank": True,
                    "default": "No default provided.",
                    "help_text": "Unicode string data. Ex: \"Hello World\"",
                    "nullable": True,
                    "readonly": True,
                    "type": "boolean",
                    "unique": False
                },
            },
            "filtering": {},
            "ordering": []
        }
        return HttpResponse(json.dumps(schema), content_type="application/json")

class AnalysisArgsResource(ModelResource):

    def prepend_urls(self):
        return [
            url(r"^(?P<resource_name>%s)/getargs%s$" % (self._meta.resource_name, trailing_slash()), self.wrap_view('get_args'), name="api_get_args"),
        ]

    def get_args(self, request, **kwargs):
        # function for OIA to retrieve default args with or without Plan specified
        planGUID = request.GET.get('planGUID', None)
        plan = None
        if planGUID:
            try:
                plan = models.PlannedExperiment.objects.filter(planGUID=planGUID)[0]
            except IndexError:
                pass

        if not plan:
            # get system default template in the same way crawler would do it
            chipType = request.GET.get('chipType', None)
            plan = models.PlannedExperiment.get_latest_plan_or_template_by_chipType(chipType)
            if not plan:
                plan = models.PlannedExperiment.get_latest_plan_or_template_by_chipType()

        if plan:
            args = plan.get_default_cmdline_args(**request.GET.dict())
        else:
            args = {}

        return self.create_response(request, args)

    class Meta:
        queryset = models.AnalysisArgs.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.AnalysisArgs._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class FileMonitorResource(ModelResource):
    class Meta:
        queryset = models.FileMonitor.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.FileMonitor._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class SupportUploadResource(ModelResource):
    result = fields.ToOneField(ResultsResource, 'result')
    file = fields.ToOneField(FileMonitorResource, 'file', null=True, full=True)

    class Meta:
        queryset = models.SupportUpload.objects.select_related('file').all()

        #allow ordering and filtering by all fields
        field_list = models.SupportUpload._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class PrepopulatedPlanningSessionResource(ModelResource):
    """Allow setting session values of the panning wizard before redirecting to the wizard."""

    def prepend_urls(self):
        urls = [url(r"^(?P<resource_name>%s)%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_session'), name="api_dispatch_session")
        ]

        return urls

    def dispatch_session(self, request, **kwargs):
        return self.dispatch('session', request, **kwargs)

    def post_session(self, request, **kwargs):
        key_value = json.loads(request.body)

        key_inserted_successfully = False
        key_name = None
        session_id = None
        while not key_inserted_successfully:
            session_id = str(uuid.uuid4())
            key_name = "prepopulated-planning-session-" + session_id
            key_inserted_successfully = cache.add(key_name, key_value, timeout=None)

        session_status = {
            "error_message": "",
            "id": session_id,
            "cache_key": key_name
        }

        return self.create_response(request, session_status)

    class Meta:

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()
        session_allowed_methods = ['post']
