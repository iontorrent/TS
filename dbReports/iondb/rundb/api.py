# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
import os
from django.db.models.aggregates import Count, Max
from tastypie.api import Api
import StringIO
import csv
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

#auth
from tastypie.authentication import BasicAuthentication, ApiKeyAuthentication, Authentication
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

from django.utils.decorators import method_decorator
from tastypie.validation import Validation, FormValidation
from tastypie.resources import ModelResource

from tastypie.http import *
from tastypie.utils import dict_strip_unicode_keys, trailing_slash
from tastypie.utils.mime import build_content_type

from iondb.rundb.views import barcodeData
from iondb.rundb import forms
import iondb.rundb.admin

from ion.utils.TSversion import findVersions
#from iondb.plugins import runner
from ion.plugin import remote

#custom query
from django.db.models.sql.constants import QUERY_TERMS, LOOKUP_SEP
from django.db import transaction
from django.db.models import Q

import socket
import datetime
import logging
from operator import itemgetter

import httplib2
from iondb.utils import toBoolean

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
from django.http import HttpResponse, HttpResponseNotFound, HttpResponseRedirect, Http404
from django.conf.urls.defaults import url

import tasks
from iondb.rundb.data import dmactions_types

from iondb.plugins.manager import pluginmanager

# shared functions for making startplugin json
from ion.utils.explogparser import getparameter, getparameter_minimal
from iondb.plugins.launch_utils import get_plugins_dict

import ast
import simplejson
import traceback
    
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

class InternalBasicAuthentication(BasicAuthentication):
    def is_authenticated(self, request, **kwargs):
        if not request.META.get('HTTP_AUTHORIZATION'):
            return False

        try:
            (auth_type, data) = request.META['HTTP_AUTHORIZATION'].split()
            if auth_type.lower() != 'basic':
                return False
            user_pass = base64.b64decode(data)
        except:
            return False

        bits = user_pass.split(':', 1)
        if len(bits) != 2:
            return False

        import base64
        username, password = base64.b64decode(bits[0], bits[1])

        from django.contrib.auth import authenticate
        user = authenticate(username=bits[0], password=bits[1])
        if user is None:
            return False

        if not self.check_active(user):
            return False

        request.user = user
        return True

# Custom Authorization Class
class IonAuthentication(object):
    """
        Derived from MultiAuthentication, but with Auth Schemes hardcoded
    """
    def __init__(self, allow_get=True):
        self.backends = [
            # Basic must be first, so it sets WWW-Authenticate header in 401.
            BasicAuthentication(realm='Torrent Browser'), ## Apache Basic Auth
            InternalBasicAuthentication(realm='Torrent Browser'),
            ApiKeyAuthentication(),
        ]
        self.allow_get = allow_get
        # NB: If BasicAuthentication is last, will return WWW-Authorize header and prompt for credentials

    def is_authenticated(self, request, **kwargs):
        """
        Identifies if the user is authenticated to continue or not.

        Should return either ``True`` if allowed, ``False`` if not or an
        ``HttpResponse`` if you need something custom.
        """
        unauthorized = False

        # Allow user with existing session - django authn via session cookie
        if hasattr(request, 'user') and request.user.is_authenticated():
            return True

        for backend in self.backends:
            check = backend.is_authenticated(request, **kwargs)

            if check:
                if isinstance(check, HttpUnauthorized):
                    unauthorized = unauthorized or check
                else:
                    request._authentication_backend = backend

                    # Did not have a session, but set and send back session cookie
                    if hasattr(request, 'user'):
                        try:
                            login(request, request.user)
                        except:
                            logger.exception("Unable to login user: '%s'", request.user.username)

                    return check

        # Allow GET OPTIONS HEAD without auth
        if self.allow_get and request.method in ('GET', 'OPTIONS', 'HEAD'):
            return True

        return unauthorized

    def get_identifier(self, request):
        """
        Provides a unique string identifier for the requestor. Delegates to Authn classes
        """
        try:
            return request._authentication_backend.get_identifier(request)
        except AttributeError:
            return 'nouser'

class CustomSerializer(Serializer):
    formats = ['json', 'jsonp', 'xml', 'yaml', 'html', 'plist', 'csv']
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
        # Untested, so this might not work exactly right.
        if data.has_key('objects'):
            objects = data['objects']
            for item in objects:
                writer = csv.DictWriter(raw_data, item.keys(), extrasaction='ignore')
                writer.writerow(item)
        return raw_data

    def from_csv(self, content):
        raw_data = StringIO.StringIO(content)
        data = []
        # Untested, so this might not work exactly right.
        for item in csv.DictReader(raw_data):
            data.append(item)
        return data

class MyModelDeclarativeMetaclass(ModelDeclarativeMetaclass):
    def __new__(cls, name, bases, attrs):
        meta = attrs.get('Meta')

        if meta and not hasattr(meta, 'serializer'):
            setattr(meta, 'serializer', CustomSerializer())

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
        user = self.cached_obj_get(request=request, **self.remove_api_resource_names(kwargs))
        if user is None:
            return HttpGone()

        user.delete()
        return HttpAccepted()

    def post_activate(self, request, **kwargs):
        if not request.user.is_superuser:
            raise HttpUnauthorized()
        user = self.cached_obj_get(request=request, **self.remove_api_resource_names(kwargs))
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
    results = fields.ToManyField('iondb.rundb.api.ResultsResource', attribute = 'results', related_name='results', null=True)
    creator = fields.ToOneField(UserResource, 'creator', full=False)
    def dehydrate(self, bundle):
        bundle.data['resultsCount'] = bundle.obj.results.count()
        return bundle
#    def apply_sorting(self, objects, options=None):
#        print objects
#        if options and options.has_key('order_by'):
#            if "resultsCount" in options['order_by']:
#                return objects.order_by()
#            if "-resultsCount" in options['order_by']:
#                return objects.annotate(resultsCount=Count('results'))
#        return super(ProjectResource, self).apply_sorting(objects, options)

# TODO: The creator field should be immutable after record insertion.
#    def hydrate_creator (self, bundle):
#        bundle = super(ProjectResource,self).hydrate_creator(bundle)
#        bundle.data['creator'] = bundle.request.user
#        return bundle
    class Meta:
        queryset = models.Project.objects.all()

#        results_allowed_methods = ['get']
        creator_allowed_methods = ['get']
        # allow ordering and filtering by all fields
        field_list = models.Project._meta.get_all_field_names()
        ordering = field_list + ['resultsCount']
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
        api_url = net_location + "/rundb/api/"

        # Mirror env structure defined in pipeline scripts
        report_root = result.get_report_dir()
        try:
          env,warn = getparameter(os.path.join(report_root,'ion_params_00.json'))
        except:
          env = getparameter_minimal(os.path.join(report_root,'ion_params_00.json'))

        env['report_root_dir'] = report_root

        env['net_location'] = net_location
        env['master_node'] = sge_master
        env['api_url'] = api_url

        env['primary_key'] = result.pk

        env.setdefault('tmap_version',settings.TMAP_VERSION)

        plugin_basefolder = "plugin_out"

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
                logger.exception("Unable to find suitable plugin for %s", plugin_name)
                return HttpGone()

            plugins_dict = get_plugins_dict([plugin_orm], result.eas.selectedPlugins)

            # Override plugin config with instance config
            plugins_dict[plugin_name]["pluginconfig"].update(config)

            # Set Started status before launching to avoid race condition
            # Set here so it appears immediately in refreshed plugin status list
            (pluginresult, created) = result.pluginresult_set.get_or_create(plugin=plugin_orm)
            pluginresult.prepare()
            pluginresult.save()

            # run plugin in either single level (default) or multilevel if indicated in plugin['runlevel']
            if len(plugins_dict[plugin_name].get('runlevel',[])) > 1 :
                # multi-runlevel plugin

                # get analyzed block directories for Proton analysis
                if env['report_type'] == 'composite':
                    env['block_dirs'] = []
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
                            env['block_dirs'].append(block_dir)

                # launch plugin for multiple levels, SGE dependency is used to wait for completion of previous level
                for runlevel in ['pre', 'block', 'default', 'post', 'last']:
                    if runlevel in plugins_dict[plugin_name]['runlevel']:
                        env['runlevel'] = runlevel
                        if runlevel == 'block':
                            if env['report_type'] == 'composite':
                              for blockId,block_dir in zip(blockIds, env['block_dirs']):
                                  env['blockId'] = blockId
                                  block_pluginbasefolder = os.path.join(block_dir,plugin_basefolder)
                                  plugins_dict, msg = remote.call_launchPluginsXMLRPC(env, plugins_dict, block_pluginbasefolder, url_root)
                                  logger.info('Plugin runlevel= %s. %s' % (runlevel, msg))
                        else:
                            plugins_dict, msg = remote.call_launchPluginsXMLRPC(env, plugins_dict, plugin_basefolder, url_root)
                            logger.info('Plugin runlevel= %s. %s' % (runlevel, msg))

            else:
                # default single level
                plugins_dict, msg = remote.call_launchPluginsXMLRPC(env, plugins_dict, plugin_basefolder, url_root)
                logger.info(msg)

            if 'ERROR' in msg:
                logger.error('Unable to launch plugin: %s', plugin_orm.name) # See ionPlugin.log for details
                pluginresult.complete('Error')
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
                    # Call trigger methods for special states
                    # - Started, Complete, Error
                    if value == 'Completed' or value=='Error':
                        pluginresult.complete(state=value)
                    elif value == 'Started':
                        pluginresult.start()
                    else:
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

    def get_pluginresults(self, request, **kwargs):
        results = self.cached_obj_get(request=request, **self.remove_api_resource_names(kwargs))
        if results is None:
            return HttpGone()

        #filter for just the major blocks
        major = request.GET.get('major',False)

        # In the order they were generated, newest plugin entry first.
        pluginresults = results.pluginresult_set.all().order_by('-starttime')
        if major:
            pluginresults = pluginresults.filter(plugin__majorBlock=True)
            major_blocks = [pluginresult.plugin.name for pluginresult in pluginresults]

        # Iterate through DB results first, find matching directories
        pluginArray = []
        seen = {}
        for pr in pluginresults:
            prpath = pr.path()
            candidates = [
                #prpath + (".%d" % pr.id), ## PluginResult ID specific output
                #prpath + ("--v%s" % pr.plugin.version), ## Plugin Version Specific Output
                prpath,
            ]
            outputpath = None
            for p in candidates:
                if not os.path.exists(p):
                    continue

                pathversion = self._startplugin_version(p)
                if pathversion:
                    if pathversion != pr.plugin.version:
                        # ~-~ these aren't the results you are looking for ~-~
                        outputpath = None
                        pr.state = pr.state + ' [Superseded]'
                        break
                else:
                    # unknown version (someone probably deleted startplugin.json)
                    # most recent plugin wins - let it fall through
                    if seen.get(os.path.basename(p), False):
                        outputpath = None
                        pr.state = pr.state + ' [Superseded?]'
                        # we are done here - we've already excluded the better matches
                        break
                # Everything matches
                #logger.debug('Got pluginresult for %s v%s at %s', pr.plugin.name, pr.plugin.version, prpath)
                outputpath = p
                break
            else:
                logger.debug("Plugin %s v%s has no plugin_out folder", pr.plugin.name, pr.plugin.version)
                outputpath = None
                pr.state = pr.state + ' [Missing]'

            if outputpath:
                # Got a matching path, populate list of files
                outputfiles = [pfile for pfile in os.listdir(outputpath) if (pfile.endswith(".html") or pfile.endswith(".php"))]
                if major:
                    def onlyBlocks(s):
                        return "_block" in s
                    outputfiles = filter(onlyBlocks, outputfiles)
                seen[os.path.basename(outputpath)] = True
            else:
                # Missing Output, or unknown version
                outputpath = prpath ## assume default path, let link be broken
                outputfiles = [] # but mask files

            data = {
                'Name': pr.plugin.name,
                'Version': pr.plugin.version,
                'State': pr.state,
                'Path': outputpath,
                'URL': "%splugin_out/%s/" % (results.reportLink, os.path.basename(outputpath)),
                'Files': outputfiles,
                'Major' : pr.plugin.majorBlock,
                'Size': pr.size,
                'inodes': pr.inodes,
                'id': pr.id
            }
            pluginArray.append(data)

        # Report base path for all plugins
        plugin_out_path = os.path.join(results.get_report_dir(),"plugin_out")
        try:
            plugin_out_listing = ( d for d in os.listdir(plugin_out_path) if os.path.isdir(d) )
        except (OSError, IOError):
            logger.info("Error listing %s", plugin_out_path)
            plugin_out_listing = []

        # Now - directories found which don't match plugin result entries
        if not major:
            import re
            for orphan in ( p for p in plugin_out_listing if p not in seen ):
                outputpath = os.path.join(plugin_out_path, orphan)
                version = self._startplugin_version(outputpath)
                if version is None:
                    version = '??'

                name = re.sub(r'_out(\.\d+)?$','',orphan)
                outputfiles = [pfile for pfile in os.listdir(outputpath) if (pfile.endswith(".html") or pfile.endswith(".php"))]

                data = {
                    'Name': name,
                    'Version': version,
                    'State': 'Unknown',
                    'Files': outputfiles,
                    'Path': outputpath,
                    'URL': "%s/plugin_out/%s/" % (results.reportLink, orphan),
                    'Major' : False, # NB: Plugins without DB record cannot be major
                    'Size': -1,
                    'inodes': -1,
                    'id': None
                }
                logger.info("Plugin folder with no db record: %s v%s at '%s'", name, version, path)
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

    def dispatch_pluginresult_set(self, request, **kwargs):
        return self.dispatch('pluginresult_set', request, **kwargs)

    def get_pluginresult_set(self, request, **kwargs):
        try:
            obj = self.cached_obj_get(request=request, **self.remove_api_resource_names(kwargs))
        except ObjectDoesNotExist:
            return HttpGone()
        except MultipleObjectsReturned:
            return HttpMultipleChoices("More than one resource is found at this URI.")

        pluginresults = PluginResultResource()
        return pluginresults.get_list(request, result_id=obj.pk)

    filesystempath = CharField('get_report_dir')
    bamLink = CharField('bamLink')
    planShortID = CharField('planShortID')
    libmetrics = ToManyField(LibMetricsResource, 'libmetrics_set', full=False)
    tfmetrics = ToManyField(TFMetricsResource, 'tfmetrics_set', full=False)
    analysismetrics = ToManyField(AnalysisMetricsResource, 'analysismetrics_set', full=False)
    qualitymetrics = ToManyField(QualityMetricsResource, 'qualitymetrics_set', full=False)
    reportstorage = fields.ToOneField(ReportStorageResource, 'reportstorage', full=True)

    sffLink = CharField('sffLinkPatch')
    tfSffLink = CharField('sffTFLinkPatch')

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
        pluginresults_allowed_methods = ['get','post','put']
        pluginresult_set_allowed_methods = ['get']
        projects_allowed_methods = ['get']
        pluginstore_allowed_methods = ['get','put']
        barcode_allowed_methods = ['get']
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

    class Meta:
        queryset = models.Sample.objects.all()

        #allow ordering and filtering by all fields
        field_list = models.Sample._meta.get_all_field_names()
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

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()
        status_allowed_methods = ['get', 'put']
        config_allowed_methods = ['get',]

class PluginResource(ModelResource):
    """Get a list of plugins"""
    isConfig = fields.BooleanField(readonly=True, attribute='isConfig')
    isPlanConfig = fields.BooleanField(readonly=True, attribute='isPlanConfig')
    isInstance = fields.BooleanField(readonly=True, attribute='isInstance')
    hasAbout = fields.BooleanField(readonly=True, attribute='hasAbout')
    versionedName = fields.CharField(readonly=True, attribute='versionedName')

    def prepend_urls(self):
        #this is meant for internal use only
        urls = [url(r"^(?P<resource_name>%s)/set/(?P<keys>\w[\w/-;]*)/type%s$" % (self._meta.resource_name,
                    trailing_slash()),self.wrap_view('get_type_set'), name="api_get_type_set"),
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
        obj = self.cached_obj_get(request=request, **self.remove_api_resource_names(kwargs))

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
        plugin = self.cached_obj_get(request=request, **self.remove_api_resource_names(kwargs))
        if not plugin:
            return HttpGone()
        info = plugin.info(use_cache=toBoolean(request.GET.get('use_cache', True)))
        if not info:
            return HttpGone()
        return self.create_response(request, info)

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

class PluginResultResource(ModelResource):
    result = fields.ToOneField(ResultsResource,'result')
    plugin = fields.ToOneField(PluginResource, 'plugin', full=True)

    path = fields.CharField(readonly=True, attribute='path')
    duration = fields.CharField(readonly=True, attribute='duration')

    # Helper methods for display / context
    resultName = fields.CharField(readonly=True, attribute='result__resultsName')
    reportLink = fields.CharField(readonly=True, attribute='result__reportLink')

    def prepend_urls(self):
        urls = [url(r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/rescan%s$" % (self._meta.resource_name, trailing_slash()),
                    self.wrap_view('dispatch_rescan'), name="api_dispatch_rescan")
        ]
        return urls

    def dispatch_rescan(self, request, **kwargs):
        return self.dispatch('rescan', request, **kwargs)

    def get_rescan(self, request, **kwargs):
        obj = self.cached_obj_get(request=request, **self.remove_api_resource_names(kwargs))
        try:
            size, inodes = obj._calc_size()
            if size > 0:
                obj.size = size
                obj.inodes = inodes
                obj.save()
        except (OSError, IOError):
            raise Http404()

        return HttpAccepted()

    class Meta:
        queryset = models.PluginResult.objects.all()
        #allow ordering and filtering by all fields
        field_list = models.PluginResult._meta.get_all_field_names()
        field_list.extend(['result','plugin', 'path', 'duration', 'id']),
        excludes = ['apikey',]
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

        rescan_allowed_methods = ['get',]

class RunTypeResource(ModelResource):

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
        letters_punct_no_spaces = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        letters_punct_with_spaces = letters_punct_no_spaces + " "
        for key, value in bundle.data.items():
            if not isinstance(value, basestring):
                continue
            if key == "sample":
                if not set(value).issubset(letters_punct_no_spaces):
                    errors[key] = "That Report sample has invalid characters. The valid values are letters, numbers, underscore, and period."
            if key == "notes":
                if not set(value).issubset(letters_punct_with_spaces):
                    errors[key] = "That Report name has invalid characters. The valid values are letters, numbers, underscore, space and period."

        return errors


class PlannedExperimentDbResource(ModelResource):
    # Backwards support - single project field
    project = fields.CharField(readonly=True, blank=True)

    projects = fields.ToManyField('iondb.rundb.api.ProjectResource', 'projects', full=False, null=True, blank=True)
    qcValues = fields.ToManyField('iondb.rundb.api.PlannedExperimentQCResource', 'plannedexperimentqc_set', full=True, null=True, blank=True)
    parentPlan = fields.CharField(blank=True, default=None)
    childPlans = fields.ListField(default=[])
    experiment = fields.ToOneField(ExperimentResource, 'experiment', full=False, null=True, blank=True)


    def hydrate_m2m(self, bundle):        
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
            project_objs = models.Project.bulk_get_or_create(projects_list, user)

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

    class Meta:
        queryset = models.PlannedExperiment.objects.select_related(
            'experiment'
        ).prefetch_related(
            'projects',
            'plannedexperimentqc_set',
            'plannedexperimentqc_set__qcType',
            'experiment__eas_set',
            'experiment__samples'
        ).all()

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()
        validation = PlannedExperimentValidation()


class PlannedExperimentResource(PlannedExperimentDbResource):
    autoAnalyze = fields.BooleanField()
    barcodedSamples = fields.CharField(blank=True, null=True)
    barcodeId = fields.CharField(blank=True, null=True)
    bedfile = fields.CharField(blank=True)
    chipType = fields.CharField()
    flows = fields.IntegerField()
    forward3primeadapter = fields.CharField(blank=True, null=True)
    library = fields.CharField(blank=True, null=True)
    libraryKey = fields.CharField(blank=True)
    librarykitname = fields.CharField(blank=True, null=True)
    notes = fields.CharField(blank=True, null=True)
    regionfile = fields.CharField(blank=True)
    reverse3primeadapter = fields.CharField(readonly = True)
    reverselibrarykey = fields.CharField(readonly = True)
    sample = fields.CharField(blank=True, null=True)
    sampleDisplayedName = fields.CharField(blank=True, null=True)
    selectedPlugins = fields.CharField(blank=True, null=True)
    sequencekitname = fields.CharField(blank=True, null=True)
    variantfrequency = fields.CharField(readonly = True)

    def dehydrate_autoAnalyze(self, bundle):
        experiment = bundle.obj.experiment
        if experiment:
            return experiment.autoAnalyze
        else:
            return False

    def dehydrate(self, bundle):
        experiment = bundle.obj.experiment
        if experiment:
            latest_eas = experiment.get_EAS()            

            bundle.data['barcodedSamples'] = latest_eas.barcodedSamples if latest_eas else ""
            bundle.data['barcodeId'] = latest_eas.barcodeKitName if latest_eas else ""
            bundle.data['bedfile'] = latest_eas.targetRegionBedFile if latest_eas else ""
            bundle.data['forward3primeadapter'] = latest_eas.threePrimeAdapter if latest_eas else ""
            bundle.data['library'] = latest_eas.reference if latest_eas else ""
            bundle.data['libraryKey'] = latest_eas.libraryKey if latest_eas else ""
            bundle.data['librarykitname'] = latest_eas.libraryKitName if latest_eas else ""
            bundle.data['regionfile'] = latest_eas.hotSpotRegionBedFile if latest_eas else ""
                        
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
                else:
                    bundle.data['variantfrequency'] = ""
            else:
                bundle.data['variantfrequency'] = ""
            
        else:
            bundle.data['barcodedSamples'] = ""  
            bundle.data['barcodeId'] = ""
            bundle.data['bedfile'] = ""
            bundle.data['forward3primeadapter'] = ""
            bundle.data['library'] = ""
            bundle.data['libraryKey'] = ""
            bundle.data['librarykitname'] = ""
            bundle.data['regionfile'] = ""
            bundle.data['sample'] = ""
            bundle.data['sampleDisplayedName'] = ""
            bundle.data['selectedPlugins'] = ""
            bundle.data['variantfrequency'] = ""

        return bundle
    

    def dehydrate_chipType(self, bundle):
        experiment = bundle.obj.experiment
        if experiment:
            return experiment.chipType
        else:
            return ""

    def dehydrate_flows(self, bundle):
        experiment = bundle.obj.experiment
        if experiment:
            return experiment.flows
        else:
            return "0"


    def dehydrate_notes(self, bundle):
        experiment = bundle.obj.experiment
        if experiment:
            return experiment.notes
        else:
            return ""


    def dehydrate_reverse3primeadapter(self, bundle):
        return ""

    def dehydrate_reverselibrarykey(self, bundle):
        return ""

    def dehydrate_sequencekitname(self, bundle):
        experiment = bundle.obj.experiment
        if experiment:
            return experiment.sequencekitname
        else:
            return ""


    def hydrate_autoAnalyze(self, bundle):
        bundle.data['x_autoAnalyze'] = bundle.data.get('autoAnalyze', True)
        return bundle

    def hydrate_barcodedSamples(self, bundle):
        barcodedSamples = bundle.data.get('barcodedSamples', "")
        # soft validation, will not raise errors
        # valid barcodedSamples format is {'sample_XXX':{'barcodes':['IonXpress_001', ]}, }
        valid = True
        if barcodedSamples:
            barcodedSamples_dict  = barcodedSamples
            
            if isinstance(barcodedSamples, str):
                #example: "barcodedSamples":"{'s1':{'barcodes': ['IonSet1_01']},'s2': {'barcodes': ['IonSet1_02']},'s3':{'barcodes': ['IonSet1_03']}}"
                barcodedSamples_dict = ast.literal_eval(barcodedSamples)
                barcodedSamples = simplejson.dumps(barcodedSamples_dict)
                
            try:
                for k,v in barcodedSamples_dict.items():
                    if isinstance(v['barcodes'],list):
                        for bc in v['barcodes']:
                            if not isinstance(bc,str):
                                valid = False
                    else:
                        valid = False
            except:
                logger.error(traceback.format_exc())                             
                valid = False

        bundle.data['x_barcodedSamples'] = barcodedSamples if valid else ""
        return bundle

    def hydrate_barcodeId(self, bundle):
        bundle.data['x_barcodeId'] = bundle.data.get('barcodeId', "")

        if bundle.data['x_barcodeId'] and bundle.data['x_barcodeId'].lower() == "none":
            bundle.data['x_barcodeId'] = ""

        return bundle

    def hydrate_bedfile(self, bundle):
        bundle.data['x_bedfile'] = bundle.data.get('bedfile', "")
        if bundle.data['x_bedfile'] and bundle.data['x_bedfile'].lower() == "none":
            bundle.data['x_bedfile'] = ""
        return bundle

    def hydrate_chipType(self, bundle):
        bundle.data['x_chipType'] = bundle.data.get('chipType', "")
        if bundle.data['x_chipType'] and bundle.data['x_chipType'].lower() == "none":
            bundle.data['x_chipType'] = ""
        return bundle

    def hydrate_flows(self, bundle):
        bundle.data['x_flows'] = bundle.data.get('flows', "0")
        return bundle

    def hydrate_forward3primeadapter(self, bundle):
        bundle.data['x_forward3primeadapter'] = bundle.data.get('forward3primeadapter', "")
        return bundle

    def hydrate_library(self,bundle):
        bundle.data['x_library'] = bundle.data.get('library', "")
        if bundle.data['x_library'] and bundle.data['x_library'].lower() == "none":
            bundle.data['x_library'] = ""
        return bundle

    def hydrate_libraryKey(self, bundle):
        bundle.data['x_libraryKey'] = bundle.data.get('libraryKey', "")
        if bundle.data['x_libraryKey'] and bundle.data['x_libraryKey'].lower() == "none":
            bundle.data['x_libraryKey'] = ""
        return bundle

    def hydrate_librarykitname(self, bundle):
        bundle.data['x_librarykitname'] = bundle.data.get('librarykitname', "")
        if bundle.data['x_librarykitname'] and bundle.data['x_librarykitname'].lower() == "none":
            bundle.data['x_librarykitname'] = ""
        return bundle

    def hydrate_notes(self, bundle):
        bundle.data['x_notes'] = bundle.data.get('notes', "")
        return bundle

    def hydrate_regionfile(self, bundle):
        bundle.data['x_regionfile'] = bundle.data.get('regionfile', "")
        if bundle.data['x_regionfile'] and bundle.data['x_regionfile'].lower() == "none":
            bundle.data['x_regionfile'] = ""
        return bundle

    def hydrate_sample(self, bundle):
        bundle.data['x_sample'] = bundle.data.get('sample', "")
        return bundle

    def hydrate_sampleDisplayedName(self, bundle):
        bundle.data['x_sampleDisplayedName'] = bundle.data.get('sampleDisplayedName', bundle.data.get('sample', ""))
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

        bundle.data['x_selectedPlugins'] = selectedPlugins if valid else ""
        return bundle

    def hydrate_sequencekitname(self, bundle):
        bundle.data['x_sequencekitname'] = bundle.data.get('sequencekitname', "")
        if bundle.data['x_sequencekitname'] and bundle.data['x_sequencekitname'].lower() == "none":
            bundle.data['x_sequencekitname'] = ""
        return bundle


    def obj_create( self, bundle, request = None, **kwargs ):
        """
        A ORM-specific implementation of ``obj_create``.
        """

        bundle.obj = self._meta.object_class()

        for key, value in kwargs.items():
            setattr(bundle.obj, key, value)

        bundle = self.full_hydrate(bundle)

        # Save FKs just in case.
        self.save_related(bundle)

        logger.debug("PDD at PlannedExperimentResource.obj_create after save_related()...")
        logger.debug("PDD ...obj_create() after save_related() bundle.data.keys=%s" %(bundle.data.keys()))
        logger.debug("PDD ...obj_create() after save_related() bundle.data.values=%s" %(bundle.data.values()))

        id = -1
        isNewPlan = True
        if bundle.obj and bundle.obj.pk:
            id = bundle.obj.pk
            isNewPlan = False

        bundle.obj.save()
        bundle.obj.save_plannedExperiment_association(isNewPlan, **bundle.data)

        # Now pick up the M2M bits.
        m2m_bundle = self.hydrate_m2m(bundle)
        self.save_m2m(m2m_bundle)
        return bundle

    def obj_update( self, bundle, request = None, skip_errors = False, **kwargs ):
        logger.debug("PDD ENTER PlannedExperimentResource.obj_update() bundle.data=%s" %(bundle.data))

        return super(PlannedExperimentResource, self).obj_update(bundle, request, skip_errors, **kwargs)


class AvailableIonChefPlannedExperimentResource(PlannedExperimentResource):
    class Meta:
        queryset = models.PlannedExperiment.objects.filter(isReusable = False, planExecuted = False, templatingKitName__in = [kit.name for kit in models.KitInfo.objects.filter(kitType = "IonChefPrepKit")])

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()
        ordering = field_list
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class AvailableIonChefPlannedExperimentSummaryResource(ModelResource):
    class Meta:
        queryset = models.PlannedExperiment.objects.filter(isReusable = False, planExecuted = False, templatingKitName__in = [kit.name for kit in models.KitInfo.objects.filter(kitType = "IonChefPrepKit")])
        
        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()
                
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()        

        metadata_allowed_methods = ['get',]


class IonChefPlanTemplateResource(PlannedExperimentResource):
    class Meta:
        queryset = models.PlannedExperiment.objects.filter(isReusable = True, templatingKitName__in = [kit.name for kit in models.KitInfo.objects.filter(kitType = "IonChefPrepKit")])

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()
        ordering = field_list
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class IonChefPlanTemplateSummaryResource(ModelResource):
    class Meta:
        queryset = models.PlannedExperiment.objects.filter(isReusable = True, templatingKitName__in = [kit.name for kit in models.KitInfo.objects.filter(kitType = "IonChefPrepKit")])

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()
        ordering = field_list
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

        metadata_allowed_methods = ['get',]        


class AvailableOneTouchPlannedExperimentResource(PlannedExperimentResource):
    class Meta:
        queryset = models.PlannedExperiment.objects.filter(isReusable = False, planExecuted = False, templatingKitName__in = [kit.name for kit in models.KitInfo.objects.filter(kitType = "TemplatingKit")])

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()
        ordering = field_list
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class OneTouchPlanTemplateResource(PlannedExperimentResource):
    class Meta:
        queryset = models.PlannedExperiment.objects.filter(isReusable = True, templatingKitName__in = [kit.name for kit in models.KitInfo.objects.filter(kitType = "TemplatingKit")])

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()
        ordering = field_list
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()


class AvailableOneTouchPlannedExperimentSummaryResource(ModelResource):
    class Meta:
        queryset = models.PlannedExperiment.objects.filter(isReusable = False, planExecuted = False, templatingKitName__in = [kit.name for kit in models.KitInfo.objects.filter(kitType = "TemplatingKit")])

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()
        
        ordering = field_list
        filtering = field_dict(field_list)
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

        metadata_allowed_methods = ['get',]


class OneTouchPlanTemplateSummaryResource(ModelResource):
    class Meta:
        queryset = models.PlannedExperiment.objects.filter(isReusable = True, templatingKitName__in = [kit.name for kit in models.KitInfo.objects.filter(kitType = "TemplatingKit")])

        #allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()
        ordering = field_list
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

        metadata_allowed_methods = ['get',]
        

class AvailablePlannedExperimentSummaryResource(ModelResource):
    class Meta:
        queryset = models.PlannedExperiment.objects.filter(isReusable = False, planExecuted = False)

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


class TorrentSuite(ModelResource):
    """Allow system updates via API"""

    def prepend_urls(self):
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

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()
        update_allowed_methods = ['get', 'put']


class PublisherResource(ModelResource):

    class Meta:
        queryset = models.Publisher.objects.all()

        # allow ordering and filtering by all fields
        field_list = models.Publisher._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

    def prepend_urls(self):
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

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()

class ContentResource(ModelResource):
    publisher = ToOneField(PublisherResource, 'publisher')
    contentupload = ToOneField(ContentUploadResource, 'contentupload')

    class Meta:
        queryset = models.Content.objects.all()

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


class CompositeLibMetricsResource(ModelResource):
    class Meta:
        queryset = models.LibMetrics.objects.all()

        #allow ordering and filtering by all fields
        field_list = ['id', 'q20_bases', 'i100Q20_reads',
            'q20_mean_alignment_length', 'aveKeyCounts']
        fields = field_list
        ordering = field_list
        filtering = field_dict(field_list)


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


class CompositeQualityMetricsResource(ModelResource):
    class Meta:
        queryset = models.QualityMetrics.objects.all()

        #allow ordering and filtering by all fields
        field_list = ['id', 'q0_bases', 'q20_bases', 'q20_mean_read_length',
        'q0_mean_read_length', 'q0_reads', 'q20_reads']
        fields = field_list
        filtering = field_dict(field_list)

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
            bundle.data['diskusage'] += bundle.obj.experiment.diskusage

        for t, dm_type in dm_types:
            try:
                dmfilestat = bundle.obj.get_filestat(dm_type)
                bundle.data['%s_state' % t] = dmfilestat.get_action_state_display()
                if not dmfilestat.isdisposed():
                    bundle.data['%s_keep' % t] = dmfilestat.getpreserved()
                if dmfilestat.in_process():
                    bundle.data['in_process'] = True
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
                    qset = qset & ~Q(dmfilestat__action_state__in=['AG','DG','AD','DD'])
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

chips = dict((c.name, c.description) for c in models.Chip.objects.all())

class MonitorExperimentResource(ModelResource):

    results = fields.ToManyField(CompositeResultResource, 'results_set', full=True, related_name='experiment')
    plan = fields.ToOneField(MonitorPlannedExperimentResource, 'plan', full=True, null=True, blank=True)

    barcodeId = fields.CharField(readonly = True, attribute='eas_set__barcodeKitName', null=True)
    library = fields.CharField(readonly = True, attribute='eas_set__reference', null=True)

    def dehydrate(self, bundle):
        bundle.data['results'].sort(key=monitor_comparator)
        bundle.data['results'].reverse()

        chipDescription = chips.get(bundle.obj.chipType, "")
        if not chipDescription:
            chipDescription = chips.get(bundle.obj.chipType[:3], bundle.obj.chipType)
        bundle.data['chipDescription'] = chipDescription
        
        try:
            qcThresholds = dict((qc.qcType.qcName, qc.threshold) for qc in
                            bundle.obj.plan.plannedexperimentqc_set.all())
        except:
            qcThresholds = {}
        bundle.data['qcThresholds'] = qcThresholds

        bundle.data['barcodeId'] = bundle.obj.get_barcodeId()
        bundle.data['library'] = bundle.obj.get_library()

        return bundle

    def get_object_list(self, request):
        one_week_ago = datetime.datetime.now() - datetime.timedelta(weeks=1)
        one_day_ago = datetime.datetime.now() - datetime.timedelta(days=1)
        return super(MonitorExperimentResource, self).get_object_list(
            request).filter(
                Q(resultDate__gte=one_day_ago) |
                (
                    ~(  Q(ftpStatus="Complete") |
                        Q(ftpStatus="Completed") |
                        Q(ftpStatus="User Aborted") |
                        Q(ftpStatus="Missing File(s)") |
                        Q(ftpStatus="Lost Chip Connection")
                    ) & Q(date__gte=one_week_ago)
                )
            )

    class Meta:
        #TODO: CONSIDER MAKING A SOUTH MIGRATION TO UPDATE EXPERIMENT WITH runMode=="" for .exclude(runMode="") using Experiment.objects.filter(runMode="").update(ftpStatus="Missing File(s)")
        queryset = models.Experiment.objects.select_related('plan').prefetch_related(
                'results_set',
                'results_set__analysismetrics',
                'results_set__libmetrics',
                'results_set__qualitymetrics',
                'results_set__eas',
                'results_set__projects',
                'plan__plannedexperimentqc_set',
                'plan__plannedexperimentqc_set__qcType'
            ).exclude(expName="NONE_ReportOnly_NONE").exclude(status="planned").order_by('-resultDate')

        field_list = ['id', 'expName', 'displayName', 'date', 'library', 'ftpStatus',
            'pgmName', 'storage_options', 'sample', 'flows', 'chipType',
            'notes', 'results', 'runMode', 'barcodeId', 'resultDate', 'results_set', 'star'
        ]
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
        fields = ['runType', 'id']
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

        chipDescription = chips.get(bundle.obj.chipType, "")
        if not chipDescription:
            chipDescription = chips.get(bundle.obj.chipType[:3], bundle.obj.chipType)
        bundle.data['chipDescription'] = chipDescription
                
        samples = bundle.obj.samples.all()
        bundle.data['sample'] = samples[0].name if samples else ""
        if bundle.data['results']:
            eas = bundle.data['results'][0].obj.eas
            if eas:
                bundle.data['barcodeId'] = eas.barcodeKitName
                bundle.data['library'] = eas.reference

        try:
            dmfilestat = bundle.obj.results_set.all()[0].get_filestat(dmactions_types.SIG)
            bundle.data['archived'] = False if not dmfilestat.isdisposed() else dmfilestat.get_action_state_display()
        except:
            bundle.data['archived'] = False
        bundle.data['keep'] = bundle.obj.storage_options == 'KI'

        return bundle


    def apply_filters(self, request, applicable_filters):
        '''
        !!DEVELOPERS!!: BE SURE TO ALSO UPDATE THE CORRESPONDING rundb.data.views.getCSV() if you ADD, MODIFY this method.
        '''
        base_object_list = super(CompositeExperimentResource, self).apply_filters(request, applicable_filters)

        name = request.GET.get('all_text', None)
        if name is not None:
            qset = (
                Q(expName__iregex=name) |
                Q(results_set__resultsName__iregex=name) |
                Q(notes__iregex=name)
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

    class Meta:
        queryset = models.Experiment.objects.select_related(
            'plan',
        ).prefetch_related(
            'results_set',
            'results_set__analysismetrics',
            'results_set__libmetrics',
            'results_set__qualitymetrics',
            'results_set__eas',
            'results_set__projects',
            'results_set__dmfilestat_set',
            'samples'
        ).exclude(expName="NONE_ReportOnly_NONE").exclude(status="planned").order_by('-resultDate')

        field_list = [
                    'barcodeId',    'chipType',     'date',
                    'expName',      'flows',        'ftpStatus',
                    'id',           'library',      'notes',
                    'pgmName',      'resultDate',   'results',
                    'results_set',  'runMode',
                    'star',         'storage_options',
        ]

        fields = field_list
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication()
        authorization = DjangoAuthorization()
        # This query is expensive, and used on main data tab.
        # Cache frequent access
        #cache = SimpleCache(timeout=17)


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
    defaultPESeqKit = fields.ToOneField(KitInfoResource, 'defaultPairedEndSequencingKit', full=True, null=True)
    defaultPELibKit = fields.ToOneField(KitInfoResource, 'defaultPairedEndLibraryKit', full=True, null=True)

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
        self.uid = user.pk
        self.username = user.username
        self.admin = user.is_superuser
        self.api_key = None
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

    def obj_get_list(self, request=None, **kwargs):
        return self.get_object_list(request)

    def obj_get(self, request=None, **kwargs):
        return AccountObject(request.user)
