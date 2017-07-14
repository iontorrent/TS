#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import SocketServer
from SimpleXMLRPCServer import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
import xmlrpclib
from distutils.version import LooseVersion

from datetime import datetime
import httplib2
import datetime
import json
import os
import sys
import traceback
import tempfile
import apt
import zipfile
import socket
import subprocess
import shutil

from djangoinit import settings
from iondb.plugins.runner import PluginRunner
from iondb.plugins.manager import pluginmanager
from iondb.plugins.launch_utils import get_plugins_to_run, add_hold_jid
from iondb.plugins.plugin_json import make_plugin_json

from iondb.rundb.models import Results, PluginResult, PluginResultJob, User, Plugin
from django.db import IntegrityError

from ion.plugin.constants import Feature, RunLevel

import logging
try:
    from logging.config import dictConfig
except ImportError:
    from django.utils.log import dictConfig

from iondb.plugins.manager import pluginmanager

__version__ = filter(str.isdigit, "$Revision$")

# Setup log file logging
try:
    log = '/var/log/ion/ionPlugin.log'
    infile = open(log, 'a')
    infile.close()
except:
    print traceback.format_exc()
    log = './ionPlugin.log'

LOGGING = {
    'version': 1,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s %(module)-17s line:%(lineno)-4d %(levelname)-8s %(message)s',
        },
        'basic': {
            'format': "%(asctime)s - %(levelname)s - %(message)s",
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'stream': 'ext://sys.stdout',
        },
        'pluginlog': {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'basic',
            'level': 'DEBUG',
            'filename': log,
            'mode': 'a',
            'maxBytes': 1024 * 1024 * 10,
            'backupCount': 5,
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['pluginlog', 'console']
    },
    'ionPlugin': {
        'level': 'DEBUG',
    },
    'ion': {
        'level': 'DEBUG',
    },
    'iondb': {
        'level': 'DEBUG',
    }
}
dictConfig(LOGGING)
logger = logging.getLogger('ionPlugin')

# distributed resource management
try:
    # We want to determine if we have the ability to make calls to the
    # drmaa interface. In order to do so, we first need to set environment
    # variables.
    if not settings.SGE_ENABLED:
        # bail if SGE (currently the only supported drmaa target) is
        # disabled from the site settings file.
        raise ImportError
    # Set env. vars necessary to talk to SGE. The SGE root contains files
    # describing where the grid master lives. The other variables determine
    # which of several possible grid masters to talk to.
    # this may or may not be needed because of the other vars that are being set by ionJobServer
    for k in ("SGE_ROOT", "SGE_CELL", "SGE_CLUSTER_NAME", "SGE_QMASTER_PORT", "SGE_EXECD_PORT", "DRMAA_LIBRARY_PATH"):
        if not k in os.environ:
            os.environ[k] = str(getattr(settings, k))
    try:
        import drmaa
    except RuntimeError:
        # drmaa will sometimes raise RuntimeError if libdrmaa1.0 is not
        # installed.
        logger.debug("Unexpected error: %s" % str(sys.exc_info()))
        raise ImportError
    import atexit  # provides cleanup of the session object
    try:
        HAVE_DRMAA = True
        # create a single drmaa session
        _session = drmaa.Session()
        try:
            _session.initialize()
            logger.debug("DRMAA session initialized")
        except:
            logger.debug("Unexpected error: %s" % str(sys.exc_info()))
        atexit.register(_session.exit)
        djs = drmaa.JobState
        # globally define some status messages
        _decodestatus = {
            djs.UNDETERMINED: 'process status cannot be determined',
            djs.QUEUED_ACTIVE: 'job is queued and active',
            djs.SYSTEM_ON_HOLD: 'job is queued and in system hold',
            djs.USER_ON_HOLD: 'job is queued and in user hold',
            djs.USER_SYSTEM_ON_HOLD: ('job is queued and in user '
                                      'and system hold'),
            djs.RUNNING: 'job is running',
            djs.SYSTEM_SUSPENDED: 'job is system suspended',
            djs.USER_SUSPENDED: 'job is user suspended',
            djs.DONE: 'job finished normally',
            djs.FAILED: 'job finished, but failed',
            }
        InvalidJob = drmaa.errors.InvalidJobException
    except drmaa.errors.InternalException:
        # If we successfully import drmaa, but it somehow wasn't configured
        # properly, we will gracefully bail by raising ImportError
        raise ImportError
except (ImportError, AttributeError):
    # drmaa import failed
    HAVE_DRMAA = False
    InvalidJob = ValueError


def logAndRaise(msg, logger):
    """
    Helper method to log an exception method and then raise it
    :param msg: The message to report and populate the exception with
    :param logger: The logger to log the message to
    """
    logger.error(msg)
    raise xmlrpclib.Fault(1, msg)


def SGEPluginJob(start_json, hold=False):
    """
    Spawn a thread that will start a SGE job, and wait for it to return
    after it has return it will make a PUT the API using to update the status
    of the run
    args is a dict of all the args that are needed by for the plugin to run
    """
    try:
        os.umask(0o0002)

        plugin_output = start_json['runinfo']['results_dir']
        # Make sure the dirs exist
        if not os.path.exists(plugin_output):
            os.makedirs(plugin_output, 0o0775)

        plugin = start_json['runinfo']['plugin']
        logger.info("Preparing for SGE submission - plugin %s --v%s on result %s (%s)", plugin['name'], plugin['version'], start_json['expmeta']['results_name'], start_json['runinfo']['pk'])

        # Branch for launch.sh vs 3.0 plugin class
        # logger.debug("Finding plugin definition: '%s':'%s'", plugin['name'], plugin['path'])
        (launch, isCompat) = pluginmanager.find_pluginscript(plugin['path'], plugin['name'])
        analysis_name = os.path.basename(start_json['runinfo']['analysis_dir'])
        if not launch or not os.path.exists(launch):
            logger.error("Analysis: %s. Path to plugin script: '%s' Does Not Exist!", analysis_name, launch)
            return 1

        # Create individual launch script from template and plugin launch.sh
        launcher = PluginRunner()
        if isCompat:
            launchScript = launcher.createPluginWrapper(launch, start_json)
        else:
            start_json.update({'command': ["python %s -vv" % launch]})
            launchScript = launcher.createPluginWrapper(None, start_json)
        launchWrapper = launcher.writePluginLauncher(plugin_output, plugin['name'], launchScript)
        # Returns filename of startpluginjson file, passed to script below
        startpluginjson = launcher.writePluginJson(start_json)

        # Prepare drmaa Job - SGE/gridengine only
        jt = _session.createJobTemplate()
        jt.nativeSpecification = " -q %s" % "plugin.q"
        if Feature.EXPORT in plugin.get('features', []):
            jt.nativeSpecification += ' -l ep=1 '

        hold_jid = plugin.get('hold_jid', [])
        if hold_jid:
            jt.nativeSpecification += " -hold_jid " + ','.join(str(jobid) for jobid in hold_jid)

        jt.workingDirectory = plugin_output
        jt.outputPath = ":" + os.path.join(plugin_output, "drmaa_stdout.txt")
        jt.joinFiles = True  # Merge stdout and stderr

        # Plugin command is: ion_pluginname_launch.sh -j startplugin.json
        jt.remoteCommand = launchWrapper
        jt.args = ["-j", startpluginjson]

        if hold:
            jt.jobSubmissionState = drmaa.JobSubmissionState.HOLD_STATE

        # Submit the job to drmaa
        jobid = _session.runJob(jt)

        logger.info("Analysis: %s. Plugin: %s. Job: %s Queued." % (analysis_name, plugin['name'], jobid))

        _session.deleteJobTemplate(jt)
        return jobid
    except:
        logger.exception("SGEPluginJob method failure")
        raise

# Threaded mix-in


class AsyncXMLRPCServer(SocketServer.ThreadingMixIn, SimpleXMLRPCServer):
    pass


class PluginServer(object):

    """
    Manages plugin functions
    """
    # the current cache for the system
    cache = apt.Cache()

    def __init__(self, allowNone=True, useDateTime=False):
        self.start_time = datetime.datetime.now()
        self.allowNone = allowNone
        self.useDateTime = useDateTime  # Support for old twisted with new python xmlrpclib

    def GetSupportedPlugins(self, pluginNames=list(), updateCache=False, onlyUpgradable=False):
        """
        Gets a list of all supported plugins
        :param pluginNames: A list of plugins in question or leave empty to all
        :param updateCache: Use this flag to force a refresh of the cache
        :param onlyUpgradable: The list returned will only be filled with upgradable items
        :return: A dictionary keyed by the names of packages and the values being a list of all of the versions available
        """

        # only call this when we have to as it will take a moment
        if updateCache:
            self.cache = apt.Cache()

        # get a list of all cached items
        if len(pluginNames) == 0:
            pluginNames = [plugin for plugin in self.cache.keys() if plugin.startswith('ion-plugin-')]

        # construct a dictionary of each plugin and all available versions
        plugins = dict()
        for pluginName in pluginNames:
            plugins[pluginName] = self.GetSupportedPluginInfo(pluginName)

        # only return the upgradeable versions
        if onlyUpgradable:
            return dict((key, value) for key, value in plugins.items() if value['UpgradeAvailable'])

        # return a dictionary of all of the plugin names
        return plugins

    def GetSupportedPluginInfo(self, pluginPackageName):
        """
        Gets information for an individual plugin
        :param pluginPackageName: The name of the package which the plugin belongs to
        :return: A dictionary information about a plugin
        Key               : Description
        AvailableVersions : List of strings of the available version for installation, sorted lowest to highest
        UpgradeAvailable  : Boolean which is true if the package manager indicates that this package can be upgraded, false otherwise
        CurrentVersion    : String representation of the current installed version according to the package manager, if not installed the string 'Not Installed'
        """

        # sanity check that we have this plugin package name
        if not self.cache.has_key(pluginPackageName):
            return

        aptPlugin = self.cache[pluginPackageName]

        plugin = dict()
        plugin['AvailableVersions'] = list()
        plugin['UpgradeAvailable'] = False
        for version in aptPlugin.versions:
            plugin['AvailableVersions'].append(version.version)
        # sort the versions from lowest to highest
        plugin['AvailableVersions'].sort(key=LooseVersion)

        if not aptPlugin.is_installed:
            plugin['CurrentVersion'] = 'Not Installed'
        elif aptPlugin.is_upgradable:
            plugin['UpgradeAvailable'] = True
            plugin['CurrentVersion'] = aptPlugin.installed.version
        else:
            plugin['CurrentVersion'] = aptPlugin.installed.version

        return plugin

    def launchPlugins(self, result_pk, plugins, net_location, username, runlevel=RunLevel.DEFAULT, params={}):
        """
        Launch multiple plugins with dependencies
        For multi-runlevel plugins the input 'plugins' is common for all runlevels
        """
        msg = ''
        logger.debug("[launchPlugins] result %s requested plugins: %s" % (result_pk, ','.join(plugins.keys())))

        try:
            # get plugins to run for this runlevel
            plugins, plugins_to_run, satisfied_dependencies = get_plugins_to_run(plugins, result_pk, runlevel)

            if len(plugins_to_run) > 0:
                logger.debug("[launchPlugins] runlevel: %s, depsolved launch order: %s" % (runlevel, ','.join(plugins_to_run)))
            else:
                logger.debug("[launchPlugins] no plugins to run at runlevel: %s" % runlevel)
                return plugins, msg

            result = Results.objects.get(pk=result_pk)
            report_dir = result.get_report_dir()
            url_root = result.reportWebLink()

            # get pluginresult owner - must be a valid TS user
            try:
                user = User.objects.get(username=username)
            except User.DoesNotExist:
                user = User.objects.get(pk=1)
                logger.error("Invalid user specified for plugin launch: %s, will use %s" % (username, user.username))

            for name in plugins_to_run:
                try:
                    p = plugins[name]
                    # get params for this plugin, make empty json value if doesn't exist
                    plugin_params = params.setdefault('plugins', {}).setdefault(name, {})

                    # Get pluginresult for multi-runlevel plugins or if specified to be reused by manual launch
                    pr = None
                    pluginresult_pk = p.get('pluginresult') or plugin_params.get('pluginresult')

                    if pluginresult_pk:
                        logger.debug("Searching for PluginResult: %s", pluginresult_pk)
                        try:
                            pr = result.pluginresult_set.get(pk=pluginresult_pk)
                        except:
                            logger.error("Failed to find pluginresult for plugin %s, result %s: %s" % (name, result.resultsName, pluginresult_pk))
                            pr = None

                    # Create new pluginresult - this is the most common path
                    if not pr:
                        pr = PluginResult.objects.create(result_id=result_pk, plugin_id=p['id'], owner=user)
                        logger.debug("New pluginresult id=%s created for plugin %s and result %s." % (pr.pk, name, result.resultsName))
                        # Always create new, unique output folder.
                        # Never fallback to old *_out format.
                        plugin_output = pr.path(create=True, fallback=False)
                    else:
                        # Use existing output folder
                        plugin_output = pr.path(create=False)

                    p['results_dir'] = plugin_output
                    p['pluginresult'] = pr.pk
                    p, holding_for = add_hold_jid(p, plugins, runlevel, satisfied_dependencies)

                    start_json = make_plugin_json(result_pk, report_dir, p, plugin_output, net_location, url_root, username, runlevel,
                                                  params.get('blockId', ''), params.get('block_dirs', ["."]), plugin_params.get('instance_config', {}))

                    # Pass on run_mode (launch source - manual/instance, pipeline)
                    start_json['runplugin']['run_mode'] = params.get('run_mode', '')

                    # add dependency info to startplugin json
                    if p.get('depends') and isinstance(p['depends'], list):
                        start_json['depends'] = {}
                        for depends_name in p['depends']:
                            if depends_name in satisfied_dependencies:
                                start_json['depends'][depends_name] = satisfied_dependencies[depends_name]
                            elif depends_name in plugins and plugins[depends_name].get('pluginresult'):
                                start_json['depends'][depends_name] = {
                                    'pluginresult': plugins[depends_name]['pluginresult'],
                                    'version': plugins[depends_name].get('version', ''),
                                    'pluginresult_path': plugins[depends_name].get('results_dir')
                                }

                    # prepare for launch: updates config, sets pluginresults status, generates api key
                    pr.prepare()
                    pr.save()
                    # update startplugin json with pluginresult info
                    start_json['runinfo']['pluginresult'] = pr.pk
                    start_json['runinfo']['api_key'] = pr.apikey

                    # NOTE: Job is held on start, and subsequently released
                    # to avoid any race condition on updating job queue status
                    # launch plugin
                    jid = SGEPluginJob(start_json, hold=True)

                    if jid:
                        # Update pluginresult status
                        prj, created = PluginResultJob.objects.get_or_create(
                            plugin_result=pr,
                            run_level=runlevel,
                            grid_engine_jobid=jid,
                            state = 'Queued',
                            config = start_json['pluginconfig'],
                        )
                        prj.save()

                        # Release now that jobid and queued state are set.
                        _session.control(jid, drmaa.JobControlAction.RELEASE)  # no return value

                    msg += 'Plugin: %s result: %s, jid %s, depends %s, holding for %s \n' % (p.get('name', ''), result.resultsName, jid, p.get('depends', []), holding_for)

                    if runlevel != RunLevel.BLOCK:
                        p['jid'] = jid
                    else:
                        p.setdefault('block_jid', []).append(jid)

                except:
                    logger.error(traceback.format_exc())
                    msg += 'ERROR: Plugin %s failed to launch.\n' % p['name']
                    pr = PluginResult.objects.get(pk=pr.pk)
        except:
            logger.error(traceback.format_exc())
            msg += 'ERROR: Failed to launch requested plugins.'

        return plugins, msg

    def sgeStop(self, jobid):
        """
        Terminate a running SGE job
        """
        logger.debug("SGE job kill request")

        if jobid is None:
            return None
        jobid = str(jobid)
        logger.info("Terminating SGE job %s", jobid)
        _session.control(jobid, drmaa.JobControlAction.TERMINATE)  # no return value
        status = "Unknown"
        try:
            jobinfo = _session.wait(jobid, timeout=20)
            # returns JobInfo object
            logger.info("Job %s wasAborted=%s with exit_status=%s", jobinfo.jobId, jobinfo.wasAborted, jobinfo.resourceUsage.get('exit_status'))
            status = "Job %s wasAborted=%s with exit_status=%s" % (jobinfo.jobId, jobinfo.wasAborted, jobinfo.resourceUsage.get('exit_status'))
        except drmaa.errors.InvalidJobException:
            status = "Job already terminated, or started previously."
            logger.warn("SGE job %s already terminated", jobid)
        except drmaa.errors.ExitTimeoutException:
            status = "Unknown. Timeout waiting for job to terminate"
            logger.warn("SGE job %s not terminated within timeout", jobid)
        return status

    def uptime(self):
        """Return the amount of time the ``ionPlugin`` has been running."""
        diff = datetime.datetime.now() - self.start_time
        seconds = float(diff.days * 24 * 3600)
        seconds += diff.seconds
        seconds += float(diff.microseconds) / 1000000.0
        logger.debug("Uptime called - %d (s)" % seconds)
        return seconds

    def pluginStatus(self, jids):
        """Get the status of the job(s)"""
        if isinstance(jids, list):
            return_list = True
        else:
            jids = [jids]
            return_list = False

        ret = []
        for jobid in jids:
            try:
                jobid = str(jobid)
                logging.debug("jobstatus for %s" % jobid)
                status = _session.jobStatus(jobid)
                ret.append(_decodestatus[status])
            except:
                logging.error(traceback.format_exc())
                ret.append("DRMAA BUG")

        return ret if return_list else ret[0]

    def updatePR(self, pk, grid_engine_job_id, state, store, jobid=None):
        """
        Because of the Apache security it is hard for crucher nodes to contact the API directly.
        This method is a simple proxy to update the status of plugins
        """
        # update() doesn't trigger special state change behaviors...
        # return PluginResult.objects.filter(id=pk).update(state=state, store=store)
        try:
            # with transaction.atomic():

            # TODO: should do an explicit check for the database object in case it's been deleted
            pr = PluginResult.objects.get(id=pk)
            if state is not None:
                pr.SetState(state, grid_engine_job_id, jobid)

            if store is not None:
                # Validate JSON?
                pr.store = store
                pr.save(update_fields=["store"])
        except Exception as exc:
            logger.exception("Failed to update plugin results: " + exc.message)
            return False

        return True

    def delete_pr_directory(self, directory):
        """We need to use the same user to delete the folder as created it"""
        def delete_error(func, path, info):
            logger.error("Failed to delete %s: %s", path, info)

        shutil.rmtree(directory, onerror=delete_error)

if __name__ == '__main__':
    # Instantiate the xmlrpc server
    server = AsyncXMLRPCServer(('', settings.IPLUGIN_PORT), SimpleXMLRPCRequestHandler, allow_none=True)
    server.timeout = 1

    # Register example object instance
    server.register_instance(PluginServer())

    # run!
    server.serve_forever()
