# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

# Celery tasks for managing plugins
from __future__ import absolute_import
import os
import logging
import traceback

import iondb.celery
from django.conf import settings
from celery.task import task, group
from celery.result import ResultSet
from celery.utils.log import get_task_logger

log = logging.getLogger(__name__)
logger = get_task_logger(__name__)

## Helper common to all methods.
def get_info_from_script(name, script, context=None, add_to_store=True):
    from ion.plugin.loader import cache
    from ion.plugin.info import PluginInfo
    mod = cache.load_module(name, script, add_to_store)

    # if the load module command returned a None object, then lets raise an exception based on what happened
    if not mod:
        if name in cache.module_errors:
            raise cache.module_errors[name]
        else:
            raise Exception("Error loading the module.")

    cls = cache.get_plugin(name)
    if not cls:
        raise Exception("The python module loaded but no classes which extend 'IonPlugin' registered themselves with the framework.")
    plugin = cls()
    return PluginInfo.from_instance(plugin)

def find_pluginscript(pluginpath, pluginname=None):
    # Legacy Launch Script
    launchsh = os.path.join(pluginpath, "launch.sh")
    if os.path.exists(launchsh):
        return launchsh

    # [3.0] Plugin Python class
    basedir = pluginpath
    while basedir and not pluginname:
        (basedir, pluginname) = os.path.split(basedir)
    plugindef = os.path.join(pluginpath, pluginname + '.py')
    if os.path.exists(plugindef):
        return plugindef

    log.error("Plugin path is missing launch script '%s' or '%s'", launchsh, plugindef)
    return None

## Derived from http://antonym.org/2005/12/dropping-privileges-in-python.html
def drop_privileges(uid_name='nobody', gid_name='nogroup'):
    import pwd, grp
    if os.getuid() != 0:
        log.info("drop_privileges: already running as non-root user '%s'",
                 pwd.getpwuid(os.getuid()))
        return

    # If we started as root, drop privs and become the specified user/group
    # Get the uid/gid from the name
    running_uid = pwd.getpwnam(uid_name)[2]
    running_gid = grp.getgrnam(gid_name)[2]

    # Try setting the new uid/gid
    try:
        os.setgroups([])
        os.setgid(running_gid)
    except OSError, e:
        log.exception('Could not set effective group id')

    try:
        os.setuid(running_uid)
    except OSError, e:
        log.exception('Could not set effective user id')

    # Ensure a very convervative umask
    old_umask = os.umask(077)

    final_uid = os.getuid()
    final_gid = os.getgid()
    log.info('drop_privileges: running as %s/%s',
             pwd.getpwuid(final_uid)[0],
             grp.getgrgid(final_gid)[0])

@task(queue="plugins", soft_time_limit=30)
def scan_plugin(data, add_to_store=True):
    """
    This method will interogate a specific plugin and get it's information
    :parameter data: A tuple of type (name, path, context) to be interrogated. Note: They do not need to be installed yet. Just name, script pairs, or name, script, context tuples.
    :parameter add_to_store: Add the plugins in the list to the store
    :returns: A PluginInfo object type
    """

    if len(data) == 2:
        (name, path) = data
        context = None
    else:
        (name, path, context) = data

    if os.path.isdir(path):
        path = find_pluginscript(path, name)

    info = {}
    try:
        info = get_info_from_script(name, path, context, add_to_store)
        if info is not None:
            info = info.todict()
    except:
        logger.error(traceback.format_exc())

    if not info:
        logger.info("Failed to get plugininfo: '%s' from '%s'", name, path)

    if info and context and 'plugin' in context:
        ppk = context["plugin"].pk
        try:
            from iondb.rundb.models import Plugin
            p = Plugin.objects.get(pk=ppk)
            p.updateFromInfo(info)
            p.save()
        except ValueError:
            # plugin version mismatch
            pass
        except:
            logger.exception("Failed to save info to plugin db cache")

    return info


def scan_all_plugins(plugin_list, add_to_store=True):
    """
     Query all plugins for their info/inspect json block
    :parameter plugin_list: Pass in list of (plugin, scriptpath) pairs, and all will be instantiated an interrogated.
    Note: They do not need to be installed yet. Just name, script pairs, or name, script, context tuples.
    :parameter add_to_store: Add the plugins in the list to the store
    :returns: A dictionary of PluginInfo object for all of the successfully loaded modules keyed by their path
    """
    plugin_info = dict()

    # fire off sub-tasks for each plugin to be scanned and collect results
    plugin_scan_tasks = [scan_plugin.s(data, add_to_store) for data in plugin_list]
    try:
        result = group(plugin_scan_tasks).apply_async().join(timeout=300)
    except Exception as exc:
        logger.error("Error while scanning plugins: " + str(exc))
    else:
        for i, data in enumerate(plugin_list):
            path = data[1]
            plugin_info[path] = result[i]

        logger.info("Rescanned %d plugins", len(plugin_list))

    return plugin_info

# Helper task to invoke PluginManager rescan, used to rescan after a delay
@task(queue="plugins", ignore_result=True)
def add_remove_plugins():
    from iondb.plugins.manager import pluginmanager
    pluginmanager.rescan()

@task(ignore_result=True)
def backfill_pluginresult_diskusage():
    '''Due to new fields (inodes), and errors with counting contents of symlinked files, this function
    updates every Result object's diskusage value.
    '''
    from django.db.models import Q
    from iondb.rundb import models
    from datetime import timedelta
    from django.utils import timezone

    # Setup log file logging
    filename = '/var/log/ion/%s.log' % 'backfill_pluginresult_diskusage'
    log = logging.getLogger('backfill_pluginresult_diskusage')
    log.propagate = False
    log.setLevel(logging.DEBUG)
    handler = logging.handlers.RotatingFileHandler(filename, maxBytes=1024 * 1024 * 10, backupCount=5)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.info("\n===== New Run =====")
    log.info("PluginResults:")
    obj_list = models.PluginResultJob.objects.filter(
        Q(plugin_result__size=-1) | Q(plugin_result__inodes=-1),
        state__in=('Complete', 'Error'),
        starttime__gte=(timezone.now() - timedelta(days=30)),
    )
    for obj in obj_list:
        try:
            obj.UpdateSizeAndINodeCount()
        except OSError:
            obj.size, obj.inodes = -1
            obj.save(update_fields=["size", "inodes"])
        except:
            log.exception(traceback.format_exc())

        log.debug("Scanned: %s at %s -- %d (%d)", str(obj), obj.default_path, obj.size, obj.inodes)


@task
def calc_size(prid):
    from iondb.rundb.models import PluginResult
    try:
        obj = PluginResult.objects.get(pk=prid)
    except (PluginResult.MultipleObjectsReturned, PluginResult.DoesNotExist):
        return
    try:
        d = obj.default_path
        if not d:
            log.info("check_size: No path: %s at %s -- %d (%d)", str(obj), d, obj.size, obj.inodes)
            return
        if not os.path.exists(obj.default_path):
            log.error("check_size: Path doesn't exist: %s at %s -- %d (%d)", str(obj), d, obj.size, obj.inodes)
            return
        obj.size, obj.inodes = obj._calc_size
        log.debug("Scanning: %s at %s -- %d (%d)", str(obj), d, obj.size, obj.inodes)
    except OSError:
        log.exception("Failed to compute plugin size: %s at '%s'", str(obj), obj.default_path)
        obj.size, obj.inodes = -1
    except:
        log.exception(traceback.format_exc())
    finally:
        obj.save(update_fields=["size", "inodes"])
    return (obj.size, obj.inodes)

@task(ignore_result=True)
def cleanup_pluginresult_state():
    ''' Fix jobs stuck in running states '''
    from django.db.models import Q
    from iondb.rundb import models
    from datetime import timedelta
    from django.utils import timezone

    # Setup log file logging
    filename = '/var/log/ion/%s.log' % 'backfill_pluginresult_diskusage'
    log = logging.getLogger('backfill_pluginresult_diskusage')
    log.propagate = False
    log.setLevel(logging.DEBUG)
    handler = logging.handlers.RotatingFileHandler(filename, maxBytes=1024 * 1024 * 10, backupCount=5)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.info("\n===== New Run =====")

    transition_states = ('Pending', 'Started', 'Queued')
    # Jobs with no timestamps - Could be large backlog.
    obj_list = models.PluginResultJob.objects.filter(starttime__isnull=True,)
    for obj in obj_list:
        obj.starttime = obj.plugin_result.result.timeStamp
        endtime = obj.endtime
        # These are handled in next query, but might as well resolve during this pass
        if obj.state in transition_states:
            obj.plugin_result.complete(obj.run_level, state='Error')
            if endtime is None:
                obj.endtime = obj.starttime + timedelta(hours=24)
        else:
            if endtime is None:
                obj.endtime = obj.starttime

        log.debug("Backfilling timestamps: %s [%s] -- %s, %s -- %d (%d)", str(obj), obj.state, obj.starttime, obj.endtime, obj.plugin_result.size, obj.plugin_result.inodes)
        obj.save()

    # Jobs stuck in SGE states
    obj_list = models.PluginResultJob.objects.filter(
        state__in=transition_states,
        starttime__lte=timezone.now() - timedelta(hours=25),
    )
    for obj in obj_list:
        endtime = obj.endtime
        obj.plugin_result.complete(obj.run_level, state='Error') # clears api_key, sets endtime, runs calc_size
        if endtime is None:
            obj.endtime = obj.starttime + timedelta(hours=24)

        log.debug("Cleaning orphan job: %s [%s] -- %s, %s -- %d (%d)", str(obj), obj.state, obj.starttime, obj.endtime, obj.plugin_result.size, obj.plugin_result.inodes)
        obj.save()