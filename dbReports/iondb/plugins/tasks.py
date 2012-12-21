# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

# Celery tasks for managing plugins
import os
import logging

from django.conf import settings
from celery.task import task

log = logging.getLogger(__name__)

## Helper common to all methods.
def get_info_from_script(name, script, context=None):
    from ion.plugin.loader import cache
    from ion.plugin.info import PluginInfo
    mod = cache.load_module(name, script)
    cls = cache.get_plugin(name)
    if not cls:
        return None
    plugin = cls()
    ## TODO plugin = cls(**context)
    info = PluginInfo.from_instance(plugin)
    return info


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


## Query all plugins for their info/inspect json block
@task(queue="plugins")
def scan_all_plugins(pluginlist):
    """
    Pass in list of (plugin, scriptpath) pairs, and all will be instantiated an interrogated.
    Note: They do not need to be installed yet. Just name, script pairs, or name, script, context tuples.
    """
    ret = {}
    logger = scan_all_plugins.get_logger()

    count = 0
    for data in pluginlist:
        if len(data) == 2:
            (name, path) = data
            context = None
        else:
            (name, path, context) = data

        if os.path.isdir(path):
            path = find_pluginscript(path, name)
        try:
            info = get_info_from_script(name, path, context)
            if info is not None:
                info = info.todict()
        except:
            logger.exception("Failed to load plugin '%s' from '%s'", name, path)
            info = None

        if info is None:
            logger.info("Failed to get plugininfo: '%s' from '%s'", name, path)
        ret[path] = info
        count += 1

        if info and context and 'plugin' in context:
            ppk = context["plugin"].pk
            try:
                from iondb.rundb.models import Plugin
                p = Plugin.objects.get(pk=ppk)
                p.updateFromInfo(info)
                p.save()
            except:
                logger.exception("Failed to save info to plugin db cache")

    logger.info("Rescanned %d plugins", count)
    return ret

# Helper task to invoke PluginManager rescan, used to rescan after a delay
@task(queue="plugins")
def add_remove_plugins():
    from iondb.plugins.manager import pluginmanager
    pluginmanager.rescan()


