# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

# Module to manage plugins

from __future__ import absolute_import
from django.conf import settings
import os
import shutil
import re
import json
import celery.exceptions
import iondb.rundb.models
from distutils.version import LooseVersion
import iondb.rundb.tasks  ## circular import
from iondb.utils.utils import getPackageName
import iondb.plugins.tasks
import json
import datetime
import logging
import xmlrpclib
import subprocess
from iondb.utils.utils import getPackageName
logger = logging.getLogger(__name__)

LEGACY_PLUGIN_SCRIPT = "launch.sh"

class PluginManager(object):
    """Class for managing plugin installation and versioning activities.

    Encapsulates install, uninstall, and upgrade.

    Also implements rescan, inactivate_missing (aka purge_plugins), and search_for_plugins,
    which handle discovery of manually installed plugins.

    TODO:
    * Allow multiple version entries
        - Legacy - one plugin entry per name, installing different version replaces
          **Name is unique identifier**
        - Transitional - one entry per name and version, old versions always inactivated
          **only one active version**
        - Multi-Version - one entry per name and version, all paths and
          messaging uniquely identify plugins

    """

    def __init__(self, gc=None):
        self.default_plugin_script = LEGACY_PLUGIN_SCRIPT
        self.pluginroot = os.path.normpath(settings.PLUGIN_PATH or "/results/plugins/")

    def rescan(self):
        """ Convenience function to purge missing and find new plugins. Logs info message"""
        removed = self.inactivate_missing()
        installed = self.search_for_plugins()
        if removed or installed:
            logger.info("Rescan Plugins '%s': %d installed/updated and %d removed", self.pluginroot, installed, removed)
        return

    def inactivate_missing(self, plugins=None):
        """
        Removes records from plugin table which no longer have corresponding folder
        on the file system.  If the folder does not exist, we assume that the plugin
        has been deleted.  In any case, one cannot execute the plugin if the plugin
        folder has been removed.
        """
        if not plugins:
            # Review all currently installed/active plugins
            activeplugins = iondb.rundb.models.Plugin.objects.filter(active=True)

        # for each record, test for corresponding folder
        count = 0
        scanned = 0
        for plugin in activeplugins:
            # if specified folder does not exist
            if plugin.path == '' or not os.path.isdir(plugin.path):
                if plugin.active:
                    plugin.active = False
                    plugin.save()
                    count += 1

                # No need to uninstall - the path is already missing
            elif not plugin.pluginscript:
                # Path but no launch script anymore. [TS-5019]
                if plugin.active:
                    plugin.active = False
                    plugin.save()
                    count += 1
            scanned += 1
        logger.debug("Scanned %d, removed %d", scanned, count)
        return count

    def find_pluginscript(self, pluginpath, pluginname):
        islaunch = True
        pluginscript = None
        # Legacy Launch Script
        launchsh = os.path.join(pluginpath, self.default_plugin_script)
        # [3.0] Plugin Python class
        plugindef = os.path.join(pluginpath, pluginname + '.py')

        # NOTE launch.sh is preferred and will be used FIRST if both exist.
        # This may change in later releases, once python class matures.
        if os.path.exists(launchsh):
            pluginscript = launchsh
            islaunch = True
        elif os.path.exists(plugindef):
            pluginscript = plugindef
            islaunch = False
        else:
            logger.error("Plugin path is missing launch script '%s' or '%s'", launchsh, plugindef)
        return (pluginscript, islaunch)


    @staticmethod
    def get_plugininfo(pluginname, pluginscript, context=None, use_cache=False):
        """
        :parameter pluginname: The simple name of the plugin
        :parameter pluginscript: The path to the script
        :parameter context: An optional pre-existing context to pull from
        :parameter  use_cache: An optional flag to use the context instead of re-probing the file system
        Query plugin script for a block of json info.
        """
        if use_cache and context and 'plugin' in context:
            # Return current content immediately without waiting
            logger.debug("Using cached plugindata: %s %s", pluginname, context['plugin'].version)
            return context['plugin'].info()

        return iondb.plugins.tasks.scan_plugin((pluginname, pluginscript, context), False)

    def get_plugininfo_list(self, updatelist):
        info = {}
        if updatelist:
            info = iondb.plugins.tasks.scan_all_plugins(updatelist)

        return info

    def set_pluginconfig(self, plugin, configfile='pluginconfig.json'):
        """ if there is data in pluginconfig json
            set the plugin.config value in the database """
        # FIXME - move to plugin model?
        pluginconfigfile = os.path.join(plugin.path, configfile)
        if not os.path.exists(pluginconfigfile):
            return False

        try:
            with open(pluginconfigfile) as f:
                config = f.read()
            config = json.loads(config)
            if plugin.config != config:
                logger.info("Setting/Refreshing pluginconfig: '%s'", pluginconfigfile)
                plugin.config = config
                return True
        except:
            logger.exception("Failed to load pluginconfig from '%s'", pluginconfigfile)

        # Invalid, or Unchanged
        return False

    def search_for_plugins(self, basedir=None):
        """ Scan folder for uninstalled or upgraded plugins
            Returns number of plugins installed or upgraded
        """
        if not basedir:
            basedir = self.pluginroot

        # Basedir - typically '/results/plugins', passed in args
        if not os.path.exists(basedir):
            return None

        # reset permissions assuming for all of the non supported plugins
        for i in os.listdir(basedir):
            if i in ["scratch", "implementations", "archive"]:
                continue
            if not getPackageName(os.path.join(basedir, i)):
                try:
                    subprocess.check_call(['sudo', '/opt/ion/iondb/bin/ion_plugin_migrate_permissions.py', i])
                except:
                    logger.exception("Failed to change permissions")


        logger.debug("Scanning %s for plugin folders", basedir)
        # only list files in the 'plugin' directory if they are actually folders
        folder_list = []
        for i in os.listdir(basedir):
            if i in ["scratch", "implementations", "archive"]:
                continue
            full_path = os.path.join(basedir, i)
            if not os.path.isdir(full_path):
                continue
            (plugin_script, islaunch) = self.find_pluginscript(full_path, i)
            if not plugin_script or not os.path.exists(plugin_script):
                logger.info("Non-plugin in plugin folder '%s': '%s', '%s'", basedir, i, plugin_script)
                continue
            # Candidate Plugin Found
            folder_list.append((i, full_path, plugin_script))

        ## Pre-scan plugins - one big celery task rather than many small ones
        pluginlist = [ (n,s,None) for n,p,s in folder_list ]
        infocache = {}
        try:
            infocache = iondb.plugins.tasks.scan_all_plugins(pluginlist)
        except Exception as exc:
            logger.exception("Failed to rescan plugin info in background task" + str(exc))

        count = 0
        for pname, full_path, plugin_script in folder_list:
            # Do early check to see if plugin script is valid
            # - cannot install without name and version.
            info = infocache.get(plugin_script)
            if not info:
                logger.error("Missing info for %s", plugin_script)
                continue

            if 'version' not in info:
                logger.error("Missing VERSION info for %s", plugin_script)
                # Cannot install versionless plugin
                continue

            # Quick skip of already installed plugins
            #if iondb.rundb.models.Plugin.objects.filter(pname=pname, version=info.get('version'), active=True).exists():
            #    continue

            # For now, install handles "reinstall" or "refresh" cases.
            try:
                (newplugin, updated) = self.install(pname, full_path, plugin_script, info)
            except ValueError:
                logger.exception("Plugin not installable due to error querying name and version '%s'", full_path)
            if updated:
                count += 1

        return count

    def install(self, pname, full_path, launch_script=None, info=None):
        """Install (or upgrade) a plugin given a src path.

        Safe to use for already installed plugins, with these cautions:
        * Will reactivate if previously marked inactive.
        * May refresh pluginconfig or upgrade to new version if found.

        TODO - handle embedded version number in path
        """

        # Cleanup, normalize and validate inputs
        pname = pname.strip()
        full_path = os.path.normpath(full_path)
        if not os.path.exists(full_path):
            logger.error("Path specified for install does not exist '%s'", full_path)

        ## Check Plugin Blacklist:
        if pname in ('scratch', 'implementations'):
            logger.error("Scratch and Implementations are reserved folders, and cannot be installed.")
            return None, False

        if not launch_script:
            (launch_script, isLaunch) = self.find_pluginscript(full_path, pname)

        logger.debug("Plugin Info: %s", info)
        if not info:
            # Worst case, should have been pre-fetched above
            logger.error("Need to rescan plugin info..")
            info = PluginManager.get_plugininfo(pname, launch_script, use_cache=False)
            logger.debug("Plugin Rescan Info: %s", info)
        if info is None:
            raise ValueError("No plugininfo for '%s' in '%s'" % (pname,launch_script))

        version = info.get('version',"0")
        majorBlock = info.get('major_block', False)

        # Only used if new plugin record is created
        packageName = getPackageName(full_path)
        plugin_defaults={
            'path':full_path,
            'date':datetime.datetime.now(),
            'active':True, # auto activate new versions
            'selected': True, # Auto enable new plugins
            'majorBlock': majorBlock,
            'description': info.get('docs', None),
            'userinputfields': info.get('config', None),
            'packageName' : packageName if not 'ion-plugins' in packageName else ''
        }

        logger.debug("Plugin Install/Upgrade checking for plugin: %s %s", pname, version)
        # needs_save is aka created. Tracks if anything changed.
        p, needs_save = iondb.rundb.models.Plugin.objects.get_or_create(name=pname, version=version, defaults=plugin_defaults)

        # update the plugin package name
        if plugin_defaults['packageName'] != p.packageName:
            p.packageName = plugin_defaults['packageName']
            p.save()

        if needs_save:
            # Newly created plugin - never before seen with this name and version
            logger.info("Installing New Plugin %s v%s at '%s'", pname, version, full_path)
            # Set pluginconfig.json if needed - only for new installs / version changes
            if self.set_pluginconfig(p):
                logger.info("Loaded new pluginconfig.json values for Plugin %s v%s at '%s'", pname, version, full_path)
        else:
            logger.debug("Existing plugin found: %s %s [%d]", p.name, p.version, p.pk)

        p.updateFromInfo(info)

        # Handle any upgrade behaviors - deactivating old versions, etc.
        (count, oldp) = self.upgrade_helper(p, current_path=full_path)

        # Preserve old selected/defaultSelected state
        if count and oldp:
            p.selected = oldp.selected
            p.defaultSelected = oldp.defaultSelected
            # Merge stock config with previous config, preserving user settings
            p.config.update(oldp.config)

        # Be careful with path handling -- FIXME
        if not p.path:
            # 1. path was empty - set path and reactivate
            # Reinstall previously removed and inactive plugin - files were restored
            logger.info("Reactivating previously uninstalled Plugin %s v%s at '%s'", pname, version, full_path)
            if self.enable(p, full_path):
                needs_save=True

        elif not os.path.exists(p.path):
            # 2. path was set, but folder is missing - replace
            # Prior path was invalid, but caller insists full_path exists
            needs_save = (p.path == full_path) # only if changed
            if needs_save: logger.info("Changing Plugin path value %s v%s to '%s' (was '%s')", pname, version, full_path, p.path)

            if os.path.exists(full_path):
                self.enable(p, full_path)

        elif p.path != full_path:
            # 3. path was set and exists
            # FIXME - for now, replace old plugin with new.
            # TODO - With multi-version install, ignore duplicate of same version
            logger.info("Found relocated Plugin %s v%s at '%s' (was '%s')", pname, version, full_path, p.path)

            # uninstall the plugin
            iondb.rundb.models.Plugin.Uninstall(p.id)

            # And restore with full_path
            self.enable(p, full_path)

        # If the path exists, it is an active. Inactive plugins are removed
        if not p.active and os.path.exists(p.path):
            logger.info("Reactivating plugin marked inactive but present on filesystem %s v%s at '%s'", pname, version, full_path)
            self.enable(p)

        if not p.script:
            p.script = os.path.basename(launch_script)
            if launch_script != os.path.join(p.path, p.script):
                logger.error("Launch Script is not in plugin folder: '%s' '%s'", p.path, launch_script)
            needs_save = True

        #create media symlinks if needed
        pluginMediaSrc =  os.path.join(p.path,"pluginMedia")
        if os.path.exists(pluginMediaSrc):
            pluginMediaDst = os.path.join("/results/pluginMedia",p.name)
            try:
                if os.path.lexists(pluginMediaDst):
                    os.unlink(pluginMediaDst)
                # note os.path.exists returns False for broken symlinks
                os.symlink(os.path.join(p.path,"pluginMedia"),
                        os.path.join("/results/pluginMedia",p.name))
            except OSError:
                logger.exception("Failed to create pluginMedia Symlink: %s", pluginMediaDst)
            # No need to set needs_save - no db data changed

        # Update majorBlock to current setting
        if p.majorBlock != majorBlock:
            p.majorBlock = majorBlock
            needs_save = True

        if needs_save:
            p.save()

        # Return true if anything changed. Always return plugin object
        return p, needs_save

    def upgrade_helper(self, plugin, current_path=None):
        # TODO: Rename this method to be _upgrade_helper
        """ Handle additional tasks during upgrade of plugin.
            Currently deactivates old versions of plugins to avoid version conflicts.
            Pass in the new plugin object - the one you want to keep.
        """
        # Some behavior here is just to disable old versions during upgrade.
        count = 0
        oldplugin = None

        # Get all plugins with the same name and different version
        for oldp in iondb.rundb.models.Plugin.objects.filter(name=plugin.name).exclude(version=plugin.version):
            # FIXME - multi-version support needs new behavior here
            if oldp.active or oldp.path:
                logger.info("Disabling old version of plugin %s v%s", oldp.name, oldp.version)
                count+=1
            else:
                continue

            # Plugins other than our version are disabled

            # Uninstall - Allows manual install of plugin previously installed via zeroinstall
            # Careful not to delete new path if installed in-place!!!
            current_path = current_path or plugin.path
            if oldp.path and oldp.path != current_path:
                # farm the uninstall off to the plugin daemon
                iondb.rundb.models.Plugin.Uninstall(oldp.id)
            else:
                oldp.path=''
            if oldp.active:
                oldp.active=False
            oldp.save()

            # Important! This is passed back to preserve autorun and selected settings
            oldplugin = oldp

        if count:
            logger.debug('Deactivated %d old versions while upgrading to %s v%s', count, plugin.name, plugin.version)

        return count, oldplugin

    # Consider moving disable and enable to plugin model
    def disable(self, plugin):
        """ Mark plugin as inactive.
        Does not remove any files. Caller must clear path and/or remove files if they might conflict with an active version

        Plugin Record is never deleted -foreign keys in PluginResults table.
        Left behind to preserve history of plugin execution on a run.

        Leave selected flags in prior state for possible reactivation.
        """
        logger.info("Disabling Plugin %s", plugin)
        if plugin.active:
            plugin.active = False
            plugin.save()
        return True

    def enable(self, plugin, path=None):
        """ Re-enables plugin
        Optional path value (as path is often cleared for inactive plugins
        Setting new path bumps plugin datestamp
        """
        if path:
            # Canonicalize before we compare or set
            path = os.path.normpath(path)
            if not os.path.exists(path):
                logger.error("Specified Plugin path does not exist '%s'", path)
                return False

            if plugin.path != path:
                plugin.path=path
                plugin.date=datetime.datetime.now()

        if not plugin.path:
            logger.error("Cannot enable plugin with no path information")
            return False

        if not plugin.pluginscript:
            logger.error("No plugin script found at path '%s'. Unable to enable plugin", plugin.path)
            return False

        plugin.active=True
        plugin.save()
        return True


pluginmanager = PluginManager()
