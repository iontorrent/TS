# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

# Module to manage plugins

import os
import shutil
import re
import json

from django.conf import settings
#from celery.task import task

import iondb.rundb.models
#from iondb.rundb.zeroinstallHelper import downloadZeroFeed

from distutils.version import LooseVersion
import zeroinstall.zerostore
import iondb.rundb.tasks  ## circular import

import json
import datetime
import logging
logger = logging.getLogger(__name__)

# Helper function
def pluginVersionGreater(a, b):
    return(LooseVersion(a.version) > LooseVersion(b.version))

class PluginManager(object):
    """Class for managing plugin installation and versioning activities.

    Encapsulates install, uninstall, and upgrade.

    Also implements rescan, inactivate_missing (aka purge_plugins), and search_for_plugins,
    which handle discovery of manually installed plugins.

    TODO:
    * Implement zeroinstall behaviors
    * Allow multiple version entries
        - Legacy - one plugin entry per name, installing different version replaces
          **Name is unique identifier**
        - Transitional - one entry per name and version, old versions always inactivated
          **only one active version**
        - Multi-Version - one entry per name and version, all paths and
          messaging uniquely identify plugins

    """
    def __init__(self, gc=None):
        if not gc:
            gc = iondb.rundb.models.GlobalConfig.objects.all().order_by('pk')[0]
        self.default_plugin_script = gc.default_plugin_script
        self.pluginroot = settings.PLUGIN_PATH or os.path.join("/results", gc.plugin_folder)


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
            plugins = iondb.rundb.models.Plugin.objects.filter(active=True).exclude(path='')

        # for each record, test for corresponding folder
        count = 0
        scanned = 0
        for plugin in plugins:
            # if specified folder does not exist
            if plugin.path == '' or not os.path.isdir(plugin.path):
                if self.disable(plugin):
                    plugin.save()
                    count += 1
                # No need to uninstall - the path is already missing
            scanned += 1
        logger.debug("Scanned %d, removed %d", scanned, count)
        return count

    def get_version(self, pluginpath, pluginscript=None):
        """ Parses version from script file. Does not require plugin instance, just path """
        if not pluginscript:
            pluginscript = os.path.join(pluginpath, self.default_plugin_script)
        if not os.path.exists(pluginscript):
            logger.error("Plugin path is missing launch script '%s'", pluginscript)
            return "0"

        # Regex to capture version strings from launch.sh
        # Flanking Quotes optional, but help delimit
        # Leading values ignored, usually '#VERSION' or '# VERSION'
        # Must be all-caps VERSION
        # Digits, dots, letters, hyphen, underscore (1.0.2-beta1_rc2)
        VERSION=re.compile(r'VERSION\s*=\s*\"?([\d\.\w\-\_]+)\"?')
        try:
            with open(pluginscript, 'r') as f:
                for line in f:
                    m = VERSION.search(line)
                    if m:
                        v = m.group(1)
                        # Validate and canonicalize version string,
                        # according to distutils.version.LooseVersion semantics
                        try:
                            v = LooseVersion(v)
                        except ValueError:
                            logger.warning("Version in file does not conform to LooseVersion rules: ", v)
                        return str(v)
        except:
            logger.exception("Failed to parse VERSION from '%s'", pluginscript)
        return "0"

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

        logger.debug("Scanning %s for plugin folders", basedir)
        # only list files in the 'plugin' directory if they are actually folders
        folder_list = []
        for i in os.listdir(basedir):
            if i in ["scratch", "implementations"]:
                continue
            full_path = os.path.join(basedir, i)
            if not os.path.isdir(full_path):
                continue
            launch_script = os.path.join(full_path, self.default_plugin_script)
            if not os.path.exists(launch_script):
                logger.info("Non-plugin (no launch.sh) in plugin folder '%s': '%s'", basedir, i)
                continue
            # Candidate Plugin Found
            folder_list.append((i, full_path, launch_script))

        count = 0
        for pname, full_path, launch_script in folder_list:
            # For now, install handles "reinstall" or "refresh" cases.
            (newplugin, updated) = self.install(pname, full_path)
            if updated:
                count += 1
        return count


    def install(self, pname, full_path):
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

        version = self.get_version(full_path)

        # Only used if new plugin record is created
        plugin_defaults={
            'path':full_path,
            'date':datetime.datetime.now(),
            'active':True, # auto activate new versions
            'selected': True, # Auto enable new plugins
        }

        # needs_save is aka created. Tracks if anything changed.
        (p, needs_save) = iondb.rundb.models.Plugin.objects.get_or_create(name=pname,
                                                              version=version, # NB: Exact Version matches only
                                                              # default values if created
                                                              defaults=plugin_defaults
                                                             )
        if needs_save:
            # Newly created plugin - never before seen with this name and version
            logger.info("Installing New Plugin %s v%s at '%s'", pname, version, full_path)
            # Set pluginconfig.json if needed - only for new installs / version changes
            if self.set_pluginconfig(p):
                logger.info("Loaded new pluginconfig.json values for Plugin %s v%s at '%s'", pname, version, full_path)

        # Handle any upgrade behaviors - deactivating old versions, etc.
        self.upgrade_helper(p)

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
            self.uninstall(plugin)
            # And restore with full_path
            self.enable(p, full_path)

        # If the path exists, it is an active. Inactive plugins are removed
        if not p.active and os.path.exists(p.path):
            logger.info("Reactivating plugin marked inactive but present on filesystem %s v%s at '%s'", pname, version, full_path)
            basedi  = settings.PLUGIN_PATH or os.path.join("/results", self.pluginroot)
            self.enable(p)

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


        if needs_save:
            p.save()
        # Return true if anything changed. Always return plugin object
        return (p, needs_save)

    def upgrade_helper(self, plugin):
        """ Handle additional tasks during upgrade of plugin.
            Currently deactivates old versions of plugins to avoid version conflicts.
            Pass in the new plugin object - the one you want to keep.
        """
        # Some behavior here is just to disable old versions during upgrade.
        count = 0
        # Upgrade - old plugins must be disabled until multiversion support is implemented

        # Get all plugins with the same name and different version
        for oldp in iondb.rundb.models.Plugin.objects.filter(name=plugin.name).exclude(version=plugin.version):
            # FIXME - multi-version support needs new behavior here
            if oldp.active or oldp.path:
                logger.info("Disabling old version of plugin %s v%s", oldp.name, oldp.version)
                count+=1

            # Plugins other than our version are disabled
            if oldp.active:
                self.disable(oldp)

            # Uninstall - Allows manual install of plugin previously installed via zeroinstall
            if oldp.path:
                # Careful not to delete new path if installed in-place!!!
                if oldp.path != plugin.path:
                    self.uninstall(oldp)
                    # uninstall clears path value
                else:
                    oldp.path=''
                    oldp.save()

        if count:
            logger.debug('Deactivated %d old versions while upgrading to %s v%s', count, plugin.name, plugin.version)

        return count

    # Consider moving disable and enable to plugin model

    def disable(self, plugin):
        """ Mark plugin as inactive.
        Does not remove any files. Caller must clear path and/or remove files if they might conflict with an active version

        Plugin Record is never deleted -foreign keys in PluginResults table.
        Left behind to preserve history of plugin execution on a run.

        Leaves autorun and selected flags in prior state for possible reactivation.
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

        plugin.active=True
        plugin.save()
        return True

    def is_plugin_store(self, plugin_path):
        norm_plugin_path=os.path.normpath(plugin_path)

        # Helper for verifying path
        def subsumes(path, maybe_child):
            """ Test if maybe_child is contained in path """
            if path == maybe_child:
                return True
            if maybe_child.find(path + '/') == 0:
                return True
            return False

        # Ensure plugin path is in a known plugin directory.
        known_plugin_paths = [ self.pluginroot, settings.PLUGIN_PATH, ]
        known_plugin_paths.extend(x.dir for x in zeroinstall.zerostore.Stores().stores)

        for known_path in known_plugin_paths:
            if subsumes(os.path.normpath(known_path), norm_plugin_path):
                return True
        else:
            return False


    def uninstall(self, plugin):
        """ Can only uninstall plugins with valid plugin.path """
        if not plugin.path:
            # API / programmer error
            logger.debug("Attempting to uninstall already uninstalled plugin: %s v%s", plugin.name, plugin.version)
            return False

        plugin_path = plugin.path

        # Clear database value early for concurrency issues
        plugin.path = '' ## None?
        plugin.active=False # Ensure plugin is inactive - it no longer has any files
        plugin.save()

        # Warning - removing symlink and removing tree may break old reports
        # In current usage, only used by instance.html, not reports.
        # If we remove the path, we must remove the symlink too,
        # or we're left with a broken symlink.
        oldPluginMedia = os.path.join('/results/pluginMedia', plugin.name)
        if os.path.lexists(oldPluginMedia):
            try:
                os.path.unlink(oldPluginMedia)
            except OSError:
                pass

        if not os.path.exists(plugin_path):
            # Files deleted elsewhere, just residual path value in db.
            logger.debug("Plugin files already removed for %s v%s. Missing path '%s'", plugin.name, plugin.version, plugin_path)
            return True ## success - it is gone, through no fault of our own

        if not self.is_plugin_store(plugin_path):
            logger.error("Attempted to delete plugin folder in unknown plugin store: '%s'", plugin_path)
            return False

        logger.warn("Purging files '%s' for plugin '%s v%s'", plugin_path, plugin.name, plugin.version)

        parent_path = os.path.normpath(os.path.dirname(plugin_path))
        # Safety checks - MUST avoid deleting parent folder unless it really is zeroinstall
        if plugin.url \
           and (parent_path not in [ settings.PLUGIN_PATH, self.pluginroot]) \
           and os.path.basename(parent_path).startswith("sha256=") \
           and os.path.exists(os.path.join(parent_path, '.manifest')):
            # Looks like zeroinstall - delete one folder up as well
            logger.debug("ZeroInstall Plugin: Removing store folder - ", parent_path)
            plugin_path = parent_path

        try:
            shutil.rmtree(plugin_path)
        except OSError:
            # Probably permission error. See if the celery async task can remove it.
            logger.debug("Deferring delete to celery task: ", plugin_path)
            ret = iondb.rundb.tasks.delete_that_folder.delay(plugin_path, "Plugin uninstallation")
            ret.get() # waits for result

        return True

