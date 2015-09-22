# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import threading
import os
import sys
import imp
import logging
import glob
import operator

from hashlib import md5

#import ion.plugin.base
#from ion.plugin.base import IonPluginBase

__all__=('cache','get_plugin')

_log = logging.getLogger(__name__)

class ModuleCache(object):
    # Borg pattern
    __shared_state = dict(
        ## keys of module store are the python modules for each plugin
        module_store = {},
        ## mapping of installed plugin names to python module for that plugin
        module_labels = {},

        ## mapping of plugin names to plugin classes
        module_cache = {},

        # Log of errors during import/loading
        module_errors = {},

        handled = {},
        loaded = False,
        write_lock = threading.RLock(),
        installed_modules = [],

        _get_plugin_cache = {}
    )

    def __init__(self):
        self.__dict__ = self.__shared_state

    def _populate(self):
        """
        Fill in all the cache information. This method is threadsafe, in the
        sense that every caller will see the same state upon return, and if the
        cache is already initialised, it does no work.
        """
        if self.loaded:
            return
        self.write_lock.acquire()
        try:
            if self.loaded:
                return
            installed_modules = self.get_installed_modules()
            if not installed_modules:
                #_log.info("No plugin modules installed")
                return ## Do not set loaded with empty installed_modules
            _log.debug("Scanning for installed modules")
            for (module_name, module_path) in installed_modules:
                if module_name in self.handled:
                    continue
                _log.debug("Searching for plugin module: '%s' from '%s'", module_name, module_path)
                if '.py' == module_path[-3:]:
                    self.load_module(module_name, module_path)
                else:
                    self.load_compat_module(module_name, module_path)

            self.loaded = True
        finally:
            self.write_lock.release()

    def set_installed_modules(self, modules):
        self.write_lock.acquire()
        try:
            self.installed_modules = modules
            self.loaded = False
        finally:
            self.write_lock.release()

    def get_installed_modules(self):
        #return models.Plugin.objects.filter(active=True)$
        return self.installed_modules or []

    def load_compat_module(self, module_name, module_path):
        import ion.plugin.launchcompat
        try:
            mod = ion.plugin.launchcompat.get_launch_class(module_name, module_path)
        except Exception as e:
            _log.exception("Failed to import legacy plugin wrapper '%s':'%s'", module_name, module_path)
            self.module_errors[module_name] = e
            return None
        if not mod:
            return None
        if mod not in self.module_store:
            self.module_store[mod] = len(self.module_store)
            self.module_labels[module_name] = mod
        return mod


    def load_module(self, module_name, module_path):
        """
        Loads the module with the provided fully qualified name.
        """
        self.handled[module_name] = None

        #for path in glob.glob(os.path.join(module_path,'[!_]*.py')): # list .py files not starting with '_'
        #    name, ext = splitext(basename(path))
        #    _log.debug("%s -- %s", name, path)
        #    #modules[name] = imp.load_source(name, path)
        module_dir, pyfile = os.path.split(module_path)
        name, ext = os.path.splitext(pyfile)
        if ext != '.py':
            return self.load_compat_module(module_name, module_path)

        try:
            _log.debug("Loading module '%s' from '%s'", module_name, module_path)
            sys.path.append(module_dir)
            mod = imp.load_source(md5(module_path).hexdigest(), module_path)
            _log.debug(mod)
            # clean up sys.path
            sys.path.remove(module_dir)
        except ImportError as e:
            _log.debug("Import Error '%s'", module_name, exc_info=True)
            self.module_errors[module_name] = e
            return None
        except Exception as e:
            _log.debug("Failed to import '%s'", module_name, exc_info=True)
            self.module_errors[module_name] = e
            return None

        if not mod:
            return None

        if mod not in self.module_store:
            self.module_store[mod] = len(self.module_store)
            self.module_labels[module_name] = mod

        return mod

    # Called by plugin metaclass to self-register on import
    def register_plugin(self, module_name, cls):
        _log.debug("Registering plugin class: %s -> %s", module_name, cls)
        self.module_cache[module_name] = cls
        pluginname = getattr(cls, 'name', cls.__name__)
        if pluginname != module_name:
            self.module_cache[pluginname] = cls
        self._get_plugin_cache.clear()
        return cls

    # Returns plugin class, which can be instantiated to interrogate or run plugin
    def get_plugin(self, name):
        cache_key = (name,)
        try:
            return self._get_plugin_cache[cache_key]
        except KeyError:
            pass
        self._populate()

        p = None
        # module_store vs module_cache, if named module is loaded, return class instance
        if name in self.module_store:
            p = self.module_cache[name]
        elif name in self.module_cache:
            p = self.module_cache[name]

        self._get_plugin_cache[cache_key] = p
        return p

    def get_plugins(self):
        return self.module_labels.keys()

    def get_module(self, name, seed_cache=True, only_installed=True):
        if seed_cache:
            self._populate()
        if only_installed and name not in self.module_cache:
            return None
        return self.module_labels.get(name)

    def get_class(self, name, seed_cache=True, only_installed=True):
        instance = self.get_instance(name, seed_cache, only_installed)
        cls = instance.getattr(name)

    def get_errors(self):
        "Returns the map of known problems with the INSTALLED_PLUGINS."
        self._populate()
        return self.module_errors

cache = ModuleCache()
get_plugin = cache.get_plugin
