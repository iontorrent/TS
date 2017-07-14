# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import logging
import json

from ion.plugin.loader import cache
from ion.plugin.constants import *
import ion.plugin.utils

from ion.plugin.runtime import IonPluginRuntime

LOG = logging.getLogger(__name__)


def lazyprop(fn, name=None):
    if not name:
        name = fn.__name__
    attr_name = '_lazy_' + name

    @property
    def _lazyprop(x):
        if not hasattr(x, attr_name):
            setattr(x, attr_name, fn(x))
        return getattr(x, attr_name)
    return _lazyprop


def lazyclassprop(fn, name=None):
    if not name:
        name = fn.__name__
    attr_name = '_lazy_' + name

    @property
    def _lazyprop(x):
        if not hasattr(x, attr_name):
            setattr(x, attr_name, fn())
        return getattr(x, attr_name)
    return _lazyprop


class IonPluginMeta(type):

    """ Metaclass for Module Classes """
    def __new__(cls, name, bases, attrs):
        super_new = super(IonPluginMeta, cls).__new__
        parents = [b for b in bases if isinstance(b, IonPluginMeta)]
        if not parents:
            # If this isn't a subclass of Modules/Plugins, don't do anything special.
            return super_new(cls, name, bases, attrs)

        # Nothing special for these abstract base classes
        if name == 'IonPlugin' or name == 'IonLaunchPlugin':
            return super_new(cls, name, bases, attrs)

        # Create the class.
        module = attrs.pop('__module__', None)
        new_class = super_new(cls, name, bases, {'__module__': module})

        # Bail out early if we have already created this class.
        add_to_store = attrs.get('add_to_store', True)
        m = cache.get_module(name, seed_cache=False, only_installed=False) if add_to_store else None
        if m is not None:
            return m

        # Enhance the class attrs
        attr = attrs.pop('', None)

        # These are class level attributes. (so we can inspect without instantiating)
        attrs.setdefault('name', name)

        # Add all attributes to the class.
        for obj_name, obj in attrs.items():
            setattr(new_class, obj_name, obj)

        # module_label = sys.modules[new_class.__module__].__name__

        new_class._prepare()
        cache.register_plugin(name, new_class)

        # return cache.get_instance(name)
        return new_class

    def _prepare(cls):
        # Create base methods
        # if hasattr(cls, 'wrapable_func'):
        #    cls.wrapable_func = update_wrapper(curry(wrapper_func,), cls.wrapable_func)

        # Set python special attributes
        py_attr = ('author', 'version')
        for name in py_attr:
            a = getattr(cls, name, None)
            # CAUTION - evaluates function at class define time
            # NOTE - all callables MUST be evaluated here - class.get does not evaluate properties
            if callable(a):
                # MUST BE CLASS METHOD
                # v = lazyprop(a,name)
                v = a()
            elif isinstance(a, property):
                # v = lazyprop(lambda: a.__get__(cls), name)
                v = a.__get__(cls)
            else:
                v = a
            if v is None:
                v = "(Unknown)"
            # Force string - python doesn't like unicode in the __name__ field
            v = str(v)
            # All plugin classes get these attrs set on class
            setattr(cls, name, v)
            setattr(cls, '__'+name+'__', v)

        # Set docstring - cls.__doc__
        docstr = getattr(cls, '__doc__')
        if docstr is None:
            import warnings
            warnings.warn("NO DOCSTRING: Please update python class documentation to provide a short description and documentation for your plugin.")
            # docstr = "[ Please update python class documentation to provide a short description and documentation for your plugin. ]"
            docstr = ""
            setattr(cls, '__doc__', docstr)

        # Upgrade "special" attributes - these can be callables, so attach property decorator
        class_attr = ('major_block', 'runtypes', 'features', 'runlevels', 'depends', 'requires', 'provides')
        for name in class_attr:
            if not hasattr(cls, name):
                continue
            a = getattr(cls, name)
            if callable(a):
                # Auto apply lazy property decorator
                setattr(cls, name, lazyclassprop(a, name))

        special_attr = (
            'output', 'results'
        )
        for name in special_attr:
            if not hasattr(cls, name):
                continue
            a = getattr(cls, name)
            if callable(a):
                # Auto apply lazy property decorator
                setattr(cls, name, lazyprop(a, name))

        return


class IonPluginBase(object):

    """
    Base Class for Plugins. Applies Metaclass
    Here so we can have IonModule which extends same infrastructure, but has different interface than plugin.
    """
    __metaclass__ = IonPluginMeta

    def __init__(self):
        pass


class IonPlugin(IonPluginBase, IonPluginRuntime):

    """ Base class for all Plugin Components """

    # Class attributes
    name = None
    version = None
    runtypes = []
    features = []
    runlevels = []
    depends = []
    major_block = False
    requires = ['BAM', ]
    output = {}
    results = {}

    def __init__(self, *args, **kwargs):
        self.context = {}
        self.blockcount = 0
        self.exit_status = -1

        # Populated from ORM
        self.plugin = None
        self.analysis = None  # result object
        self.pluginresult = None

        self.data = {}

        return super(IonPlugin, self).__init__(*args, **kwargs)

    # def __getattribute__(self, name):
    #    attr = super(IonPlugin, self).__getattribute__(name)
    #    if callable(attr):
    #        return attr()
    #    else:
    #        return attr

    # Introspection methods - FIXME - wrap functions instead
    def _get_runtypes(self):
        rt = []
        if hasattr(self, 'runtypes'):
            if callable(self.runtypes):
                rt = self.runtypes()
            else:
                rt = self.runtypes

        # If plugin reported value is empty,
        if not rt:
            # PGM runs and Thumbnails
            rt = [ion.plugin.constants.RunType.FULLCHIP,
                  ion.plugin.constants.RunType.THUMB]
            # Infer supported RunTypes from implemented methods
            if hasattr(self, 'block'):
                rt.append(ion.plugin.constants.RunType.BLOCK)
            if hasattr(self, 'thumbnail'):
                rt.append(ion.plugin.constants.RunType.THUMB)

        # vars(ion.plugin.constants.RunType).keys() ## all RunTypes
        return rt

    @classmethod
    def _metadata(cls):
        return {'name': cls.__name__,
                'version': getattr(cls, 'version', None),
                'description': cls.__doc__,
                }

    # Wrap launch class with some initial checks, dry_run, etc.
    def launch_wrapper(self, data=None, dry_run=True):
        if dry_run:
            self.log.info("Plugin dry_run: '%s' v%s", self.name, self.version)
            return True

        self.data = data or {}

        if not self.pre_launch():
            self.log.info("Plugin declined to run in pre_launch: '%s' v%s", self.name, self.version)
            return None

        self.log.info("Plugin Launch: '%s' v%s", self.name, self.version)
        status = self.launch()

        if status:
            self.generate_output()
        else:
            self.log.error("Plugin execution failed")

        if not self.exit_status:
            self.exit_status = bool(not status)

        self.log.info("Plugin complete: '%s' v%si -- exit_status = %d",
                      self.name, self.version, self.exit_status)

        return status

    def _write_json(self, fname, content):
        if not content:
            return
        if not fname:
            return
        try:
            with open(fname, 'w') as fh:
                json.dump(content, fh, indent=2)
        except (OSError, IOError, ValueError):
            self.log.exception("Failed to write '%s'", fname)
        # self.log.debug("%s: '%s'", fname, json.dumps(content, indent=2))
        return

    def generate_output(self):
        self.log.info("Generating Output Files")
        self._write_json('output.json', self.output)
        self._write_json('results.json', self.results)
        return

    # These should be overridden by plugin, if appropriate
    def pre_block(self, block):
        """ Return value indicates if block will be called """
        return hasattr(self, 'block')

    def post_block(self, block):
        """ Mark block as processed """

        return True

    def pre_launch(self, data=None):
        """ Code invoked prior to submission to queue.
            Return value determines if launch will be invoked.
        """
        return True

    def post_launch(self):
        pass

    # These must be implemented by plugins
    def launch(self, data=None):
        raise NotImplemented()

    def getUserInput(self):
        return None

    # functions for plugins that want to show barcode table UI
    def barcodetable_columns(self):
        # plugin needs to implement this function to use barcode table UI feature
        return []

    def barcodetable_data(self, data, planconfig={}, globalconfig={}):
        # optional function, if not implemented table will initialize with default (input) data
        #   data - same as barcodes.json, can be passed as is or modified or overwritten by the plugin
        #   planconfig - plugin configuration from Planning (plan.html), if any
        #   globalconfig - plugin global configuration (config.html), if any
        return data

