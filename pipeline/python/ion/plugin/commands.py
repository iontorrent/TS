# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import argparse
import atexit
import json
import types
import inspect

import ion.plugin.base

""" Plugins can append PluginCLI to the end of the Plugin Definition
    to allow running directly from the command line for testing/development purposes

    if __name__ == "__main__":
        PluginCLI(PluginClassName)

    """

import logging
LOG = logging.getLogger(__name__)

__all__ = ('PluginCLI', 'cli',)


# atexit handler
def plugin_shutdown(runningcli):
    if runningcli.ret is not None:
        sys.exit(runningcli.ret)


class PluginCLI(object):
    EXIT_SUCCESS = 0
    EXIT_ERROR = 1

    def __init__(self, plugin=None):
        self.instance = None
        if not plugin:
            # Guess module from caller
            frm = inspect.stack()[1]
            pluginmod = inspect.getmodule(frm[0])
            (name, suffix, mode, module_type) = inspect.getmoduleinfo(pluginmod.__file__)
            # name = inspect.getmodulename(pluginmod)
            self.cls = getattr(pluginmod, name)
        elif isinstance(plugin, types.TypeType):
            # Got class - recommended usage
            self.cls = plugin
        elif isinstance(plugin, types.ModuleType):
            # Got Module (returned from import. Find class of same name
            # cache.load_module(module.__name__)
            # FIXME - iterate through all module attrs to find the class
            for name, obj in inspect.getmembers(plugin):
                if hasattr(obj, "__bases__") and ion.plugin.base.IonPluginBase in obj.__bases__:
                    self.cls = obj
                    break
            else:
                LOG.error("Unable to find class in module '%s' which implements IonPlugin", plugin.__name__)
                self.cls = plugin
        elif isinstance(plugin, ion.plugin.base.IonPluginBase):
            # Got plugin instance
            self.cls = plugin.__class__
            self.instance = plugin
        elif isinstance(plugin, basestring):
            from ion.plugin.loader import cache
            self.cls = ion.plugin.loader.cache.load_module(plugin)
        else:
            LOG.fatal("Unable to recognize %s as plugin definition", plugin)
            raise ValueError("PluginCLI must be called with a class instance or string class name")
        self.ret = None

        status = self.run()
        atexit.register(plugin_shutdown, self)

    def run(self):
        self.parse_command_line()

        # Instantiate class with proper environment
        if not self.instance:
            self.instance = self.cls()
        plugin = self.instance

        if self.options.inspect:
            from ion.plugin.info import PluginInfo
            print PluginInfo.from_instance(plugin)
            return self.EXIT_SUCCESS

        if self.options.bctable_columns:
            from ion.plugin.barcodetable_columns import available_columns
            print json.dumps(available_columns(), indent=1)
            return

        if self.options.runmode == "launch":
            return plugin.launch_wrapper(dry_run=self.options.dry_run)

        if self.options.runmode == "block":
            if not self.options.block:
                LOG.fatal("Block runmode requires --block identifier")
                return self.EXIT_ERROR
            return plugin.block(self.options.block)


    def parse_command_line(self):
        version = getattr(self.cls, '__version__', getattr(self.cls, 'version', "(Unknown)"))
        docstring = inspect.getdoc(self.cls) or ''
        parser = argparse.ArgumentParser(description='Ion Plugin Command Line Interface\n' + docstring)
        parser.add_argument('--version', action='version', version='Ion Torrent Plugin - %(prog)s v' + self.cls.__version__)
        parser.add_argument('-v', '--verbose', action='count', default=0)

        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--inspect', '--info', action='store_true')
        parser.add_argument('--runmode', default="launch")
        parser.add_argument('--block', default=None)
        parser.add_argument('--bctable-columns', action='store_true')
        self.options = parser.parse_args()

        log_lvl = logging.ERROR
        if self.options.verbose: log_lvl = logging.INFO
        if self.options.verbose > 1: log_lvl = logging.DEBUG
        logging.basicConfig(level=log_lvl)

        LOG.debug("Called with: %s", self.options)
        return

cli = PluginCLI
