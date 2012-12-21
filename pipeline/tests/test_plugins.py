#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from ion.plugin.loader import cache
from ion.plugin.base import IonPlugin
from ion.plugin.info import PluginInfo

import os
import unittest
import logging
import sys

log = logging.getLogger(__name__)
#log.setLevel(logging.DEBUG)

sys.path.insert(0,os.path.join(os.path.dirname(__file__), '../python'))

def check_plugin(cls):
    assert(cls.name)
    #assert(cls.__name__ ==  cls.name)
    #assert(cls.__version__)
    #assert(cls.version)

    p = cls()
    #assertIsInstance(p, IonPlugin, msg='Plugin not instance of IonPlugin')
    assert isinstance(p, IonPlugin)
    log.info("Plugin Instance: %s", p)
    j = PluginInfo.from_instance(p)
    assert j

def run_plugin(cls):
    p = cls()

    ret =p.launch_wrapper(dry_run=True)
    if ret is not None:
        assert ret

    ret = p.launch_wrapper()
    if ret is None:
        assert ret is None
    else:
        assert ret

def setUpModule():
    plugins = []
    path = os.path.join(os.path.dirname(__file__),'plugins')
    path = os.path.normpath(os.path.abspath(path))
    assert os.path.exists(path)

    for (dirpath, dirnames, filenames) in os.walk(path):
        if '.svn' in dirpath: continue
        for name in dirnames:
            if name == '.svn': continue
            launchsh = os.path.join(dirpath, name, 'launch.sh')
            if os.path.exists(launchsh):
                log.info("Plugin [launch]: '%s'",launchsh)
                plugins.append( (name, launchsh) )
                continue
            # Find all instances of PluginName/PluginName.py
            plugindef = os.path.join(dirpath, name, name + '.py')
            if os.path.exists(plugindef):
                log.info("Plugin [class]: '%s'",plugindef)
                #sys.path.append(os.path.join(dirpath, name))
                plugins.append( (name, plugindef) )
                continue
            log.debug("Path (No Plugins): '%s/%s'",dirpath, name)

    assert plugins
    cache.set_installed_modules(plugins)

    for (name, script) in plugins:
        ret = cache.load_module(name, script)
        assert ret
    return True

def test_all_plugins():
    count = 0
    for m in cache.get_plugins():
        cls = cache.get_plugin(m)
        if not cls:
            continue
        log.info("Plugin Name: %s", m)
        count += 1
        yield check_plugin, cls
        yield run_plugin, cls
    assert count

def test_plugin_errors():
    count = 0
    for (name, err) in cache.get_errors().iteritems():
        count += 1
        if err.__class__.__name__ == 'ImportError':
            log.warn("Import Error [%s]: %s", name, err)
        else:
            log.warn("Syntax Error [%s]: %s: %s", name, err.__class__.__name__, err)
        yield assertTrue(False)
    # yield failure
    assert count==0


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(name)s@%(lineno)d: %(message)s')
    setUpModule()
    test_plugin_errors()
    test_all_plugins()



