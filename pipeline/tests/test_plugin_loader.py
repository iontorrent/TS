#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import os,sys
import logging
import json

from ion.plugin.loader import cache
from ion.plugin.info import PluginInfo

sys.path.insert(0,os.path.join(os.path.dirname(__file__), '../python'))

logger = logging.getLogger(__name__)
def test_pluginscan():
    plugins = []
    d = os.path.dirname(__file__)
    for path in [os.path.join(d,'../../plugin'), os.path.join(d,'../../rndplugins') ]:
        path = os.path.normpath(os.path.abspath(path))
        if not os.path.exists(path):
            logger.debug("skipping '%s'", path)
            continue
        for (dirpath, dirnames, filenames) in os.walk(path):
            for name in dirnames:
                launchsh = os.path.join(dirpath, name, 'launch.sh')
                if os.path.exists(launchsh):
                    plugins.append( (name, launchsh) )
                    continue
                # Find all instances of PluginName/PluginName.py
                plugindef = os.path.join(dirpath, name, name + '.py')
                if os.path.exists(plugindef):
                    #sys.path.append(os.path.join(dirpath, name))
                    plugins.append( (name, plugindef) )

    logger.debug("Plugins: '%s'", plugins)

    cache.set_installed_modules(plugins)

    for (name, err) in cache.get_errors().iteritems():
        if err.__class__.__name__ == 'ImportError':
            logger.warn("Import Error [%s]: %s", name, err)
        else:
            logger.warn("Syntax Error [%s]: %s: %s", name, err.__class__.__name__, err)

    for m in cache.get_plugins():
        cls = cache.get_plugin(m)
        if not cls:
            continue
        logger.info("Plugin Name: %s", m)
        yield check_plugin, cls


def check_plugin(cls):
    p = cls()
    if not p:
        return
    logger.info("Plugin Instance: %s", p)
    info = PluginInfo.from_instance(p)
    assert info
    assert info.version
    infojson = str(info)
    assert infojson
    infodict = json.loads(infojson)
    assert infodict['version']

    logger.debug("Plugin info:\n%s ", infojson)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(name)s@%(lineno)d: %(message)s')
    test_pluginscan()

