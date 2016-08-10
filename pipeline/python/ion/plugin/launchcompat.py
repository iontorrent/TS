# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved


import os
import subprocess
import re, tokenize, keyword
from hashlib import md5
import json
import logging

from distutils.version import LooseVersion

from ion.plugin.base import IonPlugin
from ion.plugin.constants import *

logger = logging.getLogger(__name__)


class IonLaunchPlugin(IonPlugin):

    """ Compatability with pre-3.0 launch.sh plugins
        Allows new setup and configuration definitions, but calls launch.sh
    """

    # Internal state - these are class attributes, as script is read
    # to populate VERSION and AUTORUNDISABLE in the class
    _content = {}
    _pluginsettings = {}
    _launchsh = None

    # Defaults for Launch Plugins
    # Can be overridden with pluginsettings.json file
    _runtypes = [RunType.FULLCHIP, RunType.THUMB]
    _features = []
    _runlevels = []
    _depends = []

    @classmethod
    def getContent(cls):
        if not cls._content:
            try:
                with open(cls._launchsh, 'r') as f:
                    cls._content = f.readlines()
            except:
                logger.error("Unable to read launch.sh script: '%s'", cls._launchsh)
        return cls._content

    @classmethod
    def pluginsettings(cls):
        if cls._pluginsettings:
            return cls._pluginsettings

        pluginsettingsjson = os.path.join(os.path.dirname(cls._launchsh), 'pluginsettings.json')
        if not os.path.exists(pluginsettingsjson):
            return cls._pluginsettings
        try:
            with open(pluginsettingsjson, 'r') as f:
                cls._pluginsettings = json.load(f)
        except:
            logger.error("Unable to read pluginsettings.json: '%s'", pluginsettingsjson)

        return cls._pluginsettings

    @classmethod
    def version(cls):
        # Regex to capture version strings from launch.sh
        # Flanking Quotes optional, but help delimit
        # Leading values ignored, usually '#VERSION' or '# VERSION'
        # Must be all-caps VERSION
        # Digits, dots, letters, hyphen, underscore (1.0.2-beta1_rc2)
        VERSION = re.compile(r'VERSION\s*=\s*\"?\'?([\d\.\w\-\_]+)\"?\'?')
        for line in cls.getContent():
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
        else:
            logger.warning("Plugin launch script does not define VERSION '%s'", cls._launchsh)
        return "0"

    @classmethod
    def major_block(cls):
        """ if the string AUTORUNDISABLE is in the lunch script
        don't allow the autorun settings to be changed on the config tab
        """
        for line in cls.getContent():
            if line.startswith("#MAJORBLOCK") or line.startswith("#MAJOR_BLOCK"):
                return True
        return False

    @classmethod
    def runtypes(cls):
        pluginsettings = cls.pluginsettings()
        if not pluginsettings:
            return cls._runtypes

        ret = []
        for k in pluginsettings.get('runtype', pluginsettings.get('runtypes', [])):
            c = k  # lookupEnum(RunType, k)
            if c:
                ret.append(c)
        return ret

    @classmethod
    def features(cls):
        pluginsettings = cls.pluginsettings()
        if not pluginsettings:
            return cls._features

        ret = []
        for k in pluginsettings.get('feature', pluginsettings.get('features', [])):
            c = k  # lookupEnum(Feature, k)
            if c:
                ret.append(c)
        return ret

    @classmethod
    def runlevels(cls):
        pluginsettings = cls.pluginsettings()
        if not pluginsettings:
            return cls._runlevels

        ret = []
        for k in pluginsettings.get('runlevel', pluginsettings.get('runlevels', [])):
            c = k  # lookupEnum(RunLevel, k)
            if c:
                ret.append(c)
        return ret

    @classmethod
    def depends(cls):
        pluginsettings = cls.pluginsettings()
        if not pluginsettings:
            return cls._depends

        ret = []
        for k in pluginsettings.get('depend', pluginsettings.get('depends', [])):
            c = k  # lookupEnum(Feature, k)
            if c:
                ret.append(c)
        return ret

    def launch(self):
        lenv = {}
        lenv.update(os.environ)
        # TODO setup lenv with plugin_functions variables
        # For now local ionPluginShell --local will call plugin_functions
        if not os.path.exists(self._launchsh):
            self.log.error("Unable to find launch.sh at '%s'", self._launchsh)
            return False
        outputpath = self.data['analysis_dir']
        ret = subprocess.call(["ionPluginShell", self._launchsh, "-j", "startplugin.json"], env=lenv, cwd=outputpath)
        self.exit_status = ret
        return (ret == 0)


def get_launch_class(pluginname, launch_script, add_to_store=True):
    # Build our name from user provided pluginname
    # Our class name
    if pluginname is not None:
        clsname = str(pluginname)  # de-unicodify
        m = re.match('^'+tokenize.Name+'$', clsname)
        if not m:
            logger.warn("Plugin Name: '%s' is not a valid python identifier", pluginname)
            clsname = None
        if keyword.iskeyword(pluginname):
            logger.warn("Plugin Name: '%s' is a python reserved keyword", pluginname)
            clsname = None
    else:
        clsname = None

    if not clsname:
        clsname = "P" + md5(launch_script).hexdigest()
        logger.warn("Using '%s' for plugin '%s'", clsname, pluginname)

    return type(clsname, (IonLaunchPlugin,), {'name': pluginname, '_launchsh': launch_script, '__doc__': '', 'add_to_store': add_to_store})
