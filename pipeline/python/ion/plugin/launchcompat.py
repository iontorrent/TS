# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved


import os
import subprocess
import re, tokenize, keyword
from hashlib import md5
import logging

from distutils.version import LooseVersion

from ion.plugin.base import IonPlugin
from ion.plugin.constants import RunType, Feature

logger = logging.getLogger(__name__)

class IonLaunchPlugin(IonPlugin):
    """ Compatability with pre-3.0 launch.sh plugins
        Allows new setup and configuration definitions, but calls launch.sh
    """

    ## Internal state - these are class attributes, as script is read
    ## to populate VERSION and AUTORUNDISABLE in the class
    _content=[]
    _launchsh=None

    ## Defaults for Launch Plugins
    runtypes = [ RunType.FULLCHIP ]
    features = []

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
    def version(cls):
        # Regex to capture version strings from launch.sh
        # Flanking Quotes optional, but help delimit
        # Leading values ignored, usually '#VERSION' or '# VERSION'
        # Must be all-caps VERSION
        # Digits, dots, letters, hyphen, underscore (1.0.2-beta1_rc2)
        VERSION=re.compile(r'VERSION\s*=\s*\"?([\d\.\w\-\_]+)\"?')
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
            logger.error("Plugin launch script does not define VERSION '%s'", cls._launchsh)
        return "0"

    @classmethod
    def allow_autorun(cls):
        """ if the string AUTORUNDISABLE is in the lunch script
        don't allow the autorun settings to be changed on the config tab
        """
        for line in cls.getContent():
            if line.startswith("#AUTORUNDISABLE"):
                return False
        return True

    @classmethod
    def major_block(cls):
        """ if the string AUTORUNDISABLE is in the lunch script
        don't allow the autorun settings to be changed on the config tab
        """
        for line in cls.getContent():
            if line.startswith("#MAJORBLOCK") or line.startswith("#MAJOR_BLOCK"):
                return True
        return False

    def launch(self):
        lenv = {}
        lenv.update(os.environ)
        # TODO setup lenv with plugin_functions variables
        # For now local ionPluginShell --local will call plugin_functions
        if not os.path.exists(launchsh):
            self.log.error("Unable to find launch.sh at '%s'", self.launchsh)
            return False
        outputpath = self.data['analysis_dir']
        ret = subprocess.call(["ionPluginShell", self.launchsh, "-j", "startplugin.json"], env=lenv, cwd=outputpath)
        self.exit_status = ret
        return (ret == 0)

def get_launch_class(pluginname, launch_script):
    # Build our name from user provided pluginname
    # Our class name
    if pluginname is not None:
        clsname = str(pluginname) ## de-unicodify
        m = re.match('^'+tokenize.Name+'$',clsname)
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

    return type(clsname, (IonLaunchPlugin,), {'name': pluginname, '_launchsh': launch_script, '__doc__': ''})

