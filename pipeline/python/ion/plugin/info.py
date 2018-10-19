# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import json
import logging
from copy import deepcopy

import re
from distutils.version import LooseVersion

from ion.plugin.launchcompat import get_launch_class

_logger = logging.getLogger(__name__)

# TODO: Move behavior to launchcompat


def get_version_launchsh(launch):
    # Regex to capture version strings from launch.sh
    # Flanking Quotes optional, but help delimit
    # Leading values ignored, usually '#VERSION' or '# VERSION'
    # Must be all-caps VERSION
    # Digits, dots, letters, hyphen, underscore (1.0.2-beta1_rc2)
    VERSION = re.compile(r'VERSION\s*=\s*\"?([\d\.\w\-\_]+)\"?')
    try:
        with open(launch, 'r') as f:
            for line in f:
                m = VERSION.search(line)
                if m:
                    v = m.group(1)
                    # Validate and canonicalize version string,
                    # according to distutils.version.LooseVersion semantics
                    try:
                        v = LooseVersion(v)
                    except ValueError:
                        _logger.warning("Version in file does not conform to LooseVersion rules: ", v)
                    return str(v)
    except:
        _logger.exception("Failed to parse VERSION from '%s'", pluginscript)
    return "0"


class PluginInfo(object):

    """ Class to encapsulate plugin introspection, to and from json block or plugin class instances """

    def __init__(self, infojson=None):
        self._meta = 'IonPlugin Definition format 1.0'
        self.name = None
        self.version = "0.0"
        self.config = {}  # getUserInput
        self.runtypes = []
        self.features = []
        self.runlevels = []
        self.depends = []
        self.docs = ""
        self.major_block = ""
        # self.description = ""
        self.pluginorm = None
        if infojson:
            try:
                self.parse(infojson)
            except:
                _logger.exception("Unable to inspect plugin for required parameters.")
        return

    def parse(self, data={}):
        if isinstance(data, basestring):
            data = json.loads(data)

        extract_keys = vars(self).keys()

        for k in extract_keys:
            if k in data:
                setattr(self, k, data[k])

    def todict(self):
        d = vars(self)
        # simplify to objects json can handle
        for k, v in d.iteritems():
            if isinstance(v, property):
                # d[k] = v.__get__()
                d[k] = str(v)
        return d

    def __repr__(self):
        return json.dumps(self.todict(), indent=2)

    # Set attributes of PluginInfo from an instance of an IonPlugin class
    def load_instance(self, plugin):
        try:
            self.config = plugin.getUserInput()
        except:
            _logger.exception("Failed to query plugin for getUserInput")

        for a in ('runtypes', 'features', 'runlevels', 'depends', 'major_block', 'requires_configuration'):
            v = getattr(plugin, a, None)
            if v is not None:
                setattr(self, a, v)
        self.docs = getattr(plugin, '__doc__', "")
        return self

    # Helper method for getting from an instance to a PluginInfo
    @staticmethod
    def from_instance(plugin):
        return PluginInfo(plugin._metadata()).load_instance(plugin)

    @classmethod
    def from_launch(cls, plugin, launch):
        try:
            legacyplugin = get_launch_class(plugin)
            pi = legacyplugin()

            # FIXME - inject context - plugin pk/instance is required here

            info = cls.from_instance(pi)
        except:
            _logger.exception("Failed to build and parse python class for legacy launch.sh - '%s':'%s'", plugin, launch)
            info = {
                'name': os.path.basename(os.path.dirname(launch)),
                'version': get_version_launchsh(launch),
            }
        return info
