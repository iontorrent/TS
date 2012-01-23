# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

# Load xmlrpc errors
import xmlrpclib
import socket
import time
import string

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
import iondb.settings as settings

# Out of framework django templates
import django.conf
try:
    django.conf.settings.configure(TEMPLATE_DIRS=settings.TEMPLATE_DIRS)
except:
    pass # Settings already configured

import django.template
from django.template.loader import render_to_string

from django.utils.encoding import smart_str

from iondb.anaserve import client

import logging

class PluginRunner():
    def __init__(self):

        # Default Settings
        self.context = {
            'memory': '8G',
            'debug': settings.TEST_INSTALL,
            'version': 0,
            'command': [],
            'sge': [],
        }

    def callPluginXMLRPC(self, start_json, host="127.0.0.1", port=settings.IPLUGIN_PORT):

        # Get Fault so we can catch errors
        retries = 60
        delay = 5
        # Total timeout = 60*5 == 300 == 5min
        attempts = 0

        log = logging.getLogger(__name__)
        log.propagate = False
        log.info("XMLRPC call to '%s':'%d'", host, port)

        while (attempts < retries):
            try:
                conn = client.connect(host,port)
                ret = conn.pluginStart(start_json)
                log.info("Plugin %s Queued '%s'", start_json["runinfo"]["plugin_name"], ret)
                return ret
            except (socket.error, xmlrpclib.Fault, xmlrpclib.ProtocolError, xmlrpclib.ResponseError) as f:
                log.info("XMLRPC Error: %s", f)
                if attempts < retries:
                    log.info("Error connecting to plugin daemon at %s:%d. Retrying in %d" % (host, port, delay))
                    time.sleep(delay)
                    attempts += 1
        #raise Exception("Unable to connect to plugin daemon after multiple attempts")
        return False


    def writePluginLauncher(self, pluginout, pluginname, content):
        pluginWrapperFile = os.path.join(pluginout, "ion_plugin_%s_launch.sh" % pluginname)
        with open(pluginWrapperFile, 'w') as f:
            f.write(content.encode('utf-8'))
        os.chmod(pluginWrapperFile, 0775)
        return pluginWrapperFile

    def createPluginWrapper(self, launchFile, start_json):
        c = self.context.copy()
        # Override config with plugin specific settings
        c.update(start_json['globalconfig'])
        c.update(start_json['pluginconfig'])
        c.update(start_json['runinfo'])
        c.update(start_json) # keep explicit values at original keys

        # Set essential plugin execution values to expected aliases
        c['pluginout']  = c["results_dir"]
        c['pluginpath'] = c["plugin_dir"]
        c['pluginname'] = c["plugin_name"]

        # If there's no memory request, fall back to globalconfig MEM_MAX
        if (not "memory" in c) and "MEM_MAX" in c:
            c["memory"] = c["MEM_MAX"]

        # Allow launch.sh to define some attributes... Experimental
        with open(launchFile, 'r') as launch:
            for line in launch:
                # Messy.
                if line.startswith("VERSION="):
                    # Fixme version is quoted...
                    c["version"] = line.split('=')[1].strip()

                # Skip #! and special comments
                if line.startswith("#!"):
                    pass
                elif line.startswith("#$"):
                    # SGE Resources 
                    c["sge"].append(line)
                else:
                    c["command"].append(line)

        # Flatten arrays to strings
        c["sge"] = string.join(c["sge"], '')
        c["command"] = string.join(c["command"], '')

        #ctx = django.template.Context()
        #ctx.update(c)

        # Create pluginWrapperFile from templateFile
        templateFile = "plugin/ion_plugin_wrapper.sh.tmpl"
        content = render_to_string(templateFile, c)
        return smart_str(content)

