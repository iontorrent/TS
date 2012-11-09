# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import subprocess
import logging
import httplib2
import urllib2

import json

from urllib import urlencode

def parseJsonFile(jsonFile):
    try:
        with open(jsonFile, 'r') as f:
            content = f.read()
            result = json.loads(content)
    except IOError, OSError:
        return None
    return result

## Mixin Class for some runtime methods
class IonPluginRuntime(object):
    """
    Helper functions for plugin runtime

    * self.log - python logging instance for plugin.
    * self.call - shortcut to subprocess.call. Invoke command line programs

    """

    log = logging.getLogger(__name__)
    log.__doc__ = """
    self.log - python logging instance for plugin. Usage:
        self.log.info("Message")
        self.log.error("An Error Occurred: %s", reason)
    """

    call = subprocess.call

    def get_apiurl(self):
        ## FIXME - get from  startplugin.json, django settings, or cluster_settings
        base_url = 'http://localhost'
        api_url = base_url + '/rundb/api/v1/'

        return api_url


    def get_restobj(self, resource, **kwargs):
        api_url = self.get_apiurl()
        query_url = api_url + resource + '/'

        pk = kwargs.pop('pk', None)
        if pk: query_url += pk + '/'
        if kwargs: query_url += '?' + urlencode(kwargs)

        h = httplib2.Http()
        headers = {"Content-type": "application/json","Accept": "application/json"}
        (resp, content) = h.request(query_url, "GET", headers=headers)
        status = resp.status
        if int(status) not in [200,202,304]:
            self.log.error("REST query status: '%s'\n%s", status, content)
            return None

        return content

    #@lazyprop
    def startplugin(self):
        pass
    
