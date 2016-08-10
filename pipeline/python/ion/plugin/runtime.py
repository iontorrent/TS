# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import logging
import json


def parseJsonFile(jsonFile):
    try:
        with open(jsonFile, 'r') as f:
            content = f.read()
            result = json.loads(content)
    except IOError, OSError:
        return None
    return result

# Mixin Class for some runtime methods


class IonPluginRuntime(object):

    """
    Helper functions for plugin runtime

    * self.log - python logging instance for plugin.

    """

    log = logging.getLogger(__name__)
    log.__doc__ = """
    self.log - python logging instance for plugin. Usage:
        self.log.info("Message")
        self.log.error("An Error Occurred: %s", reason)
    """

    def get_restobj(self, resource, params={}, timeout=30, attempts=5):
        def get_apiurl(resource, params={}, api_version='v1'):
            """Preprocess URL and return url, params, and headers. Override to change auth method."""
            runinfo = self.startplugin.get('runinfo', {})
            api_url = runinfo.get('api_url', "http://localhost/rundb/api")

            # Authentication via api_key as GET parameter
            api_key = runinfo.get('api_key')
            prpk = runinfo.get('pluginresult') or runinfo.get('plugin', {}).get('pluginresult')
            params.update({'pluginresult': prpk, 'api_key': api_key})

            headers = {'content-type': 'application/json'}

            query_url = '/'.join(s + '/' for s in (api_url, api_version, resource))
            return (query_url, {'params': params, 'headers': headers, })

        (url, args) = get_apiurl(resource, params)
        while True:
            try:
                response = requests.get(url, timeout=timeout, verify=False, **args)
                response.raise_for_status()
                break
            except (requests.exceptions.HTTPError) as e:
                self.log.exception("Failed")
                return None
            except (requests.exceptions.Timeout, requests.ConnectionError) as e:
                self.log.warn("%s", e)
                if type(e) == requests.exceptions.Timeout:
                    self.log.info("Increasing timeout by 5 seconds for this request.")
                    timeout += 5
                attempts -= 1
                if attempts <= 0:
                    return None

        try:
            # May fail for NaN in JSON content
            content = response.json()
        except ValueError:
            content = response.content
        return content

    @property
    def startplugin(self):
        try:
            with open('startplugin.json', 'r') as fh:
                spj = json.load(fh)
            return spj
        except:
            self.log.exception("Error reading start plugin json")
        return {}
