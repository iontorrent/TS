#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# Ion Reporter 1.0 wrapper for TS3.0

from ion.plugin import *
import urllib
import urllib2
import os
import json
import subprocess

class IonReporterUploader_V1_0(IonPlugin):
    version = "1.0.0"
    runtypes = [ RunType.FULLCHIP ]
    features = [ Feature.EXPORT ]

    def IonReporterWorkflows_1(self, config):
    
        try:
            headers = {"Authorization" : config["token"] }
            url = config["protocol"] + "://" + config["server"] + ":" + config["port"] +"/grws/analysis/wflist"
            self.log.info(url)
        except KeyError:
            error = "IonReporterUploader V1.0 Plugin Config is missing needed data."
            self.log.error(error)
            return False, error
    
        try:
            #using urllib2 right now because it does NOT verify SSL certs
            req = urllib2.Request(url = url, headers = headers)
            response = urllib2.urlopen(req, timeout=30)
            content = response.read()
            content = json.loads(content)
            workflows = content["workflows"]
            return True, workflows
        except:
            error = "IonReporterUploader V1.0 could not contact the server."
            self.log.error(error)
            return False, error


    def launch(self, data=None):
        lenv = {}
        lenv.update(os.environ)
        launchsh = os.path.join(lenv.get("RUNINFO__PLUGIN_DIR"),"old_launch.sh")
        outputpath = lenv.get("RUNINFO__RESULTS_DIR")
        if not os.path.exists(launchsh):
            self.log.error("Unable to find launch.sh at '%s'", launchsh)
            return False        
        ret = subprocess.call([launchsh, "-j", "startplugin.json"], env=lenv, cwd=outputpath)
        self.exit_status = ret
        return (ret == 0)
        

    def report(self):
        output = {
            'sections': {
                'title': 'IRU 1.0',
                'type': 'html',
                'content': '<p>Ion Reporter Uploader 1.0</p>',
            },
        }
        return output

    def getUserInput(self):
        defaultWorkFlow = {
                        'columns': [
                            {"Name": "Workflow", 
                                 "Order" : "1", 
                                 "Type" : "list", 
                                 "ValueType" : "String", 
                                 "Values" : ['A', 'B', 'C']
                            },
                        ],
                        'restrictionRules': [],
                        }
        try:
            f = urllib.urlopen("http://localhost/rundb/api/v1/plugin/?format=json&name=IonReporterUploader_V1_0&active=true")        
            d = json.loads(f.read())
            config = d["objects"][0]["config"]
            ret, workflowList = self.IonReporterWorkflows_1(config)
            if ret:
                defaultWorkFlow["columns"][0]["Values"] = ['no_workflow'] + workflowList
                return defaultWorkFlow
            else:
                return defaultWorkFlow
        except:
                return None

    def metrics(self):
        return {}

# dev use only - makes testing easier
if __name__ == "__main__": PluginCLI(IonReporterUploader_V1_0())

