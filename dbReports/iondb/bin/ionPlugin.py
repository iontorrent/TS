#!/teusr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from twisted.web import xmlrpc, server
import sys
import os
import httplib2
import datetime
import traceback
import json

from djangoinit import settings
from iondb.plugins.config import config
from iondb.plugins.runner import PluginRunner
from iondb.plugins.manager import pluginmanager

import logging
import logging.handlers

__version__ = filter(str.isdigit, "$Revision: 47186 $")

# Setup log file logging
try:
    log='/var/log/ion/ionPlugin.log'
    infile = open(log, 'a')
    infile.close()
except:
    print traceback.format_exc()
    log = './ionPlugin.log'
logger = logging.getLogger('logger')
logger.propagate = False
logger.setLevel(logging.DEBUG)
handler = logging.handlers.RotatingFileHandler(log, maxBytes=1024*1024*10, backupCount=5)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# distributed resource management
try:
    # We want to determine if we have the ability to make calls to the
    # drmaa interface. In order to do so, we first need to set environment
    # variables.
    if not settings.SGE_ENABLED:
        # bail if SGE (currently the only supported drmaa target) is
        # disabled from the site settings file.
        raise ImportError
    # Set env. vars necessary to talk to SGE. The SGE root contains files
    # describing where the grid master lives. The other variables determine
    # which of several possible grid masters to talk to.
    # this may or may not be needed because of the other vars that are being set by ionJobServer
    for k in ("SGE_ROOT", "SGE_CELL", "SGE_CLUSTER_NAME",
              "SGE_QMASTER_PORT", "SGE_EXECD_PORT", "DRMAA_LIBRARY_PATH"):
        if not k in os.environ:
            os.environ[k] = str(getattr(settings,k))
    try:
        import drmaa
    except RuntimeError:
        # drmaa will sometimes raise RuntimeError if libdrmaa1.0 is not
        # installed.
        logger.debug("Unexpected error: %s" % str(sys.exc_info()))
        raise ImportError
    import atexit # provides cleanup of the session object
    try:
        HAVE_DRMAA = True
        # create a single drmaa session
        _session = drmaa.Session()
        try:
            _session.initialize()
            logger.debug("DRMAA session initialized")
        except:
            logger.debug("Unexpected error: %s" % str(sys.exc_info()))
        atexit.register(_session.exit)
        djs = drmaa.JobState
        # globally define some status messages
        _decodestatus = {
            djs.UNDETERMINED: 'process status cannot be determined',
            djs.QUEUED_ACTIVE: 'job is queued and active',
            djs.SYSTEM_ON_HOLD: 'job is queued and in system hold',
            djs.USER_ON_HOLD: 'job is queued and in user hold',
            djs.USER_SYSTEM_ON_HOLD: ('job is queued and in user '
                                      'and system hold'),
            djs.RUNNING: 'job is running',
            djs.SYSTEM_SUSPENDED: 'job is system suspended',
            djs.USER_SUSPENDED: 'job is user suspended',
            djs.DONE: 'job finished normally',
            djs.FAILED: 'job finished, but failed',
            }
        InvalidJob = drmaa.errors.InvalidJobException
    except drmaa.errors.InternalException:
        # If we successfully import drmaa, but it somehow wasn't configured
        # properly, we will gracefully bail by raising ImportError
        raise ImportError
except (ImportError, AttributeError):
    # drmaa import failed
    HAVE_DRMAA = False
    InvalidJob = ValueError

def SGEPluginJob(start_json):
    """
    Spawn a thread that will start a SGE job, and wait for it to return
    after it has return it will make a PUT the API using to update the status
    of the run
    args is a dict of all the args that are needed by for the plugin to run
    """
    try:
        os.umask(0000)

        # Query some essential values
        plugin = start_json['runinfo']['plugin']
        resultpk = start_json['runinfo']['pk']
        analysis_name = os.path.basename(start_json['runinfo']['analysis_dir'])
        plugin_output = start_json['runinfo']['results_dir']

        logger.debug("Getting ready for queue : %s %s", resultpk, plugin['name'])

        #Make sure the dirs exist
        if not os.path.exists(plugin_output):
            os.makedirs(plugin_output, 0775)

        # Write start_json to startplugin.json file.
        startpluginjson = os.path.join(plugin_output,'startplugin.json')
        with open(startpluginjson,"w") as fp:
            json.dump(start_json,fp,indent=2)

        # Branch for launch.sh vs 3.0 plugin class
        logger.debug("Finding plugin definition: '%s':'%s'", plugin['name'], plugin['path'])
        (launch, isCompat) = pluginmanager.find_pluginscript(plugin['path'], plugin['name'])
        if not launch or not os.path.exists(launch):
            logger.error("Analysis: %s. Path to plugin script: '%s' Does Not Exist!", analysis_name, launch)
            return 1

        # Create individual launch script from template and plugin launch.sh
        launcher = PluginRunner()
        if isCompat:
            launchScript = launcher.createPluginWrapper(launch, start_json)
        else:
            start_json.update({'command': ["python %s -vv" % launch]})
            launchScript = launcher.createPluginWrapper(None, start_json)
        launchWrapper = launcher.writePluginLauncher(plugin_output, plugin['name'], launchScript)
        launcher.writePluginJson(start_json)

        # Prepare drmaa Job - SGE/gridengine only
        jt = _session.createJobTemplate()
        jt.nativeSpecification = " -q %s" % ("plugin.q")

        hold = plugin.get('hold_jid', [])
        if hold:
            jt.nativeSpecification += " -hold_jid " + ','.join(str(jobid) for jobid in hold)

        jt.workingDirectory = plugin_output
        jt.outputPath = ":" + os.path.join(plugin_output, "drmaa_stdout.txt")
        jt.joinFiles = True # Merge stdout and stderr

        # Plugin command is: ion_pluginname_launch.sh -j startplugin.json
        jt.remoteCommand = launchWrapper
        jt.args = [ "-j", startpluginjson ]

        # Submit the job to drmaa
        jobid = _session.runJob(jt)

        # Update pluginresult status
        h = httplib2.Http()
        headers = {"Content-type": "application/json","Accept": "application/json"}
        url = 'http://localhost' + '/rundb/api/v1/results/%s/plugin/' % resultpk
        resp, content = h.request(url, "PUT", body=json.dumps({plugin['name']:"Queued"}), headers=headers)
        logger.debug("PluginResult Marked Queued : %s %s" % (resp, content))

        logger.info("Analysis: %s. Plugin: %s. Job: %s Queued." % (analysis_name, plugin['name'], jobid))

        _session.deleteJobTemplate(jt)
        return jobid
    except:
        logger.critical("SGEPluginJob method failure")
        logger.critical(traceback.format_exc())

class Plugin(xmlrpc.XMLRPC):
    """Provide a way to add plugings to SGE from the API
        Also have a way to keep track of the job
    """

    def __init__(self, allowNone=True, useDateTime=False):
        self.start_time = datetime.datetime.now()
        self.allowNone = allowNone
        self.useDateTime = useDateTime # Support for old twisted with new python xmlrpclib

    def xmlrpc_pluginStart(self, start_json):
        """
        Launch the plugin defined by the start_json block
        """
        logger.debug("SGE job start request")
        try:
            jobid = SGEPluginJob(start_json)
            return jobid
        except:
            logger.error(traceback.format_exc())
            return -1
        

    def xmlrpc_sgeStop(self, x):
        """
        TODO: Write this
        """
        logger.debug("SGE job kill request")
        #X might be a dict of stuff to do.
        return x

    def xmlrpc_uptime(self):
        """Return the amount of time the ``ionPlugin`` has been running."""
        diff = datetime.datetime.now() - self.start_time
        seconds = float(diff.days*24*3600)
        seconds += diff.seconds
        seconds += float(diff.microseconds)/1000000.0
        logger.debug("Uptime called - %d (s)" % seconds)
        return seconds
        
    def xmlrpc_pluginStatus(self, jobid):
        """Get the status of the job"""
        try:
            logging.debug("jobstatus for %s" % jobid)
            status = _session.jobStatus(jobid)
        except:
            logging.error(traceback.format_exc())
            status = "DRMAA BUG"
        return status    

    def xmlrpc_update(self,pk,plugin,status):
        """
        Because of the Apache security it is hard for crucher nodes to contact the API directly.
        This method is a simple proxy to update the status of plugins
        """

        #For API updates
        h = httplib2.Http()
        headers = {"Content-type": "application/json","Accept": "application/json"}
        url = 'http://localhost' + '/rundb/api/v1/results/' + str(pk) + "/plugin/"

        logger.debug("Update called %s", url)

        #put the updated dict - partial update
        pluginUpdate = {plugin: status}
        resp, content = h.request(url, "PUT", body=json.dumps(pluginUpdate), headers=headers )
        logger.debug("Status updated : %s %s" % (resp, content))
    
        if resp.status in [200,201,202,203,204]:
            return True
        else:
            logger.error('plugin status update API error [%d]: %s', resp.status, content)
            return False


    def xmlrpc_resultsjson(self,pk,plugin,msg):

        #Upload the JSON results to the db
        try:
            JSONresults = json.loads(msg)
        except:
            logger.error("Unable to open or parse results.json")
            return False

        h = httplib2.Http()
        headers = {"Content-type": "application/json","Accept": "application/json"}
        url = 'http://localhost' + '/rundb/api/v1/results/' + str(pk) + '/pluginstore/'

        logger.debug("Updating plugin %s results.json: %s" % (plugin, url))
        update = json.dumps({plugin: JSONresults})

        #PUT the updated dict
        resp, content = h.request(url,
                            "PUT", body=update,
                            headers=headers )
        if resp.status in [200,201,202,203,204]:
            return True
        else:
            logger.error("resultsjson API PUT returned error [%d]: %s",resp.status,content)
            return False


if __name__ == '__main__':
    from twisted.internet import reactor
    r = Plugin()
    reactor.listenTCP(settings.IPLUGIN_PORT, server.Site(r))
    logger.info("ionPlugin Started Ver: %s" % __version__)
    reactor.run()
