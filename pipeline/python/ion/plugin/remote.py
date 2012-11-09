# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import xmlrpclib
import socket
import logging
import time

try:
    from django.conf import settings
except ImportError:
    import sys
    sys.path.append('/etc/')
    import torrentserver.cluster_settings as settings

def get_serverProxy():

    host = getattr(settings,'PLUGINSERVER_HOST',getattr(settings,'IPLUGIN_HOST','127.0.0.1'))
    port = getattr(settings,'PLUGINSERVER_PORT',getattr(settings,'IPLUGIN_PORT',8080))
    server = xmlrpclib.ServerProxy("http://%s:%d" % (host,port), allow_none=True)  
    return server

def callPluginXMLRPC(start_json, conn = None, retries = 60):
    """ Contact plugin daemon and make XMLRPC call to launch plugin job
        defined in start_json, which must be fully populated.
        Retries for 5 minutes on connection failure

        Return False on failure or jobid on success
    """
    delay = 5
    # Total timeout = 60*5 == 300 == 5min
    attempts = 0

    log = logging.getLogger(__name__)
    log.propagate = False

    try:
        plugin_name = start_json["runinfo"]["plugin"]["name"]
    except KeyError:
        plugin_name = start_json["runinfo"]["plugin_name"]

    while attempts < retries:
        try:
            if conn is None:
                conn = get_serverProxy()            
            jobid = conn.pluginStart(start_json)
            if jobid == -1:
                log.warn("Error Starting Plugin: '%s'", plugin_name)
                attempts+=1
                time.sleep(delay)
                continue
            log.info("Plugin %s Queued '%s'", plugin_name, jobid)
            return jobid
        except (socket.error, xmlrpclib.Fault, xmlrpclib.ProtocolError, xmlrpclib.ResponseError) as f:
            log.exception("XMLRPC Error")
            if attempts < retries:
                log.warn("Error connecting to plugin daemon. Retrying in %d", delay)
                time.sleep(delay)
                attempts += 1
    # Exceeded number of retry attempts
    log.error("Exceeded number of retry attempts. Unable to launch plugin '%s'", plugin_name)
    return jobid

def runPlugin(plugin, start_json, level = 'default', pluginserver = None, retries = 60):
  # launch a single plugin for a single runlevel and add its jobId to plugin hold list
  # multilevel plugins need level and pluginserver inputs
    debugstr='' 
    try:             
        if (level == 'pre'):
            plugin['results_dir'] = start_json['runinfo']['results_dir'] #shared folder for multilevel plugins
            
        if (level != 'block') and ('block_jid' in plugin.keys() ):
            plugin['hold_jid'] += plugin['block_jid']                                      
        if len(plugin['hold_jid']) > 0:
            plugin['hold_jid'] = list(set(plugin['hold_jid']))
            debugstr = 'hold for %s' % str(plugin['hold_jid'])
            if ( (pluginserver is not None) and ('block' not in level) and ('last' not in level)):
                # multilevel plugins: 
                # wait for previous level to start otherwise parameters may be overwritten when using same output folder
                # skip 'block' and 'last' runlevels to prevent unneeded waiting: blocks have their own folders and 'last' level is used to wait for results of other plugins.
                # SGE dependency is then used to wait for completion                
                jobIds = list(plugin['hold_jid'])
                start_json['runinfo']['plugin']['hold_jid'] = plugin['hold_jid']                
                while len(jobIds) > 0:
                    for j in jobIds:
                        status = pluginserver.pluginStatus(j)
                        if ('queued' in status) or ('on_hold' in status):
                          time.sleep(5)                    
                        else:
                          jobIds.remove(j)                    
            
        # launch plugin
        jid = callPluginXMLRPC(start_json, pluginserver, retries)
        msg = 'Launched plugin %s with jid %s %s.' % (plugin['name'],jid,debugstr)
        if jid < 0:
            msg = 'ERROR: Plugin %s failed to launch.' % plugin['name']
            return plugin, msg
            
        if level != 'block':
            plugin['hold_jid'].append(jid)
        else:
            #blocklevel plugins run asynchronously
            if ('block_jid' in plugin.keys()): 
                plugin['block_jid'].append(jid) 
            else:
                plugin['block_jid'] = [jid]
              
    except:
        msg = 'Plugin %s failed...' % plugin['name']
    return plugin, msg


