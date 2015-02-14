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
    server = xmlrpclib.ServerProxy("http://%s:%d" % (host,port), allow_none=True, verbose=True)
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
    #log.propagate = False

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


def call_launchPluginsXMLRPC(result_pk, plugins, net_location, username, runlevel='default', params={}, conn=None):
    
    if conn is None:    
        conn = get_serverProxy()
    plugins_dict, msg = conn.launchPlugins(result_pk, plugins, net_location, username, runlevel, params)

    return plugins_dict, msg

def call_pluginStatus(jobid, conn=None):
    if jobid is None:
        return "Invalid JobID"
    if conn is None:
        conn = get_serverProxy()

    ret = ""
    log = logging.getLogger(__name__)
    try:
        ret = conn.pluginStatus(jobid)
    except (socket.error, xmlrpclib.Fault, xmlrpclib.ProtocolError, xmlrpclib.ResponseError) as f:
        log.exception("XMLRPC Error")
        ret = str(f)

    return ret

def call_sgeStop(jobid, conn=None):
    if jobid is None:
        return "Invalid JobID"
    if conn is None:
        conn = get_serverProxy()

    ret = ""
    log = logging.getLogger(__name__)
    try:
        ret = conn.sgeStop(jobid)
    except (socket.error, xmlrpclib.Fault, xmlrpclib.ProtocolError, xmlrpclib.ResponseError) as f:
        log.exception("XMLRPC Error")
        ret = str(f)

    return ret

