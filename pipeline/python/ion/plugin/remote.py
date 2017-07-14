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
    host = getattr(settings, 'PLUGINSERVER_HOST', getattr(settings, 'IPLUGIN_HOST', '127.0.0.1'))
    port = getattr(settings, 'PLUGINSERVER_PORT', getattr(settings, 'IPLUGIN_PORT', 8080))
    server = xmlrpclib.ServerProxy("http://%s:%d" % (host, port), allow_none=True, verbose=False)
    return server


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
