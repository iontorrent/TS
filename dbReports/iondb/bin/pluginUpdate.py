#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved


import sys
import os
import optparse
import json

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    import iondb.settings as settings
except ImportError:
    sys.path.pop()
    sys.path.append("../")
    sys.path.append("../../")
    import iondb.settings as settings

import xmlrpclib
import time

import socket
from iondb.anaserve import client

def UpdatePluginStatus(pk,plugin, msg , host="127.0.0.1", port = settings.IPLUGIN_PORT, method="update"):
    """
    Updates a plugins status through XMLRPC
    """
    # Get Fault so we can catch errors
    retries = 10
    delay = 5
    # Total timeout = 60*5 == 300 == 5min
    attempts = 0

    while attempts < retries:
        try:
            conn = client.connect(host,port)
            methodAttach = getattr(conn,method)
            ret = methodAttach(pk,plugin,msg)
            print "Plugin Update returns: '%s'" % ret
            break
        except (socket.error, xmlrpclib.Fault, xmlrpclib.ProtocolError, xmlrpclib.ResponseError) as f:
            print "XMLRPC Error: %s" % f
            if attempts < retries:
                print "Error connecting to plugin daemon at %s:%d. Retrying in %d" % (host, port, delay)
                if attempts % 5:
                    print "Error connecting to plugin daemon at %s:%d. Retrying in %d" % (host, port, delay)
                    time.sleep(delay)
                attempts += 1
            else:
                raise "Unable to connect to plugin daemon after multiple attempts"

    print "Plugin %s Status Successfully Updated." % plugin

if __name__ == '__main__':

    options = optparse.OptionParser(description = 'Update a plugins status and json dataStore through the ionPlugin daeamon')

    options.add_option('-i', '--pk'     , dest = 'pk'     , action = 'store', default = "")
    options.add_option('-p', '--plugin' , dest = 'plugin' , action = 'store', default = "")

    options.add_option('-s', '--msg' , dest = 'msg' , action = 'store', default = "",
                       help="Status message to set (if --convert is set status codes into verbose messages)")

    options.add_option('-c', '--convert' , dest = 'convert' , action = 'store_true', default = False,
                       help="Convert Unix exit codes provided with --msg (-s) into verbose messages")

    options.add_option('-m', '--method' , dest = 'method' , action = 'store', default = "update",
                       help="The XML-RPC Method to use, defaults to 'update'")

    options.add_option('-j', '--jsonpath' , dest = 'jsonpath' , action = 'store', default = "",
                       help="Path to the JSON file to PUT")


    (options, argv) = options.parse_args(sys.argv)

    opts = { "pk" : options.pk, "plugin" : options.plugin }

    goodArgs = True
    for key, value in opts.iteritems():
        if not value:
            print "You must give the value for",
            print key
            goodArgs = False
        else:
            print key,
            print value

    if not goodArgs:
        print "Bad args"
        sys.exit(1)

    # Borrowed from TLScript
    try:
        # SHOULD BE settings.QMASTERHOST, but this always seems to be "localhost"
        act_qmaster = os.path.join(settings.SGE_ROOT, settings.SGE_CELL, 'common', 'act_qmaster')
        master_node = open(act_qmaster).readline().strip()
    except IOError:
        master_node = "localhost"

    updateMessage = options.msg

    if options.convert:
        if updateMessage == "1":
            updateMessage = "Error"
        elif updateMessage == "0":
            updateMessage = "Completed"
        else:
            updateMessage = "Error"


    if options.jsonpath:
        try:
            updateMessage = open(os.path.join(options.jsonpath,"results.json")).read()
        except:
            print "Error could not read the json file ", options.jsonpath
            sys.exit(1)

        try:
            updateJSON = json.loads(updateMessage)
        except:
            print "Error could not parse the json file ", options.jsonpath
            sys.exit(1)

    updater = UpdatePluginStatus(options.pk,
                                 options.plugin,
                                 updateMessage,
                                 master_node,
                                 settings.IPLUGIN_PORT,
                                 options.method
                                 )

    print "Set the Status of the plugin %s to %s" % (options.pk, updateMessage)



