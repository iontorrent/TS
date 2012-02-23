#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved


import sys
import os
import optparse
import json

import djangoinit
from django.conf import settings

import xmlrpclib
import time

import socket
from iondb.anaserve import client

def UpdatePluginStatus(pk, plugin, version, msg, host="127.0.0.1", port = settings.IPLUGIN_PORT, method="update"):
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

    print "Plugin '%s' Successfully Updated" % plugin

if __name__ == '__main__':

    options = optparse.OptionParser(description = 'Update a plugins status and json dataStore through the ionPlugin daeamon')

    options.add_option('-i', '--pk'     , dest = 'pk'     , action = 'store', default = "")
    options.add_option('-p', '--plugin' , dest = 'plugin' , action = 'store', default = "") ## Plugin Name
    options.add_option('-v', '--version' , dest = 'pluginversion' , action = 'store', default = "") ## Plugin Version

    # More precise identifiers
    #options.add_option('-k', '--pluginkey' , dest = 'pluginkey' , action = 'store', default = "") ## Plugin DB Key
    options.add_option('-r', '--resultkey', dest = 'resultkey' , action = 'store', default = "") ## ResultKey

    options.add_option('-s', '--msg' , dest = 'msg' , action = 'store', default = "",
                       help="Status message to set (if --convert is set status codes into verbose messages)")

    options.add_option('-c', '--convert' , dest = 'convert' , action = 'store_true', default = False,
                       help="Convert Unix exit codes provided with --msg (-s) into verbose messages")

    options.add_option('-m', '--method' , dest = 'method' , action = 'store', default = "update",
                       help="[DEPRECATED, IGNORED] The XML-RPC Method to use, defaults to 'update'")

    options.add_option('-j', '--jsonpath' , dest = 'jsonpath' , action = 'store', default = "",
                       help="Path to the JSON file to PUT")


    (options, argv) = options.parse_args(sys.argv)

    # Provide a placeholder value for unspecified version
    if not options.pluginversion: options.pluginversion = '0'

    # Check required options
    opts = { "pk" : options.pk, "plugin" : options.plugin, "version": options.pluginversion }

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

    # Set a status using 'update' method.
    if options.msg:
        updateMessage = options.msg

        if options.convert:
           if updateMessage == "1":
               updateMessage = "Error"
           elif updateMessage == "0":
               updateMessage = "Completed"
           else:
               updateMessage = "Error"

        # TODO Ensure message is in allowed values

        updater = UpdatePluginStatus(options.pk,
                                     options.plugin, options.pluginversion,
                                     updateMessage,
                                     settings.IPLUGIN_HOST,
                                     settings.IPLUGIN_PORT,
                                     'update'
                                     )
        print "Set the Status of the plugin '%s' on report '%s' to %s" % (options.plugin, options.pk,options.msg)


    if options.jsonpath:
        try:
            updateJSON = open(os.path.join(options.jsonpath,"results.json")).read()
        except:
            print "Error could not read the json file ", options.jsonpath
            sys.exit(1)

        try:
            updateJSON = json.loads(updateMessage)
        except:
            print "Error could not parse the json file ", options.jsonpath
            sys.exit(1)

        updater = UpdatePluginStatus(options.pk,
                                     options.plugin, options.pluginversion,
                                     updateMessage,
                                     settings.IPLUGIN_HOST,
                                     settings.IPLUGIN_PORT,
                                     'resultsjson'
                                     )

        print "Posted results store for plugin '%s'" % options.plugin

