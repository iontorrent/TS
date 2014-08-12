#!/usr/bin/expect
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

set PLUGINCONFIG__USER_PASSWORD [lindex $argv 0]
set PLUGINCONFIG__USER_NAME [lindex $argv 1]
set PLUGINCONFIG__IP [lindex $argv 2]

spawn pscp -pw $PLUGINCONFIG__USER_PASSWORD $PLUGINCONFIG__USER_NAME@$PLUGINCONFIG__IP:/tmp/ ./
expect "Store key in cache" {
    send "y\r"
    expect "password"
}
exit