#!/usr/bin/expect
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

set PLUGINCONFIG__USER_NAME [lindex $argv 0]
set PLUGINCONFIG__IP [lindex $argv 1]

spawn ssh $PLUGINCONFIG__USER_NAME@$PLUGINCONFIG__IP
expect "yes" {
    send "yes\r"
    expect "password"
}
exit