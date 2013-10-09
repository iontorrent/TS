#!/usr/bin/expect
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
#exp_internal 1
set timeout 9
set PLUGINCONFIG__USER_NAME [lindex $argv 0]
set PLUGINCONFIG__USER_PASS [lindex $argv 1]
set PLUGINCONFIG__IP [lindex $argv 2]
set PLUGINCONFIG__FILENAME [lindex $argv 3]
set PLUGINCONFIG__DESTDIR [lindex $argv 4]
log_user 0

# Command format:
# rsync [OPTIONS] filename username@server:destination_dir
spawn rsync -av $PLUGINCONFIG__FILENAME $PLUGINCONFIG__USER_NAME@$PLUGINCONFIG__IP:$PLUGINCONFIG__DESTDIR
expect {
   "*yes/no*" { send "yes\r" ; exp_continue }
   "Store key in cache" { send "y\r" ; exp_continue }
   "*assword:"
}
send "$PLUGINCONFIG__USER_PASS\r"
send_user "Transferring $PLUGINCONFIG__FILENAME..."
expect {
    "*otal size*" { send_user "complete.\n" ; exit }
    "*ailed:*" { send_user "failed.\n" ; exit 1}
    timeout { send_user "time-out " ; exp_continue }
    eof { exit }
}
