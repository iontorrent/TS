#!/bin/bash
#-------------------------------------------------------------------------------
# Copyright 2008-2010 Connexed.
#
# $Id: reverse_ssh.sh 2163 2010-06-30 04:31:26Z bruce $
#
# Edited 5 Mar 2015 Thermo Fisher Scientific
#
# Description:
#
#   This script creates a secure shell listener session from the supplied
#   remote host/port/user-login indicated to the supplied local host/port. 
#
#   When attempting to 'start' a listener, if a similar command is 
#   already running then nothing is done.
#
# Arguments:
#
#  TBD: options:
#    --action 
#    --lport
#    --aport
#    --oport
#    --host
#    --user
#    --pass
#
#  $1 - restart|start|stop (restart and stop will kill any 'similar'
#       background ssh processes)
#  $2 - The 'local' IP port (default: 22)
#  $3 - The remote 'access' IP port (default: 22)
#  $4 - The remote 'operator login' IP port (default: 15022)
#  $5 - The remote host IP address/name (default: 'rssh.iontorrent.net')
#  $6 - The remote host username (default: 'root')
#  $7 - The remote host password (undefined if unused)
#-------------------------------------------------------------------------------

# This function invokes an ssh command and injects the password 
# if prompted; It also handles prompts for remote key update.

function start_session
{
  command="$1"
  password="${2//\$/\\\$}" # Escape the dollar sign.

  /usr/bin/expect -c "
  log_user 1
  set timeout 5
  spawn $command
  expect {
	eof {
		puts \"Connection failed.\"
		exit
	}

    -re \".*Are.*.*yes.*no.*\" {
      send \"yes\r\"
      exp_continue
    }

    -re \"\r\nPermission denied, please try again.\r\r\.*@.*'s password:\" { 
		puts \"\r\nConnection failed.\r\nBad username or password.\"
		exit
	}

    -re \".*assword:\" {
      send -- \"$password\r\"
      exp_continue
    }

  }
  interact"
}

default_action="start"
default_local_port=22
default_remote_access_port=22
default_operator_login_port=15022
default_remote_address="rssh.iontorrent.net"
default_remote_username="root"

# Check the first argument to ensure it is a valid action (use 
# default if it is missing).

if [ -z "$1" ];then
  action=$default_action
else
  action=$1
fi

case $action in

  start)
    ;;

  stop)
    ;;

  restart)
   $0 stop
   $0 start $2 $3 $4 $5 $6 $7
   exit $?
   ;;

  *)
    echo "Usage: $0 restart|start|stop <local-ip-port> <remote-access-ip-port> <operator-login-ip-port> <remote-ip-address> <remote-user> <remote-password>"
    echo "Defaults: action = $default_action"
    echo "          local-ip-port = $default_local_port"
    echo "          remote-access-ip-port = $default_remote_access_port"
    echo "          operator-login-ip-port = $default_operator_login_port"
    echo "          remote-ip-address = $default_remote_address"
    echo "          remote-user = $default_remote_username"
    echo "          remote_password = unused if undefined"
    exit 1
    ;;

esac

# Assign the remaining arguments to variables (or use defaults if missing).

if [ -z "$2" ];then
  local_port=$default_local_port
else
  local_port=$2
fi

if [ -z "$3" ];then
  remote_access_port=$default_remote_access_port
else
  remote_access_port=$3
fi

if [ -z "$4" ];then
  operator_login_port=$default_operator_login_port
else
  operator_login_port=$4
fi

if [ -z "$5" ];then
  remote_address=$default_remote_address
else
  remote_address=$5
fi

if [ -z "$6" ];then
  remote_username=$default_remote_username
else
  remote_username=$6
fi

# Ok to use empty password if one is not defined.

remote_password="$7"

# Create the 'preamble' that contains a known static portion of the 
# connect command string used for searching process list output.

preamble="ssh -N"

# Determine if a similar command is already running (e.g. different port).

pgrep -f "$preamble" > /dev/null 2>&1 

if [ $? -eq 0 ] ; then

  # Matching processes were found.

  # If this is a 'stop' action then kill any matches first, then exit;
  # (the start action should do nothing).

  if [ "$action" == "stop" ] ; then

    echo -n "$0 $* - Terminating process: "
    ps -ef | egrep "$preamble"
    pkill -f "$preamble"

  fi

  exit 0

else

  # No matching processes were found.

  # If this is a start action then create the session
  # (the stop action should do nothing).

  if [ "$action" == "start" ] ; then

    command="$preamble -p $remote_access_port -l $remote_username -o StrictHostKeyChecking=no -R ${operator_login_port}:localhost:${local_port} $remote_address"
    start_session "$command" "$remote_password"
    
  fi

fi
