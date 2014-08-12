#!/bin/bash
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
# Adds passwordless execution of commands for ionadmin user. Idempotent.
if [ -z "$1" ]; then
  export EDITOR=$0 && sudo -E visudo
else
  echo "Changing sudoers"
  sed -i '/^Defaults:ionadmin/ d' $1
  echo "Defaults:ionadmin !authenticate" >> $1
fi
