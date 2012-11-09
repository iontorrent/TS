#!/bin/bash
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
## SGE Epilog script

makeVarsFromJson ()
{
    local IFS=$'\n'
    for line in $(jsonpipe < $1); do
        if echo $line|egrep -q '(\{}|\[])'; then
            :
        else
            line=$(echo $line|sed 's:^/::g')
            var_name=$(echo $line|awk '{print $1}')
            var_val=$(echo $line|awk -F'\t' '{print $2}'|sed 's/"//g;s/^=//')
            index=$(basename $var_name)
            # var_name ends in a number, its an array variable
            if [ $index -eq $index 2>/dev/null ]; then
                #strip off number
                var_name=$(dirname "$var_name")
                #convert slashes to double underscore, convert to uppercase
                var_name=$(echo ${var_name}|sed -E 's:(.*):\U\1:g;s:/:__:g')
                eval $var_name[$index]="$var_val"
                export ${var_name}
            else
                #convert slashes to double underscore, convert to uppercase
                var_name=$(echo ${var_name}|sed -E 's:(.*):\U\1:g;s:/:__:g')
                export eval "${var_name}"="${var_val}"
            fi
        fi
    done
}

# should define exit_status
. ${SGE_JOB_SPOOL_DIR}/usage
echo "SGE exit_status: ${exit_status-255}"

if [ ${exit_status-255} -gt 126 ]; then
  makeVarsFromJson "./startplugin.json"
  echo "==============================================================================="
  date +'end time=%Y-%m-%d %k:%M:%S.%N'
  ## Set Failed status - error not caught in script itself. probably killed due to memory/time)
  ion-plugin-status --pk ${RUNINFO__PK} --plugin "${RUNINFO__PLUGIN__NAME}" --version "${RUNINFO__PLUGIN__VERSION-0}" -s 'Error'
fi

