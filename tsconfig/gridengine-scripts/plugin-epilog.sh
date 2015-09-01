#!/bin/bash
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
## SGE Epilog script

makeVarsFromJson()
{
    local IFS=$'\n'
    for line in $(jsonpipe < $1);do
        if echo $line|egrep -q '(\{}|\[]|barcodedSamples|sampleinfo)'; then
            :
        else
            line=${line#/}
            var_name=$(echo $line|awk '{print $1}') # =${line%%	*}
            var_val=$(echo $line|awk -F'\t' '{print $2}'|sed 's/"//g') #=${line##*	}  #
            index=$(basename $var_name) #=${var_name%%/*}

            # Sanitize
            #(sampleinfo can contain more chars than are valid variable names)
            var_name=${var_name//-/_} # $(echo $line|sed 's:-:_:g')

            # var_name ends in a number, its an array variable
            # (test fails with "integer expression expected" for non-numbers)
            if [ "$index "-eq "$index" 2>/dev/null ]; then
                #strip off number
                var_name=$(dirname $var_name) #=${var_name%%/$index}
                #convert slashes to double underscore
                #convert to uppercase
                var_name=$(echo ${var_name^^}|sed 's:/:__:g')
                #echo $var_name[$index]=$var_val
                eval $var_name[$index]=\""$var_val"\"
                export ${var_name}
            else
                #convert slashes to double underscore
                #convert to uppercase
                var_name=$(echo ${var_name^^}|sed 's:/:__:g')
                #echo $var_name=$var_val
                export eval "${var_name}"="${var_val}"
            fi
        fi
    done
}


/bin/cp -u ${SGE_JOB_SPOOL_DIR}/usage .
# should define exit_status

. ${SGE_JOB_SPOOL_DIR}/usage
echo "SGE exit_status: ${exit_status-128}"

if [ ${exit_status} -eq 2 ] || [ ${exit_status} -gt 128 ]; then
  makeVarsFromJson "./startplugin.json"
  echo "==============================================================================="
  date +'end time=%Y-%m-%d %k:%M:%S.%N'
  ## Set Failed status - error not caught in script itself. probably killed due to memory/time)
  echo "FATAL: JOB Killed by SGE [${exit_status}]."
  case $exit_status in
      2)
          echo "PLUGIN DEVELOPER: Syntax Error in plugin launch script!"
          ;;
      134)
          echo "Plugin exceeded maximum memory requested for plugin queue."
          ;;
      137)
          echo "Plugin exceeded maximum runtime requested for plugin queue."
          ;;
      143)
          echo "Plugin Terminated by user request"
          ;;
  esac
  ion-plugin-status --pk ${RUNINFO__PK} -s 'Error'
fi
