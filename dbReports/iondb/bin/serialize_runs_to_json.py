#!/usr/bin/python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import pwd
import grp
import os
import shutil
import traceback
import argparse
from iondb.bin import djangoinit
from iondb.rundb.models import (
    Results,
    Experiment,
    ExperimentAnalysisSettings,
    DMFileSet,
    DMFileStat,
)
import iondb.rundb.data.dmactions_types as dmactions_types

from django.core import serializers
from django.core.serializers.json import DjangoJSONEncoder


def write_serialized_json(result, skip_if_exists=True):
    destination = result.get_report_path()
    if not os.path.exists(destination):
        print(
            "Error: no report directory for %s (%d)." % (result.resultsName, result.pk)
        )
        return

    sfile = os.path.join(destination, "serialized_%s.json" % result.resultsName)
    # skip if already exists
    if skip_if_exists and os.path.exists(sfile):
        print("File %s exists, skipping." % sfile)
        return

    serialize_objs = [result, result.experiment]
    for obj in [
        result.experiment.plan,
        result.eas,
        result.analysismetrics,
        result.libmetrics,
        result.qualitymetrics,
    ]:
        if obj:
            serialize_objs.append(obj)
    serialize_objs += list(result.pluginresult_set.all())

    try:
        with open(sfile, "wt") as f:
            obj_json = serializers.serialize(
                "json", serialize_objs, indent=2, use_natural_keys=True
            )

            dmfilesets = DMFileSet.objects.filter(dmfilestat__result=result)
            obj_json = obj_json.rstrip(" ]\n") + ","
            obj_json += serializers.serialize(
                "json", dmfilesets, indent=2, fields=("type", "version")
            ).lstrip("[")

            f.write(obj_json)
        print(
            "Result: %s (%d), saved serialized objs file %s"
            % (result.resultsName, result.pk, sfile)
        )

        try:
            # TODO: Set ownership to www-data user and group
            uid = pwd.getpwnam("www-data").pw_uid
            gid = grp.getgrnam("www-data").gr_gid
            os.chown(sfile, uid, gid)
            os.chmod(sfile, 0o0664)
        except Exception:
            print("Unable to set the ownership to www-data user.")
            print(traceback.format_exc())

        # also want the serialized json in raw data folder so Sigproc and proton
        # onboard Basecalling Input files can be imported
        expDir = result.experiment.expDir
        try:
            if os.path.exists(expDir):
                shutil.copy2(sfile, expDir)
        except Exception:
            print("Unable to copy serialized json file to %s" % expDir)
            print(traceback.format_exc())

    except Exception:
        print(
            "Error: Unable to save serialized.json for %s (%d)."
            % (result.resultsName, result.pk)
        )
        print(traceback.format_exc())


if __name__ == "__main__":
    """
    Serializes objects needed for Data Import and saves serialized_resultsName.json in report root dir
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reportName",
        dest="name",
        action="store",
        default="",
        help="created serialized json for a report",
    )
    parser.add_argument(
        "--expName",
        dest="exp_name",
        action="store",
        default="",
        help="created serialized json for all reports of an experiment",
    )
    parser.add_argument(
        "--skip-existing",
        dest="skip_if_exists",
        action="store",
        default="on",
        help="skip json creation if file already exists [default:on]",
    )

    args = parser.parse_args()

    if args.name:
        results = Results.objects.filter(resultsName=args.name)
        if not results:
            print("Error: result %s is not found, exiting." % args.name)
    elif args.exp_name:
        results = Results.objects.filter(experiment__expName__contains=args.exp_name)
        if not results:
            print(
                "Error: no results found for experiment name %s, exiting."
                % args.exp_name
            )
    else:
        # dmfilestats =
        # DMFileStat.objects.exclude(dmfileset__type=dmactions_types.INTR).exclude(action_state__in=['AG','DG','AD','DD'])
        dmfilestats = DMFileStat.objects.filter(
            dmfileset__type=dmactions_types.BASE
        ).exclude(action_state__in=["AG", "DG", "AD", "DD"])
        results = Results.objects.filter(dmfilestat__in=dmfilestats).distinct()

    print("Processing %d results." % len(results))

    for r in results:
        write_serialized_json(r, args.skip_if_exists == "on")
