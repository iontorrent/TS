#!/usr/bin/env python
# coding=utf-8
# Copyright (C) 2018 Ion Torrent Systems, Inc. All Rights Reserved

import subprocess
import argparse

parser = argparse.ArgumentParser(description='This script enables and disables Thermo telemetry services.')
parser.add_argument('--enable', action="store_true", default=False)
parser.add_argument('--disable', action="store_true", default=False)

args = parser.parse_args()

if all([args.enable, args.disable]) or all([not args.enable, not args.disable]):
    print("Specify either --enable or --disable!")
    exit(1)

for service_name in ["deeplaser", "RSM_Launch"]:
    if args.enable:
        subprocess.check_output(["sudo", "update-rc.d", service_name, "enable"])
        subprocess.check_output(["sudo", "service", service_name, "start"])
    elif args.disable:
        subprocess.check_output(["sudo", "update-rc.d", service_name, "disable"])
        subprocess.check_output(["sudo", "service", service_name, "stop"])
