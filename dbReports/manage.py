#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import os
import sys

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "iondb.settings")

    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)
