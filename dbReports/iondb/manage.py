#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import os
import sys

if __name__ == "__main__":
    import logging
    logger = logging.getLogger(__name__)
    logger.info("WARNING: Legacy manage script running. Please use /opt/ion/manage.py instead.")
    #execute_manager(settings)

    ## Newer style manage script - Django 1.4+
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "iondb.settings")

    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)

