#!/usr/bin/env python
# Copyright (C) 2019 Ion Torrent Systems, Inc. All Rights Reserved


import iondb.bin.djangoinit
import iondb.plugins.tasks as tasks

tasks.add_remove_plugins()
