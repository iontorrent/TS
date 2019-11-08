#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved
""" This script takes a json file containing flow orders and adds to the database,
    if flow order with same name already exists it will be updated.
"""
import os
import sys
import json
from iondb.bin import djangoinit
from iondb.rundb.models import FlowOrder


def add_floworders_from_file(path):
    with open(path) as f:
        contents = json.load(f)

    for item in contents:
        name = item["name"]
        flowOrder = item["flowOrder"]
        description = item.get("description") or name

        obj, created = FlowOrder.objects.get_or_create(
            name=name, defaults={"flowOrder": flowOrder, "description": description}
        )
        if not created:
            obj.flowOrder = flowOrder
            obj.description = description
            obj.save()

        print("Created" if created else "Updated", "flowOrder %s" % name)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: add_floworders.py FILE_PATH")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print("File not found: %s" % path)
        sys.exit(1)

    add_floworders_from_file(path)
