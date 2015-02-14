#!/usr/bin/env python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
'''
Write Test Fragment templates to the database Template table.  This uses the existing
json file template_init.json but ignore everything except the fields values to create/update
the database records.  This avoids having to hard-code the pk values in the json.
'''
import sys

import iondb.bin.djangoinit
from iondb.rundb.models import Template
from django.core import serializers

def main(filename):
    '''Main function which adds, or updates test fragment objects'''
    with open(filename) as json_data:
        objects = serializers.deserialize('json', json_data)

        for obj in objects:
            try:
                existing = Template.objects.get(name=obj.object.name)
                # Use the pk for the matching object name
                obj.object.pk = existing.pk
                created = False
            except Template.DoesNotExist:
                # Ignore the pk in the file and insert as new record
                obj.object.pk = None
                created = True
            except Template.MultipleObjectsReturned:
                first = True
                for existing in Template.objects.filter(name=obj.object.name):
                    if first:
                        obj.object.pk = existing.pk
                        created = False
                        first = False
                    else:
                        # Purge entries with a duplicate name
                        print("Removing Duplicate entry %d for %s" % (existing.id, existing.name))
                        existing.delete()

            obj.save()

            print("%s %s test fragment" % ("Added new" if created else "Updated", obj.object.name))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Need to specify a filename"
        sys.exit(1)
    else:
        sys.exit(main(sys.argv[1]))
