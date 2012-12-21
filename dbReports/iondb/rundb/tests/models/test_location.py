# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.test import TestCase
from iondb.rundb.models import Location

def save_location(self):
    l = Location()
    l.defaultlocation = True
    l.name = 'HomeTest'
    l.save()
    self.assertIsNotNone(l.id, 'Location id is None')
    return l

class LocationModelTest(TestCase):

    def test_save(self):
        l = save_location(self)
        return l
    
    def test_save_method(self):
        l = Location()
        l.defaultlocation = True
        l.name = 'HomeTest'
        l.save()
        _id = l.id

        l = Location()
        l.defaultlocation = True
        l.name = 'HomeTest'
        l.save()

        for l in Location.objects.exclude(id=l.id):
            self.assertFalse(l.defaultlocation, 'Should not be default location, only one is allowed')

    def test_unicode(self):
        l = self.test_save()
        self.assertEquals(unicode(l), "%s" % (l.name))
