# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from django.test import TestCase
from iondb.rundb.models import Project
from django.contrib.auth.models import User
from django.db.utils import IntegrityError, DatabaseError

def bulk_get_or_create_create_multi_projects(self):
    name = [' a ', 'b', '', None, ' ', ' a b ']
    projects = Project.bulk_get_or_create(name)
    self.assertEqual(len(projects), 3, 'Incorrect number of projects created')
    Project.objects.get(name='a')
    Project.objects.get(name='b')
    Project.objects.get(name='a_b')
    return projects

class ProjectTest(TestCase):
    fixtures = ['iondb/rundb/tests/models/fixtures/groups.json', 'iondb/rundb/tests/models/fixtures/users.json']
    
    
    def setUp(self):
        self.ionadmin = User.objects.get(username='ionadmin')
    
    
    def test_save(self):
        project = Project()
        project.name = 'abc'
        project.creator = self.ionadmin
        
        project.save()
        
        self.assertIsNotNone(project.id, 'Project id is None')
        return project
        
        
    def test_unicode(self):
        proj = self.test_save()
        self.assertEquals(unicode(proj), proj.name)
        
        
    def test_name_unique(self):
        p1 = Project.objects.create(name='abc', creator=self.ionadmin)
        with self.assertRaises(IntegrityError):
            p2 = Project.objects.create(name='abc', creator=self.ionadmin)

    def test_name_maxlength(self):
        max = 64
        with self.assertRaises(DatabaseError):
            p1 = Project.objects.create(name='a'* (max + 1), creator=self.ionadmin)
        
        
    def test_bulk_get_or_create_create_multi_projects(self):
        return bulk_get_or_create_create_multi_projects(self)
        
        
    def _test_bulk_get_or_create_create_single_without_user(self, name = 'a'):
        projects = Project.bulk_get_or_create(name)
        self.assertIsNotNone(projects[0].id, 'Project id is None')
        user = User.objects.order_by('pk')[0]
        self.assertEqual(projects[0].creator, user, 'Default user not same')
        print 'created %s ' % projects[0].id
        
        
    def test_bulk_get_or_create_create_single_without_user(self, name = 'a'):
        self.assertEquals(0, len(Project.objects.all()), 'Not empty')
        self._test_bulk_get_or_create_create_single_without_user(name)
        
        
    def test_bulk_get_or_create_get_single_without_user(self):
        created = self.test_save()
        self._test_bulk_get_or_create_create_single_without_user(name = created.name)
        
        