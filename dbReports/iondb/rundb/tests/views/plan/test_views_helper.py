# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.test import TestCase
from iondb.rundb.plan.views_helper import get_projects, get_projects_helper
from iondb.rundb.models import Project
from django.contrib.auth.models import User
from iondb.rundb.tests.models import test_project
import logging
logger = logging.getLogger(__name__)


class SavePlanOrTemplateHelpers(TestCase):
    fixtures = ['iondb/rundb/tests/models/fixtures/groups.json', 'iondb/rundb/tests/models/fixtures/users.json']

    
    def setUp(self):
        self.ionadmin = User.objects.get(username='ionadmin')


    def test_get_projects(self):
        data = {'projects':[],
                'newProjects': []}
#        data = simplejson.loads(data)
        result =  get_projects('ionadmin', data)
        self.assertFalse(result, 'list should be empty')


    def test_get_projects_new(self):
        newProjects = ','.join([' a ', 'b', '', ' ', ' a b '])
        data = {'projects':[],
                'newProjects': newProjects}
        result = get_projects('ionadmin', data)
        self.assertEqual(len(result), 3, 'Incorrect number of projects created')
        Project.objects.get(name='a')
        Project.objects.get(name='b')
        Project.objects.get(name='a_b')


    def test_get_projects_helper(self):
        projects = test_project.bulk_get_or_create_create_multi_projects(self)
        self.assertEqual(len(projects), 3, 'Incorrect number of projects created')
        Project.objects.get(name='a')
        Project.objects.get(name='b')
        Project.objects.get(name='a_b')
        projectNameAndIds = [str(p.id) + '|' + p.name for p in projects]
        logger.info(projectNameAndIds)
        projectNameAndIds = ','.join(projectNameAndIds)
        result, missing = get_projects_helper(projectNameAndIds)
        self.assertFalse(len(missing),'No projects should be missing - %s ' % missing)
        self.assertEqual(len(result), len(projects), 'Incorrect number of projects found')
        Project.objects.get(name='a')
        Project.objects.get(name='b')
        Project.objects.get(name='a_b')


    def test_get_projects_existing(self):
        projects = test_project.bulk_get_or_create_create_multi_projects(self)
        self.assertEqual(len(projects), 3, 'Incorrect number of projects created')
        Project.objects.get(name='a')
        Project.objects.get(name='b')
        Project.objects.get(name='a_b')
        projectNameAndIds = [str(p.id) + '|' + p.name for p in projects]
        
        data = {'projects':projectNameAndIds,
                'newProjects': []}
        result = get_projects('ionadmin', data)
        self.assertEqual(len(result), len(projects), 'Incorrect number of projects retrieved')
        Project.objects.get(name='a')
        Project.objects.get(name='b')
        Project.objects.get(name='a_b')
