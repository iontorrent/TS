# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.contrib.auth.models import User
from django.test import TestCase
import json
import logging
logger = logging.getLogger(__name__)


class SavePlanOrTemplate(TestCase):
    fixtures = ['iondb/rundb/tests/models/fixtures/groups.json', 'iondb/rundb/tests/models/fixtures/users.json']
    
    
    def setUp(self):
        self.ionadmin = User.objects.get(username='ionadmin')
        self.client.login(username='ionadmin', password='ionadmin')

    def test_non_post_fails(self):
        _id = 0
        response = self.client.get('/plan/save/%d/' % _id, data = {}, content_type = "application/json")
        self.assertEqual(200, response.status_code)  
        self.assertTrue(str(response).find('{"error": "Error, unsupported HTTP Request method (GET) for plan update."}') >= 0)
    
    def test_savePlan_error_template_name_required(self):
        _id = 0
        data = {'submitIntent':'savePlan'}
        data = json.dumps(data)
        response = self.client.post('/plan/save/%d/' % _id, data = data, content_type = "application/json")
        logger.debug(response)
        self.assertTrue(str(response).find('{"error": "Error, please enter a Run Plan Name."}') >= 0)

    def test_savePlan_template_name_invalidChars(self):
        _id = 0
        data = {'submitIntent':'savePlan', 'planDisplayedName': ' a! '}
        data = json.dumps(data)
        response = self.client.post('/plan/save/%d/' % _id, data = data, content_type = "application/json")
        logger.debug(response)
        self.assertEqual(200, response.status_code)
        self.assertTrue(str(response).find('{"error": "Error, Run Plan Name should contain only numbers, letters, spaces, and the following: . - _"}') >= 0)

    def test_savePlan_template_note_invalidChars(self):
        _id = 0
        data = {'submitIntent':'savePlan', 'planDisplayedName': ' a ', 'notes_workaround': ' a!'}
        data = json.dumps(data)
        response = self.client.post('/plan/save/%d/' % _id, data = data, content_type = "application/json")
        logger.debug(response)
        self.assertEqual(200, response.status_code)
        self.assertTrue(str(response).find('{"error": "Error, Run Plan note should contain only numbers, letters, spaces, and the following: . - _"}') >= 0)
