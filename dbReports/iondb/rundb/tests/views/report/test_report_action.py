# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from mockito import mock, when, any, verify, contains
from django.test import TestCase
import xmlrpclib
from iondb.rundb.models import Results, Experiment, Message
from datetime import datetime
from django.contrib.auth.models import User
from iondb.rundb.tests.models.test_results import create_result

def verifyMessage(self, status, _resultsName):
    self.assertEqual(1, Message.objects.count(), 'Message should be created.')
    message = Message.objects.latest('time')
    self.assertEqual(self.ionadmin.username, message.route, 'Message not routed to username')
    self.assertTrue(str(message.body).find("(%s)" % _resultsName) >= 0)
    if status:
        self.assertTrue(str(message.body).find("complete") >= 0)
    else:
        self.assertTrue(str(message.body).find("failed") >= 0)


class ReportActionTest(TestCase):
    fixtures = ['iondb/rundb/tests/views/report/fixtures/globalconfig.json', 
                'iondb/rundb/tests/models/fixtures/groups.json', 
                'iondb/rundb/tests/models/fixtures/users.json']

    def setUp(self):
        self.ionadmin = User.objects.get(username='ionadmin')
        self.client.login(username='ionadmin', password='ionadmin')
        self.report = create_result(self)

    def test_only_post_allowed(self):
        ''' test only HTTP POST allowed '''
        _id = 0
        action = '_'
        response = self.client.get('/report/action/%d/%s' %(_id, action))
        self.assertEqual(405, response.status_code)
        
    
    def _verifyMessage(self, status, _resultsName):
        verifyMessage(self, status, _resultsName)

    def test_archive_report_default_comment(self):
        _id = self.report.id
        action = 'A'
        proxy = mock()
        when(xmlrpclib).ServerProxy(any(), allow_none=any()).thenReturn(proxy)
        when(proxy).archive_report(contains(str(_id)), any()).thenReturn(False)
        response = self.client.post('/report/action/%d/%s' %(_id, action))
        self.assertEqual(200, response.status_code)
        verify(proxy).archive_report(contains(str(_id)), contains(str("No Comment")))
        self._verifyMessage(False, self.report.resultsName)
        
    def test_archive_report_with_comment(self):
        comment = 'foo'
        _id = self.report.id
        action = 'A'
        proxy = mock()
        when(xmlrpclib).ServerProxy(any(), allow_none=any()).thenReturn(proxy)
        when(proxy).archive_report(contains(str(_id)), any()).thenReturn(False)
        response = self.client.post('/report/action/%d/%s' %(_id, action), data = {'comment':comment})
        self.assertEqual(200, response.status_code)
        verify(proxy).archive_report(contains(str(_id)), contains(comment))
        self._verifyMessage(False, self.report.resultsName)
    
    def test_export_report(self):
        _id = self.report.id
        action = 'E'
        proxy = mock()
        when(xmlrpclib).ServerProxy(any(), allow_none=any()).thenReturn(proxy)
        when(proxy).export_report(contains(str(_id)), any()).thenReturn(False)
        response = self.client.post('/report/action/%d/%s' %(_id, action))
        self.assertEqual(200, response.status_code)
        verify(proxy).export_report(contains(str(_id)), contains(str("No Comment")))
        self._verifyMessage(False, self.report.resultsName)

    def test_prune_report(self):
        _id = self.report.id
        action = 'P'
        proxy = mock()
        when(xmlrpclib).ServerProxy(any(), allow_none=any()).thenReturn(proxy)
        when(proxy).prune_report(contains(str(_id)), any()).thenReturn(False)
        response = self.client.post('/report/action/%d/%s' %(_id, action))
        self.assertEqual(200, response.status_code)
        verify(proxy).prune_report(contains(str(_id)), contains(str("No Comment")))
        self._verifyMessage(False, self.report.resultsName)

    def test_auto_exempt_404(self):
        _id = 0
        action = 'Z'
        response = self.client.post('/report/action/%d/%s' %(_id, action))
        self.assertEqual(404, response.status_code)
        
    def test_auto_exempt(self):
        _id = self.report.id
        action = 'Z'
        response = self.client.post('/report/action/%d/%s' %(_id, action))
        self.assertEqual(200, response.status_code)
        found = Results.objects.get(id=self.report.id)
        self.assertTrue(found.autoExempt, 'should be True')
        
        response = self.client.post('/report/action/%d/%s' %(_id, action))
        self.assertEqual(200, response.status_code)
        found = Results.objects.get(id=self.report.id)
        self.assertFalse(self.report.autoExempt, 'should be False')
