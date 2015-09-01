# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import urllib2
import httplib
import base64

from django.contrib.auth.models import User
from django.conf import settings
from django.utils import simplejson
from django.test import Client
from django.test import LiveServerTestCase
from django.core.urlresolvers import reverse

from selenium.webdriver.remote.remote_connection import LOGGER

import logging
logger = logging.getLogger(__name__)


from selenium.common.exceptions import NoSuchElementException

from iondb.rundb.tests.selenium.webdriver import CustomWebDriver

class SeleniumTestCase(LiveServerTestCase):
    """
    A base test case for selenium, providing hepler methods for generating
    clients and logging in profiles.

    Since we are using LiveServerTestCase, the database is created and torn down for each
    of the individual tests, therefore, it's safe to add a user in the setUp function
    """

    @classmethod
    def setUpClass(cls):
        if settings.GUIDISPLAY == False:
            settings.DISPLAY.start()  
        LOGGER.setLevel(logging.WARNING)
        
        # Instantiating the WebDriver will load your browser
        cls.wd = CustomWebDriver()
        cls.server_url = settings.TEST_SERVER_URL
        ##cls.server_url = 'http://ts-sandbox.itw'
        cls.wd.get("%s%s" % (cls.server_url, '/login'))
        ##the login now persists between tests
        ##so we only need to login with the username/password if the page 
        ##has the username fields
        try:
            #enter the username and password
            cls.wd.find_css('#id_username').send_keys('ionadmin')
            cls.wd.find_css("#id_password").send_keys('ionadmin')
            #click the login link
            cls.wd.find_element_by_xpath('//button[@type="submit"]').click()
            #wait for the Ajax on the HOME page
            cls.wd.wait_for_ajax()
        except NoSuchElementException:
            pass


    def delete_planned_experiment(self, pk):
        logger.info(">>>> Going to delete_planned_experiment... pk=%s" %(str(pk)))
        
        self.latest_pe_api_url = '/rundb/api/v1/plannedexperiment/{0}/'.format(pk)
        
        host = settings.TEST_SERVER_URL
        url = host + self.latest_pe_api_url
        request = urllib2.Request(url)
        
        base64String = base64.encodestring("%s:%s" %("ionadmin", "ionadmin")).replace("\n", "")
        request.add_header("Authorization", "Basic %s" %(base64String))
        request.get_method = lambda : 'DELETE'
        response = urllib2.urlopen(request)

        #logger.info(">>>> delete_planned_experiment... DELETE response=%s" %(response))
        
#        json = simplejson.loads(response.read())
#        return json['objects'][0]

    def get_latest_all_nth_planned_experiment(self,n): 
        try:
            self.latest_pe_api_url = '/rundb/api/v1/plannedexperiment/?format=json&order_by=-id'+"&limit="+str(n)

            host = settings.TEST_SERVER_URL
            url = host + self.latest_pe_api_url
            request = urllib2.Request(url)

            base64String = base64.encodestring("%s:%s" %("ionadmin", "ionadmin")).replace("\n", "")
            request.add_header("Authorization", "Basic %s" %(base64String))
            
            response = urllib2.urlopen(request)
            json = simplejson.loads(response.read())
            return json['objects']
        except Exception, e:
            #TO-DO: do something with the exception
            raise e
        
    def get_latest_planned_experiment(self):
        try:
            self.latest_pe_api_url = '/rundb/api/v1/plannedexperiment/?format=json&order_by=-id'

            host = settings.TEST_SERVER_URL
            url = host + self.latest_pe_api_url
            request = urllib2.Request(url)

            base64String = base64.encodestring("%s:%s" %("ionadmin", "ionadmin")).replace("\n", "")
            request.add_header("Authorization", "Basic %s" %(base64String))
            
            response = urllib2.urlopen(request)
            json = simplejson.loads(response.read())
            return json['objects'][0]
        except Exception, e:
            #TO-DO: do something with the exception
            raise e

    @classmethod
    def tearDownClass(cls):
        if settings.GUIDISPLAY == False:
            settings.DISPLAY.stop()
        cls.wd.quit()

    def open(self, url):
        self.wd.get("%s%s" % (self.server_url, url))