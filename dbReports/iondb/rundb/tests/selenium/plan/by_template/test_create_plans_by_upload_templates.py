# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
import time
from datetime import datetime
from django.core.urlresolvers import reverse
from django.test import LiveServerTestCase
from iondb.rundb.test import SeleniumTestCase

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

from selenium.webdriver.remote.remote_connection import LOGGER
import logging
logger = logging.getLogger(__name__)


import os
import glob


class TestCreatePlansCsvUpload(SeleniumTestCase):  

    @classmethod
    def setUpClass(cls):
        super(TestCreatePlansCsvUpload, cls).setUpClass()
    
    def test_create_csv_plans(self):
        home_dir = os.path.abspath(os.path.join(os.path.join(os.path.join(os.getcwd()))))
        csvdir="/iondb/rundb/tests/test_data/"
        csvfilepaths= glob.glob(home_dir+csvdir+"/*.csv")
        for csvfilepath in csvfilepaths:
            self.test_create_csv_plan_with_plugins(csvfilepath=None)    
        
    def test_create_csv_plan_with_plugins(self,csvfilepath=None): 
        """
        TS-7986: validation failed when csv file has plugins
        """
          
        self.open(reverse('plan_templates'))
          
        #now we need to wait for the page to load
        self.wd.wait_for_ajax()
  
        #button click on Upload Plans
        self.wd.find_element_by_link_text("Upload Plans").click()
        time.sleep(3)
        if not csvfilepath:
            #change directory: go to the dbReports home directory
            home_dir = os.path.abspath(os.path.join(os.path.join(os.path.join(os.getcwd()))))            
            csvfilepath=home_dir +"/iondb/rundb/tests/test_data/autoTest_batchPlanning_selectedPlugins.csv"

  
        self.wd.find_css("#postedfile").send_keys(csvfilepath)
          
        #button click on Upload CSV for batch planning button
        #self.wd.find_element_by_xpath('//a[@class="btn btn-primary submitUpload"]').click()
        self.wd.find_element_by_id("submitUpload").click()
        self.wd.wait_for_ajax()
  
        uploaded_plan = self.get_latest_planned_experiment()
        #now examine the selectedPlugins blob for IonReporterUploader
        selected_plugins = uploaded_plan['selectedPlugins']
        logger.info("test_create_csv_plan_with_plugins... selected_plugins=%s" %(selected_plugins))
                    
        if 'variantCaller' in selected_plugins:
            self.assertTrue(True)
              
            if 'coverageAnalysis' in selected_plugins:
                self.assertTrue(True)
            else:
                self.assertTrue(False)                
        else:
            self.assertTrue(False)
              
        #tear down the uploaded plan 
        logger.info( ">>> test_create_csv_plan_with_plugins...- ALL PASS - GOING to delete plan.id=%s" %(uploaded_plan['id']))
        self.delete_planned_experiment(uploaded_plan['id'])  
          
          