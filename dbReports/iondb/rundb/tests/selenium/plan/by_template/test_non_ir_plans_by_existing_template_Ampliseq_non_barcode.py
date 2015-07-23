# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
import time
from datetime import datetime
from django.core.urlresolvers import reverse
from django.test import LiveServerTestCase
from iondb.rundb.test import SeleniumTestCase
from django.conf import settings
import requests

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

from selenium.webdriver.remote.remote_connection import LOGGER
import logging
logger = logging.getLogger(__name__)


import os, sys
try: 
    import simplejson as json
except ImportError: import json

class TestCreatePlansByTemplate(SeleniumTestCase):  

    @classmethod
    def setUpClass(cls):
        super(TestCreatePlansByTemplate, cls).setUpClass()      
        
    def test_create_multi_chip_no_ir_from_non_barcoded_custom_ampliseq_dna_chef_template_plan(self):
        """
        This is a very basic plan wizard
        - To create 2 run plans from an existing user created template
        - each plan has its own sample tube label value
        - check the plans' notes and planStatus values
        - clean up when done
        """

        user = "ionadmin"
        base_url = settings.TEST_SERVER_URL+ "/rundb/api/v1/"
        plugin_url = "plannedexperiment"
        params = "?format=json&planDisplayedName__iexact=Ion AmpliSeq Cancer Panel&limit=1"

        url=base_url + plugin_url + params
        print url
  
        resp = requests.get(url, auth=('ionadmin', 'ionadmin'))
        non_barcoded_template_id= json.loads(resp.content)['objects'][0]['id']
      
        #non_barcoded_template_id=id
        self.open(reverse('page_plan_new_plan', args=(non_barcoded_template_id,)))
         
        #test requires a default IR account be set and TS can establish connection with it
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()
        current_dt = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')  
        self.wd.find_element_by_id("planName").send_keys('PLAN-ampliseq-dna-{0}'.format(current_dt))
 
        self.wd.find_element_by_id('default_targetBedFile').find_elements_by_tag_name('option')[1].click()      
        self.wd.find_element_by_id("numRows").clear()
        self.wd.find_element_by_id("numRows").send_keys('2')
        self.wd.find_element_by_id("numRowsClick").click()
         
        #wait
        self.wd.wait_for_ajax()
        #now navigate to the Kits chevron and choose the Chip-Type
        self.wd.wait_for_css("#Kits").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_css("#chipType").find_elements_by_tag_name('option')[2].click()#(314v2)Ion 314v2 Chip
  
        self.wd.find_element_by_xpath('//input[@name="templatekitType" and @value="IonChef"]').click() 
        time.sleep(2)
                
        select = Select(self.wd.find_element_by_id("templateKit"))
        select.select_by_visible_text("ION PGM IC 200 KIT")
        time.sleep(2)
                
        #now navigate to the Save chevron
        self.wd.find_element_by_id("Save_plan").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_ajax()       
             
        logger.info( ">>> test_create_multi_chip_no_ir_plan_from_non_barcoded_ampliseq_dna_chef_template... non_barcoded_template_id=%d" %(non_barcoded_template_id))
        time.sleep(2)
         
       
         
        #now enter two samples
        sample_name_1 = 'nonBC-Sample-Name-10'
        sample_ext_id_1 = 'nonBC-Sample-External-Id-10'
        sample_description_1 = 'nonBC-Sample-Desc-10'
        sample_tube_label_1 = "selenium_tube_151828"
 
        sample_name_2 = 'nonBC-Sample-Name-11'
        sample_ext_id_2 = 'nonBC-Sample-External-Id-11'
        sample_description_2 = 'nonBC-Sample-Desc-11'
        sample_tube_label_2 = "selenium_tube_A4zk1845"
         
        sampleTable = self.wd.find_element_by_id("chipsets")
         
        trs = sampleTable.find_elements(By.TAG_NAME, "tr")
          
        tds = trs[1].find_elements(By.TAG_NAME, "td")
                 
        for element in tds:
            if element.find_elements(By.NAME, "sampleName"):
                element.find_element_by_name("sampleName").send_keys(sample_name_1)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_1)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_1)
            if element.find_elements(By.NAME, "tubeLabel"):
                element.find_element_by_name("tubeLabel").send_keys(sample_tube_label_1)
                                     
        tds = trs[2].find_elements(By.TAG_NAME, "td")    
         
        for element in tds:
            if element.find_elements(By.NAME, "sampleName"):
                element.find_element_by_name("sampleName").send_keys(sample_name_2)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_2)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_2)
            if element.find_elements(By.NAME, "tubeLabel"):
                element.find_element_by_name("tubeLabel").send_keys(sample_tube_label_2)
                 
        #enter notes for the plan
        PLAN_NOTES = 'selenium no-IR-no-BC test PLAN based on a selenium template'
        self.wd.find_element_by_id("note").clear()  #clear template's notes first
        self.wd.find_element_by_id("note").send_keys(PLAN_NOTES)
        time.sleep(3)
         
        #now save the plan
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_ajax()
 
        self.open(reverse('plan_templates'))
        time.sleep(3)
         
        #now retrieve the latest PlannedExperiment (The Plan you just saved) and verify the data you entered
        latest_plan = self.get_latest_planned_experiment()
         
        ##logger.info( ">>> test_create_multi_chip_no_ir_plan_from_non_barcoded_ampliseq_dna_chef_template... #1 latest_plan=%s" %(latest_plan))
                 
        #verify the notes
        self.assertEqual(latest_plan['notes'], PLAN_NOTES)
        #verify the plan status
        self.assertEqual(latest_plan['planStatus'], 'pending')
        #verify sample tube label
        self.assertEqual(latest_plan['sampleTubeLabel'], sample_tube_label_2)
                 
        #now delete the planned run
        logger.info( ">>> test_create_multi_chip_no_ir_plan_from_non_barcoded_ampliseq_dna_template... - PASS - GOING to delete #1 plan.id=%s" %(latest_plan['id']))
         
        self.delete_planned_experiment(latest_plan['id']) 
        time.sleep(2)
         
        #now retrieve the latest PlannedExperiment (The Plan you just saved) and verify the data you entered
        latest_plan = self.get_latest_planned_experiment()
         
        ##logger.info( ">>> test_create_multi_chip_no_ir_plan_from_non_barcoded_ampliseq_dna_chef_template... #2 latest_plan=%s" %(latest_plan))
                 
        #verify the notes
        self.assertEqual(latest_plan['notes'], PLAN_NOTES)
        #verify the plan status
        self.assertEqual(latest_plan['planStatus'], 'pending')
        #verify sample tube label
        self.assertEqual(latest_plan['sampleTubeLabel'], sample_tube_label_1)
                 
        #now delete the planned run
        logger.info( ">>> test_create_multi_chip_no_ir_plan_from_non_barcoded_ampliseq_dna_template...- PASS - GOING to delete #2 plan.id=%s" %(latest_plan['id']))
         
        self.delete_planned_experiment(latest_plan['id']) 
        time.sleep(2)
                   
        #and then delete the template
        #logger.info( ">>> test_create_multi_chip_no_ir_plan_from_non_barcoded_ampliseq_dna_template...- ALL PASS - GOING to delete template.id=%s" %(non_barcoded_template_id))
        #self.delete_planned_experiment(non_barcoded_template_id)
        #time.sleep(2)
        