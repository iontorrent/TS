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
from selenium.webdriver.common.keys import Keys
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
        

    def test_create_no_ir_same_ref_by_barcode_plan_from_ampliseq_template_plan (self):
        """
        Test plan with same ref info for each barcode
        """ 
        template_name="Ion AmpliSeq Colon and Lung Cancer Panel v2"
        base_url = settings.TEST_SERVER_URL+ "/rundb/api/v1/"
        plugin_url = "plannedexperiment"
        params = "?format=json&planDisplayedName__iexact="+template_name+"&limit=1"

        url=base_url + plugin_url + params

  
        resp = requests.get(url, auth=('ionadmin', 'ionadmin'))
        barcoded_template_id= json.loads(resp.content)['objects'][0]['id']
        
        print url +'\t'+str(barcoded_template_id)
        
        self.open(reverse('page_plan_new_plan', args=(barcoded_template_id,)))
        self.wd.wait_for_ajax()
        
        select = Select(self.wd.find_element_by_id("default_reference"))
        select.select_by_visible_text("hg19(Homo sapiens)")

        self.wd.find_element_by_id('default_targetBedFile').find_elements_by_tag_name('option')[1].click()   
        self.wd.wait_for_ajax()
         
        current_dt = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
            
        self.wd.find_element_by_id("planName").send_keys('ref-info-PLAN-target-seq-{0}'.format(current_dt))
        
        self.wd.find_element_by_id("numRows").send_keys(Keys.CONTROL + "a")
        time.sleep(1)
        self.wd.find_element_by_id("numRows").send_keys(Keys.BACKSPACE)
        time.sleep(1)
        self.wd.find_element_by_id("numRows").send_keys('2')
        self.wd.find_element_by_id("numRowsClick").click()
         
        #wait
        self.wd.wait_for_ajax()
         
        SAMPLE_TUBE_LABEL = "selenium_BC tube_9952dfk"
        self.wd.find_element_by_id("barcodeSampleTubeLabel").send_keys(SAMPLE_TUBE_LABEL)
         
        #now enter two samples
        sample_name_1 = 'BC-Sample-Name-10'
        sample_ext_id_1 = 'BC-Sample-External-Id-10'
        sample_description_1 = 'BC-Sample-Desc-10'

        sample_name_2 = 'BC-Sample-Name-11'
        sample_ext_id_2 = 'BC-Sample-External-Id-11'
        sample_description_2 = 'BC-Sample-Desc-11'
         
        sampleTable = self.wd.find_element_by_id("chipsets")
         
        trs = sampleTable.find_elements(By.TAG_NAME, "tr")
          
        tds = trs[1].find_elements(By.TAG_NAME, "td")
                 
        for element in tds:
            if element.find_elements(By.NAME, "sampleName"):
                element.find_element_by_name("sampleName").clear()
                element.find_element_by_name("sampleName").send_keys(sample_name_1)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_1)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_1)
                                     
        tds = trs[2].find_elements(By.TAG_NAME, "td")    
         
        for element in tds:
            if element.find_elements(By.NAME, "sampleName"):
                element.find_element_by_name("sampleName").clear()
                element.find_element_by_name("sampleName").send_keys(sample_name_2)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_2)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_2)
                 
        #enter notes for the plan
        PLAN_NOTES = 'selenium no-IR- same ref info per barcoded sample test PLAN'
        self.wd.find_element_by_id("note").clear()  #clear template's notes first
        self.wd.find_element_by_id("note").send_keys(PLAN_NOTES)
        time.sleep(3)
         
        #now navigate to the Kits chevron and choose the Chip-Type
        self.wd.wait_for_css("#Kits").find_elements_by_tag_name('a')[0].click()
        #self.wd.wait_for_css("#chipType").find_elements_by_tag_name('option')[4].click()#(316v2)Ion 316v2 Chip
        self.wd.find_element_by_xpath("//select[@name='chipType']/option[@value='P1.1.17']").click()
        self.wd.find_element_by_id("OneTouch__templatekitType").click()
        self.wd.wait_for_css("#templateKit").find_elements_by_tag_name('option')[1].click() ##ION PGM Template OT2 200 Kit
        self.wd.wait_for_css("#barcodeId").find_elements_by_tag_name('option')[1].click()#IonSet1

        self.wd.wait_for_css("#Save_plan").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_ajax()                 
        #now save the plan
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_ajax()
  
        self.open(reverse('plan_templates'))
        time.sleep(3)
         
        #now retrieve the latest PlannedExperiment (The Plan you just saved) and verify the data you entered
        latest_plan = self.get_latest_planned_experiment()
                 
        #verify the notes
        self.assertEqual(latest_plan['notes'], PLAN_NOTES)
        #verify the plan status
        self.assertEqual(latest_plan['planStatus'], 'planned')
        #verify sample tube label
        self.assertEqual(latest_plan['sampleTubeLabel'], SAMPLE_TUBE_LABEL)
 
        logger.info( ">>> test_create_no_ir_same_ref_by_barcode_plan_from_target_seq_template... barcodedSamples=%s" %(latest_plan['barcodedSamples']))
          
        #verify the barcoded samples
        self.assertEqual(len(latest_plan['barcodedSamples']), 2)
        barcodes = latest_plan['barcodedSamples'].keys()
            
        if sample_name_1 in barcodes and sample_name_2 in barcodes:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
    
        REFERENCE = latest_plan['library']
        HOT_SPOT_SUB_STRING = latest_plan["regionfile"]      
        TARGET_REGION_SUB_STRING = latest_plan["bedfile"]
        
        sample1 = latest_plan['barcodedSamples'][sample_name_1]
        barcodedSamples = sample1['barcodeSampleInfo']
            
        _ionSet1Dict = barcodedSamples['IonSelect-1']
        self.assertEqual(_ionSet1Dict['reference'], REFERENCE)
        
        expected_match = True        
        actual_match = False
        if HOT_SPOT_SUB_STRING in _ionSet1Dict['hotSpotRegionBedFile']:
            actual_match = True
        self.assertEqual(actual_match, expected_match)
        
        actual_match = False
        if TARGET_REGION_SUB_STRING in _ionSet1Dict['targetRegionBedFile']:
            actual_match = True
        self.assertEqual(actual_match, expected_match)
            
        sample2 = latest_plan['barcodedSamples'][sample_name_2]
        barcodedSamples = sample2['barcodeSampleInfo']
            
        _ionSet2Dict = barcodedSamples['IonSelect-2']
        self.assertEqual(_ionSet1Dict['reference'], REFERENCE) 
        
        expected_match = True        
        actual_match = False
        if HOT_SPOT_SUB_STRING in _ionSet1Dict['hotSpotRegionBedFile']:
            actual_match = True
        self.assertEqual(actual_match, expected_match)
        
        actual_match = False
        if TARGET_REGION_SUB_STRING in _ionSet1Dict['targetRegionBedFile']:
            actual_match = True
        self.assertEqual(actual_match, expected_match)
    
        #now delete the planned run
        logger.info( ">>> test_create_no_ir_same_ref_by_barcode_plan_from_target_seq_template... GOING to delete plan.id=%s" %(latest_plan['id']))         
        self.delete_planned_experiment(latest_plan['id'])   
           
        time.sleep(3)
