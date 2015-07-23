# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
import time
from datetime import datetime
from django.core.urlresolvers import reverse
from django.test import LiveServerTestCase
from iondb.rundb.test import SeleniumTestCase

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys

from selenium.webdriver.remote.remote_connection import LOGGER
import logging
logger = logging.getLogger(__name__)


import os

class TestCreatePlansByTemplate(SeleniumTestCase):  

    @classmethod
    def setUpClass(cls):
        super(TestCreatePlansByTemplate, cls).setUpClass()      
        
    def test_create_csv_plan_with_plugins(self): 
        """
        TS-7986: validation failed when csv file has plugins
        """
         
        self.open(reverse('plan_templates'))
         
        #now we need to wait for the page to load
        self.wd.wait_for_ajax()
 
        #button click on Upload Plans
        self.wd.find_element_by_link_text("Upload Plans").click()
        time.sleep(3)
                 
        #change directory: go to the dbReports home directory
        home_dir = os.path.abspath(os.path.join(os.path.join(os.path.join(os.getcwd()))))
 
        self.wd.find_css("#postedfile").send_keys(home_dir +"/iondb/rundb/tests/test_data/autoTest_batchPlanning_selectedPlugins.csv")
         
        #button click on Upload CSV for batch planning button
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary submitUpload"]').click()
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
         
  
     
    def test_create_csv_chef_plan_with_chefInfo(self): 
        """
         A non-barcoded Chef plan with 
         - Chef templating kit, 
         - templating size, 
         - library read length and 
         - tube label specified
         The saved planStatus should be pending
        """
       
        self.open(reverse('plan_templates'))
         
        #now we need to wait for the page to load
        self.wd.wait_for_ajax()
 
        #button click on Upload Plans
        self.wd.find_element_by_link_text("Upload Plans").click()
        time.sleep(3)
                 
        #change directory: go to the dbReports home directory
        home_dir = os.path.abspath(os.path.join(os.path.join(os.path.join(os.getcwd()))))
 
        self.wd.find_css("#postedfile").send_keys(home_dir +"/iondb/rundb/tests/test_data/autoTest_batchPlanning_chef.csv")
         
        #button click on Upload CSV for batch planning button
        ##ElementNotVisibleException: Message: u'Element is not currently visible and so may not be interacted with' ; Stacktrace: 
        ##self.wd.find_element_by_xpath('//a[@class="btn btn-primary submitUpload"]').click()
  
        ##workaround - use find_element_by_id instead
        self.wd.find_element_by_id("submitUpload").click()
         
        self.wd.wait_for_ajax()
 
        uploaded_plan = self.get_latest_planned_experiment()
         
        #examine the chip value
        chip = uploaded_plan["chipType"]
        #customer facing namer is 318v2 while internal chip name is 318
        self.assertEqual(chip, "318")
         
        #examine the templating kit value
        templating_kit = uploaded_plan["templatingKitName"]
        self.assertEqual(templating_kit, "Ion PGM Hi-Q Chef Kit")
         
        #examine the templating size
        templating_size = uploaded_plan["templatingSize"]
        self.assertEqual(int(templating_size), 400)
         
        #examine the library read length
        library_read_length = uploaded_plan["libraryReadLength"]
        self.assertEqual(int(library_read_length), 550)
         
        #examine the flow count
        flows = uploaded_plan["flows"]
        self.assertEqual(int(flows), 1100)
        
        #examine the sample tube label
        sample_tube_label = uploaded_plan["sampleTubeLabel"]
        self.assertEqual(sample_tube_label, "csv TL 2015X5J")
        
        #examine the plan status
        plan_status = uploaded_plan["planStatus"]
        self.assertEqual(plan_status, "pending")
             
             
        #tear down the uploaded plan 
        logger.info( ">>> test_create_csv_chef_plan_with_chefInfo...- ALL PASS - GOING to delete plan.id=%s" %(uploaded_plan['id']))
        self.delete_planned_experiment(uploaded_plan['id'])  
         
 
 
    def test_create_multi_chip_no_ir_plan_from_non_barcoded_ampliseq_dna_chef_template(self):
        """
        This is a very basic plan wizard
        - create a Chef non-IR, non-barcoded, no reference template for AmpliSeq DNA application
        - create 2 plans from this template
        - each plan has its own sample tube label value
        - check the plans' notes and planStatus values
        - clean up when done
        """
 
        self.open(reverse('page_plan_new_template', args=(1,)))
        ##self.open(reverse('page_plan_new_plan_from_code', args=(1,)))
         
        #test requires a default IR account be set and TS can establish connection with it
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()
 
        #select None for IR account
        self.wd.find_element_by_xpath('//input[@name="irOptions" and @value="0"]').click() #None
        #select Self as Sample Grouping
        self.wd.find_element_by_xpath('//input[@name="sampleGrouping" and @value="2"]').click() #Self
        #wait
        self.wd.wait_for_ajax()
 
        #now navigate to the Kits chevron and choose the Chip-Type
        self.wd.wait_for_css("#Kits").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_css("#chipType").find_elements_by_tag_name('option')[2].click()#(314v2)Ion 314v2 Chip
  
        self.wd.find_element_by_xpath('//input[@name="templatekitType" and @value="IonChef"]').click() 
        time.sleep(2)
                
        ##self.wd.wait_for_css("#templateKit").find_elements_by_tag_name('option')[1].click() ##ION PGM IC 200 KIT
        ##self.wd.find_element_by_id("templateKit").find_elements_by_tag_name('option')[1].click() ##ION PGM IC 200 KIT
        select = Select(self.wd.find_element_by_id("templateKit"))
        select.select_by_visible_text("ION PGM IC 200 KIT")
        time.sleep(2)
                
        #now navigate to the Save chevron
        self.wd.find_element_by_id("Save_template").find_elements_by_tag_name('a')[0].click()
         
        #select None for reference
        self.wd.find_element_by_id("default_reference").find_elements_by_tag_name('option')[0].click() ##None
         
        current_dt = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
        self.wd.find_element_by_id("templateName").send_keys('ampliseq-dna-{0}'.format(current_dt))
 
        TEMPLATE_NOTES = 'selenium no-IR-no-BC test template'
        self.wd.find_element_by_id("note").send_keys(TEMPLATE_NOTES)        
        time.sleep(2)
                 
        #now save the template
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
 
        non_barcoded_template_id = self.get_latest_planned_experiment()['id']
 
        logger.info( ">>> test_create_multi_chip_no_ir_plan_from_non_barcoded_ampliseq_dna_chef_template... non_barcoded_template_id=%d" %(non_barcoded_template_id))
         
        #now open the plan run link
        self.open(reverse('page_plan_new_plan', args=(non_barcoded_template_id,)))
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()
         
        self.wd.find_element_by_id("planName").send_keys('PLAN-ampliseq-dna-{0}'.format(current_dt))
 
        self.wd.find_element_by_id("numRows").clear()
        self.wd.find_element_by_id("numRows").send_keys('2')
        self.wd.find_element_by_id("numRowsClick").click()
         
        #wait
        self.wd.wait_for_ajax()
         
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
        logger.info( ">>> test_create_multi_chip_no_ir_plan_from_non_barcoded_ampliseq_dna_template...- ALL PASS - GOING to delete template.id=%s" %(non_barcoded_template_id))
        self.delete_planned_experiment(non_barcoded_template_id)
        time.sleep(2)
        


    def test_create_no_ir_same_ref_by_barcode_plan_from_target_seq_template(self):
        """
        Test plan with same ref info for each barcode
        """ 
        self.open(reverse('page_plan_new_template', args=(2,)))
        ##self.open(reverse('page_plan_new_plan_from_code', args=(2,)))
         
        #test requires a default IR account be set and TS can establish connection with it
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()
 
        #select None for IR account
        self.wd.find_element_by_xpath('//input[@name="irOptions" and @value="0"]').click() #None
        #select Self as Sample Grouping
        self.wd.find_element_by_xpath('//input[@name="sampleGrouping" and @value="2"]').click() #Self
        #wait
        self.wd.wait_for_ajax()
 
        #now navigate to the Kits chevron and choose the Chip-Type
        self.wd.wait_for_css("#Kits").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_css("#chipType").find_elements_by_tag_name('option')[4].click()#(316v2)Ion 316v2 Chip
   
        ##self.wd.wait_for_css("#OneTouch__templatekitType").click()
        self.wd.find_element_by_id("OneTouch__templatekitType").click()
        self.wd.wait_for_css("#templateKit").find_elements_by_tag_name('option')[1].click() ##ION PGM Template OT2 200 Kit
        self.wd.wait_for_css("#barcodeId").find_elements_by_tag_name('option')[1].click()#IonSet1
                         
        #now navigate to the Save chevron
        self.wd.find_element_by_id("Save_template").find_elements_by_tag_name('a')[0].click()
         
        #select ecoli for reference
        #self.wd.find_element_by_id("default_reference").find_elements_by_tag_name('option')[0].click() ##None
        select = Select(self.wd.find_element_by_id("default_reference"))
        select.select_by_visible_text("e_coli_dh10b(E. coli DH10B)")
         
        select = Select(self.wd.find_element_by_id("default_targetBedFile"))
        select.select_by_visible_text("ecoli.bed")
         
        #select = Select(self.wd.find_element_by_id("default_hotSpotBedFile"))
        #select.select_by_visible_text("ecoli_hotspot.bed")
         
        current_dt = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
        self.wd.find_element_by_id("templateName").send_keys('target-seq-{0}'.format(current_dt))
 
        TEMPLATE_NOTES = 'selenium no-IR-BC target seq template'
        self.wd.find_element_by_id("note").send_keys(TEMPLATE_NOTES)        
        time.sleep(2)
                 
        #now save the template
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
 
        barcoded_template_id = self.get_latest_planned_experiment()['id']
 
        logger.info( ">>> test_create_no_ir_same_ref_by_barcode_plan_from_target_seq_template... barcoded_template_id=%d" %(barcoded_template_id))
         
        #now open the plan run link
        self.open(reverse('page_plan_new_plan', args=(barcoded_template_id,)))
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()
         
        self.wd.find_element_by_id("planName").send_keys('ref-info-PLAN-target-seq-{0}'.format(current_dt))
        
        numRows_control = self.wd.find_element_by_id("numRows")
        logger.info("test_create_no_ir_same_ref_by_barcode_plan_from_target_seq_template B4 should be 1 numRows=%s" %(numRows_control.get_attribute('value')))

        #20150304-problem: this could be Firefox and Selenium compatibility issue; clearing a numeric input results in NaN
        #http://stackoverflow.com/questions/23412912/selenium-send-keys-doesnt-work-if-input-type-number
        ##numRows_control.clear()
        #20150304-workaround: use backspace instead of clear
        numRows_control.send_keys(Keys.BACKSPACE)
        #logger.info("test_create_no_ir_same_ref_by_barcode_plan_from_target_seq_template AFTER CLEAR numRows=%s" %(numRows_control.get_attribute('value')))

        numRows_control.send_keys('2')
        logger.info("test_create_no_ir_same_ref_by_barcode_plan_from_target_seq_template AFTER SEND_KEYS numRows=%s" %(numRows_control.get_attribute('value')))
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
            
        _ionSet1Dict = barcodedSamples['IonSet1_01']
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
            
        _ionSet2Dict = barcodedSamples['IonSet1_02']
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
           
        #and then delete the template
        logger.info( ">>> test_create_no_ir_same_ref_by_barcode_plan_from_target_seq_template... - ALL PASS - GOING to delete template.id=%s" %(barcoded_template_id))        
        self.delete_planned_experiment(barcoded_template_id)
        time.sleep(3)

        


