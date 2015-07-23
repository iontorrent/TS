# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
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

class TestCreatePlansByTemplate(SeleniumTestCase):  

    @classmethod
    def setUpClass(cls):
        super(TestCreatePlansByTemplate, cls).setUpClass()
        

    def test_create_ir_plan_from_barcoded_ampliseq_dna_template(self):
        """
        Test happy path - create an IR, barcoded AmpliSeq DNA plan from a created-from-scratch template
        - no reference selected
        Verify what has been persisted in
        - barcodedSamples JSON
        - selectedPlugins JSON
        """          
        self.open(reverse('page_plan_new_template', args=(1,)))
           
        #test requires a default IR account be set and TS can establish connection with it
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()
        #select the Tumor_Normal Sample GroupingTumor_Normal
        self.wd.find_element_by_xpath('//input[@name="sampleGrouping" and @value="3"]').click() #Tumor_Normal
        #wait
        self.wd.wait_for_ajax()
           
        irworkflow = self.wd.find_element_by_xpath('//select[@name="irworkflow"]')
        #choose the third workflow option
        irworkflow.find_elements_by_tag_name('option')[3].click()#AmpliSeq CCP tumor-normal pair
        time.sleep(3)
          
        #now navigate to the Kits chevron and choose the Chip-Type
        self.wd.wait_for_css("#Kits").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_css("#chipType").find_elements_by_tag_name('option')[2].click()#(314v2)Ion 314v2 Chip
        self.wd.wait_for_css("#barcodeId").find_elements_by_tag_name('option')[1].click()#IonSet1
    
        self.wd.find_element_by_xpath('//input[@name="templatekitType" and @value="IonChef"]').click()
        time.sleep(2)
        
                
        select = Select(self.wd.find_element_by_id("templateKit"))
        select.select_by_visible_text("ION PGM IC 200 KIT")
        time.sleep(2)
        #now navigate to the Save chevron
        self.wd.find_element_by_id("Save_template").find_elements_by_tag_name('a')[0].click()

        #select None for reference
        self.wd.find_element_by_id("default_reference").find_elements_by_tag_name('option')[0].click() ##None
           
        current_dt = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
        self.wd.find_element_by_id("templateName").send_keys('ampliseq-dna-{0}'.format(current_dt))
        self.wd.find_element_by_id("note").send_keys('this is a selenium test template')        
                   
        #now save the template
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
   
        barcoded_template_id = self.get_latest_planned_experiment()['id']
   
        logger.info(">>> test_create_ir_plan_from_barcoded_ampliseq_dna_template barcoded_template_id=%d" %(barcoded_template_id))
           
        #now open the plan run link
        self.open(reverse('page_plan_new_plan', args=(barcoded_template_id,)))
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()
        self.wd.find_element_by_id("planName").clear()
        self.wd.find_element_by_id("planName").send_keys('PLAN-ampliseq-dna-{0}'.format(current_dt))

        self.wd.find_element_by_id("numRows").clear()
        self.wd.find_element_by_id("numRows").send_keys('2')
        self.wd.find_element_by_id("numRowsClick").click()
        time.sleep(3)
  
        #wait
        self.wd.wait_for_ajax()
           
        #now enter two samples
        sample_name_1 = 'Barcoded-Sample-Name-10'
        sample_ext_id_1 = 'Barcoded-Sample-External-Id-10'
        sample_description_1 = 'Barcoded-Sample-Desc-10'
   
        sample_name_2 = 'Barcoded-Sample-Name-11'
        sample_ext_id_2 = 'Barcoded-Sample-External-Id-11'
        sample_description_2 = 'Barcoded-Sample-Desc-11'
   
        barcodedSampleTable = self.wd.find_element_by_id("chipsets")
           
        trs = barcodedSampleTable.find_elements(By.TAG_NAME, "tr")
            
        tds = trs[1].find_elements(By.TAG_NAME, "td")
        
        #logger.info( "test_create_ir_plan_from_barcoded_ampliseq_dna_template... tds[1]=%s" %(tds))
           
        for element in tds:
            if element.find_elements(By.NAME, "sampleName"):
                #clear auto-generated sample name first
                element.find_element_by_name("sampleName").clear()
                element.find_element_by_name("sampleName").send_keys(sample_name_1)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_1)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_1)

            if element.find_elements(By.NAME, "irWorkflow"):
                ##element.find_element_by_name("irWorkflow").find_elements_by_tag_name('option')[1].click()#non-Upload Only workflow
                select = Select(element.find_element_by_name("irWorkflow"))
                #select.select_by_visible_text("AmpliSeq Colon-Lung tumor-normal pair")
                select.select_by_visible_text("AmpliSeq CCP tumor-normal pair")
   
            if element.find_elements(By.NAME, "irGender"):
                element.find_element_by_name("irGender").find_elements_by_tag_name('option')[1].click()#Male
            if element.find_elements(By.NAME, "irRelationRole"):
                element.find_element_by_name("irRelationRole").find_elements_by_tag_name('option')[1].click()#Tumor
               
            #ElementNotVisibleException: Message: Element is not currently visible and so may not be interacted with
#            if element.find_elements(By.NAME, "irRelationshipType"):
#                element.find_element_by_name("irRelationshipType").send_keys("Tumor_Normal")            
            if element.find_elements(By.NAME, "irSetID"):
                element.find_element_by_name("irSetID").send_keys("1")
                #wait
                self.wd.wait_for_ajax()
                       
        tds = trs[2].find_elements(By.TAG_NAME, "td")    

        for element in tds:
            if element.find_elements(By.NAME, "sampleName"):
                #clear auto-generated sample name first
                element.find_element_by_name("sampleName").clear()                
                element.find_element_by_name("sampleName").send_keys(sample_name_2)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_2)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_2)

            if element.find_elements(By.NAME, "irWorkflow"):
                ##element.find_element_by_name("irWorkflow").find_elements_by_tag_name('option')[1].click()#non-Upload Only workflow
                select = Select(element.find_element_by_name("irWorkflow"))
                #select.select_by_visible_text("AmpliSeq Colon-Lung tumor-normal pair")
                select.select_by_visible_text("AmpliSeq CCP tumor-normal pair")
   
            if element.find_elements(By.NAME, "irGender"):
                element.find_element_by_name("irGender").find_elements_by_tag_name('option')[2].click()#Female
            if element.find_elements(By.NAME, "irRelationRole"):
                element.find_element_by_name("irRelationRole").find_elements_by_tag_name('option')[2].click()#Normal
                   
            #ElementNotVisibleException: Message: Element is not currently visible and so may not be interacted with                
#            if element.find_elements(By.NAME, "irRelationshipType"):
#                element.find_element_by_name("irRelationshipType").send_keys("Tumor_Normal")                
            if element.find_elements(By.NAME, "irSetID"):
                element.find_element_by_name("irSetID").send_keys("1")
                #wait
                self.wd.wait_for_ajax()
   
        #enter notes for the plan
        self.wd.find_element_by_id("note").clear()  #clear template's notes first
        self.wd.find_element_by_id("note").send_keys('this is a selenium test plan based on a selenium template')
           
        #now save the plan
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_ajax()
   
        self.open(reverse('plan_templates'))
        time.sleep(3)
           
        #now retrieve the latest PlannedExperiment (The Plan you just saved) and verify the data you entered
        latest_plan = self.get_latest_planned_experiment()
           
        #verify the barcoded samples
        self.assertEqual(len(latest_plan['barcodedSamples']), 2)
        barcodes = latest_plan['barcodedSamples'].keys()

        logger.info(">>> test_create_ir_plan_from_barcoded_ampliseq_dna_template...  barcodedSamples.keys=%s" %(barcodes))
          
        if sample_name_1 in barcodes and sample_name_2 in barcodes:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
   
        sample1 = latest_plan['barcodedSamples'][sample_name_1]
        barcodedSamples = sample1['barcodeSampleInfo']
           
        logger.info( "test_create_ir_plan_from_barcoded_ampliseq_dna_template... ASSERT.... barcodedSample1 - barcodeSampleInfo=%s" %(barcodedSamples))
           
        _ionSet1Dict = barcodedSamples['IonSet1_01']
        self.assertEqual(_ionSet1Dict['description'], sample_description_1)
        self.assertEqual(_ionSet1Dict['externalId'], sample_ext_id_1)
   
        sample2 = latest_plan['barcodedSamples'][sample_name_2]
        barcodedSamples = sample2['barcodeSampleInfo']
           
        _ionSet2Dict = barcodedSamples['IonSet1_02']
        self.assertEqual(_ionSet2Dict['description'], sample_description_2)
        self.assertEqual(_ionSet2Dict['externalId'], sample_ext_id_2)  
   
        #now examine the selectedPlugins blob for IonReporterUploader
        selected_plugins = latest_plan['selectedPlugins']
   
        logger.info( "test_create_ir_plan_from_barcoded_ampliseq_dna_template... selected_plugins=%s" %(selected_plugins))
   
        if 'IonReporterUploader' in selected_plugins:
            iru = selected_plugins['IonReporterUploader']
            if 'userInput' in iru:
                ui = iru['userInput']
                if 'userInputInfo' in ui:
                    uii = ui['userInputInfo']
                    if len(uii) != 2: self.assertTrue(False)
   
                    index = 0
                    for _d in uii:
                        index += 1
                        #make sure all the keys are in each dictionary of the userInputInfo list
                        if not 'ApplicationType' in _d or not 'Gender' in _d or not 'Relation' in _d\
                            or not 'RelationRole' in _d or not 'Workflow' in _d or not 'barcodeId' in _d or not 'sample' in _d \
                            or not 'sampleDescription' in _d or not 'sampleExternalId' in _d or not 'sampleName' in _d \
                            or not 'setid' in _d:
                            self.assertTrue(False)
   
                        #make sure the set id has the suffix
                        if not '__' in _d['setid']: self.assertTrue(False)
   
                        if index == 1:
                            #check the sample name
                            if _d['sample'] != sample_name_1 or _d['sampleName'] != sample_name_1: self.assertTrue(False)
#                            #check the gender, relation and relationRole
#                            if _d['Gender'] != 'Male' or _d['Relation'] != 'Tumor_Normal' or _d['RelationRole'] != 'Tumor': self.assertTrue(False)
                            #check the gender and relationRole
                            if _d['Gender'] != 'Male' or _d['RelationRole'] != 'Tumor': self.assertTrue(False)
                            #check the barcodeId
                            if _d['barcodeId'] != 'IonSet1_01': self.assertTrue(False)
   
                        else:
                            #check the sample name
                            if _d['sample'] != sample_name_2 or _d['sampleName'] != sample_name_2: self.assertTrue(False)
#                            #check the gender, relation and relationRole
#                            if _d['Gender'] != 'Female' or _d['Relation'] != 'Tumor_Normal' or _d['RelationRole'] != 'Normal': self.assertTrue(False)
                               
                            #check the gender and relationRole
                            if _d['Gender'] != 'Female' or _d['RelationRole'] != 'Normal': self.assertTrue(False)
                            #check the barcodeId
                            if _d['barcodeId'] != 'IonSet1_02': self.assertTrue(False)
   
   
            else:
                self.assertTrue(False)
        else:
            self.assertTrue(False)
   
        #now delete the planned run
        logger.info( "test_create_ir_plan_from_barcoded_ampliseq_dna_template... GOING to delete plan.id=%s" %(latest_plan['id']))
           
        self.delete_planned_experiment(latest_plan['id']) 
             
        #and then delete the template
        logger.info( "test_create_ir_plan_from_barcoded_ampliseq_dna_template... - ALL PASS - GOING to delete template.id=%s" %(barcoded_template_id))
        self.delete_planned_experiment(barcoded_template_id)
        time.sleep(2)

    def test_create_plan_from_barcoded_ampliseq_dna_template_that_will_fail_ir_validation(self):
        """
        Test if a plan that fails IR validation and then corrected can be saved successfully
        - no reference selected
        Verify what has been persisted in
        - barcodedSamples JSON
        - selectedPlugins JSON
        """    
        self.open(reverse('page_plan_new_template', args=(1,)))
            
        #test requires a default IR account be set and TS can establish connection with it        
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()
    
        #select the Trio Sample Grouping
        self.wd.find_element_by_xpath('//input[@name="sampleGrouping" and @value="4"]').click() #Trio
    
        irworkflow = self.wd.find_element_by_xpath('//select[@name="irworkflow"]')
        #choose the AmpliSeq Exome trio workflow option
        irworkflow.find_elements_by_tag_name('option')[1].click()#AmpliSeq Exome trio
    
        #now navigate to the Kits chevron and choose the Chip-Type
        self.wd.wait_for_css("#Kits").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_css("#chipType").find_elements_by_tag_name('option')[2].click()#(314v2)Ion 314v2 Chip
        self.wd.wait_for_css("#barcodeId").find_elements_by_tag_name('option')[1].click()#IonSet1
    
        self.wd.find_element_by_xpath('//input[@name="templatekitType" and @value="IonChef"]').click()
        time.sleep(2)
        
                
        select = Select(self.wd.find_element_by_id("templateKit"))
        select.select_by_visible_text("ION PGM IC 200 KIT")
        time.sleep(2)

        
        #now navigate to the Save chevron
        self.wd.find_element_by_id("Save_template").find_elements_by_tag_name('a')[0].click()
          
        current_dt = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
        self.wd.find_element_by_id("templateName").send_keys('to-fail-validation-ampliseq-dna-{0}'.format(current_dt))

        #select None for reference
        self.wd.find_element_by_id("default_reference").find_elements_by_tag_name('option')[0].click() ##None
                          
        #now save the template
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
    
        barcoded_template_id = self.get_latest_planned_experiment()['id']

        logger.info( ">>> test_create_plan_from_barcoded_ampliseq_dna_template_that_will_fail_ir_validation... barcoded_template_id=%s" %(barcoded_template_id))         
    
        #now open the plan run link
        self.open(reverse('page_plan_new_plan', args=(barcoded_template_id,)))
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()
        time.sleep(5)

        self.wd.find_element_by_id("planName").clear()
        self.wd.find_element_by_id("planName").send_keys('PLAN-TO-FAIL-VALIDATION-ampliseq-dna-{0}'.format(current_dt))
    
        self.wd.find_element_by_id("numRows").clear()
        self.wd.find_element_by_id("numRows").send_keys('3')
        self.wd.find_element_by_id("numRowsClick").click()
        time.sleep(3)

        #wait
        self.wd.wait_for_ajax()
                    
        #now enter three samples
        sample_name_1 = 'Failing-Sample-Name-1'
        sample_ext_id_1 = 'Failing-Sample-External-Id-1'
        sample_description_1 = 'Failing-Sample-Desc-1'
    
        sample_name_2 = 'Failing-Sample-Name-2'
        sample_ext_id_2 = 'Failing-Sample-External-Id-2'
        sample_description_2 = 'Failing-Sample-Desc-2'
    
        sample_name_3 = 'Failing-Sample-Name-3'
        sample_ext_id_3 = 'Failing-Sample-External-Id-3'
        sample_description_3 = 'Failing-Sample-Desc-3'
    
        barcodedSampleTable = self.wd.find_element_by_id("chipsets")
            
        trs = barcodedSampleTable.find_elements(By.TAG_NAME, "tr")
             
        tds = trs[1].find_elements(By.TAG_NAME, "td")
            
        for element in tds:
            if element.find_elements(By.NAME, "sampleName"):
                element.find_element_by_name("sampleName").clear()
                element.find_element_by_name("sampleName").send_keys(sample_name_1)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_1)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_1)

            if element.find_elements(By.NAME, "irWorkflow"):
                ##element.find_element_by_name("irWorkflow").find_elements_by_tag_name('option')[1].click()#non-Upload Only workflow
                select = Select(element.find_element_by_name("irWorkflow"))
                select.select_by_visible_text("AmpliSeq Exome trio")
                  
            if element.find_elements(By.NAME, "irRelationRole"):
            	##element.find_element_by_name("irRelationRole").find_elements_by_tag_name("option")[1].click() #Father
                select = Select(element.find_element_by_name("irRelationRole"))
                select.select_by_visible_text("Father")
                  
            if element.find_elements(By.NAME, "irGender"):
                ##element.find_element_by_name("irGender").find_elements_by_tag_name('option')[1].click()#Male
                select = Select(element.find_element_by_name("irGender"))
                select.select_by_visible_text("Male")
                  
            #ElementNotVisibleException: Message: Element is not currently visible and so may not be interacted with
#            if element.find_elements(By.NAME, "irRelationshipType"):
#                element.find_element_by_name("irRelationshipType").send_keys("Tumor_Normal")            
            if element.find_elements(By.NAME, "irSetID"):
                element.find_element_by_name("irSetID").send_keys("2")
                #wait
                self.wd.wait_for_ajax()
                        
        tds = trs[2].find_elements(By.TAG_NAME, "td")    
            
        for element in tds:
            if element.find_elements(By.NAME, "sampleName"):
                element.find_element_by_name("sampleName").clear()
                element.find_element_by_name("sampleName").send_keys(sample_name_2)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_2)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_2)

            if element.find_elements(By.NAME, "irWorkflow"):
                ##element.find_element_by_name("irWorkflow").find_elements_by_tag_name('option')[1].click()#non-Upload Only workflow
                select = Select(element.find_element_by_name("irWorkflow"))
                select.select_by_visible_text("AmpliSeq Exome trio")
                  
            if element.find_elements(By.NAME, "irRelationRole"):
            	##element.find_element_by_name("irRelationRole").find_elements_by_tag_name("option")[1].click() #Father
                select = Select(element.find_element_by_name("irRelationRole"))
                select.select_by_visible_text("Father")
                    
            if element.find_elements(By.NAME, "irGender"):
                ##
                ##element.find_element_by_name("irGender").find_elements_by_tag_name('option')[2].click()#Female
                select = Select(element.find_element_by_name("irGender"))
                select.select_by_visible_text("Unknown")
                                                  
            if element.find_elements(By.NAME, "irSetID"):
                element.find_element_by_name("irSetID").send_keys("2")
                #wait
                self.wd.wait_for_ajax()        
                       
        tds = trs[3].find_elements(By.TAG_NAME, "td")    
            
        for element in tds:
            if element.find_elements(By.NAME, "sampleName"):
                element.find_element_by_name("sampleName").clear()
                element.find_element_by_name("sampleName").send_keys(sample_name_3)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_3)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_3)

            if element.find_elements(By.NAME, "irWorkflow"):
                ##element.find_element_by_name("irWorkflow").find_elements_by_tag_name('option')[1].click()#non-Upload Only workflow
                select = Select(element.find_element_by_name("irWorkflow"))
                select.select_by_visible_text("AmpliSeq Exome trio")
                  
            if element.find_elements(By.NAME, "irRelationRole"):
            	##element.find_element_by_name("irRelationRole").find_elements_by_tag_name("option")[3].click() #Proband
                select = Select(element.find_element_by_name("irRelationRole"))
                select.select_by_visible_text("Proband")
                    
            if element.find_elements(By.NAME, "irGender"):
                element.find_element_by_name("irGender").find_elements_by_tag_name('option')[0].click()#blank
               
            if element.find_elements(By.NAME, "irSetID"):
                element.find_element_by_name("irSetID").send_keys("2")
                #wait
                self.wd.wait_for_ajax()                
    
        #now try to save the plan
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_ajax()
    
        time.sleep(3)
    
#        #now click on FIX ERRORS
#        self.wd.find_element_by_xpath('//button[@value="cancel"]').click()
#        #now fix sample 1 and change to FATHER
#        self.wd.find_element_by_xpath('//select[@name="irRelationRole81"]').find_elements_by_tag_name('option')[1].click()#Father
    
        #acknowledge IRU validation popup error message
        #element is at the third level from root within a div element
        #html > body > div.appriseOuter > div.appriseInner > div.aButtons
        ##self.wd.find_element_by_xpath('//button[@value="ok"]').click()
            
        self.wd.find_element_by_xpath('//*/div/button[@value="ok"]').click()

        #wait
        self.wd.wait_for_ajax()   
        #sleep so we can see the error messages
        time.sleep(3)
          
        #now try to fix the samples                        
        tds = trs[2].find_elements(By.TAG_NAME, "td")    
            
        for element in tds:
            if element.find_elements(By.NAME, "irRelationRole"):
                ##element.find_element_by_name("irRelationRole").find_elements_by_tag_name('option')[2].click()#Mother  
                select = Select(element.find_element_by_name("irRelationRole"))
                select.select_by_visible_text("Mother")
            if element.find_elements(By.NAME, "irGender"):
                ##element.find_element_by_name("irGender").find_elements_by_tag_name('option')[1].click()#Female     
                select = Select(element.find_element_by_name("irGender"))
                select.select_by_visible_text("Female")
                                                                     
        tds = trs[3].find_elements(By.TAG_NAME, "td")    
            
        for element in tds:
#             if element.find_elements(By.NAME, "irRelationRole"):
#                 ##element.find_element_by_name("irRelationRole").find_elements_by_tag_name('option')[3].click()#Proband              
            if element.find_elements(By.NAME, "irGender"):
                ##element.find_element_by_name("irGender").find_elements_by_tag_name('option')[1].click()#Female
                select = Select(element.find_element_by_name("irGender"))
                select.select_by_visible_text("Female")   
                            
        #now save the plan.  We except the plan to PASS validation
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_ajax()
    
        time.sleep(3)
            
        self.open(reverse('plan_templates'))
    
        #now retrieve the latest PlannedExperiment (The Plan you just saved) and verify the data you entered
        latest_plan = self.get_latest_planned_experiment()
            
        #verify the barcoded samples
        self.assertEqual(len(latest_plan['barcodedSamples']), 3)
        barcodes = latest_plan['barcodedSamples'].keys()
            
        if sample_name_1 in barcodes and sample_name_2 in barcodes:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
    
        sample1 = latest_plan['barcodedSamples'][sample_name_1]
        barcodedSamples = sample1['barcodeSampleInfo']
            
        _ionSet1Dict = barcodedSamples['IonSet1_01']
        self.assertEqual(_ionSet1Dict['description'], sample_description_1)
        self.assertEqual(_ionSet1Dict['externalId'], sample_ext_id_1)
    
        sample2 = latest_plan['barcodedSamples'][sample_name_2]
        barcodedSamples = sample2['barcodeSampleInfo']
            
        _ionSet2Dict = barcodedSamples['IonSet1_02']
        self.assertEqual(_ionSet2Dict['description'], sample_description_2)
        self.assertEqual(_ionSet2Dict['externalId'], sample_ext_id_2)  
    
        #now delete the planned run
        logger.info( ">>> test_create_plan_from_barcoded_ampliseq_dna_template_that_will_fail_ir_validation... GOING to delete plan.id=%s" %(latest_plan['id']))         
        self.delete_planned_experiment(latest_plan['id'])   
           
        #and then delete the template
        logger.info( ">>> test_create_plan_from_barcoded_ampliseq_dna_template_that_will_fail_ir_validation... - ALL PASS - GOING to delete template.id=%s" %(barcoded_template_id))        
        self.delete_planned_experiment(barcoded_template_id)
        time.sleep(3)


    def test_edit_ir_plan_sampleTubeLabel_n_notes(self):
        """
        Test if editing a generic sequencing, barcoded plan for sampleTubeLabel and notes can be saved successfully
        Verify what has been persisted in
        - barcodedSamples JSON
        - selectedPlugins JSON
        """          
        #generic sequencing template
        self.open(reverse('page_plan_new_template', args=(0,)))
         
        #test requires a default IR account be set        
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()
         
        #select the Trio Sample Grouping
        self.wd.find_element_by_xpath('//input[@name="sampleGrouping" and @value="4"]').click() #Trio
 
        irworkflow = self.wd.find_element_by_xpath('//select[@name="irworkflow"]')
        #choose the AmpliSeq Exome trio workflow option
        irworkflow.find_elements_by_tag_name('option')[1].click()#AmpliSeq Exome trio
 
        #now navigate to the Kits chevron and choose the Chip-Type
        self.wd.wait_for_css("#Kits").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_css("#chipType").find_elements_by_tag_name('option')[2].click()#(314v2)Ion 314v2 Chip
        self.wd.wait_for_css("#barcodeId").find_elements_by_tag_name('option')[1].click()#IonSet1
         
        self.wd.find_element_by_xpath('//input[@name="templatekitType" and @value="IonChef"]').click()
        time.sleep(2)
        
        select = Select(self.wd.find_element_by_id("templateKit"))
        select.select_by_visible_text("ION PGM IC 200 KIT")
        time.sleep(2)
 
        #now navigate to the Save chevron
        ##self.wd.wait_for_css("#Save_template").find_elements_by_tag_name('a')[0].click()
        self.wd.find_element_by_id("Save_template").find_elements_by_tag_name('a')[0].click()
        current_dt = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
        ##self.wd.wait_for_css("#templateName").send_keys('to-be-edited-ampliseq-dna-{0}'.format(current_dt))
        self.wd.find_element_by_id("templateName").send_keys('to-be-edited-generic-seq-{0}'.format(current_dt))
        #now save the template
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_ajax()   
        time.sleep(3)
         
        #need to delay the Driver so that we can pick up the latest PlannedExperiment
        self.open(reverse('plan_templates'))
 
        barcoded_template_id = self.get_latest_planned_experiment()['id']
        
        logger.info( ">>> test_edit_ir_plan_sampleTubeLabel_n_notes... just created template.id=%s" %(barcoded_template_id))        
 
        #now open the plan run link
        self.open(reverse('page_plan_new_plan', args=(barcoded_template_id,)))
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()
 
        ##self.wd.wait_for_css("#planName").send_keys('NEW-PLAN-TO-BE-EDITED-ampliseq-dna-{0}'.format(current_dt))
        self.wd.find_element_by_id("planName").send_keys('NEW-PLAN-TO-BE-EDITED-generic-seq-{0}'.format(current_dt))

        self.wd.find_element_by_id("numRows").clear()
        self.wd.find_element_by_id("numRows").send_keys('3')
        self.wd.find_element_by_id("numRowsClick").click()
         
        #wait        
        self.wd.wait_for_ajax()
         
        #now enter three samples
        sample_name_1 = 'Passing-Sample-Name-1'
        sample_ext_id_1 = 'Passing-Sample-External-Id-1'
        sample_description_1 = 'Passing-Sample-Desc-1'
 
        sample_name_2 = 'Passing-Sample-Name-2'
        sample_ext_id_2 = 'Passing-Sample-External-Id-2'
        sample_description_2 = 'Passing-Sample-Desc-2'
 
        sample_name_3 = 'Passing-Sample-Name-3'
        sample_ext_id_3 = 'Passing-Sample-External-Id-3'
        sample_description_3 = 'Passing-Sample-Desc-3'
 
        barcodedSampleTable = self.wd.find_element_by_id("chipsets")
         
        trs = barcodedSampleTable.find_elements(By.TAG_NAME, "tr")
          
        tds = trs[1].find_elements(By.TAG_NAME, "td")
         
        for element in tds:
            if element.find_elements(By.NAME, "sampleName"):
                element.find_element_by_name("sampleName").clear()
                element.find_element_by_name("sampleName").send_keys(sample_name_1)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_1)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_1)

            if element.find_elements(By.NAME, "irWorkflow"):
                ##element.find_element_by_name("irWorkflow").find_elements_by_tag_name('option')[1].click()#non-Upload Only workflow
                select = Select(element.find_element_by_name("irWorkflow"))
                select.select_by_visible_text("AmpliSeq Exome trio")
                                 
            if element.find_elements(By.NAME, "irRelationRole"):
                element.find_element_by_name("irRelationRole").find_elements_by_tag_name('option')[1].click()#Father
            if element.find_elements(By.NAME, "irGender"):
                element.find_element_by_name("irGender").find_elements_by_tag_name('option')[1].click()#Male
 
           
            if element.find_elements(By.NAME, "irSetID"):
                element.find_element_by_name("irSetID").send_keys("3")
                #wait
                self.wd.wait_for_ajax()
                     
        tds = trs[2].find_elements(By.TAG_NAME, "td")    
         
        for element in tds:
            if element.find_elements(By.NAME, "sampleName"):
                element.find_element_by_name("sampleName").clear()                
                element.find_element_by_name("sampleName").send_keys(sample_name_2)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_2)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_2)

            if element.find_elements(By.NAME, "irWorkflow"):
                ##element.find_element_by_name("irWorkflow").find_elements_by_tag_name('option')[1].click()#non-Upload Only workflow
                select = Select(element.find_element_by_name("irWorkflow"))
                select.select_by_visible_text("AmpliSeq Exome trio")
                                 
            if element.find_elements(By.NAME, "irRelationRole"):
                element.find_element_by_name("irRelationRole").find_elements_by_tag_name('option')[2].click()#Mother
            if element.find_elements(By.NAME, "irGender"):
                element.find_element_by_name("irGender").find_elements_by_tag_name('option')[1].click()#Female
                                
            if element.find_elements(By.NAME, "irSetID"):
                element.find_element_by_name("irSetID").send_keys("3")
                #wait
                self.wd.wait_for_ajax()        
                    
        tds = trs[3].find_elements(By.TAG_NAME, "td")    
         
        for element in tds:
            if element.find_elements(By.NAME, "sampleName"):
                element.find_element_by_name("sampleName").clear()                
                element.find_element_by_name("sampleName").send_keys(sample_name_3)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_3)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_3)

            if element.find_elements(By.NAME, "irWorkflow"):
                ##element.find_element_by_name("irWorkflow").find_elements_by_tag_name('option')[1].click()#non-Upload Only workflow
                select = Select(element.find_element_by_name("irWorkflow"))
                select.select_by_visible_text("AmpliSeq Exome trio")
                 
            if element.find_elements(By.NAME, "irRelationRole"):
                element.find_element_by_name("irRelationRole").find_elements_by_tag_name('option')[3].click()#Proband
            if element.find_elements(By.NAME, "irGender"):
                element.find_element_by_name("irGender").find_elements_by_tag_name('option')[2].click()#Unknown
 
                
            if element.find_elements(By.NAME, "irSetID"):
                element.find_element_by_name("irSetID").send_keys("3")
                #wait
                self.wd.wait_for_ajax()                
 
        #now save the plan
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_ajax()
 
        self.open(reverse('plan_templates'))
 
        latest_plan = self.get_latest_planned_experiment()
        #now open that plan for editing
        self.open(reverse('page_plan_edit_plan', args=(latest_plan['id'],)))
        self.wd.wait_for_ajax()
 
        time.sleep(3)
         
        notesValue = 'automated testing the notes here'
        sampleTubeLabelValue = 'automated tube 101'

        self.wd.find_element_by_id("note").clear()
        self.wd.find_element_by_id("note").send_keys(notesValue)
        self.wd.find_element_by_id("barcodeSampleTubeLabel").clear()
        self.wd.find_element_by_id("barcodeSampleTubeLabel").send_keys(sampleTubeLabelValue)  
                 
        #now save the plan.  We except the plan to PASS validation
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_ajax()
 
        latest_plan = self.get_latest_planned_experiment()
         
        #verify the barcoded samples
        self.assertEqual(len(latest_plan['barcodedSamples']), 3)
        barcodes = latest_plan['barcodedSamples'].keys()
         
        if sample_name_1 in barcodes and sample_name_2 in barcodes:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
 
        db_tubeLabel = latest_plan['sampleTubeLabel']
        self.assertEqual(db_tubeLabel, sampleTubeLabelValue, "sampleTubeLabel value validation")
 
        db_notes = latest_plan['notes']
        self.assertEqual(db_notes, notesValue, "notes value validation")
         
         
        sample1 = latest_plan['barcodedSamples'][sample_name_1]
        barcodedSamples = sample1['barcodeSampleInfo']
         
        _ionSet1Dict = barcodedSamples['IonSet1_01']
        self.assertEqual(_ionSet1Dict['description'], sample_description_1)
        self.assertEqual(_ionSet1Dict['externalId'], sample_ext_id_1)
 
        sample2 = latest_plan['barcodedSamples'][sample_name_2]
        barcodedSamples = sample2['barcodeSampleInfo']
         
        _ionSet2Dict = barcodedSamples['IonSet1_02']
        self.assertEqual(_ionSet2Dict['description'], sample_description_2)
        self.assertEqual(_ionSet2Dict['externalId'], sample_ext_id_2)  
 
        #now delete the planned run
        logger.info( ">>> test_edit_ir_plan_sampleTubeLabel_n_notes... GOING to delete plan.id=%s" %(latest_plan['id']))        
        self.delete_planned_experiment(latest_plan['id'])
          
        #and then delete the template
        logger.info( ">>> test_edit_ir_plan_sampleTubeLabel_n_notes... - ALL PASS - GOING to delete template.id=%s" %(barcoded_template_id))                
        self.delete_planned_experiment(barcoded_template_id)

    def test_edit_ir_plan_chevron_kits_n_attributes(self):
        """
        Test kits chevron attributes can be edited/updated and saved successfully
        Verify what has been persisted in Kits attributes:
             samplePrepKitName, controlSequencekitname
             librarykitname, chipType, templatingKitName, flows
             barcodeId, sequencekitname, base_recalibration_mode
        """
        """
        Case 1:
            create a template -> enter values on the Kits chevron -> save the template
                -> create a plan from this template -> enter values on the Kits chevron
                     -> save the plan -> check the saved values
        """
        #generic sequencing template
        self.open(reverse('page_plan_new_template', args=(0,)))

        #test requires a default IR account be set
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()

        #select the Trio Sample Grouping
        self.wd.find_element_by_xpath('//input[@name="sampleGrouping" and @value="4"]').click() #Trio

        irworkflow = self.wd.find_element_by_xpath('//select[@name="irworkflow"]')
        #choose the AmpliSeq Exome trio workflow option
        irworkflow.find_elements_by_tag_name('option')[1].click()#AmpliSeq Exome trio

        #now navigate to the Kits chevron and choose the Chip-Type
        self.wd.wait_for_css("#Kits").find_elements_by_tag_name('a')[0].click()
        #self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_css("#chipType").find_elements_by_tag_name('option')[2].click()#(314v2)Ion 314v2 Chip
        self.wd.wait_for_css("#barcodeId").find_elements_by_tag_name('option')[1].click()#IonSet1

        self.wd.find_element_by_xpath('//input[@name="templatekitType" and @value="IonChef"]').click()
        time.sleep(2)

        select = Select(self.wd.find_element_by_id("templateKit"))
        select.select_by_visible_text("ION PGM IC 200 KIT")
        time.sleep(2)

        #now navigate to the Save chevron
        self.wd.find_element_by_id("Save_template").find_elements_by_tag_name('a')[0].click()
        current_dt = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
        self.wd.find_element_by_id("templateName").send_keys('Chevron Kits-Edit TestCase-{0}'.format(current_dt))

        #now save the template
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        #need to delay the Driver so that we can pick up the latest PlannedExperiment
        self.wd.wait_for_ajax()
        time.sleep(3)

        self.open(reverse('plan_templates'))

        barcoded_template_id = self.get_latest_planned_experiment()['id']
        logger.info( ">>> test_edit/update_ir_plan_kits_chevron_n_attributes... created template.id=%s" %(barcoded_template_id))

        #now open the plan run link
        self.open(reverse('page_plan_new_plan', args=(barcoded_template_id,)))
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()
        self.wd.wait_for_css("#Ionreporter").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_ajax()
        time.sleep(2)

        #select the Self Sample Grouping - this is selected in order to modify optional control sequence in Kits
        self.wd.find_element_by_xpath('//input[@name="sampleGrouping" and @value="2"]').click() #self

        #navigate to Application and select DNA and AmpliSeq DNA
        self.wd.wait_for_css("#Application").find_elements_by_tag_name('a')[0].click()
        self.wd.find_element_by_xpath('//*[@id="step_form"]/div[1]/div[1]/div/div/label[1]/input').click()
        self.wd.find_element_by_xpath('//*[@id="runTypeHolder"]/div/div/label[2]/input').click()

        #now navigate to the Kits chevron and modify all the attributes of kits chevron
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_css("#samplePreparationKit").find_elements_by_tag_name('option')[2].click()#Ion AmpliSeq Exome Kit
        self.wd.wait_for_css("#controlsequence").find_elements_by_tag_name('option')[1].click()#Ion AmpliSeq Sample Id panel
        self.wd.wait_for_css("#libraryKitType").find_elements_by_tag_name('option')[2].click()#Ion AmpliSeq 2.0 Library Kit
        self.wd.wait_for_css("#chipType").find_elements_by_tag_name('option')[4].click()#Ion 316 Chip v2
        self.wd.wait_for_css("#barcodeId").find_elements_by_tag_name('option')[2].click()#Ionxpress
        self.wd.find_element_by_xpath('//input[@name="templatekitType" and @value="OneTouch"]').click()#OneTouch
        time.sleep(2)
        self.wd.wait_for_css("#templateKit").find_elements_by_tag_name('option')[2].click()#Ion PGM Hi-Q OT2 Kit - 400
        self.wd.wait_for_css("#sequenceKit").find_elements_by_tag_name('option')[6].click()#Ion PGM Sequencing 200 Kit v2
        self.wd.wait_for_css("#base_recalibrate").find_elements_by_tag_name('option')[1].click()#Enable Calibration Standard

        self.wd.find_element_by_id("flows").clear()
        self.wd.find_element_by_id("flows").send_keys('200')

        #now navigate to the Save chevron
        self.wd.wait_for_css("#Save_plan").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_ajax()
        time.sleep(3)

        #Update Default Reference & BED Files
        select = Select(self.wd.find_element_by_id("default_reference"))
        select.select_by_visible_text("e_coli_dh10b(E. coli DH10B)")

        select = Select(self.wd.find_element_by_id("default_targetBedFile"))
        select.select_by_visible_text("ecoli.bed")

        select = Select(self.wd.find_element_by_id("default_hotSpotBedFile"))
        select.select_by_visible_text("None")
        #select.select_by_visible_text("ecoli_hotspot.bed")

        logger.info( ">>> test_edit/update_ir_plan_kits_chevron_n_attributes... created template.id=%s" %(barcoded_template_id))

        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_ajax()
        time.sleep(3)

        self.open(reverse('plan_templates'))
        self.wd.wait_for_ajax()
        time.sleep(3)

        latest_runplan = self.get_latest_planned_experiment()

        #Validation of all the modified attributes of kits Chevron
        self.assertEqual(latest_runplan['samplePrepKitName'],"Ion AmpliSeq Exome Kit")
        self.assertEqual(latest_runplan['controlSequencekitname'],"Ion AmpliSeq Sample ID Panel")
        self.assertEqual(latest_runplan['librarykitname'],"Ion AmpliSeq 2.0 Library Kit")
        self.assertEqual(latest_runplan['chipType'],"316v2")
        self.assertEqual(latest_runplan['templatingKitName'],"Ion PGM Hi-Q OT2 Kit - 400")
        self.assertEqual(latest_runplan['barcodeId'],"IonXpress")
        self.assertEqual(latest_runplan['sequencekitname'],"IonProtonIHiQ")
        self.assertEqual(latest_runplan['base_recalibration_mode'],"panel_recal")
        self.assertEqual(latest_runplan['flows'],200)

        # Edit the existing plan->kits chevron attributes from the edit page and verify the data integrity
        """
        Case II:
                -> EDIT the saved plan -> make sure ALL attributes on the Kits chevron are still editable
                   -> change the chip type and other attributes -> update the plan -> check the attributes are saved correctly
        """
        self.open(reverse('page_plan_edit_plan', args=(latest_runplan['id'],)))
        self.wd.wait_for_css("#Kits").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_css("#samplePreparationKit").find_elements_by_tag_name('option')[1].click()#Ion TargetSeq Exome Kit (12rxn)
        self.wd.wait_for_css("#controlsequence").find_elements_by_tag_name('option')[0].click()#Ion AmpliSeq Sample Id panel
        self.wd.wait_for_css("#libraryKitType").find_elements_by_tag_name('option')[11].click()#Ion PicoPlex
        self.wd.wait_for_css("#chipType").find_elements_by_tag_name('option')[7].click()#Ion 520 (P1.0.20 Software windowed) chip
        self.wd.wait_for_css("#barcodeId").find_elements_by_tag_name('option')[6].click()#RNA_Barcode_None
        self.wd.wait_for_css("#templateKit").find_elements_by_tag_name('option')[4].click()#Ion PGM Template OT2 200 Kit
        self.wd.wait_for_css("#sequenceKit").find_elements_by_tag_name('option')[3].click()#Ion PGM Install Kit
        self.wd.wait_for_css("#base_recalibrate").find_elements_by_tag_name('option')[0].click()#Default Calibration
        self.wd.find_element_by_id("flows").clear()
        self.wd.find_element_by_id("flows").send_keys('400')

        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_ajax()
        time.sleep(2)

        #navigate to the Save chevron
        self.wd.wait_for_css("#Save_plan").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_ajax()

        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        time.sleep(2)
        self.wd.wait_for_ajax()

        self.open(reverse('planned'))
        self.wd.wait_for_ajax()
        time.sleep(2)

        latest_edited_plan = self.get_latest_planned_experiment()

        #Validation of all the modified attributes of kits Chevron
        self.assertEqual(latest_edited_plan['samplePrepKitName'],"Ion AmpliSeq CCP")
        self.assertEqual(latest_edited_plan['controlSequencekitname'],"")
        self.assertEqual(latest_edited_plan['librarykitname'],"IonPicoPlex")
        self.assertEqual(latest_edited_plan['chipType'],"520")
        self.assertEqual(latest_edited_plan['templatingKitName'],"Ion PGM Template OT2 200 Kit")
        self.assertEqual(latest_edited_plan['barcodeId'],"RNA_Barcode_None")
        self.assertEqual(latest_edited_plan['sequencekitname'],"IonPGMInstallKit")
        self.assertEqual(latest_edited_plan['base_recalibration_mode'],"standard_recal")
        self.assertEqual(latest_edited_plan['flows'],400)

        #Delete the planned run
        logger.info( ">>> test_edit/update_ir_plan_kits_chevron_n_attributes... GOING to delete plan.id=%s" %(latest_runplan['id']))
        self.delete_planned_experiment(latest_runplan['id'])

        #Delete the template
        logger.info( ">>> test_edit/update_ir_plan_kits_chevron_n_attributes... - ALL PASS - GOING to delete template.id=%s" %(barcoded_template_id))
        self.delete_planned_experiment(barcoded_template_id)
        time.sleep(3)

    def test_data_persistance_when_traversing_chevron_before_saving_plan(self):
        """
        TS-10480
        Plan wizard - selecting an IR workflow should cascade to the sample config table
        """
        self.open(reverse('page_plan_new_template', args=(0,)))

        #test requires a default IR account be set
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()

        #select the Trio Sample Grouping
        self.wd.find_element_by_xpath('//input[@name="sampleGrouping" and @value="4"]').click() #Trio

        irworkflow = self.wd.find_element_by_xpath('//select[@name="irworkflow"]')
        #choose the AmpliSeq Exome trio workflow option
        irworkflow.find_elements_by_tag_name('option')[1].click()#AmpliSeq trio trio

        self.wd.wait_for_css("#Kits").find_elements_by_tag_name('a')[0].click()
        #self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_css("#chipType").find_elements_by_tag_name('option')[2].click()#(314v2)Ion 314v2 Chip
        self.wd.wait_for_css("#barcodeId").find_elements_by_tag_name('option')[1].click()#IonSet1

        self.wd.find_element_by_xpath('//input[@name="templatekitType" and @value="IonChef"]').click()
        time.sleep(2)

        select = Select(self.wd.find_element_by_id("templateKit"))
        select.select_by_visible_text("ION PGM IC 200 KIT")
        time.sleep(2)

        #now navigate to the Save chevron
        self.wd.find_element_by_id("Save_template").find_elements_by_tag_name('a')[0].click()
        current_dt = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
        self.wd.find_element_by_id("templateName").send_keys('IR Workflow check in sample config table TestCase-{0}'.format(current_dt))

        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        time.sleep(2)
        self.wd.wait_for_ajax()

        barcoded_template_id = self.get_latest_planned_experiment()['id']
        logger.info( ">>> test_update_ir_workflow_and_verify_in_sample_config_table... created template.id=%s" %(barcoded_template_id))

        #now open the plan run link
        self.open(reverse('page_plan_new_plan', args=(barcoded_template_id,)))
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()
        time.sleep(2)

        #Update Default Reference & BED Files
        select = Select(self.wd.find_element_by_id("default_reference"))
        select.select_by_visible_text("hg19(Homo sapiens)")

        select = Select(self.wd.find_element_by_id("default_targetBedFile"))
        select.select_by_visible_text("BRCA1_2.20131001.designed.bed")

        select = Select(self.wd.find_element_by_id("default_hotSpotBedFile"))
        select.select_by_visible_text("BRCA1_2.20131001.hotspots.bed")

        barcodedSampleTable = self.wd.find_element_by_id("chipsets")
        trs = barcodedSampleTable.find_elements(By.TAG_NAME, "tr")
        tds = trs[1].find_elements(By.TAG_NAME, "td")

        #Update the attributes in Sample config table
        #Traverse to kits chevron and verify the data is persistent
        #Do not save the plan and traverse
        for element in tds:
            if element.find_elements(By.NAME, "barcode"):
                select = Select(element.find_element_by_name("barcode")).first_selected_option
                selected_barcode = select.text
                #Verify the default barcode value for the selected bar code under Kits
                self.assertEqual(selected_barcode,"IonSet1_01 (TACTCACGATA)")  
            if element.find_elements(By.NAME, "sampleName"):
                element.find_element_by_name("sampleName").clear()                
                element.find_element_by_name("sampleName").send_keys("sample_traversing_check")
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys("sample_ext_id_1")
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys("sample_description_1")
            if element.find_elements(By.NAME, "ircancerType"):
                element.find_element_by_name("ircancerType").find_elements_by_tag_name('option')[11].click()#LiverCancer
            if element.find_elements(By.NAME, "ircellularityPct"):
                element.find_element_by_name("ircellularityPct").send_keys("20") 
            if element.find_elements(By.NAME, "irRelationRole"):
                element.find_element_by_name("irRelationRole").find_elements_by_tag_name('option')[2].click()#Mother
            if element.find_elements(By.NAME, "irGender"):
                element.find_element_by_name("irGender").find_elements_by_tag_name('option')[1].click()#Female
            if element.find_elements(By.NAME, "irSetID"):
                element.find_element_by_name("irSetID").send_keys("1")
                self.wd.wait_for_ajax()
        # traverse to chevron KITS and modify the attributes
        self.wd.wait_for_css("#Kits").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_css("#chipType").find_elements_by_tag_name('option')[4].click()#(314v2)Ion 314 Chip
        self.wd.wait_for_css("#barcodeId").find_elements_by_tag_name('option')[3].click()#IonXpressRNA

        #navigate to the Save chevron and verify the data is persistent
        self.wd.wait_for_css("#Save_plan").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_ajax()

        barcodedSampleTable = self.wd.find_element_by_id("chipsets")
        trs = barcodedSampleTable.find_elements(By.TAG_NAME, "tr")
        tds = trs[1].find_elements(By.TAG_NAME, "td")
        ##Verify the modified attributes values are persistent under plan tab
        for element in tds:
            if element.find_elements(By.NAME, "barcode"):
                select = Select(element.find_element_by_name("barcode")).first_selected_option
                selected_barcode = select.text
                self.assertEqual(selected_barcode,"IonXpressRNA_001 (CTAAGGTAAC)")
            if element.find_elements(By.NAME, "sampleName"):
                sample = element.find_element_by_name("sampleName")
                actual_sample_text = sample.get_attribute("value")
                self.assertEqual(actual_sample_text,"sample_traversing_check")
            if element.find_elements(By.NAME, "sampleExternalId"):
                sample = element.find_element_by_name("sampleExternalId")
                actual_sample_extID = sample.get_attribute("value")
                self.assertEqual(actual_sample_extID,"sample_ext_id_1")
            if element.find_elements(By.NAME, "sampleDescription"):
                sample = element.find_element_by_name("sampleDescription")
                actual_sample_extID = sample.get_attribute("value")
                self.assertEqual(actual_sample_extID,"sample_description_1")
            if element.find_elements(By.NAME, "ircancerType"):
                select = Select(element.find_element_by_name("ircancerType")).first_selected_option
                selected_cancer_type = select.text
                self.assertEqual(selected_cancer_type,"Liver Cancer")
            if element.find_elements(By.NAME, "ircellularityPct"):
                sample = element.find_element_by_name("ircellularityPct")
                actual_sample_extID = sample.get_attribute("value")
                self.assertEqual(actual_sample_extID,"20")
            if element.find_elements(By.NAME, "irRelationRole"):
                select = Select(element.find_element_by_name("irRelationRole")).first_selected_option
                selected_relation = select.text
                self.assertEqual(selected_relation,"Mother")
            if element.find_elements(By.NAME, "irGender"):
                select = Select(element.find_element_by_name("irGender")).first_selected_option
                selected_relation = select.text
                self.assertEqual(selected_relation,"Female")
            if element.find_elements(By.NAME, "irSetID"):
                sample = element.find_element_by_name("irSetID")
                actual_sample_extID = sample.get_attribute("value")
                self.assertEqual(actual_sample_extID,"1")
    
        #Delete the template
        logger.info( ">>> test_data_persistance_when_traversing_chevron_before_saving_plan... ALL PASS GOING to delete plan.id=%s" %(barcoded_template_id))
        self.delete_planned_experiment(barcoded_template_id)

    def test_select_irworkflow_should_cascade_sample_config_table(self):
        """
        TS-10481
        Plan wizard - selecting an IR workflow should cascade to the sample config table
        """
        self.open(reverse('page_plan_new_template', args=(0,)))

        #test requires a default IR account be set
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()

        #select the Trio Sample Grouping
        self.wd.find_element_by_xpath('//input[@name="sampleGrouping" and @value="4"]').click() #Trio

        irworkflow = self.wd.find_element_by_xpath('//select[@name="irworkflow"]')
        #choose the AmpliSeq Exome trio workflow option
        irworkflow.find_elements_by_tag_name('option')[1].click()#AmpliSeq Exome trio

        #now navigate to the Save chevron
        self.wd.find_element_by_id("Save_template").find_elements_by_tag_name('a')[0].click()
        current_dt = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
        self.wd.find_element_by_id("templateName").send_keys('IR Workflow check in sample config table TestCase-{0}'.format(current_dt))

        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        time.sleep(2)
        self.wd.wait_for_ajax()

        barcoded_template_id = self.get_latest_planned_experiment()['id']
        logger.info( ">>> test_update_ir_workflow_and_verify_in_sample_config_table... created template.id=%s" %(barcoded_template_id))

        #now open the plan run link
        self.open(reverse('page_plan_new_plan', args=(barcoded_template_id,)))
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()

        barcodedSampleTable = self.wd.find_element_by_id("chipsets")
        trs = barcodedSampleTable.find_elements(By.TAG_NAME, "tr")
        tds = trs[1].find_elements(By.TAG_NAME, "td")

        #Verify irWorkflow in config table before modification
        for element in tds:
            if element.find_elements(By.NAME, "irWorkflow"):
                select = Select(element.find_element_by_name("irWorkflow")).first_selected_option
                selected_irworkflow = select.get_attribute("value")
                self.assertEqual(selected_irworkflow,"AmpliSeq Exome trio")

        self.wd.wait_for_css("#Ionreporter").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_ajax()
        time.sleep(2)

        #Modify irworkflow data and verify that it is cascaded correctly to the sample config table

        #select the Trio Sample Grouping
        self.wd.find_element_by_xpath('//input[@name="sampleGrouping" and @value="4"]').click() #Trio

        irworkflow = self.wd.find_element_by_xpath('//select[@name="irworkflow"]')
        #choose the AmpliSeq IDP trio workflow option
        irworkflow.find_elements_by_tag_name('option')[2].click()#AmpliSeq IDP trio
        self.wd.wait_for_ajax()
        time.sleep(2)

        #navigate to Plan tab but do not save the plan
        self.wd.wait_for_css("#Save_plan").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_ajax()
        time.sleep(2)

        barcodedSampleTable = self.wd.find_element_by_id("chipsets")
        trs = barcodedSampleTable.find_elements(By.TAG_NAME, "tr")
        tds = trs[1].find_elements(By.TAG_NAME, "td")

        for element in tds:
            if element.find_elements(By.NAME, "irWorkflow"):
                select = Select(element.find_element_by_name("irWorkflow")).first_selected_option
                selected_irworkflow = select.get_attribute("value")
                self.assertEqual(selected_irworkflow,"AmpliSeq IDP trio")

        #Delete the template
        logger.info( ">>> test_select_irworkflow_should_cascade_sample_config_table... ALL PASS GOING to delete template.id=%s" %(barcoded_template_id))
        self.delete_planned_experiment(barcoded_template_id)