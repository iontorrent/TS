# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import time
from datetime import datetime
from django.core.urlresolvers import reverse
from django.test import LiveServerTestCase
from iondb.rundb.test import SeleniumTestCase

from selenium.webdriver.common.by import By

from selenium.webdriver.remote.remote_connection import LOGGER
import logging
logger = logging.getLogger(__name__)


import os

class TestCreatePlansByTemplate(SeleniumTestCase):  

    @classmethod
    def setUpClass(cls):
        super(TestCreatePlansByTemplate, cls).setUpClass()      
        
    def test_create_plan_with_plugins_from_csv_upload(self): 
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
        if 'variantCaller' in selected_plugins:
            self.assertTrue(True)
            
            if 'coverageAnalysis' in selected_plugins:
                self.assertTrue(True)
            else:
                self.assertTrue(False)                
        else:
            self.assertTrue(False)
            
        #tear down the uploaded plan
        self.delete_planned_experiment(uploaded_plan['id'])   


    def test_create_ir_plan_from_barcoded_ampliseq_dna_template(self):
        self.open(reverse('page_plan_new_template', args=(1,)))
        
        #test requires a default IR account be set and TS can establish connection with it
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()

        #select the Tumor_Normal Sample Grouping
        self.wd.find_element_by_xpath('//input[@name="sampleGrouping" and @value="3"]').click() #Tumor_Normal
        #wait
        self.wd.wait_for_ajax()
        
        irworkflow = self.wd.find_element_by_xpath('//select[@name="irworkflow"]')
        #choose the second workflow option
        irworkflow.find_elements_by_tag_name('option')[2].click()#AmpliSeq CCP tumor-normal pair

        #now navigate to the Kits chevron and choose the Chip-Type
        self.wd.wait_for_css("#Kits").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_css("#chipType").find_elements_by_tag_name('option')[2].click()#(314v2)Ion 314v2 Chip
        self.wd.wait_for_css("#barcodeId").find_elements_by_tag_name('option')[1].click()#IonSet1
 
        self.wd.wait_for_css("#IonChef__templatekitType").click()
        self.wd.wait_for_css("#IonChef__templatingKit").find_elements_by_tag_name('option')[1].click()
                
        #now navigate to the Save chevron
        self.wd.wait_for_css("#Save_template").find_elements_by_tag_name('a')[0].click()
        current_dt = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
        self.wd.wait_for_css("#templateName").send_keys('ampliseq-dna-{0}'.format(current_dt))
        #now save the template
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()

        barcoded_template_id = self.get_latest_planned_experiment()['id']

        #now open the plan run link
        self.open(reverse('page_plan_new_plan', args=(barcoded_template_id,)))
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()

        self.wd.wait_for_css("#planName").send_keys('PLAN-ampliseq-dna-{0}'.format(current_dt))

        self.wd.wait_for_css("#numRows").clear()
        self.wd.wait_for_css("#numRows").send_keys('2')
        self.wd.wait_for_css("#numRowsClick").click()
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
                element.find_element_by_name("sampleName").send_keys(sample_name_1)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_1)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_1)

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
                element.find_element_by_name("sampleName").send_keys(sample_name_2)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_2)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_2)

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

        #now save the plan
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_ajax()

        self.open(reverse('plan_templates'))

        #now retrieve the latest PlannedExperiment (The Plan you just saved) and verify the data you entered
        latest_plan = self.get_latest_planned_experiment()
        
        #verify the barcoded samples
        self.assertEqual(len(latest_plan['barcodedSamples']), 2)
        barcodes = latest_plan['barcodedSamples'].keys()
        
        if sample_name_1 in barcodes and sample_name_2 in barcodes:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

        sample1 = latest_plan['barcodedSamples'][sample_name_1]
        barcodedSamples = sample1['barcodeSampleInfo']
        
        logger.info( "test_create_ir_plan_from_barcoded_ampliseq_dna_template... ASSERT.... barcodedSamples=%s" %(barcodedSamples))
        
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
        logger.info( "test_create_ir_plan_from_barcoded_ampliseq_dna_template... GOING to delete template.id=%s" %(barcoded_template_id))
        self.delete_planned_experiment(barcoded_template_id)


    def test_create_plan_from_barcoded_ampliseq_dna_template_that_will_fail_validation(self):

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

        self.wd.wait_for_css("#IonChef__templatekitType").click()
        self.wd.wait_for_css("#IonChef__templatingKit").find_elements_by_tag_name('option')[1].click()
        
        #now navigate to the Save chevron
        self.wd.wait_for_css("#Save_template").find_elements_by_tag_name('a')[0].click()
        current_dt = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
        self.wd.wait_for_css("#templateName").send_keys('to-fail-validation-ampliseq-dna-{0}'.format(current_dt))
        #now save the template
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()

        barcoded_template_id = self.get_latest_planned_experiment()['id']

        #now open the plan run link
        self.open(reverse('page_plan_new_plan', args=(barcoded_template_id,)))
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()

        self.wd.wait_for_css("#planName").send_keys('PLAN-TO-FAIL-VALIDATION-ampliseq-dna-{0}'.format(current_dt))

        self.wd.wait_for_css("#numRows").clear()
        self.wd.wait_for_css("#numRows").send_keys('3')
        self.wd.wait_for_css("#numRowsClick").click()
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
                element.find_element_by_name("sampleName").send_keys(sample_name_1)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_1)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_1)
            if element.find_elements(By.NAME, "irRelationRole"):
            	element.find_element_by_name("irRelationRole").find_elements_by_tag_name("option")[1].click() #Father

            if element.find_elements(By.NAME, "irGender"):
                element.find_element_by_name("irGender").find_elements_by_tag_name('option')[1].click()#Male
            
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
                element.find_element_by_name("sampleName").send_keys(sample_name_2)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_2)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_2)
            if element.find_elements(By.NAME, "irRelationRole"):
            	element.find_element_by_name("irRelationRole").find_elements_by_tag_name("option")[1].click() #Father

            if element.find_elements(By.NAME, "irGender"):
                element.find_element_by_name("irGender").find_elements_by_tag_name('option')[2].click()#Female

            if element.find_elements(By.NAME, "irSetID"):
                element.find_element_by_name("irSetID").send_keys("2")
                #wait
                self.wd.wait_for_ajax()        
                   
        tds = trs[3].find_elements(By.TAG_NAME, "td")    
        
        for element in tds:
            if element.find_elements(By.NAME, "sampleName"):
                element.find_element_by_name("sampleName").send_keys(sample_name_3)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_3)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_3)
            if element.find_elements(By.NAME, "irRelationRole"):
            	element.find_element_by_name("irRelationRole").find_elements_by_tag_name("option")[3].click() #Proband

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


        #now try to fix the samples
        for element in tds:
            if element.find_elements(By.NAME, "irRelationRole"):
                element.find_element_by_name("irRelationRole").find_elements_by_tag_name('option')[1].click()#Father

            if element.find_elements(By.NAME, "irGender"):
                element.find_element_by_name("irGender").find_elements_by_tag_name('option')[1].click()#Male

                    
        tds = trs[2].find_elements(By.TAG_NAME, "td")    
        
        for element in tds:
            if element.find_elements(By.NAME, "irRelationRole"):
                element.find_element_by_name("irRelationRole").find_elements_by_tag_name('option')[2].click()#Mother            
            if element.find_elements(By.NAME, "irGender"):
                element.find_element_by_name("irGender").find_elements_by_tag_name('option')[1].click()#Female     
                   
        tds = trs[3].find_elements(By.TAG_NAME, "td")    
        
        for element in tds:
            if element.find_elements(By.NAME, "irRelationRole"):
                element.find_element_by_name("irRelationRole").find_elements_by_tag_name('option')[3].click()#Proband
            if element.find_elements(By.NAME, "irGender"):
                element.find_element_by_name("irGender").find_elements_by_tag_name('option')[1].click()#Female
  
                        
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
        self.delete_planned_experiment(latest_plan['id'])   
        #and then delete the template
        self.delete_planned_experiment(barcoded_template_id)


    def test_edit_ir_plan_sampleTubeLabel_n_notes(self):

        self.open(reverse('page_plan_new_template', args=(1,)))
        
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
        
        self.wd.wait_for_css("#IonChef__templatekitType").click()
        self.wd.wait_for_css("#IonChef__templatingKit").find_elements_by_tag_name('option')[1].click()

        #now navigate to the Save chevron
        self.wd.wait_for_css("#Save_template").find_elements_by_tag_name('a')[0].click()
        current_dt = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
        self.wd.wait_for_css("#templateName").send_keys('to-be-edited-ampliseq-dna-{0}'.format(current_dt))
        #now save the template
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()

        #need to delay the Driver so that we can pick up the latest PlannedExperiment
        self.open(reverse('plan_templates'))

        barcoded_template_id = self.get_latest_planned_experiment()['id']

        #now open the plan run link
        self.open(reverse('page_plan_new_plan', args=(barcoded_template_id,)))
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()

        self.wd.wait_for_css("#planName").send_keys('NEW-PLAN-TO-BE-EDITED-ampliseq-dna-{0}'.format(current_dt))

        self.wd.wait_for_css("#numRows").clear()
        self.wd.wait_for_css("#numRows").send_keys('3')
        self.wd.wait_for_css("#numRowsClick").click()
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
                element.find_element_by_name("sampleName").send_keys(sample_name_1)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_1)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_1)
                
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
                element.find_element_by_name("sampleName").send_keys(sample_name_2)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_2)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_2)
                
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
                element.find_element_by_name("sampleName").send_keys(sample_name_3)
            if element.find_elements(By.NAME, "sampleExternalId"):
                element.find_element_by_name("sampleExternalId").send_keys(sample_ext_id_3)
            if element.find_elements(By.NAME, "sampleDescription"):
                element.find_element_by_name("sampleDescription").send_keys(sample_description_3)

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

        self.wd.wait_for_css("#note").send_keys(notesValue)
        self.wd.wait_for_css("#barcodeSampleTubeLabel").send_keys(sampleTubeLabelValue)  
                
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
        self.delete_planned_experiment(latest_plan['id'])   
        #and then delete the template
        self.delete_planned_experiment(barcoded_template_id)
