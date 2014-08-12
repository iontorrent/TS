# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import time
from datetime import datetime
from django.core.urlresolvers import reverse
from django.test import LiveServerTestCase
from iondb.rundb.test import SeleniumTestCase

import os

class TestCreatePlansByTemplate(SeleniumTestCase):  

    @classmethod
    def setUpClass(cls):
        super(TestCreatePlansByTemplate, cls).setUpClass()      
        
    def test_create_plan_with_plugins_from_csv_upload(self): 
        """
        TS-7986: validation failed when csv file has plugins
        """
        
        self.open(reverse('plans'))
        
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
        
        #test requires a default IR account be set
        #now we need to wait for the ajax call which loads the IR accounts to finish
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

        #now enter two samples
        sample_name_1 = 'Barcoded-Sample-Name-10'
        sample_ext_id_1 = 'Barcoded-Sample-External-Id-10'
        sample_description_1 = 'Barcoded-Sample-Desc-10'

        sample_name_2 = 'Barcoded-Sample-Name-11'
        sample_ext_id_2 = 'Barcoded-Sample-External-Id-11'
        sample_description_2 = 'Barcoded-Sample-Desc-11'

        self.wd.find_element_by_xpath('//input[@name="barcodeSampleName81"]').send_keys(sample_name_1)
        self.wd.find_element_by_xpath('//input[@name="barcodeSampleExternalId81"]').send_keys(sample_ext_id_1)
        self.wd.find_element_by_xpath('//input[@name="barcodeSampleDescription81"]').send_keys(sample_description_1)

        self.wd.find_element_by_xpath('//select[@name="irGender81"]').find_elements_by_tag_name('option')[1].click()#Male
        self.wd.find_element_by_xpath('//select[@name="irRelation81"]').find_elements_by_tag_name('option')[1].click()#Tumor_Normal
        self.wd.find_element_by_xpath('//select[@name="irRelationRole81"]').find_elements_by_tag_name('option')[1].click()#Tumor

        self.wd.find_element_by_xpath('//input[@name="barcodeSampleName82"]').send_keys(sample_name_2)
        self.wd.find_element_by_xpath('//input[@name="barcodeSampleExternalId82"]').send_keys(sample_ext_id_2)
        self.wd.find_element_by_xpath('//input[@name="barcodeSampleDescription82"]').send_keys(sample_description_2) 

        self.wd.find_element_by_xpath('//select[@name="irGender82"]').find_elements_by_tag_name('option')[2].click()#Female
        self.wd.find_element_by_xpath('//select[@name="irRelation82"]').find_elements_by_tag_name('option')[1].click()#Tumor_Normal
        self.wd.find_element_by_xpath('//select[@name="irRelationRole82"]').find_elements_by_tag_name('option')[2].click()#Normal

        #now save the plan
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_ajax()

        self.open(reverse('plans'))

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
                            #check the gender, relation and relationRole
                            if _d['Gender'] != 'Male' or _d['Relation'] != 'Tumor_Normal' or _d['RelationRole'] != 'Tumor': self.assertTrue(False)
                            #check the barcodeId
                            if _d['barcodeId'] != 'IonSet1_01': self.assertTrue(False)

                        else:
                            #check the sample name
                            if _d['sample'] != sample_name_2 or _d['sampleName'] != sample_name_2: self.assertTrue(False)
                            #check the gender, relation and relationRole
                            if _d['Gender'] != 'Female' or _d['Relation'] != 'Tumor_Normal' or _d['RelationRole'] != 'Normal': self.assertTrue(False)
                            #check the barcodeId
                            if _d['barcodeId'] != 'IonSet1_02': self.assertTrue(False)


            else:
                self.assertTrue(False)
        else:
            self.assertTrue(False)

        #now delete the planned run
        self.delete_planned_experiment(latest_plan['id'])   
        #and then delete the template
        self.delete_planned_experiment(barcoded_template_id)


    def test_create_plan_from_barcoded_ampliseq_dna_template_that_will_fail_validation(self):

        self.open(reverse('page_plan_new_template', args=(1,)))
        
        #test requires a default IR account be set        
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()

        #select the Trio Sample Grouping
        self.wd.find_element_by_xpath('//input[@name="sampleGrouping" and @value="4"]').click()#Trio

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
        self.wd.wait_for_css("#templateName").send_keys('to-fail-validaiton-ampliseq-dna-{0}'.format(current_dt))
        #now save the template
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()

        barcoded_template_id = self.get_latest_planned_experiment()['id']

        #now open the plan run link
        self.open(reverse('page_plan_new_plan', args=(barcoded_template_id,)))
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()

        self.wd.wait_for_css("#planName").send_keys('PLAN-TO-FAIL-VALIDATION-ampliseq-dna-{0}'.format(current_dt))

        #now enter two samples
        sample_name_1 = 'Failing-Sample-Name-1'
        sample_ext_id_1 = 'Failing-Sample-External-Id-1'
        sample_description_1 = 'Failing-Sample-Desc-1'

        sample_name_2 = 'Failing-Sample-Name-2'
        sample_ext_id_2 = 'Failing-Sample-External-Id-2'
        sample_description_2 = 'Failing-Sample-Desc-2'

        sample_name_3 = 'Failing-Sample-Name-3'
        sample_ext_id_3 = 'Failing-Sample-External-Id-3'
        sample_description_3 = 'Failing-Sample-Desc-3'

        self.wd.find_element_by_xpath('//input[@name="barcodeSampleName81"]').send_keys(sample_name_1)
        self.wd.find_element_by_xpath('//input[@name="barcodeSampleExternalId81"]').send_keys(sample_ext_id_1)
        self.wd.find_element_by_xpath('//input[@name="barcodeSampleDescription81"]').send_keys(sample_description_1)

        self.wd.find_element_by_xpath('//select[@name="irGender81"]').find_elements_by_tag_name('option')[1].click()#Male
        self.wd.find_element_by_xpath('//select[@name="irRelation81"]').find_elements_by_tag_name('option')[1].click()#Trio
        #Adding MOTHER as the incorrect choice that will fail
        self.wd.find_element_by_xpath('//select[@name="irRelationRole81"]').find_elements_by_tag_name('option')[2].click()#Mother

        self.wd.find_element_by_xpath('//input[@name="barcodeSampleName82"]').send_keys(sample_name_2)
        self.wd.find_element_by_xpath('//input[@name="barcodeSampleExternalId82"]').send_keys(sample_ext_id_2)
        self.wd.find_element_by_xpath('//input[@name="barcodeSampleDescription82"]').send_keys(sample_description_2) 

        self.wd.find_element_by_xpath('//select[@name="irGender82"]').find_elements_by_tag_name('option')[2].click()#Female
        self.wd.find_element_by_xpath('//select[@name="irRelation82"]').find_elements_by_tag_name('option')[1].click()#Trio
        self.wd.find_element_by_xpath('//select[@name="irRelationRole82"]').find_elements_by_tag_name('option')[2].click()#Mother


        self.wd.find_element_by_xpath('//input[@name="barcodeSampleName83"]').send_keys(sample_name_3)
        self.wd.find_element_by_xpath('//input[@name="barcodeSampleExternalId83"]').send_keys(sample_ext_id_3)
        self.wd.find_element_by_xpath('//input[@name="barcodeSampleDescription83"]').send_keys(sample_description_3) 

        self.wd.find_element_by_xpath('//select[@name="irGender83"]').find_elements_by_tag_name('option')[3].click()#Unknown
        self.wd.find_element_by_xpath('//select[@name="irRelation83"]').find_elements_by_tag_name('option')[1].click()#Trio
        self.wd.find_element_by_xpath('//select[@name="irRelationRole83"]').find_elements_by_tag_name('option')[3].click()#Proband

        #now save the plan.  We except the plan to FAIL validation
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_ajax()

        time.sleep(3)

        #now click on FIX ERRORS
        self.wd.find_element_by_xpath('//button[@value="cancel"]').click()
        #now fix sample 1 and change to FATHER
        self.wd.find_element_by_xpath('//select[@name="irRelationRole81"]').find_elements_by_tag_name('option')[1].click()#Father

        #now save the plan.  We except the plan to PASS validation
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_ajax()

        self.open(reverse('plans'))

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

    def test_edit_a_plan_and_make_it_fail_validation(self):

        self.open(reverse('page_plan_new_template', args=(1,)))
        
        #test requires a default IR account be set        
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()

        #select the Trio Sample Grouping
        self.wd.find_element_by_xpath('//input[@name="sampleGrouping" and @value="4"]').click()#Trio

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
        self.open(reverse('plans'))

        barcoded_template_id = self.get_latest_planned_experiment()['id']

        #now open the plan run link
        self.open(reverse('page_plan_new_plan', args=(barcoded_template_id,)))
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()

        self.wd.wait_for_css("#planName").send_keys('NEW-PLAN-TO-BE-EDITED-ampliseq-dna-{0}'.format(current_dt))

        #now enter two samples
        sample_name_1 = 'Passing-Sample-Name-1'
        sample_ext_id_1 = 'Passing-Sample-External-Id-1'
        sample_description_1 = 'Passing-Sample-Desc-1'

        sample_name_2 = 'Passing-Sample-Name-2'
        sample_ext_id_2 = 'Passing-Sample-External-Id-2'
        sample_description_2 = 'Passing-Sample-Desc-2'

        sample_name_3 = 'Passing-Sample-Name-3'
        sample_ext_id_3 = 'Passing-Sample-External-Id-3'
        sample_description_3 = 'Passing-Sample-Desc-3'

        self.wd.find_element_by_xpath('//input[@name="barcodeSampleName81"]').send_keys(sample_name_1)
        self.wd.find_element_by_xpath('//input[@name="barcodeSampleExternalId81"]').send_keys(sample_ext_id_1)
        self.wd.find_element_by_xpath('//input[@name="barcodeSampleDescription81"]').send_keys(sample_description_1)

        self.wd.find_element_by_xpath('//select[@name="irGender81"]').find_elements_by_tag_name('option')[1].click()#Male
        self.wd.find_element_by_xpath('//select[@name="irRelation81"]').find_elements_by_tag_name('option')[1].click()#Trio
        self.wd.find_element_by_xpath('//select[@name="irRelationRole81"]').find_elements_by_tag_name('option')[1].click()#Father

        self.wd.find_element_by_xpath('//input[@name="barcodeSampleName82"]').send_keys(sample_name_2)
        self.wd.find_element_by_xpath('//input[@name="barcodeSampleExternalId82"]').send_keys(sample_ext_id_2)
        self.wd.find_element_by_xpath('//input[@name="barcodeSampleDescription82"]').send_keys(sample_description_2) 

        self.wd.find_element_by_xpath('//select[@name="irGender82"]').find_elements_by_tag_name('option')[2].click()#Female
        self.wd.find_element_by_xpath('//select[@name="irRelation82"]').find_elements_by_tag_name('option')[1].click()#Trio
        self.wd.find_element_by_xpath('//select[@name="irRelationRole82"]').find_elements_by_tag_name('option')[2].click()#Mother


        self.wd.find_element_by_xpath('//input[@name="barcodeSampleName83"]').send_keys(sample_name_3)
        self.wd.find_element_by_xpath('//input[@name="barcodeSampleExternalId83"]').send_keys(sample_ext_id_3)
        self.wd.find_element_by_xpath('//input[@name="barcodeSampleDescription83"]').send_keys(sample_description_3) 

        self.wd.find_element_by_xpath('//select[@name="irGender83"]').find_elements_by_tag_name('option')[3].click()#Unknown
        self.wd.find_element_by_xpath('//select[@name="irRelation83"]').find_elements_by_tag_name('option')[1].click()#Trio
        self.wd.find_element_by_xpath('//select[@name="irRelationRole83"]').find_elements_by_tag_name('option')[3].click()#Proband

        #now save the plan.  We except the plan to PASS validation
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_ajax()

        self.open(reverse('plans'))

        latest_plan = self.get_latest_planned_experiment()
        #now open that plan for editing
        self.open(reverse('page_plan_edit_plan', args=(latest_plan['id'],)))
        self.wd.wait_for_ajax()

        #Change Father to Mother
        self.wd.find_element_by_xpath('//select[@name="irRelationRole81"]').find_elements_by_tag_name('option')[2].click()#Mother 

        #now save the plan.  We except the plan to FAIL validation
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_ajax()
        
        time.sleep(3)

        #now click on IGNORE ERRORS
        self.wd.find_element_by_xpath('//button[@value="cancel"]').click()

        #now fix sample 1 and change to FATHER
        self.wd.find_element_by_xpath('//select[@name="irRelationRole81"]').find_elements_by_tag_name('option')[1].click()#Father
        
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
