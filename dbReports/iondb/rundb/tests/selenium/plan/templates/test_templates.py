# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
import time
from datetime import datetime
from django.core.urlresolvers import reverse
from django.test import LiveServerTestCase
from iondb.rundb.test import SeleniumTestCase

from selenium.webdriver.support.ui import Select

import logging
logger = logging.getLogger(__name__)

class TestCreateNewTemplate(SeleniumTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestCreateNewTemplate, cls).setUpClass()       

    def test_create_ampliseq_dna_no_ref_template(self):
        """
        Test if an AmpliSeq DNA template with no reference can be created from scratch
        - IR-enabled
        - non-barcoded
        - no reference or BED files selected
        """  
        self.open(reverse('page_plan_new_template', args=(1,)))
         
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()
        irworkflow = self.wd.find_element_by_xpath('//select[@name="irworkflow"]')
        #choose the second workflow option
        irworkflow.find_elements_by_tag_name('option')[1].click()
        #now navigate to the Kits chevron and choose the Chip-Type
        self.wd.wait_for_css("#Kits").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_css("#chipType").find_elements_by_tag_name('option')[2].click()#(314v2)Ion 314v2 Chip
 
        self.wd.find_element_by_id("OneTouch__templatekitType").click()
        self.wd.wait_for_css("#templateKit").find_elements_by_tag_name('option')[1].click() ##ION PGM
        self.wd.wait_for_css("#barcodeId").find_elements_by_tag_name('option')[1].click()#IonSet1
         
        #now navigate to the Save chevron
        self.wd.wait_for_css("#Save_template").find_elements_by_tag_name('a')[0].click()
        current_dt = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
        self.wd.wait_for_css("#templateName").send_keys('ampliseq-dna-no-ref-{0}'.format(current_dt))
 
        #select None for reference
        self.wd.find_element_by_id("default_reference").find_elements_by_tag_name('option')[0].click() ##None
                         
        #now save the template
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
         
        self.open(reverse('plan_templates'))
 
        #now we pull the template from the API
        latest_template_as_json = self.get_latest_planned_experiment()
         
        logger.info( ">>> test_create_ampliseq_dna_no_ref_template... latest_template_as_json=%s" %(latest_template_as_json)) 
              
        self.assertEqual(latest_template_as_json['planDisplayedName'], 'ampliseq-dna-no-ref-{0}'.format(current_dt))
        self.assertEqual(latest_template_as_json['chipType'], '314v2')
        self.assertEqual(latest_template_as_json['isReusable'], True)
        self.assertEqual(latest_template_as_json['isSystemDefault'], False)
                         
        #now delete the template
        self.delete_planned_experiment(latest_template_as_json['id'])
 
    
    def test_create_ampliseq_dna_with_ref_template(self):
        """
        Test if an AmpliSeq DNA template can be created from scratch
        - IR-enabled
        - non-barcoded
        - reference and BED files selected
        """ 
        self.open(reverse('page_plan_new_template', args=(1,)))
         
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()
        irworkflow = self.wd.find_element_by_xpath('//select[@name="irworkflow"]')
        #choose the second workflow option
        irworkflow.find_elements_by_tag_name('option')[1].click()
        #now navigate to the Kits chevron and choose the Chip-Type
        self.wd.wait_for_css("#Kits").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_css("#chipType").find_elements_by_tag_name('option')[2].click()#(314v2)Ion 314v2 Chip
 
        self.wd.find_element_by_id("OneTouch__templatekitType").click()
        self.wd.wait_for_css("#templateKit").find_elements_by_tag_name('option')[1].click() ##ION PGM
        self.wd.wait_for_css("#barcodeId").find_elements_by_tag_name('option')[1].click()#IonSet1
          
        #now navigate to the Save chevron
        self.wd.wait_for_css("#Save_template").find_elements_by_tag_name('a')[0].click()
        current_dt = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
        self.wd.wait_for_css("#templateName").send_keys('ampliseq-dna-ref-{0}'.format(current_dt))
 
        #not all TS instances will have hg19 ref, select ecoli instead for this test
        ##self.wd.find_element_by_id("default_reference").find_elements_by_tag_name('option')[0].click() ##None
        select = Select(self.wd.find_element_by_id("default_reference"))
        select.select_by_visible_text("e_coli_dh10b(E. coli DH10B)")
         
        select = Select(self.wd.find_element_by_id("default_targetBedFile"))
        select.select_by_visible_text("ecoli.bed")
         
        select = Select(self.wd.find_element_by_id("default_hotSpotBedFile"))
        select.select_by_visible_text("ecoli_hotspot.bed")
         
        #now save the template
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
         
        self.open(reverse('plan_templates'))
 
        #now we pull the template from the API
        latest_template_as_json = self.get_latest_planned_experiment()
         
        logger.info( ">>> test_create_ampliseq_dna_no_ref_template... latest_template_as_json=%s" %(latest_template_as_json)) 
              
        self.assertEqual(latest_template_as_json['planDisplayedName'], 'ampliseq-dna-ref-{0}'.format(current_dt))
        self.assertEqual(latest_template_as_json['chipType'], '314v2')
        self.assertEqual(latest_template_as_json['isReusable'], True)
        self.assertEqual(latest_template_as_json['isSystemDefault'], False)
                         
        #now delete the template
        self.delete_planned_experiment(latest_template_as_json['id'])


    def test_create_ampliseq_dna_no_name_with_ref_no_target_template(self):
        """
        Test if validation error messages are displayed 
        - create an AmpliSeq DNA template with 
        - no template name
        - reference selected but no target region BED file
        """
        self.open(reverse('page_plan_new_template', args=(1,)))
        
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()
        irworkflow = self.wd.find_element_by_xpath('//select[@name="irworkflow"]')
        #choose the second workflow option
        irworkflow.find_elements_by_tag_name('option')[1].click()
        #now navigate to the Kits chevron and choose the Chip-Type
        self.wd.wait_for_css("#Kits").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_css("#chipType").find_elements_by_tag_name('option')[2].click()#(314v2)Ion 314v2 Chip

        self.wd.find_element_by_id("OneTouch__templatekitType").click()
        self.wd.wait_for_css("#templateKit").find_elements_by_tag_name('option')[1].click() ##ION PGM
        self.wd.wait_for_css("#barcodeId").find_elements_by_tag_name('option')[1].click()#IonSet1
        
        #now navigate to the Save chevron
        self.wd.wait_for_css("#Save_template").find_elements_by_tag_name('a')[0].click()

        #not all TS instances will have hg19 ref, select ecoli instead for this test
        ##self.wd.find_element_by_id("default_reference").find_elements_by_tag_name('option')[0].click() ##None
        select = Select(self.wd.find_element_by_id("default_reference"))
        select.select_by_visible_text("e_coli_dh10b(E. coli DH10B)")

        #now save the template
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        self.wd.wait_for_ajax()
        time.sleep(3)
           
        #verify if error messages are displayed 
        message_count = 0
        expected_message_count = 2
                
        webElements = self.wd.find_elements_by_tag_name("p")

        for webElement in webElements:
            content = webElement.get_attribute("outerHTML")
            if "Template Name is required" in content:
                message_count += 1
                logger.info( ">>> test_create_ampliseq_dna_no_name_with_ref_no_target_template... content=%s" %(content)) 
            if "Target Regions BED File is required" in content:
                message_count += 1
                logger.info( ">>> test_create_ampliseq_dna_no_name_with_ref_no_target_template... content=%s" %(content)) 

        self.assertEqual(message_count, expected_message_count)
        
