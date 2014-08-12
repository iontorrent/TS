# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from datetime import datetime
from django.core.urlresolvers import reverse
from django.test import LiveServerTestCase
from iondb.rundb.test import SeleniumTestCase

class TestCreateNewTemplate(SeleniumTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestCreateNewTemplate, cls).setUpClass()       

    def test_create_ampliseq_dna_template(self):

        self.open(reverse('page_plan_new_template', args=(1,)))
        
        #now we need to wait for the ajax call which loads the IR accounts to finish
        self.wd.wait_for_ajax()
        irworkflow = self.wd.find_element_by_xpath('//select[@name="irworkflow"]')
        #choose the second workflow option
        irworkflow.find_elements_by_tag_name('option')[1].click()
        #now navigate to the Kits chevron and choose the Chip-Type
        self.wd.wait_for_css("#Kits").find_elements_by_tag_name('a')[0].click()
        self.wd.wait_for_css("#chipType").find_elements_by_tag_name('option')[2].click()#(314v2)Ion 314v2 Chip

        self.wd.wait_for_css("#OneTouch__templatekitType").click()
        self.wd.wait_for_css("#OneTouch__templatingKit").find_elements_by_tag_name('option')[1].click()
        
        #now navigate to the Save chevron
        self.wd.wait_for_css("#Save_template").find_elements_by_tag_name('a')[0].click()
        current_dt = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
        self.wd.wait_for_css("#templateName").send_keys('ampliseq-dna-{0}'.format(current_dt))
        #now save the template
        self.wd.find_element_by_xpath('//a[@class="btn btn-primary btn-100 pull-right"]').click()
        
        self.open(reverse('plans'))

        #now we pull the template from the API
        latest_template_as_json = self.get_latest_planned_experiment()
        self.assertEqual(latest_template_as_json['planDisplayedName'], 'ampliseq-dna-{0}'.format(current_dt))
        self.assertEqual(latest_template_as_json['chipType'], '314v2')
        #now delete the template
        self.delete_planned_experiment(latest_template_as_json['id'])

        
