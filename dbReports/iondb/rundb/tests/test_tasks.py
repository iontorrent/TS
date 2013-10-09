# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

'''
    iondb.bin.djangoinit messes up logging, which is important for this script, so i had to roll my own.
'''
import os
os.environ['DJANGO_SETTINGS_MODULE'] = "iondb.settings"
import sys
dbreports_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if dbreports_abs_path not in sys.path:
    sys.path.insert(0, dbreports_abs_path)
import iondb.settings
assert iondb.settings.CELERY_ALWAYS_EAGER

from iondb.rundb import models
import tempfile
import shutil
import iondb.rundb.tasks as tasks
import datetime
import mmap
import zipfile
import json
from optparse import OptionParser
import logging
logger = logging.getLogger(__name__)


usage = '''
            Verify that your django setting CELERY_ALWAYS_EAGER is set to true in either settings or local settings.
            Empty out your ContentUpload and Content tables.
            Remove all FileMonitor table entries that are named fake_fixed_design.zip
            Make sure that you have a valid hg19 genome reference.
            Make sure that you have a working BED publisher.
            All option arguments are mandatory.
            usage:
            ionadmin@TSVMware:~$ cd TS_/dbReports/
            ionadmin@TSVMware:~/TS_/dbReports$ionadmin@TSVMware:~/TS_/dbReports$ python iondb/rundb/tests/test_tasks.py -bhttps://uat-8145154.us-east-1.elb.amazonaws.com -uusername -ppassword
        '''

parser = OptionParser(usage)
parser.add_option("-b", "--base_url", dest="base_url",
                  help="Base ampliseq url without a trailing slash e.g. https://ampliseq.com")
parser.add_option("-u", "--username", dest="username",
                  help="The ampliseq username that has permissions to view the list of fixed panels")
parser.add_option("-p", "--password", dest="password",
                  help="The ampliseq password")

class TasksTest():
    FILE_NAME_TO_BAD_WORDS = {"pre_process.py_standard_error.log": ["error", "Error", "ERROR"],
                              "pre_process.py_standard_output.log": ["error", "Error", "ERROR"],
                              "register.py_standard_output.log": ["error", "Error", "ERROR"],
                              "validate.py_standard_error.log": ["error", "Error", "ERROR"],
                              "validate.py_standard_output.log": ["HTTP/1.1 400", "HTTP/1.1 401", "HTTP/1.1 403", "HTTP/1.1 500"]}
    
    def __init__(self, base_url, username, password):
        self.base_url = base_url
        self.username = username
        self.password = password
    
    def does_file_have_word(self, file_path, word):
        ''' check if a file contains a word without loading it into memory '''
        f = open(file_path)
        s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        if word in s:
            return True
        return False

    def get_plan_json(self, zip_path):
        panel_zip = zipfile.ZipFile(zip_path, 'r')
        outer_json = json.loads(panel_zip.read("plan.json"))
        panel_zip.close()
        return outer_json

    def get_expected_content_count(self, outer_json):
        ''' check to see if plan for 3.6 has a non-None designed_bed and hotspot_bed, return 8 if thats the case, 4 otherwise.'''
        inner_plan = outer_json["plan"]["3.6"]
        if "designed_bed" in inner_plan and inner_plan["designed_bed"] and "hotspot_bed" in inner_plan and inner_plan["hotspot_bed"]:
            return 8
        else:
            return 4

    def getReady(self, zip_location):
        '''
            Create a publisher and a monitor, also copy the panel archive under test
            into a temp folder, which ampliseq_zip_upload can do whatever it wants with.
        '''
        monitor = models.FileMonitor()
        monitor.name = "fake_fixed_design.zip"
        monitor.url = "fake_url"
        monitor.celery_task_id = "fake_celery_task_id"
        monitor.created = datetime.datetime.now()
        monitor.updated = datetime.datetime.now()
        monitor.save()
        
        self.temp_intial_dir = tempfile.mkdtemp()
        self.full_path = os.path.join(self.temp_intial_dir, 'fake_fixed_panel.zip')
        shutil.copy(zip_location, self.full_path)

    def finishUp(self):
        '''
            Remove the data left after verifying a single zip file.
            Removing the bed publisher will also nuke Contents and ContentUploads
        '''
        shutil.rmtree(self.temp_intial_dir, ignore_errors=True)
        models.ContentUpload.objects.all().delete()
        models.FileMonitor.objects.filter(name="fake_fixed_design.zip").delete()

    def verify_ampliseq_zip_upload(self, zip_location, outer_json):
        '''
            Run a single ampliseq fixed panel zip through ampliseq_zip_upload. Verify that
            the publisher scripts do not have any errors in the logs and make sure that the Content objects
            get created and saved.
        '''
        self.getReady(zip_location)
        try:
            #make sure that the db only has the publisher and monitor we expect
            pub = models.Publisher.objects.get(name="BED")
            monitor = models.FileMonitor.objects.get(name="fake_fixed_design.zip")
            assert pub
            assert monitor
            
            #run the task
            tasks.ampliseq_zip_upload(args=(self.full_path, monitor.pk), meta='{"reference":"hg19"}')
            
            #verify that the Content and ContentUpload objects got created and saved.
            assert 1 == models.ContentUpload.objects.all().count()
            assert "Successfully Completed" == models.ContentUpload.objects.all()[0:1][0].status
            upload = models.ContentUpload.objects.all()[0:1][0]
            assert self.get_expected_content_count(outer_json) == models.Content.objects.all().count()
            
            #check the logs for any error like keywords
            log_dir = os.path.dirname(upload.file_path)
            for key, value in TasksTest.FILE_NAME_TO_BAD_WORDS.items():
                log_file_path = os.path.join(log_dir, key)
                if os.path.exists(log_file_path):
                    for bad_word in value:
                        assert not self.does_file_have_word(log_file_path, bad_word)
        finally:
            self.finishUp()

    def download_something_sync(self, results_dir, url, name=""):
        '''
            This method needs CELERY_ALWAYS_EAGER=True in your settings or local_settings to work.
        '''
        
        retval = tasks.download_something.delay(url=url,
                                              download_monitor_pk=None,
                                              dir=results_dir,
                                              name=name,
                                              auth=(self.username, self.password))
        path, monitor_id = retval.result
        models.FileMonitor.objects.get(pk=monitor_id).delete()
        return path

    def fill_folder_with_active_zips(self, zip_folder):
        temp_dir = tempfile.mkdtemp()
        active_panel_json_path = self.download_something_sync(temp_dir, "%s/ws/tmpldesign/list/active" % self.base_url)
        json_obj = json.loads(open(active_panel_json_path, 'r').read())
        
        for panel in json_obj["TemplateDesigns"]:
            if panel["pipeline"] == "DNA" and panel["genome"] == "HG19": 
                results_uri = panel["resultsUri"]
                zip_path = self.download_something_sync(results_dir=zip_folder,
                                                      url=self.base_url+results_uri,
                                                      name="panel%s.zip" % str(json_obj["TemplateDesigns"].index(panel)))
                shutil.move(zip_path, zip_folder)
                shutil.rmtree(os.path.dirname(zip_path))
        
        shutil.rmtree(temp_dir)

    def test_all_panel_zips(self):
        '''
            Run each panel zip at base_url through the ampliseq_zip_upload method,
            verify that the invoked scripts produce good results.
            
            Prereqs:
            - hg19 genome reference installed and enabled.
            - working BED publisher must be installed
            - ContentUpload, Content tables completely empty
        '''
        
        zip_folder = tempfile.mkdtemp()
        self.fill_folder_with_active_zips(zip_folder)
        
        #zip_dir = os.path.join(os.path.dirname(__file__), 'res')
        good = 0
        error = 0
        for zip_file in os.listdir(zip_folder):
            if zip_file.endswith("zip"):
                zip_path = os.path.join(zip_folder, zip_file)
                outer_json = self.get_plan_json(zip_path)
                try:
                    self.verify_ampliseq_zip_upload(zip_path, outer_json)
                    logger.info("Successfully validated zip file %s for panel named %s" % (zip_path, outer_json["design_name"]))
                    good += 1
                except:
                    error_str = "Failed to validate zip file %s, for panel named %s, with id %s and short name %s"
                    error_str = error_str % (zip_path, outer_json["design_name"], outer_json["id"], outer_json["design_id"])
                    logger.exception(error_str)
                    error += 1
        logger.info("Verified %s panels, %s were good, %s had errors" % (str(good+error), str(good), str(error)))
        shutil.rmtree(zip_folder)


if __name__ == "__main__":
    (options, args) = parser.parse_args()
    if options.base_url and options.username and options.password:
        tt = TasksTest(base_url=options.base_url,
                       username=options.username,
                       password=options.password)
        tt.test_all_panel_zips()
    else:
        parser.print_usage()
        exit(1)