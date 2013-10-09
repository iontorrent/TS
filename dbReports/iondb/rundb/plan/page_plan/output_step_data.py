# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
import logging
from iondb.rundb.models import Project
import re
from iondb.rundb.plan.page_plan.step_names import StepNames
logger = logging.getLogger(__name__)

class OutputFieldNames():

    PROJECTS = 'projects'
    NEW_PROJECTS = 'newProjects'



class OutputStepData(AbstractStepData):

    def __init__(self):
        super(OutputStepData, self).__init__()
        self.resourcePath = 'rundb/plan/page_plan/page_plan_output.html'
        self.prepopulatedFields[OutputFieldNames.PROJECTS] = list(Project.objects.filter(public=True).order_by('name'))
        self.savedFields[OutputFieldNames.PROJECTS] = None
        self.savedFields[OutputFieldNames.NEW_PROJECTS] = None
        self.savedListFieldNames.append(OutputFieldNames.PROJECTS)

    def validateField(self, field_name, new_field_value):
        if field_name == OutputFieldNames.NEW_PROJECTS and new_field_value:
            valid = True
            if not re.match('[a-zA-Z0-9\\-_\\., ]+$', new_field_value):
                valid = False
                self.validationErrors[field_name] = "Project names should contain only letters, numbers, dashes, and underscores."

            projects = new_field_value.split(',');
            for project in projects:
                if len(re.sub('\\s+|\\s+$', '', project)) > 64:
                    valid = False
                    self.validationErrors[field_name] = ' Project name length should be 64 characters maximum.';

            if valid:
                self.validationErrors.pop(field_name, None)
        elif field_name == OutputFieldNames.NEW_PROJECTS:
            self.validationErrors.pop(field_name, None)
            

    def getStepName(self):
        return StepNames.OUTPUT
    
    def updateSavedObjectsFromSavedFields(self):
        pass
    
    def updateFromStep(self, updated_step):
        pass