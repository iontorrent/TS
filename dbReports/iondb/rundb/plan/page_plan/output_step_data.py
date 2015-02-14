# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.models import Project
import re
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.plan_validator import validate_projects


class OutputFieldNames():

    PROJECTS = 'projects'
    NEW_PROJECTS = 'newProjects'



class OutputStepData(AbstractStepData):

    def __init__(self, sh_type):
        super(OutputStepData, self).__init__(sh_type)
        self.resourcePath = 'rundb/plan/page_plan/page_plan_output.html'
        self.prepopulatedFields[OutputFieldNames.PROJECTS] = list(Project.objects.filter(public=True).order_by('name'))
        self.savedFields[OutputFieldNames.PROJECTS] = None
        self.savedFields[OutputFieldNames.NEW_PROJECTS] = None
        self.savedListFieldNames.append(OutputFieldNames.PROJECTS)

        self.sh_type = sh_type
        
    def validateField(self, field_name, new_field_value):
        if field_name == OutputFieldNames.NEW_PROJECTS:
            self.validationErrors.pop(field_name, None)
            if new_field_value:
                errors, trimmed_projectNames = validate_projects(new_field_value)
                if errors:
                    self.validationErrors[field_name] = '\n'.join(errors)
            

    def getStepName(self):
        return StepNames.OUTPUT
    
    def updateSavedObjectsFromSavedFields(self):
        pass
    
    def updateFromStep(self, updated_step):
        pass
