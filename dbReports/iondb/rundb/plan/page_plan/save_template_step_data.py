# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.plan_validator import validate_plan_name, validate_notes, validate_QC

from iondb.rundb.models import QCType

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import logging
logger = logging.getLogger(__name__)

class SaveTemplateStepDataFieldNames():
    RESOURCE_PATH = "rundb/plan/page_plan/page_plan_save_template.html"
    TEMPLATE_NAME = "templateName";
    SET_AS_FAVORITE = "setAsFavorite";
    NOTE = "note";

    LIMS_META = 'LIMS_meta'
    META = 'meta' 


class MonitoringFieldNames():
    QC_TYPES = 'qcTypes'
    
class SaveTemplateStepData(AbstractStepData):

    def __init__(self, sh_type):
        super(SaveTemplateStepData, self).__init__(sh_type)
        self.resourcePath = SaveTemplateStepDataFieldNames.RESOURCE_PATH
        self.savedFields = OrderedDict()
                
        self.savedFields[SaveTemplateStepDataFieldNames.TEMPLATE_NAME] = None
        self.savedFields[SaveTemplateStepDataFieldNames.SET_AS_FAVORITE] = None
        self.savedFields[SaveTemplateStepDataFieldNames.NOTE] = None

        self.savedFields[SaveTemplateStepDataFieldNames.LIMS_META] = None
        self.savedFields[SaveTemplateStepDataFieldNames.META] = {}
        
        self.sh_type = sh_type

        # Monitoring
        self.qcNames = []
        all_qc_types = list(QCType.objects.all().order_by('qcName'))
        self.prepopulatedFields[MonitoringFieldNames.QC_TYPES] = all_qc_types
        for qc_type in all_qc_types:
            self.savedFields[qc_type.qcName] = qc_type.defaultThreshold
            self.qcNames.append(qc_type.qcName)

        
    def validateField(self, field_name, new_field_value):
        self.validationErrors.pop(field_name, None)
        
        if field_name == SaveTemplateStepDataFieldNames.TEMPLATE_NAME:
            errors = validate_plan_name(new_field_value, 'Template Name')
            if errors:
                self.validationErrors[field_name] = '\n'.join(errors)
        elif field_name == SaveTemplateStepDataFieldNames.NOTE:
            errors = validate_notes(new_field_value)
            if errors:
                self.validationErrors[field_name] = '\n'.join(errors)    

        # validate all qc thresholds must be positive integers
        elif field_name in self.qcNames:
            errors = validate_QC(new_field_value, field_name)
            if errors:
                self.validationErrors[field_name] = errors[0]

        else:
            for section, sectionObj in self.step_sections.items():
                self.validationErrors.pop(field_name, None)

                
    def getStepName(self):
        return StepNames.SAVE_TEMPLATE
    
    def updateSavedObjectsFromSavedFields(self):
        pass
    

    def getDefaultSection(self):
        """
        Sections are optional for a step.  Return the default section      
        """
        if not self.step_sections:
            return None
        return self.step_sections.get(StepNames.REFERENCE, None)


    def getDefaultSectionSavedFieldDict(self):
        """
        Sections are optional for a step.  Return the savedFields dictionary of the default section if it exists.
        Otherwise, return an empty dictionary        
        """
        default_value = {}
        if not self.step_sections:
            return default_value

        sectionObj = self.step_sections.get(StepNames.REFERENCE, None)
        if  sectionObj:
            #logger.debug("save_template_step_data.getDefaultSectionSavedFieldDict() sectionObj.savedFields=%s" %(sectionObj.savedFields))
            return sectionObj.savedFields
        else:
            return default_value


    def getDefaultSectionPrepopulatedFieldDict(self):
        """
        Sections are optional for a step.  Return the prepopuldatedFields dictionary of the default section if it exists.
        Otherwise, return an empty dictionary        
        """
        default_value = {}
        if not self.step_sections:
            return default_value

        sectionObj = self.step_sections.get(StepNames.REFERENCE, None)
        if  sectionObj:
            ##logger.debug("save_template_step_data.getDefaultSectionPrepopulatedFieldsFieldDict() sectionObj.prepopulatedFields=%s" %(sectionObj.prepopulatedFields))
            return sectionObj.prepopulatedFields
        else:
            return default_value