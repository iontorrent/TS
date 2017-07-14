# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
'''
Created on May 21, 2013

@author: ionadmin
'''
import logging
logger = logging.getLogger(__name__)


class AbstractStepData(object):

    '''
    Superclass for stepdata classes. SavedFields are fields that users set values for,
    PrepopulatedFields are fields that the step page shows to the user.
    '''

    def __init__(self, sh_type):
        self.resourcePath = None
        self.prev_step_url = '#'
        self.next_step_url = '#'
        self.savedFields = {}
        self.savedListFieldNames = []
        self.savedObjects = {}
        self.prepopulatedFields = {}
        self.validationErrors = {}
        self._dependsOn = []
        self._changedFields = {}
        self.warnings = []

        # some section can appear in multiple chevrons, key is the step name and value is the step_data object
        self.step_sections = {}

        self.sh_type = sh_type

    def getCurrentSavedFieldDict(self):
        return self.savedFields

    def getCurrentSavedObjectDict(self):
        return self.savedObjects

    def getSectionSavedFieldDict(self, sectionName):
        """
        Sections are optional for a step.  Return the savedFields dictionary of the section if it exists.
        Otherwise, return an empty dictionary
        """
        default_value = {}

        if not self.step_sections:
            return default_value
        if sectionName in self.step_sections.keys():
            sectionObj = self.step_sections[sectionName]
            if sectionObj:
                # logger.debug("abstract_step_data.getSectionSavedFieldDict() sectionObj.savedFields=%s" %(sectionObj.savedFields))
                return sectionObj.savedFields
            else:
                return default_value
        return default_value

    def hasStepSections(self):
        """
        Sections are optional for a step. Assume no section by default.
        """
        if self.step_sections:
            return True
        return False

    def getDefaultSection(self):
        """
        Sections are optional for a step.  Let subclass override this.
        TODO: add more section support
        """
        return None

    def getDefaultSectionSavedFieldDict(self):
        """
        Sections are optional for a step.  Let subclass override this.
        TODO: add more section support
        """
        return {}

    def getDefaultSectionPrepopulatedFieldDict(self):
        """
        Sections are optional for a step.  Let subclass override this.
        TODO: add more section support
        """
        return {}

    def updateSavedFieldValuesFromRequest(self, request):
        changed = False
        for key in self.savedFields.keys():
            if self.updateSavedFieldValueFromRequest(request, key):
                changed = True

        for sectionKey, sectionObj in self.step_sections.items():
            if sectionObj:
                for key in sectionObj.getCurrentSavedFieldDict().keys():
                    if sectionObj.updateSavedFieldValueFromRequest(request, key):
                        changed = True

        if changed:
            self.warnings = []

        self.validate()
        return changed

    def validate(self):
        for key in self.savedFields.keys():
            self.validateField(key, self.savedFields[key])

        for sectionKey, sectionObj in self.step_sections.items():
            if sectionObj:
                # logger.debug("abstract_step_data.validate() sectionKey=%s" %(sectionKey))
                for key in sectionObj.getCurrentSavedFieldDict().keys():
                    self.validationErrors.pop(key, None)
                    sectionObj.validateField_in_section(key, sectionObj.savedFields[key])

                # if sectionObj.validationErrors:
                if (len(sectionObj.validationErrors) > 0):
                    logger.debug("after validateField_in_section sectionObj.validationErrors=%s" % (sectionObj.validationErrors))
                    self.validationErrors.update(sectionObj.validationErrors)

        self.validateField_crossField_dependencies(self.savedFields.keys(), self.savedFields)

        self.validateStep()

    def updateSavedFieldValueFromRequest(self, request, saved_field_name):
        retval = False
        if request.POST.has_key(saved_field_name):
            if saved_field_name in self.savedListFieldNames:
                new_value = request.POST.getlist(saved_field_name)
            else:
                new_value = request.POST.get(saved_field_name, None)
            if new_value != self.savedFields[saved_field_name] and str(new_value) != str(self.savedFields[saved_field_name]):
                self.updateChangedFields(saved_field_name, self.savedFields[saved_field_name], new_value)
                self.savedFields[saved_field_name] = new_value
                retval = True
        elif self.savedFields[saved_field_name]:
            self.savedFields[saved_field_name] = None
            retval = True
        return retval

    def updateChangedFields(self, key, old_value, new_value):
        if key in self._changedFields:
            self._changedFields[key][1] = new_value
        else:
            self._changedFields[key] = [old_value, new_value]

    def validateStep(self):
        '''
        default overall validation does nothing
        '''
        return

    def validateField(self, field_name, new_field_value):
        '''
        default field validation does nothing.
        '''
        return

    def validateField_crossField_dependencies(self, fieldNames, fieldValues):
        '''
        default overall cross field validations does nothing
        '''
        return

    def validateField_in_section(self, field_name, new_field_value):
        """
        field validation for a step that acts as a section to another step
        """
        return

    def updateSavedObjectsFromSavedFields(self):
        raise NotImplementedError('you must use a subclass to invoke this method')

    def getStepName(self):
        raise NotImplementedError('you must use a subclass to invoke this method')

    def updateFromStep(self, step_depended_on):
        raise NotImplementedError('you must use a subclass to invoke this method')

    def alternateUpdateFromStep(self, step_depended_on):
        """
        update a step or section with alternate logic based on the step it is depending on
        """
        return

    def getPrePopulatedFieldDict(self):
        return self.prepopulatedFields

    def hasErrors(self):
        return len(self.validationErrors) > 0
