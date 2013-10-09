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

    def __init__(self):
        self.resourcePath=None
        self.savedFields = {}
        self.savedListFieldNames = []
        self.savedObjects = {}
        self.prepopulatedFields = {}
        self.validationErrors = {}
        self._dependsOn = []

    def getCurrentSavedFieldDict(self):
        return self.savedFields

    def updateSavedFieldValuesFromRequest(self, request):
        changed = False
        for key in self.savedFields.keys():
            if self.updateSavedFieldValueFromRequest(request, key):
                changed = True
        
        self.validate()
        return changed

    def validate(self):
        for key in self.savedFields.keys():
            self.validateField(key, self.savedFields[key])
        self.validateStep()

    def updateSavedFieldValueFromRequest(self, request, saved_field_name):
        retval = False
        if request.POST.has_key(saved_field_name):
            if saved_field_name in self.savedListFieldNames:
                new_value = request.POST.getlist(saved_field_name)
            else:
                new_value = request.POST.get(saved_field_name, None)
            if new_value != self.savedFields[saved_field_name] and str(new_value) != str(self.savedFields[saved_field_name]):
                self.savedFields[saved_field_name] = new_value
                retval = True
        elif self.savedFields[saved_field_name]:
            self.savedFields[saved_field_name] = None
            retval = True
        return retval

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

    def updateSavedObjectsFromSavedFields(self):
        raise NotImplementedError('you must use a subclass to invoke this method')

    def getStepName(self):
        raise NotImplementedError('you must use a subclass to invoke this method')

    def updateFromStep(self, step_depended_on):
        raise NotImplementedError('you must use a subclass to invoke this method')

    def getPrePopulatedFieldDict(self):
        return self.prepopulatedFields
    
    def hasErrors(self):
        return len(self.validationErrors) > 0