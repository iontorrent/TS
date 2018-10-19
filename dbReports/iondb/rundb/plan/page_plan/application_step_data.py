# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
'''
Created on May 21, 2013

@author: ionadmin
'''
import logging
from django.core.urlresolvers import reverse

from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.models import RunType, ApplProduct, ApplicationGroup
from iondb.rundb.plan.page_plan.step_names import StepNames


logger = logging.getLogger(__name__)


class ApplicationFieldNames():

    APPL_PRODUCT = 'applProduct'
    APPL_PRODUCTS = 'applProducts'   #this is for all applProduct definitions for the selected application and target technique
    APPL_PRODUCTS_CATEGORIZED = 'applProducts_categorized'   #this is for all categorized applProduct definitions for the selected application and target technique    
    RUN_TYPE = 'runType'
    APPLICATION_GROUP_NAME = "applicationGroupName"
    SAMPLE_GROUPING = 'sampleGrouping'
    RUN_TYPES = 'runTypes'
    APPLICATION_GROUPS = 'applicationGroups'
    SAMPLE_GROUPINGS = 'sampleGroupings'
    COLUMNS = 'columns'
    NAME = 'Name'
    RELATIONSHIP_TYPE = 'RelationshipType'
    VALUES = 'Values'
    PLAN_STATUS = "planStatus"
    UPDATE_KITS_DEFAULTS = 'updateKitsDefaults'
    CATEGORIES = "categories"
    INSTRUMENT_TYPE = 'instrumentType'


class ApplicationStepData(AbstractStepData):

    def __init__(self, sh_type):
        super(ApplicationStepData, self).__init__(sh_type)
        self.resourcePath = 'rundb/plan/page_plan/page_plan_application.html'
        self.prev_step_url = reverse("page_plan_ionreporter")
        self.next_step_url = reverse("page_plan_kits")

        # self._dependsOn = [StepNames.IONREPORTER]

        self.savedFields[ApplicationFieldNames.RUN_TYPE] = None
        self.savedFields[ApplicationFieldNames.APPLICATION_GROUP_NAME] = ""
        self.savedFields[ApplicationFieldNames.SAMPLE_GROUPING] = None
        self.prepopulatedFields[ApplicationFieldNames.PLAN_STATUS] = ""

        self.savedObjects[ApplicationFieldNames.RUN_TYPE] = None
        self.savedObjects[ApplicationFieldNames.APPL_PRODUCT] = None
        self.prepopulatedFields[ApplicationFieldNames.APPL_PRODUCTS] = None
        self.prepopulatedFields[ApplicationFieldNames.APPL_PRODUCTS_CATEGORIZED] = None
        self.prepopulatedFields[ApplicationFieldNames.INSTRUMENT_TYPE] = None
        self.savedObjects[ApplicationFieldNames.UPDATE_KITS_DEFAULTS] = True
        self.prepopulatedFields[ApplicationFieldNames.RUN_TYPES] = list(RunType.objects.filter(isActive=True).order_by('description'))

        self.prepopulatedFields[ApplicationFieldNames.CATEGORIES] = ''

#        isSupported = isOCP_enabled()
#        if isSupported:
#            self.prepopulatedFields[ApplicationFieldNames.APPLICATION_GROUPS] = ApplicationGroup.objects.filter(isActive=True).order_by('uid')
#        else:
#            self.prepopulatedFields[ApplicationFieldNames.APPLICATION_GROUPS] = ApplicationGroup.objects.filter(isActive=True).exclude(name = "DNA + RNA").order_by('uid')

        self.prepopulatedFields[ApplicationFieldNames.APPLICATION_GROUPS] = ApplicationGroup.objects.filter(isActive=True).order_by('description')
        
        # self.prepopulatedFields[ApplicationFieldNames.SAMPLE_GROUPINGS] = SampleGroupType_CV.objects.filter(isActive=True).order_by('uid')
        # self._dependsOn = [StepNames.EXPORT]

        self.sh_type = sh_type

    def getStepName(self):
        return StepNames.APPLICATION

    def updateSavedObjectsFromSavedFields(self):
        # logger.debug("ENTER application_step_data.updateSavedObjectsFromSavedFields() self.savedFields=%s" %(self.savedFields))
        previous_run_type = self.savedObjects[ApplicationFieldNames.RUN_TYPE]
        previous_appl_product = self.savedObjects[ApplicationFieldNames.APPL_PRODUCT]

        if self.savedFields[ApplicationFieldNames.RUN_TYPE]:
            self.savedObjects[ApplicationFieldNames.RUN_TYPE] = RunType.objects.get(pk=self.savedFields[ApplicationFieldNames.RUN_TYPE])
            self.savedObjects[ApplicationFieldNames.APPL_PRODUCT] = ApplProduct.get_default_for_runType(self.savedObjects[ApplicationFieldNames.RUN_TYPE].runType,
                            applicationGroupName=self.savedFields[ApplicationFieldNames.APPLICATION_GROUP_NAME],
                            instrumentType = self.prepopulatedFields[ApplicationFieldNames.INSTRUMENT_TYPE])

            self.prepopulatedFields[ApplicationFieldNames.APPL_PRODUCTS] = ApplProduct.objects.filter(isActive=True, isDefaultForInstrumentType=True,
                            applType__runType=self.savedObjects[ApplicationFieldNames.RUN_TYPE].runType,
                            applicationGroup = None)


            if self.savedFields[ApplicationFieldNames.APPLICATION_GROUP_NAME]:
                moreApplProducts = ApplProduct.objects.filter(isActive=True, isDefaultForInstrumentType=True,
                                                          applType__runType=self.savedObjects[ApplicationFieldNames.RUN_TYPE].runType,
                                                          applicationGroup__name=self.savedFields[ApplicationFieldNames.APPLICATION_GROUP_NAME])
                
                # moreApplProducts cannot filtered purely by runType alone since runType and applicationGroup has many-to-many relationship
                # client code applProductToInstrumentType assumes there is only applProduct entry for a given runType + applicationGroup + instrumentType
                if moreApplProducts:
                    self.prepopulatedFields[ApplicationFieldNames.APPL_PRODUCTS] = moreApplProducts

                # an applProduct entry can be categorized for certain specific business requirements
                # client code reads the first the 1st entry of applProductToCategories matching a given runType + applicationGroup + categories
                # do not want to mix this with applProducts since the latter is intended for default characters matching a given runType + applicationGroup + instrumentType
                # assumption: applProduct's categories attribute contains only 1 category value (instead of chained categories) 
                categorizedApplProducts = ApplProduct.objects.filter(isActive=True,
                                                          applType__runType=self.savedObjects[ApplicationFieldNames.RUN_TYPE].runType,
                                                          applicationGroup__name=self.savedFields[ApplicationFieldNames.APPLICATION_GROUP_NAME]).exclude(categories = "")

                if categorizedApplProducts:
                    self.prepopulatedFields[ApplicationFieldNames.APPL_PRODUCTS_CATEGORIZED] = categorizedApplProducts
                else:
                    self.prepopulatedFields[ApplicationFieldNames.APPL_PRODUCTS_CATEGORIZED] = None
                                                                                                   
        else:
            self.savedObjects[ApplicationFieldNames.RUN_TYPE] = None
            self.savedObjects[ApplicationFieldNames.APPL_PRODUCT] = None
            self.prepopulatedFields[ApplicationFieldNames.APPL_PRODUCTS] = None
            self.prepopulatedFields[ApplicationFieldNames.APPL_PRODUCTS_CATEGORIZED] = None

        self.savedObjects[ApplicationFieldNames.UPDATE_KITS_DEFAULTS] = (previous_run_type != self.savedObjects[ApplicationFieldNames.RUN_TYPE]) or (previous_appl_product != self.savedObjects[ApplicationFieldNames.APPL_PRODUCT])

    def updateFromStep(self, step_depended_on):
        pass
    #     if step_depended_on.getStepName() != StepNames.EXPORT:
    #         return
        # ir_sample_groupings = None
        # if step_depended_on.savedFields[ExportFieldNames.IR_OPTIONS] == ExportFieldNames.IR_VERSION_40:
        #     ir_qs = Plugin.objects.filter(name__icontains='IonReporter')
        #     irExists = ir_qs.count() > 0
        #     if irExists and ir_qs[0:1][0].userinputfields and ApplicationFieldNames.COLUMNS in ir_qs[0:1][0].userinputfields:
        #         configJson = ir_qs[0:1][0].userinputfields
        #         for column in configJson[ApplicationFieldNames.COLUMNS]:
        #             if column[ApplicationFieldNames.NAME] == ApplicationFieldNames.RELATIONSHIP_TYPE:
        #                 ir_sample_groupings = column[ApplicationFieldNames.VALUES]
        #                 break

        # if ir_sample_groupings:
        #     self.prepopulatedFields[ApplicationFieldNames.SAMPLE_GROUPINGS] = \
        #         SampleGroupType_CV.objects.filter(isActive=True, iRValue__in=ir_sample_groupings).order_by('uid')
        # else:
        #     self.prepopulatedFields[ApplicationFieldNames.SAMPLE_GROUPINGS] = SampleGroupType_CV.objects.filter(isActive=True).order_by('uid')

    def validateField(self, field_name, new_field_value):
        self.validationErrors.pop(field_name, None)

        if field_name == ApplicationFieldNames.RUN_TYPE:
            if not new_field_value:
                self.validationErrors[field_name] = 'Please select Target Technique'
