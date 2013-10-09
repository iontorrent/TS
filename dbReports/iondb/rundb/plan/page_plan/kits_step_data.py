# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import logging
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.models import KitInfo, Chip, dnaBarcode, LibraryKey,\
    ThreePrimeadapter, GlobalConfig
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.page_plan.application_step_data import ApplicationFieldNames

logger = logging.getLogger(__name__)


class KitsFieldNames():

    SAMPLE_PREPARATION_KIT = 'samplePreparationKit'
    SAMPLE_PREP_KITS = 'samplePrepKits'
    LIBRARY_KIT_NAME = 'librarykitname'
    LIB_KITS = 'libKits'
    LIBRARY_KEY = 'libraryKey'
    FORWARD_LIB_KEYS = 'forwardLibKeys'
    FORWARD_3_PRIME_ADAPTER = 'forward3primeAdapter'
    FORWARD_3_ADAPTERS = 'forward3Adapters'
    TEMPLATE_KIT_NAME = 'templatekitname'
    ONE_TOUCH = 'OneTouch' 
    KIT_VALUES = 'kit_values'
    APPLICATION_DEFAULT = 'applDefault'
    TEMPLATE_KIT_TYPES = 'templateKitTypes'
    ION_CHEF = 'IonChef'
    TEMPLATE_KIT_TYPE = 'templatekitType'
    SEQUENCE_KIT_NAME = 'sequencekitname'
    
    SEQ_KITS = 'seqKits'
    CONTROL_SEQUENCE = 'controlsequence'
    CONTROL_SEQ_KITS = 'controlSeqKits'
    CHIP_TYPE = 'chipType'
    CHIP_TYPES = 'chipTypes'
    BARCODE_ID = 'barcodeId'
    BARCODES = 'barcodes'
    IS_DUPLICATED_READS = 'isDuplicateReads'
    FLOWS = 'flows'
    
    TEMPLATE_KITS = "templateKits"
    
    ONE_TOUCH_AVALANCHE = "Avalanche"   
    AVALANCHE_FORWARD_3_PRIME_ADAPTERS = 'avalancheForward3PrimeAdapters'
    AVALANCHE_FORWARD_3_PRIME_ADAPTER = 'avalancheForward3PrimeAdapter'
    AVALANCHE_TEMPLATE_KIT_NAME = 'avalancheTemplateKitName'
    AVALANCHE_SEQUENCE_KIT_NAME = 'avalancheSequencekitname'
    AVALANCHE_FLOWS = "avalancheFlows"

    NON_AVALANCHE_FORWARD_3_PRIME_ADAPTER = 'nonAvalancheForward3PrimeAdapter'
    NON_AVALANCHE_TEMPLATE_KIT_NAME = 'nonAvalancheTemplateKitName'
    NON_AVALANCHE_SEQUENCE_KIT_NAME = 'nonAvalancheSequencekitname' 
  
class KitsStepData(AbstractStepData):


    def __init__(self):
        super(KitsStepData, self).__init__()
        self.resourcePath = 'rundb/plan/page_plan/page_plan_kits.html'
        
        #20130827-test
        ##self._dependsOn.append(StepNames.IONREPORTER) 
                
        self._dependsOn.append(StepNames.APPLICATION) 
        self._dependsOn.append(StepNames.BARCODE_BY_SAMPLE)
        
        self.savedFields[KitsFieldNames.SAMPLE_PREPARATION_KIT] = None
        self.prepopulatedFields[KitsFieldNames.SAMPLE_PREP_KITS] = KitInfo.objects.filter(kitType='SamplePrepKit', isActive=True).order_by('name')
        
        self.savedFields[KitsFieldNames.LIBRARY_KIT_NAME] = None
        self.prepopulatedFields[KitsFieldNames.LIB_KITS] = KitInfo.objects.filter(kitType='LibraryKit', isActive=True).order_by("name")
        
        self.savedFields[KitsFieldNames.LIBRARY_KEY] = None
        self.prepopulatedFields[KitsFieldNames.FORWARD_LIB_KEYS] = LibraryKey.objects.filter(direction='Forward', runMode='single').order_by('-isDefault', 'name')
        self.savedFields[KitsFieldNames.LIBRARY_KEY] = self.prepopulatedFields[KitsFieldNames.FORWARD_LIB_KEYS][0].sequence
        
        self.savedFields[KitsFieldNames.FORWARD_3_PRIME_ADAPTER] = None
        self.prepopulatedFields[KitsFieldNames.FORWARD_3_ADAPTERS] = ThreePrimeadapter.objects.filter(direction='Forward', runMode='single').order_by('-isDefault', 'chemistryType', 'name')
        self.savedFields[KitsFieldNames.FORWARD_3_PRIME_ADAPTER] = self.prepopulatedFields[KitsFieldNames.FORWARD_3_ADAPTERS][0].sequence
        
        self.savedFields[KitsFieldNames.AVALANCHE_FORWARD_3_PRIME_ADAPTER] = None
        self.prepopulatedFields[KitsFieldNames.AVALANCHE_FORWARD_3_PRIME_ADAPTERS] = ThreePrimeadapter.objects.filter(direction='Forward', runMode='single', chemistryType = 'avalanche').order_by('-isDefault', 'name')
        self.savedFields[KitsFieldNames.AVALANCHE_FORWARD_3_PRIME_ADAPTER] = self.prepopulatedFields[KitsFieldNames.AVALANCHE_FORWARD_3_PRIME_ADAPTERS][0].sequence
        
        self.savedFields[KitsFieldNames.TEMPLATE_KIT_NAME] = None
        self.savedFields[KitsFieldNames.TEMPLATE_KIT_TYPE] = KitsFieldNames.ONE_TOUCH
        
        oneTouchDict = {
                KitsFieldNames.KIT_VALUES          : KitInfo.objects.filter(kitType__in=['TemplatingKit', 'AvalancheTemplateKit'], isActive=True).order_by("name"),
                KitsFieldNames.APPLICATION_DEFAULT : None
                }
        
        ionChefDict = {
                KitsFieldNames.KIT_VALUES          : KitInfo.objects.filter(kitType='IonChefPrepKit', isActive=True).order_by("name"),
                KitsFieldNames.APPLICATION_DEFAULT : None
                }
        
        oneTouchAvalancheDict = {
                KitsFieldNames.KIT_VALUES          : KitInfo.objects.filter(kitType__in=['AvalancheTemplateKit'], isActive=True).order_by("name"),
                KitsFieldNames.APPLICATION_DEFAULT : None
                }
        
        self.prepopulatedFields[KitsFieldNames.TEMPLATE_KIT_TYPES] = {
                                                                KitsFieldNames.ONE_TOUCH : oneTouchDict, 
                                                                KitsFieldNames.ION_CHEF  : ionChefDict,
                                                                KitsFieldNames.ONE_TOUCH_AVALANCHE : oneTouchAvalancheDict
                                                                }
        
        self.savedFields[KitsFieldNames.SEQUENCE_KIT_NAME] = None
        self.prepopulatedFields[KitsFieldNames.SEQ_KITS] = KitInfo.objects.filter(kitType='SequencingKit', isActive=True).order_by("name")
                            
        self.savedFields[KitsFieldNames.TEMPLATE_KIT_NAME] = None
        self.prepopulatedFields[KitsFieldNames.TEMPLATE_KITS] = KitInfo.objects.filter(kitType__in=['TemplatingKit', 'AvalancheTemplateKit'], isActive=True).order_by("name")

        self.savedFields[KitsFieldNames.CONTROL_SEQUENCE] = None
        self.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', isActive=True).order_by("name")
        
        self.savedFields[KitsFieldNames.CHIP_TYPE] = None
        self.prepopulatedFields[KitsFieldNames.CHIP_TYPES] = list(Chip.objects.filter(isActive=True).order_by('description', 'name').distinct('description'))
        
        self.savedFields[KitsFieldNames.BARCODE_ID] = None
        self.prepopulatedFields[KitsFieldNames.BARCODES] = list(dnaBarcode.objects.values('name').distinct().order_by('name'))
        
        self.savedFields[KitsFieldNames.IS_DUPLICATED_READS] = GlobalConfig.objects.all()[0].mark_duplicates
        
        self.savedFields[KitsFieldNames.FLOWS] = 0


    def getStepName(self):
        return StepNames.KITS

    def updateSavedObjectsFromSavedFields(self):
        pass
    
    def updateFromStep(self, updated_step):

        logger.debug("ENTER kits_step_data.updateFromStep() updated_step.stepName=%s" %(updated_step.getStepName()))

        if updated_step.getStepName() == StepNames.BARCODE_BY_SAMPLE:
            self.savedFields[KitsFieldNames.BARCODE_ID] = updated_step.savedFields[KitsFieldNames.BARCODE_ID]

        if updated_step.getStepName() == StepNames.APPLICATION and updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]:
            applProduct = updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]
            logger.debug("kits_step_data.updateFromStep() Updating kits for applproduct %s" % applProduct.productCode)
            
            if applProduct.defaultSamplePrepKit:
                self.savedFields[KitsFieldNames.SAMPLE_PREPARATION_KIT] = applProduct.defaultSamplePrepKit.name
            
            if applProduct.defaultLibraryKit:
                self.savedFields[KitsFieldNames.LIBRARY_KIT_NAME] = applProduct.defaultLibraryKit.name
            
            if applProduct.defaultTemplateKit:
                self.prepopulatedFields[KitsFieldNames.TEMPLATE_KIT_TYPES][KitsFieldNames.ONE_TOUCH][KitsFieldNames.APPLICATION_DEFAULT] = applProduct.defaultTemplateKit
            
            if applProduct.defaultIonChefPrepKit:
                self.prepopulatedFields[KitsFieldNames.TEMPLATE_KIT_TYPES][KitsFieldNames.ION_CHEF][KitsFieldNames.APPLICATION_DEFAULT] = applProduct.defaultIonChefPrepKit
                   
            if applProduct.defaultAvalancheTemplateKit:
                self.prepopulatedFields[KitsFieldNames.TEMPLATE_KIT_TYPES][KitsFieldNames.ONE_TOUCH_AVALANCHE][KitsFieldNames.APPLICATION_DEFAULT] = applProduct.defaultAvalancheTemplateKit
                self.savedFields[KitsFieldNames.AVALANCHE_TEMPLATE_KIT_NAME] = applProduct.defaultAvalancheTemplateKit.name
                
                logger.debug("kits_step_data.updateFromStep() defaultAvalancheTemplateKit=%s" %(applProduct.defaultAvalancheTemplateKit.name))
                                
            if applProduct.defaultTemplateKit:
                self.savedFields[KitsFieldNames.TEMPLATE_KIT_NAME] = applProduct.defaultTemplateKit.name

   
            if applProduct.defaultSequencingKit:
                self.savedFields[KitsFieldNames.SEQUENCE_KIT_NAME] = applProduct.defaultSequencingKit.name
            
            if applProduct.defaultAvalancheSequencingKit:
                self.savedFields[KitsFieldNames.AVALANCHE_SEQUENCE_KIT_NAME] = applProduct.defaultAvalancheSequencingKit.name
                
            if applProduct.defaultControlSeqKit:
                self.savedFields[KitsFieldNames.CONTROL_SEQUENCE] = applProduct.defaultControlSeqKit.name
            
            if applProduct.isDefaultBarcoded and applProduct.defaultBarcodeKitName:
                self.savedFields[KitsFieldNames.BARCODE_ID] = applProduct.defaultBarcodeKitName
            
            if applProduct.defaultFlowCount > 0:
                self.savedFields[KitsFieldNames.FLOWS] = applProduct.defaultFlowCount
                
                logger.debug("kits_step_data.updateFromStep() USE APPLPRODUCT - flowCount=%s" %(str(self.savedFields[KitsFieldNames.FLOWS])))
            else:
                if applProduct.isDefaultPairedEnd and applProduct.defaultPairedEndSequencingKit:
                    self.savedFields[KitsFieldNames.FLOWS] = applProduct.defaultPairedEndSequencingKit.flowCount
                elif applProduct.defaultSequencingKit:
                    self.savedFields[KitsFieldNames.FLOWS] = applProduct.defaultSequencingKit.flowCount                
                                
                logger.debug("kits_step_data.updateFromStep() USE SEQ KIT- flowCount=%s" %(str(self.savedFields[KitsFieldNames.FLOWS])))


            nonAvalanche3PrimeAdapters = ThreePrimeadapter.objects.filter(direction='Forward', runMode='single').exclude(chemistryType = 'avalanche').order_by('-isDefault', 'name')
            self.savedFields[KitsFieldNames.NON_AVALANCHE_FORWARD_3_PRIME_ADAPTER] = nonAvalanche3PrimeAdapters[0].sequence
            if applProduct.defaultTemplateKit:
                self.savedFields[KitsFieldNames.NON_AVALANCHE_TEMPLATE_KIT_NAME] = applProduct.defaultTemplateKit.name
            if applProduct.defaultSequencingKit:
                self.savedFields[KitsFieldNames.NON_AVALANCHE_SEQUENCE_KIT_NAME] = applProduct.defaultSequencingKit.name

            #for key, value in self.savedFields.items():
            #    logger.debug("kits_step_data.updateFromStep() savedFields.key=%s; value=%s" %(key, value))
            #for key, value in self.prepopulatedFields.items():
            #    logger.info("kits_step_data.updateFromStep() prepopulatedFields.key=%s; value=%s" %(key, value))
                
                

    def validateField(self, field_name, new_field_value):
        '''
        Flows qc value must be a positive integer
        Chip type is required
        '''
        if field_name == KitsFieldNames.FLOWS:
            valid = False
            try:
                int_val = int(new_field_value)
                if int_val >= 0:
                    valid = True
            except:
                pass
            if valid and field_name in self.validationErrors:
                self.validationErrors.pop(field_name, None)
            elif not valid:
                self.validationErrors[field_name] = "%s must be a positive integer." % field_name

        if field_name == KitsFieldNames.CHIP_TYPE:
            if new_field_value:
                self.validationErrors.pop(field_name, None)
            else:
                self.validationErrors[field_name] = "Chip Type must be selected."
            