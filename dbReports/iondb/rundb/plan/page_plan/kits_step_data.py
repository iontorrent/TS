# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import logging
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.models import KitInfo, Chip, dnaBarcode, LibraryKey,\
    ThreePrimeadapter, GlobalConfig
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.page_plan.application_step_data import ApplicationFieldNames
from iondb.utils import validation
from iondb.rundb.plan.plan_validator import validate_flows

logger = logging.getLogger(__name__)


class KitsFieldNames():

    SAMPLE_PREPARATION_KIT = 'samplePreparationKit'
    SAMPLE_PREP_KITS = 'samplePrepKits'
    LIBRARY_KIT_NAME = 'librarykitname'
    LIB_KITS = 'libKits'
    LIBRARY_KEY = 'libraryKey'
    TF_KEY = 'tfKey'
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
    BARCODES_SUBSET = 'barcodes_subset'
    IS_DUPLICATED_READS = 'isDuplicateReads'
    BASE_RECALIBRATE = 'base_recalibrate'
    REALIGN = 'realign'
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
  
    IS_BARCODE_KIT_SELECTION_REQUIRED = "isBarcodeKitSelectionRequired"
    BARCODE_KIT_NAME = "barcodeId"
    
class KitsStepData(AbstractStepData):


    def __init__(self):
        super(KitsStepData, self).__init__()
        self.resourcePath = 'rundb/plan/page_plan/page_plan_kits.html'
        
        #20130827-test
        ##self._dependsOn.append(StepNames.IONREPORTER) 
                
        self._dependsOn.append(StepNames.APPLICATION) 
        self._dependsOn.append(StepNames.BARCODE_BY_SAMPLE)
        
        self.savedFields[KitsFieldNames.SAMPLE_PREPARATION_KIT] = None
        self.prepopulatedFields[KitsFieldNames.SAMPLE_PREP_KITS] = KitInfo.objects.filter(kitType='SamplePrepKit', isActive=True).order_by('description')
        
        self.savedFields[KitsFieldNames.LIBRARY_KIT_NAME] = None
        self.prepopulatedFields[KitsFieldNames.LIB_KITS] = KitInfo.objects.filter(kitType='LibraryKit', isActive=True).order_by("description")
        
        self.savedFields[KitsFieldNames.LIBRARY_KEY] = None
        self.prepopulatedFields[KitsFieldNames.FORWARD_LIB_KEYS] = LibraryKey.objects.filter(direction='Forward', runMode='single').order_by('-isDefault', 'name')
        self.savedFields[KitsFieldNames.LIBRARY_KEY] = self.prepopulatedFields[KitsFieldNames.FORWARD_LIB_KEYS][0].sequence
        
        self.savedFields[KitsFieldNames.TF_KEY] = GlobalConfig.objects.all()[0].default_test_fragment_key
        
        self.savedFields[KitsFieldNames.FORWARD_3_PRIME_ADAPTER] = None
        self.prepopulatedFields[KitsFieldNames.FORWARD_3_ADAPTERS] = ThreePrimeadapter.objects.filter(direction='Forward', runMode='single').order_by('-isDefault', 'chemistryType', 'name')
        self.savedFields[KitsFieldNames.FORWARD_3_PRIME_ADAPTER] = self.prepopulatedFields[KitsFieldNames.FORWARD_3_ADAPTERS][0].sequence
        
        self.savedFields[KitsFieldNames.AVALANCHE_FORWARD_3_PRIME_ADAPTER] = None
        self.prepopulatedFields[KitsFieldNames.AVALANCHE_FORWARD_3_PRIME_ADAPTERS] = ThreePrimeadapter.objects.filter(direction='Forward', runMode='single', chemistryType = 'avalanche').order_by('-isDefault', 'name')
        self.savedFields[KitsFieldNames.AVALANCHE_FORWARD_3_PRIME_ADAPTER] = self.prepopulatedFields[KitsFieldNames.AVALANCHE_FORWARD_3_PRIME_ADAPTERS][0].sequence
        
        self.savedFields[KitsFieldNames.TEMPLATE_KIT_NAME] = None
        ##no longer default to OneTouch
        #self.savedFields[KitsFieldNames.TEMPLATE_KIT_TYPE] = KitsFieldNames.ONE_TOUCH
        self.savedFields[KitsFieldNames.TEMPLATE_KIT_TYPE] = None
        
        oneTouchDict = {
                KitsFieldNames.KIT_VALUES          : KitInfo.objects.filter(kitType__in=['TemplatingKit', 'AvalancheTemplateKit'], isActive=True).order_by("description"),
                KitsFieldNames.APPLICATION_DEFAULT : None
                }
        
        ionChefDict = {
                KitsFieldNames.KIT_VALUES          : KitInfo.objects.filter(kitType='IonChefPrepKit', isActive=True).order_by("description"),
                KitsFieldNames.APPLICATION_DEFAULT : None
                }
        
        oneTouchAvalancheDict = {
                KitsFieldNames.KIT_VALUES          : KitInfo.objects.filter(kitType__in=['AvalancheTemplateKit'], isActive=True).order_by("description"),
                KitsFieldNames.APPLICATION_DEFAULT : None
                }
        
        self.prepopulatedFields[KitsFieldNames.TEMPLATE_KIT_TYPES] = {
                                                                KitsFieldNames.ONE_TOUCH : oneTouchDict, 
                                                                KitsFieldNames.ION_CHEF  : ionChefDict,
                                                                KitsFieldNames.ONE_TOUCH_AVALANCHE : oneTouchAvalancheDict
                                                                }
        
        self.savedFields[KitsFieldNames.SEQUENCE_KIT_NAME] = None
        self.prepopulatedFields[KitsFieldNames.SEQ_KITS] = KitInfo.objects.filter(kitType='SequencingKit', isActive=True).order_by("description")
                            
        self.savedFields[KitsFieldNames.TEMPLATE_KIT_NAME] = None
        self.prepopulatedFields[KitsFieldNames.TEMPLATE_KITS] = KitInfo.objects.filter(kitType__in=['TemplatingKit', 'AvalancheTemplateKit'], isActive=True).order_by("description")

        self.savedFields[KitsFieldNames.CONTROL_SEQUENCE] = None
        self.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', isActive=True).order_by("description")
        
        self.savedFields[KitsFieldNames.CHIP_TYPE] = None
        self.prepopulatedFields[KitsFieldNames.CHIP_TYPES] = list(Chip.objects.filter(isActive=True).order_by('description', 'name').distinct('description'))
        
        self.savedFields[KitsFieldNames.BARCODE_ID] = None
        self.prepopulatedFields[KitsFieldNames.BARCODES] = list(dnaBarcode.objects.values('name').distinct().order_by('name'))
        
        gc = GlobalConfig.objects.all()[0]
        self.savedFields[KitsFieldNames.IS_DUPLICATED_READS] = gc.mark_duplicates
        self.savedFields[KitsFieldNames.BASE_RECALIBRATE] = gc.base_recalibrate
        self.savedFields[KitsFieldNames.REALIGN] = gc.realign        
        
        self.savedFields[KitsFieldNames.FLOWS] = 0

        self.prepopulatedFields[KitsFieldNames.IS_BARCODE_KIT_SELECTION_REQUIRED] = False


    def getStepName(self):
        return StepNames.KITS

    def updateSavedObjectsFromSavedFields(self):
        pass
    
    def updateFromStep(self, updated_step):

        logger.debug("ENTER kits_step_data.updateFromStep() updated_step.stepName=%s" %(updated_step.getStepName()))

        if updated_step.getStepName() == StepNames.BARCODE_BY_SAMPLE:
            self.savedFields[KitsFieldNames.BARCODE_ID] = updated_step.savedFields['barcodeSet']

        #if user has not clicked on the Application chevron, we need to try to do some catch up 
        if updated_step.getStepName() == StepNames.APPLICATION and updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]:
            applProduct = updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]
            logger.debug("kits_step_data.updateFromStep() Updating kits for applproduct %s" % applProduct.productCode)

            if applProduct.applType.runType in ["AMPS", "AMPS_EXOME"]:
                self.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType__in =["", "DNA", "AMPS_ANY"], isActive=True).order_by("name")            
            elif applProduct.applType.runType in ["AMPS_RNA"]:
                 self.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType__in =["", "RNA", "AMPS_ANY"], isActive=True).order_by("name")
            elif applProduct.applType.runType in ["RNA"]:
                self.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType = "RNA", isActive=True).order_by("name")            
            elif applProduct.applType.runType in ["AMPS_DNA_RNA"]:
                self.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType__in =["", "DNA", "RNA", "AMPS_ANY"], isActive=True).order_by("name")
            else:
                self.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType__in =["", "DNA"], isActive=True).order_by("name")
                 
            self.prepopulatedFields[KitsFieldNames.BARCODES] = list(dnaBarcode.objects.values('name').distinct().order_by('name'))                              
            if applProduct.barcodeKitSelectableType == "":              
                self.prepopulatedFields[KitsFieldNames.BARCODES_SUBSET] = list(dnaBarcode.objects.values('name').filter(type__in =["", "none"]).distinct().order_by('name'))   
            elif applProduct.barcodeKitSelectableType == "dna":
                self.prepopulatedFields[KitsFieldNames.BARCODES_SUBSET] = list(dnaBarcode.objects.values('name').filter(type = "dna").distinct().order_by('name'))            
            elif applProduct.barcodeKitSelectableType == "rna":
                self.prepopulatedFields[KitsFieldNames.BARCODES_SUBSET] = list(dnaBarcode.objects.values('name').filter(type = "rna").distinct().order_by('name'))            
            elif applProduct.barcodeKitSelectableType == "dna+":
                self.prepopulatedFields[KitsFieldNames.BARCODES_SUBSET] = list(dnaBarcode.objects.values('name').filter(type__in =["dna", "", "none"]).distinct().order_by('name'))            
            elif applProduct.barcodeKitSelectableType == "rna+":
                self.prepopulatedFields[KitsFieldNames.BARCODES_SUBSET] = list(dnaBarcode.objects.values('name').filter(type__in =["rna", "", "none"]).distinct().order_by('name'))            
            else:
                 self.prepopulatedFields[KitsFieldNames.BARCODES_SUBSET] = list(dnaBarcode.objects.values('name').distinct().order_by('name')) 
                                            
            if applProduct.defaultTemplateKit:
                self.prepopulatedFields[KitsFieldNames.TEMPLATE_KIT_TYPES][KitsFieldNames.ONE_TOUCH][KitsFieldNames.APPLICATION_DEFAULT] = applProduct.defaultTemplateKit
            
            if applProduct.defaultIonChefPrepKit:
                self.prepopulatedFields[KitsFieldNames.TEMPLATE_KIT_TYPES][KitsFieldNames.ION_CHEF][KitsFieldNames.APPLICATION_DEFAULT] = applProduct.defaultIonChefPrepKit

            self.prepopulatedFields[KitsFieldNames.IS_BARCODE_KIT_SELECTION_REQUIRED] = applProduct.isBarcodeKitSelectionRequired
            
            if updated_step.savedObjects[ApplicationFieldNames.UPDATE_KITS_DEFAULTS]:
                self.updateFieldsFromDefaults(applProduct)
            
            
    def updateFieldsFromDefaults(self, applProduct):
            
            if not applProduct.defaultChipType:
                self.savedFields[KitsFieldNames.CHIP_TYPE] = None    
            
            if applProduct.defaultSamplePrepKit:
                self.savedFields[KitsFieldNames.SAMPLE_PREPARATION_KIT] = applProduct.defaultSamplePrepKit.name
            else:
                self.savedFields[KitsFieldNames.SAMPLE_PREPARATION_KIT] = None
            
            if applProduct.defaultLibraryKit:
                self.savedFields[KitsFieldNames.LIBRARY_KIT_NAME] = applProduct.defaultLibraryKit.name
            
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
            else:
                self.savedFields[KitsFieldNames.CONTROL_SEQUENCE] = None
            
            if applProduct.isDefaultBarcoded and applProduct.defaultBarcodeKitName:
                self.savedFields[KitsFieldNames.BARCODE_ID] = applProduct.defaultBarcodeKitName
            elif not applProduct.isDefaultBarcoded:
                self.savedFields[KitsFieldNames.BARCODE_ID] = None
            
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


    def validateField(self, field_name, new_field_value):
        '''
        Flows qc value must be a positive integer
        Chip type is required
        '''
        if field_name == KitsFieldNames.FLOWS:
            errors = validate_flows(new_field_value)
            if errors:
                self.validationErrors[field_name] = ' '.join(errors)
            else:
                self.validationErrors.pop(field_name, None)

        if field_name == KitsFieldNames.CHIP_TYPE and self.prepopulatedFields.get('is_chipType_required',True):
            if validation.has_value(new_field_value):
                self.validationErrors.pop(field_name, None)
            else:
                self.validationErrors[field_name] = validation.required_error("Chip Type")
        
        if field_name == KitsFieldNames.TEMPLATE_KIT_NAME:
            if validation.has_value(new_field_value):
                self.validationErrors.pop(field_name, None)
            else:
                self.validationErrors[field_name] = validation.required_error("Template Kit")


    def validateField_crossField_dependencies(self, fieldNames, fieldValues):
        if KitsFieldNames.LIBRARY_KIT_NAME in fieldNames and KitsFieldNames.BARCODE_KIT_NAME:
            newFieldValue = fieldValues.get(KitsFieldNames.LIBRARY_KIT_NAME, "")
            dependentFieldValue = fieldValues.get(KitsFieldNames.BARCODE_KIT_NAME, "")
           
            self.validateField_crossField_dependency(KitsFieldNames.LIBRARY_KIT_NAME, newFieldValue, KitsFieldNames.BARCODE_KIT_NAME, dependentFieldValue)   


    def validateField_crossField_dependency(self, field_name, new_field_value, dependent_field_name, dependent_field_value):                         
        isBarcodeKitRequired = False
        
        if field_name == KitsFieldNames.LIBRARY_KIT_NAME:
            if new_field_value:
                libKit_objs = KitInfo.objects.filter(kitType='LibraryKit', name = new_field_value).order_by("-isActive")
                if libKit_objs and len(libKit_objs) > 0:
                    libKit_obj = libKit_objs[0]
                    if libKit_obj.categories and ("bcrequired" in libKit_obj.categories.lower()):
                        isBarcodeKitRequired = True         
                        
        if dependent_field_name == KitsFieldNames.BARCODE_KIT_NAME:            
            if self.prepopulatedFields[KitsFieldNames.IS_BARCODE_KIT_SELECTION_REQUIRED] or isBarcodeKitRequired:
                
                if validation.has_value(dependent_field_value):
                    self.validationErrors.pop(dependent_field_name, None)
                else:
                    self.validationErrors[dependent_field_name] = validation.required_error("Barcode Set")    
            else:
                self.validationErrors.pop(dependent_field_name, None)    
                      
