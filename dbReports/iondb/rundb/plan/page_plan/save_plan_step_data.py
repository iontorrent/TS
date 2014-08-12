# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.models import dnaBarcode, Plugin, SampleAnnotation_CV, KitInfo, RunType
from iondb.utils import validation
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.page_plan.application_step_data import ApplicationFieldNames
from iondb.rundb.plan.page_plan.reference_step_data import ReferenceFieldNames
from iondb.rundb.plan.page_plan.kits_step_data import KitsFieldNames
from iondb.rundb.plan.page_plan.export_step_data import ExportFieldNames
from iondb.rundb.plan.plan_validator import validate_plan_name, validate_notes, validate_sample_name, validate_sample_tube_label, validate_barcode_sample_association

from iondb.rundb.plan.views_helper import convert

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

##from iondb.utils import toBoolean

import json
import uuid
import logging
logger = logging.getLogger(__name__)

MAX_LENGTH_SAMPLE_DESCRIPTION = 1024
MAX_LENGTH_SAMPLE_NAME = 127

class SavePlanFieldNames():

    UPLOADERS = 'uploaders'
    SET_ID = 'setid'
    EXTERNAL_ID = 'externalId'
    WORKFLOW = 'Workflow'
    GENDER = 'Gender'
    CANCER_TYPE = "cancerType"
    CELLULARITY_PCT = "cellularityPct" 
    NUCLEOTIDE_TYPE = "NucleotideType"
    RELATIONSHIP_TYPE = 'Relation'
    RELATION_ROLE = 'RelationRole'
    PLAN_NAME = 'planName'
    SAMPLE = 'sample'
    NOTE = 'note'
    SELECTED_IR = 'selectedIr'
    IR_CONFIG_JSON = 'irConfigJson'
    BARCODE_SET = 'barcodeSet'
    BARCODE_SAMPLE_TUBE_LABEL = 'barcodeSampleTubeLabel'
    BARCODE_TO_SAMPLE = 'barcodeToSample'
    BARCODE_SETS = 'barcodeSets'
    BARCODE_SETS_SUBSET = "barcodeSets_subset"
    BARCODE_SETS_BARCODES = 'barcodeSets_barcodes'
    SAMPLE_TO_BARCODE = 'sampleToBarcode'
    BARCODED_IR_PLUGIN_ENTRIES = 'barcodedIrPluginEntries'
    SAMPLE_EXTERNAL_ID = 'sampleExternalId'
    SAMPLE_NAME = 'sampleName'
    SAMPLE_DESCRIPTION = 'sampleDescription'
    TUBE_LABEL = 'tubeLabel'
    IR_GENDER = 'irGender'

    IR_CANCER_TYPE = "ircancerType"
    IR_CELLULARITY_PCT = "ircellularityPct"
    
    IR_WORKFLOW = 'irWorkflow'
    IR_RELATION_ROLE = 'irRelationRole'
    IR_RELATIONSHIP_TYPE = 'irRelationshipType'
    IR_SET_ID = 'irSetID'

    BAD_SAMPLE_NAME = 'bad_sample_name'
    BAD_SAMPLE_EXTERNAL_ID = 'bad_sample_external_id'
    BAD_SAMPLE_DESCRIPTION = 'bad_sample_description'
    BAD_TUBE_LABEL = 'bad_tube_label'
    BARCODE_SAMPLE_NAME = 'barcodeSampleName'
    BARCODE_SAMPLE_DESCRIPTION = 'barcodeSampleDescription'
    BARCODE_SAMPLE_EXTERNAL_ID = 'barcodeSampleExternalId'

    BARCODE_SAMPLE_NUCLEOTIDE_TYPE = "nucleotideType"
    BARCODE_SAMPLE_REFERENCE = "reference"
    BARCODE_SAMPLE_TARGET_REGION_BED_FILE = "targetRegionBedFile"
    BARCODE_SAMPLE_HOTSPOT_REGION_BED_FILE = "hotSpotRegionBedFile"
    BARCODE_SAMPLE_CONTROL_SEQ_TYPE = "controlSequenceType"
    
    BARCODE_SAMPLE_INFO = 'barcodeSampleInfo'
    NO_SAMPLES = 'no_samples'
    DESCRIPTION = 'description'
    BAD_IR_SET_ID = 'badIrSetId'
    SAMPLE_ANNOTATIONS = 'sampleAnnotations'

    CONTROL_SEQ_TYPES = "controlSeqTypes"
    BARCODE_KIT_SELECTABLE_TYPE = "barcodeKitSelectableType"
    
    PLAN_REFERENCE = "plan_reference"
    PLAN_TARGET_REGION_BED_FILE = "plan_targetRegionBedFile"
    PLAN_HOTSPOT_REGION_BED_FILE = "plan_hotSpotRegionBedFile"
    
    ##IS_REFERENCE_BY_SAMPLE_SUPPORTED = "isReferenceBySampleSupported"
    RUN_TYPE = "runType"
    ONCO_SAME_SAMPLE = "isOncoSameSample"

    REFERENCE_STEP_HELPER = "referenceStepHelper"

    NO_BARCODE = 'no_barcode'
    BAD_BARCODES = 'bad_barcodes'
    
class SavePlanStepData(AbstractStepData):

    def __init__(self):
        super(SavePlanStepData, self).__init__()
        self.resourcePath = 'rundb/plan/page_plan/page_plan_save_plan.html'
        
        self.savedFields[SavePlanFieldNames.PLAN_NAME] = None
        self.savedFields[SavePlanFieldNames.NOTE] = None
        self.savedFields['applicationType'] = ''
        self.savedFields['irDown'] = '0'    
            
        self.savedFields[SavePlanFieldNames.BARCODE_SET] = ''

        self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE] = ""
        self.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE] = ""
        self.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE] = ""
                    
        self.prepopulatedFields[SavePlanFieldNames.SELECTED_IR] = None
        self.prepopulatedFields[SavePlanFieldNames.IR_CONFIG_JSON] = None
        self.prepopulatedFields[SavePlanFieldNames.SAMPLE_ANNOTATIONS] = list(SampleAnnotation_CV.objects.all().order_by("annotationType", "iRValue"))
        self.savedFields[SavePlanFieldNames.BARCODE_SAMPLE_TUBE_LABEL] = None
        self.savedObjects[SavePlanFieldNames.BARCODE_TO_SAMPLE] = OrderedDict()
        
        self.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS] = list(dnaBarcode.objects.values_list('name',flat=True).distinct().order_by('name'))
        all_barcodes = {}
        for bc in dnaBarcode.objects.order_by('name', 'index').values('name', 'id_str','sequence'):
            all_barcodes.setdefault(bc['name'],[]).append(bc)
        self.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS_BARCODES] = json.dumps(all_barcodes)
        
        self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE] = OrderedDict()
        self.savedObjects[SavePlanFieldNames.BARCODED_IR_PLUGIN_ENTRIES] = []
        self.prepopulatedFields['fireValidation'] = "1"

        self.savedObjects['samplesTableList'] = [{"row":"1"}]
        self.savedFields['samplesTable'] = json.dumps(self.savedObjects['samplesTableList'])

        self.savedFields[SavePlanFieldNames.ONCO_SAME_SAMPLE] = False
        
        #logger.debug("save_plan_step_data samplesTable=%s" %(self.savedFields['samplesTable']))
        
        self.prepopulatedFields[SavePlanFieldNames.CONTROL_SEQ_TYPES] = KitInfo.objects.filter(kitType='ControlSequenceKitType', isActive=True).order_by("name")        

        self.savedObjects[SavePlanFieldNames.REFERENCE_STEP_HELPER] = None
                          
        self.updateSavedObjectsFromSavedFields()

        ##self.prepopulatedFields[SavePlanFieldNames.IS_REFERENCE_BY_SAMPLE_SUPPORTED] = False
        
        self._dependsOn.append(StepNames.APPLICATION)
        self._dependsOn.append(StepNames.KITS)
        self._dependsOn.append(StepNames.REFERENCE)
            

    def getStepName(self):
        return StepNames.SAVE_PLAN
    
    def validateField(self, field_name, new_field_value):
        self.validationErrors.pop(field_name, None)
        
        if field_name == SavePlanFieldNames.PLAN_NAME:
            errors = validate_plan_name(new_field_value, 'Plan Name')
            if errors:
                self.validationErrors[field_name] = '\n'.join(errors)
        elif field_name == SavePlanFieldNames.NOTE:
            errors = validate_notes(new_field_value)
            if errors:
                self.validationErrors[field_name] = '\n'.join(errors)
        elif field_name == SavePlanFieldNames.BARCODE_SAMPLE_TUBE_LABEL:
            errors = validate_sample_tube_label(new_field_value)
            if errors:
                self.validationErrors[field_name] = '\n'.join(errors)
            
    def validateStep(self):
        any_samples = False
        self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_NAME] = []
        self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_EXTERNAL_ID] = []
        self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_DESCRIPTION] = []
        self.validationErrors[SavePlanFieldNames.BAD_TUBE_LABEL] = []
        self.validationErrors[SavePlanFieldNames.BAD_IR_SET_ID] = []

        self.validationErrors.pop(SavePlanFieldNames.NO_BARCODE,None)
        self.validationErrors.pop(SavePlanFieldNames.BAD_BARCODES,None)

        barcodeSet = self.savedFields[SavePlanFieldNames.BARCODE_SET]
        selectedBarcodes = []
        
        samplesTable = json.loads(self.savedFields['samplesTable'])            

        #logger.debug("save_plan_step_data - anySamples? samplesTable=%s" %(samplesTable))
        
        for row in samplesTable:
            sample_name = row.get(SavePlanFieldNames.SAMPLE_NAME,'').strip()

            #logger.debug("save_plan_step_data - anySamples? sampleName=%s" %(sample_name))
            
            if sample_name:
                any_samples = True
                if validate_sample_name(sample_name):
                    self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_NAME].append(sample_name)
                
                external_id = row.get(SavePlanFieldNames.SAMPLE_EXTERNAL_ID,'')
                if external_id:
                    self.validate_field(external_id, self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_EXTERNAL_ID])
                    
                description = row.get(SavePlanFieldNames.SAMPLE_DESCRIPTION,'')
                if description:
                    self.validate_field(description, self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_DESCRIPTION], False,
                                        MAX_LENGTH_SAMPLE_DESCRIPTION)

                ir_set_id = row.get('irSetId','')
                if ir_set_id and not (str(ir_set_id).isdigit()):
                    self.validationErrors[SavePlanFieldNames.BAD_IR_SET_ID].append(ir_set_id)

                tube_label = row.get('tubeLabel','')
                if validate_sample_tube_label(tube_label):
                    self.validationErrors[SavePlanFieldNames.BAD_TUBE_LABEL].append(tube_label)

                if barcodeSet:
                    selectedBarcodes.append(row.get('barcodeId'))
                
        
        if any_samples:
            self.validationErrors.pop(SavePlanFieldNames.NO_SAMPLES, None)
        else:
            self.validationErrors[SavePlanFieldNames.NO_SAMPLES] = "You must enter at least one sample"
            
        if not self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_NAME]:
            self.validationErrors.pop(SavePlanFieldNames.BAD_SAMPLE_NAME, None)
        
        if not self.validationErrors[SavePlanFieldNames.BAD_TUBE_LABEL]:
            self.validationErrors.pop(SavePlanFieldNames.BAD_TUBE_LABEL, None)
        
        if not self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_EXTERNAL_ID]:
            self.validationErrors.pop(SavePlanFieldNames.BAD_SAMPLE_EXTERNAL_ID, None)
        
        if not self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_DESCRIPTION]:
            self.validationErrors.pop(SavePlanFieldNames.BAD_SAMPLE_DESCRIPTION, None)
            
        if not self.validationErrors[SavePlanFieldNames.BAD_IR_SET_ID]:
            self.validationErrors.pop(SavePlanFieldNames.BAD_IR_SET_ID, None)

        if barcodeSet:
            errors = validate_barcode_sample_association(selectedBarcodes, barcodeSet)
                        
            myErrors = convert(errors)
            if myErrors.get("MISSING_BARCODE", ""):
                self.validationErrors[SavePlanFieldNames.NO_BARCODE] = myErrors.get("MISSING_BARCODE", "")
            if myErrors.get("DUPLICATE_BARCODE", ""):
                self.validationErrors[SavePlanFieldNames.BAD_BARCODES] = myErrors.get("DUPLICATE_BARCODE", "")
                 

    def validate_field(self, value, bad_samples, validate_leading_chars=True, max_length=MAX_LENGTH_SAMPLE_NAME):
        exists = False
        if value:
            exists = True
            if not validation.is_valid_chars(value):
                bad_samples.append(value)
            
            if validate_leading_chars and value not in bad_samples and not validation.is_valid_leading_chars(value):
                bad_samples.append(value)
            
            if value not in bad_samples and not validation.is_valid_length(value, max_length):
                bad_samples.append(value)
        
        return exists


    def updateSavedObjectsFromSavedFields(self):        
        self.prepopulatedFields["fireValidation"] = "0"

        self.savedObjects['samplesTableList'] = json.loads(self.savedFields['samplesTable'])

        #logger.debug("save_plan_step_data.updateSavedObjectsFromSavedFields() ORIGINAL type(self.savedFields[samplesTable])=%s; self.savedFields[samplesTable]=%s" %(type(self.savedFields['samplesTable']), self.savedFields['samplesTable']))     
        #logger.debug("save_plan_step_data.updateSavedObjectsFromSavedFields() AFTER JSON.LOADS... type(self.savedObjects[samplesTableList])=%s; self.savedObjects[samplesTableList]=%s" %(type(self.savedObjects['samplesTableList']), self.savedObjects['samplesTableList']))       

        if self.savedFields[SavePlanFieldNames.BARCODE_SET]:
            planned_dnabarcodes = list(dnaBarcode.objects.filter(name=self.savedFields[SavePlanFieldNames.BARCODE_SET]).order_by('id_str'))
            self.prepopulatedFields['planned_dnabarcodes'] = planned_dnabarcodes

            planReference = self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE]
            planHotSptRegionBedFile = self.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE]
            planTargetRegionBedFile = self.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE]
    
            logger.debug("save_plan_step_data.updateSavedObjectsFromSavedFields() BARCODE_SET PLAN_REFERENCE=%s; TARGET_REGION=%s; HOTSPOT_REGION=%s;" %(planReference, planTargetRegionBedFile, planHotSptRegionBedFile))

            self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE] = {}
            self.savedObjects[SavePlanFieldNames.BARCODED_IR_PLUGIN_ENTRIES] = []
            for row in self.savedObjects['samplesTableList']:

                logger.debug("save_plan_step_data.updateSavedObjectsFromSavedFields() BARCODE_SET LOOP row=%s" %(row)) 
                               
                sample_name = row.get(SavePlanFieldNames.SAMPLE_NAME,'').strip()
                if sample_name:
                    id_str = row['barcodeId']
                    
                    # update barcodedSamples dict
                    if sample_name not in self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE]:
                        self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE][sample_name] = {
                            KitsFieldNames.BARCODES: [],
                            SavePlanFieldNames.BARCODE_SAMPLE_INFO: {}
                        }

                   
                    sample_nucleotideType = row.get(SavePlanFieldNames.BARCODE_SAMPLE_NUCLEOTIDE_TYPE, "")
                    
                    sampleReference = row.get(SavePlanFieldNames.BARCODE_SAMPLE_REFERENCE, "")
                    sampleHotSpotRegionBedFile = row.get(SavePlanFieldNames.BARCODE_SAMPLE_HOTSPOT_REGION_BED_FILE, "")
                    sampleTargetRegionBedFile = row.get(SavePlanFieldNames.BARCODE_SAMPLE_TARGET_REGION_BED_FILE, "")
                    
                    ##isReferenceBySample = toBoolean(self.prepopulatedFields[SavePlanFieldNames.IS_REFERENCE_BY_SAMPLE_SUPPORTED], False)
                    runType = self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE]
                    
                    #logger.debug("save_plan_step_data SETTING reference step helper runType=%s; sample_nucleotideType=%s; sampleReference=%s" %(runType, sample_nucleotideType, sampleReference))
                    
                    if runType == "AMPS_DNA_RNA" and sample_nucleotideType == "DNA":
                        reference_step_helper = self.savedObjects[SavePlanFieldNames.REFERENCE_STEP_HELPER]
                        if reference_step_helper:
                            if sampleReference != planReference:
                                reference_step_helper.savedFields[ReferenceFieldNames.REFERENCE] = sampleReference
                                #logger.debug("save_plan_step_data DIFF SETTING reference step helper reference=%s" %(sampleReference))
                                
                            if sampleHotSpotRegionBedFile != planHotSptRegionBedFile:
                                reference_step_helper.savedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE] = sampleHotSpotRegionBedFile
                                #logger.debug("save_plan_step_data DIFF SETTING reference step helper hotSpot=%s" %(sampleHotSpotRegionBedFile))
                                
                            if sampleTargetRegionBedFile != planTargetRegionBedFile:
                                reference_step_helper.savedFields[ReferenceFieldNames.TARGET_BED_FILE] = sampleTargetRegionBedFile
                                #logger.debug("save_plan_step_data DIFF SETTING reference step helper targetRegions=%s" %(sampleTargetRegionBedFile))

                    #if reference info is not settable in the sample config table, use the up-to-date reference selection from the reference chevron
                    self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE][sample_name][KitsFieldNames.BARCODES].append(id_str)
                    self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE][sample_name][SavePlanFieldNames.BARCODE_SAMPLE_INFO][id_str] = \
                        {
                            SavePlanFieldNames.EXTERNAL_ID : row.get(SavePlanFieldNames.SAMPLE_EXTERNAL_ID,''),
                            SavePlanFieldNames.DESCRIPTION : row.get(SavePlanFieldNames.SAMPLE_DESCRIPTION,''),

                            SavePlanFieldNames.BARCODE_SAMPLE_NUCLEOTIDE_TYPE : sample_nucleotideType,

                            SavePlanFieldNames.BARCODE_SAMPLE_REFERENCE : row.get(SavePlanFieldNames.BARCODE_SAMPLE_REFERENCE, ""),
                            SavePlanFieldNames.BARCODE_SAMPLE_TARGET_REGION_BED_FILE : row.get(SavePlanFieldNames.BARCODE_SAMPLE_TARGET_REGION_BED_FILE, ""),
                            SavePlanFieldNames.BARCODE_SAMPLE_HOTSPOT_REGION_BED_FILE : row.get(SavePlanFieldNames.BARCODE_SAMPLE_HOTSPOT_REGION_BED_FILE, ""), 
                                                       
                            SavePlanFieldNames.BARCODE_SAMPLE_CONTROL_SEQ_TYPE : row.get(SavePlanFieldNames.BARCODE_SAMPLE_CONTROL_SEQ_TYPE, ""),
                        }

                    #logger.debug("save_plan_step_data.updateSavedObjectsFromSaveFields() sampleName=%s; id_str=%s; savedObjects=%s" %(sample_name, id_str, self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE][sample_name][SavePlanFieldNames.BARCODE_SAMPLE_INFO][id_str]))
                    #logger.debug("save_plan_step_date.updateSavedObjectsFromSaveFields() savedObjects=%s" %(self.savedObjects));
                    ##logger.debug("save_plan_step_date.updateSavedObjectsFromSaveFields() savedFields=%s" %(self.savedFields));
                                                                         
                    # update barcoded IR fields
                    barcode_ir_userinput_dict = {
                        KitsFieldNames.BARCODE_ID             : id_str,
                        SavePlanFieldNames.SAMPLE             : sample_name,
                        SavePlanFieldNames.SAMPLE_NAME        : sample_name.replace(' ', '_'),
                        SavePlanFieldNames.SAMPLE_EXTERNAL_ID : row.get(SavePlanFieldNames.SAMPLE_EXTERNAL_ID,''),
                        SavePlanFieldNames.SAMPLE_DESCRIPTION : row.get(SavePlanFieldNames.SAMPLE_DESCRIPTION,''),
                        
                        SavePlanFieldNames.WORKFLOW           : row.get(SavePlanFieldNames.IR_WORKFLOW,''),
                        SavePlanFieldNames.GENDER             : row.get(SavePlanFieldNames.IR_GENDER,''),
                        SavePlanFieldNames.NUCLEOTIDE_TYPE    : row.get(SavePlanFieldNames.BARCODE_SAMPLE_NUCLEOTIDE_TYPE, ''),
                                                                            
                        SavePlanFieldNames.CANCER_TYPE        : row.get(SavePlanFieldNames.IR_CANCER_TYPE, ""),
                        SavePlanFieldNames.CELLULARITY_PCT    : row.get(SavePlanFieldNames.IR_CELLULARITY_PCT, ""),
                                                    
                        SavePlanFieldNames.RELATION_ROLE      : row.get(SavePlanFieldNames.IR_RELATION_ROLE,''),
                        SavePlanFieldNames.RELATIONSHIP_TYPE  : row.get(SavePlanFieldNames.IR_RELATIONSHIP_TYPE,''),
                        SavePlanFieldNames.SET_ID             : row.get(SavePlanFieldNames.IR_SET_ID, '')
                    }

                    #logger.debug("save_plan_step_data.updateSavedObjectsFromSavedFields() barcode_ir_userinput_dict=%s" %(barcode_ir_userinput_dict))
                            
                    self.savedObjects[SavePlanFieldNames.BARCODED_IR_PLUGIN_ENTRIES].append(barcode_ir_userinput_dict)

        #logger.debug("EXIT save_plan_step_date.updateSavedObjectsFromSaveFields() type(self.savedObjects['samplesTableList'])=%s; self.savedObjects['samplesTableList']=%s" %(type(self.savedObjects['samplesTableList']), self.savedObjects['samplesTableList']));
       
    
    def updateFromStep(self, updated_step): 
        #logger.debug("ENTER save_plan_step_data.updateFromStep() updated_step.stepName=%s; self.savedFields=%s" %(updated_step.getStepName(), self.savedFields))
 
        if updated_step.getStepName() not in self._dependsOn:
            return

        if updated_step.getStepName() == StepNames.APPLICATION:
            if updated_step.savedObjects[ApplicationFieldNames.RUN_TYPE]:
                self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE] = updated_step.savedObjects[ApplicationFieldNames.RUN_TYPE].runType
            else:
                self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE] = ""

            logger.debug("save_plan_step_data.updateFromStep() going to update RUNTYPE value=%s" %(self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE]))

            
        if updated_step.getStepName() == StepNames.APPLICATION and updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]:         
            applProduct = updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]
            
            barcodeDetails = None  
            self.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS] = list(dnaBarcode.objects.values_list('name',flat=True).distinct().order_by('name'))   
            if applProduct.barcodeKitSelectableType == "":
                self.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS_SUBSET] = list(dnaBarcode.objects.values_list('name',flat=True).filter(type__in =["", "none"]).distinct().order_by('name'))
            elif applProduct.barcodeKitSelectableType == "dna":
                self.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS_SUBSET] = list(dnaBarcode.objects.values_list('name',flat=True).filter(type = "dna").distinct().order_by('name'))   
            elif applProduct.barcodeKitSelectableType == "rna":
                self.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS_SUBSET] = list(dnaBarcode.objects.values_list('name',flat=True).filter(type = "rna").distinct().order_by('name')) 
            elif applProduct.barcodeKitSelectableType == "dna+":
                self.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS_SUBSET] = list(dnaBarcode.objects.values_list('name',flat=True).filter(type__in =["dna", "", "none"]).distinct().order_by('name')) 
            elif applProduct.barcodeKitSelectableType == "rna+":
                self.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS_SUBSET] = list(dnaBarcode.objects.values_list('name',flat=True).filter(type__in =["rna", "", "none"]).distinct().order_by('name')) 
            else:
                self.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS_SUBSET] = list(dnaBarcode.objects.values_list('name',flat=True).distinct().order_by('name'))  

            barcodeDetails = dnaBarcode.objects.order_by('name', 'index').values('name', 'id_str','sequence')                      
                
            all_barcodes = {}
            for bc in barcodeDetails:
                all_barcodes.setdefault(bc['name'],[]).append(bc)
            self.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS_BARCODES] = json.dumps(all_barcodes) 

            ##self.prepopulatedFields[SavePlanFieldNames.IS_REFERENCE_BY_SAMPLE_SUPPORTED] = applProduct.isReferenceBySampleSupported    
            ##logger.debug("save_plan_step_data.updateFromStep() APPLPRODUCT... isReferenceBySampleSupported=%s;" %(self.prepopulatedFields[SavePlanFieldNames.IS_REFERENCE_BY_SAMPLE_SUPPORTED]))
                                                        
        if updated_step.getStepName() == StepNames.KITS:
            barcode_set = updated_step.savedFields[KitsFieldNames.BARCODE_ID]
            if str(barcode_set) != str(self.savedFields[SavePlanFieldNames.BARCODE_SET]):
                self.savedFields[SavePlanFieldNames.BARCODE_SET] = barcode_set
                if barcode_set:
                    barcodes = list(dnaBarcode.objects.filter(name=barcode_set).order_by('id_str'))
                    self.prepopulatedFields['planned_dnabarcodes'] = barcodes

                    bc_count = min(len(barcodes), len(self.savedObjects['samplesTableList']))
                    self.savedObjects['samplesTableList'] = self.savedObjects['samplesTableList'][:bc_count]
                    for i in range(bc_count):
                        self.savedObjects['samplesTableList'][i]['barcodeId'] = barcodes[i].id_str
                    
                    self.savedFields['samplesTable'] = json.dumps(self.savedObjects['samplesTableList'])
                
        elif updated_step.getStepName() == StepNames.REFERENCE:
            self.savedObjects[SavePlanFieldNames.REFERENCE_STEP_HELPER] = updated_step
            
            self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE] = updated_step.savedFields.get(ReferenceFieldNames.REFERENCE, "")
            self.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE] = updated_step.savedFields.get(ReferenceFieldNames.TARGET_BED_FILE, "")
            self.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE] = updated_step.savedFields.get(ReferenceFieldNames.HOT_SPOT_BED_FILE, "")

            #logger.debug("save_plan_step_data.updateFromStep() REFERENCE type(plan_reference)=%s; plan_reference=%s" %(type(self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE]), self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE]))
            #logger.debug("save_plan_step_data.updateFromStep() REFERENCE reference.savedFields=%s" %(updated_step.savedFields))
           
            self.updateSavedFieldsForSamples()        
                
        elif updated_step.getStepName() == StepNames.EXPORT:
            if ExportFieldNames.IR_ACCOUNT_ID not in updated_step.savedFields\
                or not updated_step.savedFields[ExportFieldNames.IR_ACCOUNT_ID]\
                or updated_step.savedFields[ExportFieldNames.IR_ACCOUNT_ID] == '0':
                
                if SavePlanFieldNames.BAD_IR_SET_ID in self.validationErrors:
                    self.validationErrors.pop(SavePlanFieldNames.BAD_IR_SET_ID, None)
                    
        logger.debug("EXIT save_plan_step_data.updateFromStep() self.savedFields=%s" %(self.savedFields))



    def updateSavedFieldsForSamples(self):        
        ##logger.debug("ENTER save_plan_step_data.updateSavedFieldsForSamples() B4 type=%s; self.savedFields[samplesTable]=%s" %(type(self.savedFields['samplesTable']), self.savedFields['samplesTable']))       
        ##logger.debug("ENTER save_plan_step_data.updateSavedFieldsForSamples() B4 type=%s; self.savedObjects[samplesTableList]=%s" %(type(self.savedObjects['samplesTableList']), self.savedObjects['samplesTableList']))       

        if self.savedFields[SavePlanFieldNames.BARCODE_SET]:
            #convert tuple to string
            planReference = str(self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE])
            planHotSptRegionBedFile = str(self.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE])
            planTargetRegionBedFile = str(self.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE])
                    
            #logger.debug("save_plan_step_data.updateSavedFieldsForSamples() type(planReference)=%s; planReference=%s" %(type(planReference), planReference))
            #logger.debug("save_plan_step_data.updateSavedFieldsForSamples() type(self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE])=%s; self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE]=%s" %(type(self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE]), self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE]))

            hasAnyChanges = False
            myTable = json.loads(self.savedFields['samplesTable'])
            #convert unicode to str
            myTable = convert(myTable)
            
            for index, row in enumerate(myTable):
                ##logger.debug("save_plan_step_data.updateSavedFieldsForSamples() B4 CHANGES... BARCODE_SET LOOP row=%s" %(row))                   
                                              
                sample_name = row.get(SavePlanFieldNames.SAMPLE_NAME,'').strip()
                if sample_name:
                   
                    sample_nucleotideType = row.get(SavePlanFieldNames.BARCODE_SAMPLE_NUCLEOTIDE_TYPE, "")
                    
                    sampleReference = row.get(SavePlanFieldNames.BARCODE_SAMPLE_REFERENCE, "")
                    sampleHotSpotRegionBedFile = row.get(SavePlanFieldNames.BARCODE_SAMPLE_HOTSPOT_REGION_BED_FILE, "")
                    sampleTargetRegionBedFile = row.get(SavePlanFieldNames.BARCODE_SAMPLE_TARGET_REGION_BED_FILE, "")

                    runType = self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE]

                    if runType == "AMPS_DNA_RNA" and sample_nucleotideType == "RNA":
                        newSampleReference = sampleReference
                        newSampleHotspotRegionBedFile = sampleHotSpotRegionBedFile
                        newSampleTargetRegionBedFile = sampleTargetRegionBedFile
                    else:
                        newSampleReference = planReference
                        newSampleHotspotRegionBedFile = planHotSptRegionBedFile
                        newSampleTargetRegionBedFile = planTargetRegionBedFile

                    hasChanged = False
                    if newSampleReference != sampleReference:
                        row[SavePlanFieldNames.BARCODE_SAMPLE_REFERENCE] = newSampleReference
                        hasChanged = True
                    if newSampleHotspotRegionBedFile != sampleHotSpotRegionBedFile:
                        row[SavePlanFieldNames.BARCODE_SAMPLE_HOTSPOT_REGION_BED_FILE] = newSampleHotspotRegionBedFile
                        hasChanged = True
                    if newSampleTargetRegionBedFile != sampleTargetRegionBedFile:
                        row[SavePlanFieldNames.BARCODE_SAMPLE_TARGET_REGION_BED_FILE] = newSampleTargetRegionBedFile
                        hasChanged = True
                        
                    if hasChanged:
                        myTable[index] = row 
                        #logger.debug("save_plan_step_data.updateSavedFieldsForSamples() AFTER CHANGES  BARCODE_SET LOOP myTable[index]=%s" %(myTable[index]))                                     
                        hasAnyChanges = True
                        
 
            if hasAnyChanges:    
                logger.debug("save_plan_step_data.updateSavedFieldsForSamples() hasAnyChanges AFTER CHANGES... type=%s; myTable=%s" %(type(myTable), myTable))       

                #convert list with single quotes to str with double quotes. Then convert it to be unicode
                self.savedFields['samplesTable'] = unicode(json.dumps(myTable))
                
                #logger.debug("save_plan_step_data.updateSavedFieldsForSamples() hasAnyChanges AFTER unicode(json.dumps)... type=%s; self.savedFields[samplesTable]=%s" %(type(self.savedFields['samplesTable']), self.savedFields['samplesTable']))       

                self.savedObjects['samplesTableList'] = json.loads(self.savedFields['samplesTable'])                
                logger.debug("save_plan_step_data.updateSavedFieldsForSamples() hasAnyChanges AFTER json.loads... type=%s; self.savedObjects[samplesTableList]=%s" %(type(self.savedObjects['samplesTableList']), self.savedObjects['samplesTableList']))       
                
