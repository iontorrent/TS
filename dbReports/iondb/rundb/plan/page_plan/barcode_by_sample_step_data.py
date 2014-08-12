# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.models import dnaBarcode, Sample, SampleAnnotation_CV
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.page_plan.application_step_data import ApplicationFieldNames
from iondb.rundb.plan.page_plan.kits_step_data import KitsFieldNames
from iondb.rundb.plan.page_plan.reference_step_data import ReferenceFieldNames
from iondb.rundb.plan.page_plan.save_plan_step_data import SavePlanFieldNames
from iondb.rundb.plan.plan_validator import validate_sample_tube_label, validate_barcode_sample_association

from iondb.rundb.plan.views_helper import convert

import json
import logging
logger = logging.getLogger(__name__)

class BarcodeBySampleFieldNames():
    # NOTE: this step uses field names from save_plan_step_data for consistency
    SAMPLESET_ITEMS = 'samplesetitems'

    ONCO_SAME_SAMPLE = "isOncoSameSample"

class BarcodeBySampleStepData(AbstractStepData):

    def __init__(self):
        super(BarcodeBySampleStepData, self).__init__()

        self.resourcePath = 'rundb/plan/page_plan/page_plan_by_sample_barcode.html'
        
        self.savedFields = OrderedDict()
        self.savedFields[SavePlanFieldNames.BARCODE_SET] = ''
        self.savedFields[SavePlanFieldNames.BARCODE_SAMPLE_TUBE_LABEL] = ''
        self.savedFields['applicationType'] = ''
        self.savedFields['irDown'] = '0'

        self.prepopulatedFields[BarcodeBySampleFieldNames.SAMPLESET_ITEMS] = []
        self.prepopulatedFields[SavePlanFieldNames.SAMPLE_ANNOTATIONS] = list(SampleAnnotation_CV.objects.all().order_by("annotationType", "iRValue"))
        self.prepopulatedFields['fireValidation'] = "1"

        self.savedObjects[SavePlanFieldNames.BARCODED_IR_PLUGIN_ENTRIES] = []
        self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE] = OrderedDict()
        
        self.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS] = list(dnaBarcode.objects.values_list('name',flat=True).distinct().order_by('name'))
        all_barcodes = {}
        for bc in dnaBarcode.objects.order_by('name', 'index').values('name', 'id_str','sequence'):
            all_barcodes.setdefault(bc['name'],[]).append(bc)
        self.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS_BARCODES] = json.dumps(all_barcodes)

        self.savedObjects['samplesTableList'] = [{"row":"1"}]
        self.savedFields['samplesTable'] = json.dumps(self.savedObjects['samplesTableList'])

        self.savedFields[SavePlanFieldNames.ONCO_SAME_SAMPLE] = False

        self.savedObjects[SavePlanFieldNames.REFERENCE_STEP_HELPER] = None
        
        self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE] = ""
        self.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE] = ""
        self.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE] = ""

        self._dependsOn.append(StepNames.IONREPORTER)
        self._dependsOn.append(StepNames.APPLICATION)
        self._dependsOn.append(StepNames.REFERENCE)

    def getStepName(self):
        return StepNames.BARCODE_BY_SAMPLE

    
    def updateFromStep(self, updated_step):
        if updated_step.getStepName() not in self._dependsOn:
            return

        if updated_step.getStepName() == StepNames.APPLICATION:
            if updated_step.savedObjects[ApplicationFieldNames.RUN_TYPE]:
                self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE] = updated_step.savedObjects[ApplicationFieldNames.RUN_TYPE].runType
            else:
                self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE] = ""

            logger.debug("barcode_by_sample_step_data.updateFromStep() APPLICATION going to update RUNTYPE value=%s" %(self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE]))
       
        if updated_step.getStepName() == StepNames.IONREPORTER:
            
            # update samples table with saved sampleset items fields for IR 
            if updated_step.savedFields['irAccountId'] and self.prepopulatedFields.get('selectedIr') != updated_step.savedFields['irAccountId']:
                sorted_sampleSetItems = self.prepopulatedFields['samplesetitems']
                for row in self.savedObjects['samplesTableList']:
                    row[SavePlanFieldNames.IR_WORKFLOW] = updated_step.savedFields['irworkflow']
                    sampleset_item = [item for item in sorted_sampleSetItems if item.sample.displayedName == row['sampleName']]
                    if len(sampleset_item) > 0:
                        row[SavePlanFieldNames.IR_GENDER] = sampleset_item[0].gender
                        row[SavePlanFieldNames.IR_RELATION_ROLE] = sampleset_item[0].relationshipRole
                        row[SavePlanFieldNames.IR_SET_ID] = sampleset_item[0].relationshipGroup
                        #20140306-WIP-TO-BE-TESTED
                        row[SavePlanFieldNames.IR_CANCER_TYPE] = sampleset_item[0].cancerType
                        row[SavePlanFieldNames.IR_CELLULARITY_PCT] = sampleset_item[0].cellularityPct                        
                        
                self.savedFields['samplesTable'] = json.dumps(self.savedObjects['samplesTableList'])
            
            self.prepopulatedFields['irworkflow'] = updated_step.savedFields['irworkflow']
            self.prepopulatedFields['selectedIr'] = updated_step.savedFields['irAccountId']
 
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
                
        elif updated_step.getStepName() == StepNames.REFERENCE:
            self.savedObjects[SavePlanFieldNames.REFERENCE_STEP_HELPER] = updated_step

            self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE] = updated_step.savedFields.get(ReferenceFieldNames.REFERENCE, "")
            self.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE] = updated_step.savedFields.get(ReferenceFieldNames.TARGET_BED_FILE, "")
            self.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE] = updated_step.savedFields.get(ReferenceFieldNames.HOT_SPOT_BED_FILE, "")

            #logger.debug("barcode_by_sample_step_data.updateFromStep() REFERENCE plan_reference=%s" %(self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE]))
       
            self.updateSavedFieldsForSamples()
            logger.debug("barcode_by_sample_step_data.updateFromStep() REFERENCE reference.savedFields=%s" %(updated_step.savedFields))


    def updateSavedFieldsForSamples(self):        
        if self.savedFields[SavePlanFieldNames.BARCODE_SET]:
            #convert tuple to string
            planReference = str(self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE])
            planHotSptRegionBedFile = str(self.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE])
            planTargetRegionBedFile = str(self.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE])
                    
            #logger.debug("barcode_by_sample_step_data.updateSavedFieldsForSamples() type(planReference)=%s; planReference=%s" %(type(planReference), planReference))
            #logger.debug("barcode_by_sample_step_data.updateSavedFieldsForSamples() type(self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE])=%s; self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE]=%s" %(type(self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE]), self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE]))

            hasAnyChanges = False
            myTable = json.loads(self.savedFields['samplesTable'])
            #convert unicode to str
            myTable = convert(myTable)
            
            for index, row in enumerate(myTable):
                #logger.debug("barcode_by_sample_step_data.updateSavedFieldsForSamples() B4 CHANGES... BARCODE_SET LOOP row=%s" %(row))                   
                                              
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
                        
                                                
                    #logger.debug("barcode_by_sample_step_data.updateSavedFieldsForSamples() BARCODE_SET LOOP  type(newSampleReference)=%s; newSampleReference=%s" %(type(newSampleReference), newSampleReference))

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
                        #logger.debug("barcode_by_sample_step_data.updateSavedFieldsForSamples() AFTER CHANGES  BARCODE_SET LOOP myTable[index]=%s" %(myTable[index]))                                     
                        hasAnyChanges = True
                        
 
            if hasAnyChanges:    
                #logger.debug("barcode_by_sample_step_data.updateSavedFieldsForSamples() hasAnyChanges AFTER CHANGES... type=%s; myTable=%s" %(type(myTable), myTable))       

                #convert list with single quotes to str with double quotes. Then convert it to be unicode
                self.savedFields['samplesTable'] = unicode(json.dumps(myTable))
                
                #logger.debug("barcode_by_sample_step_data.updateSavedFieldsForSamples() hasAnyChanges AFTER unicode(json.dumps)... type=%s; self.savedFields[samplesTable]=%s" %(type(self.savedFields['samplesTable']), self.savedFields['samplesTable']))       
 
                self.savedObjects['samplesTableList'] = json.loads(self.savedFields['samplesTable'])
                self.updateSavedObjectsFromSavedFields()
                #logger.debug("barcode_by_sample_step_data.updateSavedFieldsForSamples() hasAnyChanges AFTER json.loads... type=%s; self.savedObjects[samplesTableList]=%s" %(type(self.savedObjects['samplesTableList']), self.savedObjects['samplesTableList']))       
                
                                                         

                
    def validateField(self, field_name, new_field_value):
        self.validationErrors.pop(field_name, None)
        errors = None

        if field_name == SavePlanFieldNames.BARCODE_SAMPLE_TUBE_LABEL:
            errors = validate_sample_tube_label(new_field_value)

        if errors:
            self.validationErrors[field_name] = '\n'.join(errors)

    def validateStep(self):
        self.validationErrors.pop(SavePlanFieldNames.NO_BARCODE,None)
        self.validationErrors.pop(SavePlanFieldNames.BAD_BARCODES,None)
                
        samplesTable = json.loads(self.savedFields['samplesTable'])
        barcodeSet = self.savedFields[SavePlanFieldNames.BARCODE_SET]

        selectedBarcodes = []

        for row in samplesTable:
            sample_name = row.get(SavePlanFieldNames.SAMPLE_NAME,'').strip()
            if sample_name:
                if barcodeSet:
                    selectedBarcodes.append(row.get('barcodeId'))

        if barcodeSet:
            errors = validate_barcode_sample_association(selectedBarcodes, barcodeSet)
            
            myErrors = convert(errors)
            if myErrors.get("MISSING_BARCODE", ""):
                self.validationErrors[SavePlanFieldNames.NO_BARCODE] = myErrors.get("MISSING_BARCODE", "")
            if myErrors.get("DUPLICATE_BARCODE", ""):
                self.validationErrors[SavePlanFieldNames.BAD_BARCODES] = myErrors.get("DUPLICATE_BARCODE", "")
 
        self.prepopulatedFields['fireValidation'] = "0"


    def updateSavedObjectsFromSavedFields(self):
        #logger.debug("ENTER barcode_by_sample_step_data.updateSavedObjectsFromSavedFields()")
        
        self.savedObjects['samplesTableList'] = json.loads(self.savedFields['samplesTable'])
        
        if self.savedFields[SavePlanFieldNames.BARCODE_SET]:

            #logger.debug("barcode_by_sample_step_data.updateSavedObjectsFromSavedFields() BARCODE_SET self.savedFields=%s" %(self.savedFields))
            #logger.debug("barcode_by_sample_step_data.updateSavedObjectsFromSavedFields() BARCODE_SET self=%s" %(self))
            

            planReference = self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE]
            planHotSptRegionBedFile = self.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE]
            planTargetRegionBedFile = self.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE]
    
            logger.debug("barcode_by_sample_step_data.updateSavedObjectsFromSavedFields() BARCODE_SET PLAN_REFERENCE=%s; TARGET_REGION=%s; HOTSPOT_REGION=%s;" %(planReference, planTargetRegionBedFile, planHotSptRegionBedFile))

            self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE] = {}
            self.savedObjects[SavePlanFieldNames.BARCODED_IR_PLUGIN_ENTRIES] = []
            for row in self.savedObjects['samplesTableList']:
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
                    
                    runType = self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE]                    
                    
                    #logger.debug("barcode_by_sample_step_data SETTING reference step helper runType=%s; sample_nucleotideType=%s; sampleReference=%s" %(runType, sample_nucleotideType, sampleReference))
                    
                    if runType == "AMPS_DNA_RNA" and sample_nucleotideType == "DNA":
                        reference_step_helper = self.savedObjects[SavePlanFieldNames.REFERENCE_STEP_HELPER]
                        if reference_step_helper:
                            if sampleReference != planReference:
                                reference_step_helper.savedFields[ReferenceFieldNames.REFERENCE] = sampleReference
                                 
                            if sampleHotSpotRegionBedFile != planHotSptRegionBedFile:
                                reference_step_helper.savedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE] = sampleHotSpotRegionBedFile
                                
                            if sampleTargetRegionBedFile != planTargetRegionBedFile:
                                reference_step_helper.savedFields[ReferenceFieldNames.TARGET_BED_FILE] = sampleTargetRegionBedFile
                                         
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
                   
                    self.savedObjects[SavePlanFieldNames.BARCODED_IR_PLUGIN_ENTRIES].append(barcode_ir_userinput_dict)

        #logger.debug("EXIT barcode_by_sample_step_date.updateSavedObjectsFromSaveFields() type(self.savedObjects['samplesTableList'])=%s; self.savedObjects['samplesTableList']=%s" %(type(self.savedObjects['samplesTableList']), self.savedObjects['samplesTableList']));
       
