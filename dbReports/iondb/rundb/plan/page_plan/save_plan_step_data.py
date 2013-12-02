# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.models import dnaBarcode, Plugin, SampleAnnotation_CV
from iondb.rundb.plan.views_helper import is_valid_chars, is_valid_length,\
    is_invalid_leading_chars
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.page_plan.kits_step_data import KitsFieldNames
from iondb.rundb.plan.page_plan.export_step_data import ExportFieldNames
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import json
import uuid
import logging
logger = logging.getLogger(__name__)

MAX_LENGTH_PLAN_NAME = 512
MAX_LENGTH_NOTES = 1024
MAX_LENGTH_SAMPLE_DESCRIPTION = 1024
MAX_LENGTH_SAMPLE_NAME = 127
MAX_LENGTH_SAMPLE_TUBE_LABEL = 512

class SavePlanFieldNames():

    UPLOADERS = 'uploaders'
    SET_ID = 'setid'
    EXTERNAL_ID = 'externalId'
    WORKFLOW = 'Workflow'
    GENDER = 'Gender'
    RELATION = 'Relation'
    RELATION_ROLE = 'RelationRole'
    PLAN_NAME = 'planName'
    SAMPLE = 'sample'
    NOTE = 'note'
    SELECTED_IR = 'selectedIr'
    IR_CONFIG_JSON = 'irConfigJson'
    PREV_BARCODE_ID = 'prevBarcodeId'
    BARCODE_SAMPLE_TUBE_LABEL = 'barcodeSampleTubeLabel'
    BARCODE_TO_SAMPLE = 'barcodeToSample'
    SAMPLE_TO_BARCODE = 'sampleToBarcode'
    BARCODED_IR_PLUGIN_ENTRIES = 'barcodedIrPluginEntires'
    SAMPLE_EXTERNAL_ID = 'sampleExternalId'
    SAMPLE_NAME = 'sampleName'
    SAMPLE_DESCRIPTION = 'sampleDescription'
    TUBE_LABEL = 'tubeLabel'
    IR_GENDER = 'irGender'
    IR_WORKFLOW = 'irWorkflow'
    IR_RELATION = 'irRelation'
    IR_RELATION_ROLE = 'irRelationRole'
    IR_SET_ID = 'irSetID'
    CHIP_RANGE = 'chipRange'
    CHIP_TO_SAMPLES = 'chipToSamples'
    BAD_SAMPLE_NAME = 'bad_sample_name'
    BAD_SAMPLE_EXTERNAL_ID = 'bad_sample_external_id'
    BAD_SAMPLE_DESCRIPTION = 'bad_sample_description'
    BAD_TUBE_LABEL = 'bad_tube_label'
    BARCODE_SAMPLE_NAME = 'barcodeSampleName'
    BARCODE_SAMPLE_DESCRIPTION = 'barcodeSampleDescription'
    BARCODE_SAMPLE_EXTERNAL_ID = 'barcodeSampleExternalId'
    BARCODE_SAMPLE_INFO = 'barcodeSampleInfo'
    NO_SAMPLES = 'no_samples'
    DESCRIPTION = 'description'
    BAD_IR_SET_ID = 'badIrSetId'
    SAMPLE_ANNOTATIONS = 'sampleAnnotations'


class SavePlanStepData(AbstractStepData):

    def __init__(self):
        super(SavePlanStepData, self).__init__()
        self.resourcePath = 'rundb/plan/page_plan/page_plan_save_plan.html'
        
        self.savedFields[SavePlanFieldNames.PLAN_NAME] = None
        self.savedFields[SavePlanFieldNames.NOTE] = None
        self.savedFields['applicationType'] = ''
        self.savedFields['irDown'] = '0'
        
        self.prepopulatedFields[SavePlanFieldNames.SELECTED_IR] = None
        self.prepopulatedFields[SavePlanFieldNames.IR_CONFIG_JSON] = None
        self.prepopulatedFields[SavePlanFieldNames.SAMPLE_ANNOTATIONS] = list(SampleAnnotation_CV.objects.all())
        self.prepopulatedFields[SavePlanFieldNames.PREV_BARCODE_ID] = None
        self.prepopulatedFields[KitsFieldNames.BARCODES] = []
        self.savedFields[SavePlanFieldNames.BARCODE_SAMPLE_TUBE_LABEL] = None
        self.savedObjects[SavePlanFieldNames.BARCODE_TO_SAMPLE] = OrderedDict()
        self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE] = OrderedDict()
        self.savedObjects[SavePlanFieldNames.BARCODED_IR_PLUGIN_ENTRIES] = []
        self.prepopulatedFields['fireValidation'] = "1"


        self.prepopulatedFields[SavePlanFieldNames.CHIP_RANGE] = [i + 1 for i in range(20)]
        self.savedObjects[SavePlanFieldNames.CHIP_TO_SAMPLES] = OrderedDict()
        for i in self.prepopulatedFields[SavePlanFieldNames.CHIP_RANGE]:
            self.savedFields[self.__get_index_key(SavePlanFieldNames.SAMPLE_EXTERNAL_ID, i)] = None
            self.savedFields[self.__get_index_key(SavePlanFieldNames.SAMPLE_NAME, i)] = None
            self.savedFields[self.__get_index_key(SavePlanFieldNames.SAMPLE_DESCRIPTION, i)] = None
            self.savedFields[self.__get_index_key(SavePlanFieldNames.TUBE_LABEL, i)] = None
            self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_GENDER, i)] = None
            self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_WORKFLOW, i)] = None
            self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_RELATION, i)] = None
            self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_RELATION_ROLE, i)] = None
            self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_SET_ID, i)] = 0
        self.updateSavedObjectsFromSavedFields()
        self._dependsOn = [StepNames.KITS]

    def getStepName(self):
        return StepNames.SAVE_PLAN
    
    def validateField(self, field_name, new_field_value):
        if field_name == SavePlanFieldNames.PLAN_NAME:
            self.validatePlanName(new_field_value)
        elif field_name == SavePlanFieldNames.NOTE:
            self.validateNote(new_field_value)
        elif field_name == SavePlanFieldNames.BARCODE_SAMPLE_TUBE_LABEL:
            self.validateBarcodeSampleTubeLabel(new_field_value)
            
    def validateStep(self):
        any_samples = False
        self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_NAME] = []
        self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_EXTERNAL_ID] = []
        self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_DESCRIPTION] = []
        self.validationErrors[SavePlanFieldNames.BAD_TUBE_LABEL] = []
        self.validationErrors[SavePlanFieldNames.BAD_IR_SET_ID] = []

        if self.prepopulatedFields[KitsFieldNames.BARCODES]:
            for barcode in self.prepopulatedFields[KitsFieldNames.BARCODES]:
                if self.validate_field(self.savedFields[self.__get_index_key(SavePlanFieldNames.BARCODE_SAMPLE_NAME, barcode.pk)],
                                             self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_NAME]):
                    any_samples = True
                    
                external_id = self.savedFields[self.__get_index_key(SavePlanFieldNames.BARCODE_SAMPLE_EXTERNAL_ID, barcode.pk)]
                if external_id:
                    self.validate_field(external_id, self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_EXTERNAL_ID])
                    
                description = self.savedFields[self.__get_index_key(SavePlanFieldNames.BARCODE_SAMPLE_DESCRIPTION, barcode.pk)]
                if description:
                    self.validate_field(description, self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_DESCRIPTION], False,
                                        MAX_LENGTH_SAMPLE_DESCRIPTION)
                    
                ir_set_id = self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_SET_ID, barcode.pk)]
                if ir_set_id and not (str(ir_set_id).isdigit()):
                    self.validationErrors[SavePlanFieldNames.BAD_IR_SET_ID].append(ir_set_id)
        else:
            for i in self.prepopulatedFields[SavePlanFieldNames.CHIP_RANGE]:
                if self.validate_field(self.savedFields[self.__get_index_key(SavePlanFieldNames.SAMPLE_NAME, i)],
                                             self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_NAME]):
                    any_samples = True
                tube_label = self.savedFields[self.__get_index_key(SavePlanFieldNames.TUBE_LABEL, i)]
                if tube_label and not is_valid_length(tube_label, MAX_LENGTH_SAMPLE_TUBE_LABEL):
                    self.validationErrors[SavePlanFieldNames.BAD_TUBE_LABEL].append(tube_label)
                
                external_id = self.savedFields[self.__get_index_key(SavePlanFieldNames.SAMPLE_EXTERNAL_ID, i)]
                if external_id:
                    self.validate_field(external_id, self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_EXTERNAL_ID])
                
                description = self.savedFields[self.__get_index_key(SavePlanFieldNames.SAMPLE_DESCRIPTION, i)]
                if description:
                    self.validate_field(description, self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_DESCRIPTION], False,
                                        MAX_LENGTH_SAMPLE_DESCRIPTION)
                
                ir_set_id = self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_SET_ID, i)]
                if ir_set_id and not (str(ir_set_id).isdigit()):
                    self.validationErrors[SavePlanFieldNames.BAD_IR_SET_ID].append(ir_set_id)
        
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

    def validate_field(self, value, bad_samples, validate_leading_chars=True, max_length=MAX_LENGTH_SAMPLE_NAME):
        exists = False
        if value:
            exists = True
            
            if not is_valid_chars(value):
                bad_samples.append(value)
            
            if validate_leading_chars and value not in bad_samples and is_invalid_leading_chars(value):
                bad_samples.append(value)
            
            if value not in bad_samples and not is_valid_length(value, max_length):
                bad_samples.append(value)
        
        return exists

    def validatePlanName(self, new_plan_name):
        field_name = SavePlanFieldNames.PLAN_NAME
        self.validationErrors[field_name] = []
        valid = True
        if not new_plan_name:
            self.validationErrors[field_name].append('Error, please enter a Plan Name.')
            valid = False
        
        if not is_valid_chars(new_plan_name):
            self.validationErrors[field_name].append('Error, Plan Name should contain only numbers, letters, spaces, and the following: . - _')
            valid = False
            
        if not is_valid_length(new_plan_name, MAX_LENGTH_PLAN_NAME):
            self.validationErrors[field_name].append('Error, Plan Name length should be %s characters maximum. It is currently %s characters long.' % (str(MAX_LENGTH_PLAN_NAME), str(len(new_plan_name))))
            valid = False
            
        if valid:
            self.validationErrors.pop(field_name, None)
            
    def validateNote(self, new_note_value):
        field_name = SavePlanFieldNames.NOTE
        if new_note_value:
            valid = True
            self.validationErrors[field_name] = []
            if not is_valid_chars(new_note_value):
                valid = False
                self.validationErrors[field_name].append('Error, Note  should contain only numbers, letters, spaces, and the following: . - _')
            if not is_valid_length(new_note_value, MAX_LENGTH_NOTES):
                valid = False
                self.validationErrors[field_name].append('Error, Note length should be %s characters maximum.' % str(MAX_LENGTH_NOTES))
        
            if valid:
                self.validationErrors.pop(field_name, None)
        else:
            self.validationErrors.pop(field_name, None)
    
    def validateBarcodeSampleTubeLabel(self, new_barcode_sample_tube_label):
        field_name = SavePlanFieldNames.BARCODE_SAMPLE_TUBE_LABEL
        if new_barcode_sample_tube_label:
            valid = True
            self.validationErrors[field_name] = []
            if not is_valid_length(new_barcode_sample_tube_label, MAX_LENGTH_PLAN_NAME):
                valid = False
                self.validationErrors[field_name].append('Error, Tube Label length should be %s characters maximum.' % str(MAX_LENGTH_PLAN_NAME))
            
            if valid:
                self.validationErrors.pop(field_name, None)
        else:
            self.validationErrors.pop(field_name, None)


    def create_barcode_dict_from_sample(self, i):
        return {
                SavePlanFieldNames.SAMPLE_EXTERNAL_ID : self.savedFields[self.__get_index_key(SavePlanFieldNames.SAMPLE_EXTERNAL_ID, i)],
                SavePlanFieldNames.SAMPLE_NAME        : self.savedFields[self.__get_index_key(SavePlanFieldNames.SAMPLE_NAME, i)],
                SavePlanFieldNames.SAMPLE_DESCRIPTION : self.savedFields[self.__get_index_key(SavePlanFieldNames.SAMPLE_DESCRIPTION, i)],
                SavePlanFieldNames.TUBE_LABEL         : self.savedFields[self.__get_index_key(SavePlanFieldNames.TUBE_LABEL, i)],
            }

    def create_barcode_dict_from_barcode(self, sample_name, i):
        return {
                SavePlanFieldNames.SAMPLE_EXTERNAL_ID : self.savedFields[self.__get_index_key(SavePlanFieldNames.BARCODE_SAMPLE_EXTERNAL_ID, i)],
                SavePlanFieldNames.SAMPLE_NAME        : sample_name,
                SavePlanFieldNames.SAMPLE_DESCRIPTION : self.savedFields[self.__get_index_key(SavePlanFieldNames.BARCODE_SAMPLE_DESCRIPTION, i)],
            }

    def update_barcode_dict(self, i):
        return {
                SavePlanFieldNames.IR_GENDER          : self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_GENDER, i)],
                SavePlanFieldNames.IR_WORKFLOW        : self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_WORKFLOW, i)],
                SavePlanFieldNames.IR_RELATION        : self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_RELATION, i)],
                SavePlanFieldNames.IR_RELATION_ROLE   : self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_RELATION_ROLE, i)],
                SavePlanFieldNames.IR_SET_ID          : self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_SET_ID, i)]
            }

    def create_barcode_ir_userinput_dict(self, barcode, sample_name, barcode_dict):
        return {
            KitsFieldNames.BARCODE_ID             : barcode.id_str,
            SavePlanFieldNames.SAMPLE             : sample_name,
            SavePlanFieldNames.SAMPLE_NAME        : sample_name.strip().replace(' ', '_'),
            SavePlanFieldNames.SAMPLE_EXTERNAL_ID : barcode_dict[SavePlanFieldNames.SAMPLE_EXTERNAL_ID],
            SavePlanFieldNames.SAMPLE_DESCRIPTION : barcode_dict[SavePlanFieldNames.SAMPLE_DESCRIPTION],
            SavePlanFieldNames.WORKFLOW           : self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_WORKFLOW, barcode.pk)],
            SavePlanFieldNames.GENDER             : self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_GENDER, barcode.pk)],
            SavePlanFieldNames.RELATION           : self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_RELATION, barcode.pk)],
            SavePlanFieldNames.RELATION_ROLE      : self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_RELATION_ROLE, barcode.pk)]
        }
        
    def updateSavedObjectsFromSavedFields(self):
        self.prepopulatedFields["fireValidation"] = "0"
        for i in self.prepopulatedFields[SavePlanFieldNames.CHIP_RANGE]:
            self.savedObjects[SavePlanFieldNames.CHIP_TO_SAMPLES][i] = self.create_barcode_dict_from_sample(i)
            self.savedObjects[SavePlanFieldNames.CHIP_TO_SAMPLES][i].update(self.update_barcode_dict(i))

        self.savedObjects[SavePlanFieldNames.BARCODE_TO_SAMPLE].clear()
        self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE].clear()
        self.savedObjects[SavePlanFieldNames.BARCODED_IR_PLUGIN_ENTRIES] = []
        for barcode in self.prepopulatedFields[KitsFieldNames.BARCODES]:
            sample_name = self.savedFields[self.__get_index_key(SavePlanFieldNames.BARCODE_SAMPLE_NAME, barcode.pk)]
            barcode_dict = self.create_barcode_dict_from_barcode(sample_name, barcode.pk)
            barcode_dict.update(self.update_barcode_dict(barcode.pk))
            
            self.savedObjects[SavePlanFieldNames.BARCODE_TO_SAMPLE][barcode] = barcode_dict
            
            if sample_name and sample_name not in self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE]:
                self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE][sample_name] = {KitsFieldNames.BARCODES: [], SavePlanFieldNames.BARCODE_SAMPLE_INFO: {}}
            if sample_name:
                self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE][sample_name][KitsFieldNames.BARCODES].append(barcode.id_str)
                self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE][sample_name][SavePlanFieldNames.BARCODE_SAMPLE_INFO][barcode.id_str] = \
                            {
                                SavePlanFieldNames.EXTERNAL_ID : barcode_dict[SavePlanFieldNames.SAMPLE_EXTERNAL_ID], 
                                SavePlanFieldNames.DESCRIPTION : barcode_dict[SavePlanFieldNames.SAMPLE_DESCRIPTION]
                            }
                
            if sample_name:
                barcode_ir_userinput_dict = self.create_barcode_ir_userinput_dict(barcode, sample_name, barcode_dict)
                
                if self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_SET_ID, barcode.pk)]:
                    barcode_ir_userinput_dict[SavePlanFieldNames.SET_ID] = self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_SET_ID, barcode.pk)]
                else:
                    barcode_ir_userinput_dict[SavePlanFieldNames.SET_ID] = None
                self.savedObjects[SavePlanFieldNames.BARCODED_IR_PLUGIN_ENTRIES].append(barcode_ir_userinput_dict)
        # raise RuntimeError(self.savedObjects[SavePlanFieldNames.BARCODE_TO_SAMPLE])
    
    def __get_index_key(self, key_starts_with, index):
        return key_starts_with + str(index)
    
    def updateFromStep(self, updated_step):
        if updated_step.getStepName() not in self._dependsOn:
            return
        
        if updated_step.getStepName() == StepNames.KITS:
            self.__update_from_kits(updated_step)
        elif updated_step.getStepName() == StepNames.EXPORT:
            
            if ExportFieldNames.IR_ACCOUNT_ID not in updated_step.savedFields\
                or not updated_step.savedFields[ExportFieldNames.IR_ACCOUNT_ID]\
                or updated_step.savedFields[ExportFieldNames.IR_ACCOUNT_ID] == '0':
                if SavePlanFieldNames.BAD_IR_SET_ID in self.validationErrors:
                    self.validationErrors.pop(SavePlanFieldNames.BAD_IR_SET_ID, None)
                for i in self.prepopulatedFields['chipRange']:
                    self.__update_ir_saved_fields_from_kits(i)
                for barcode in self.prepopulatedFields[KitsFieldNames.BARCODES]:
                    self.__update_ir_saved_fields_from_kits(barcode.pk)
                self.updateSavedObjectsFromSavedFields()
        

    def __update_barcodesample_saved_fields_from_kits(self, i):
        self.savedFields[self.__get_index_key(SavePlanFieldNames.BARCODE_SAMPLE_EXTERNAL_ID, i)] = None
        self.savedFields[self.__get_index_key(SavePlanFieldNames.BARCODE_SAMPLE_NAME, i)] = None
        self.savedFields[self.__get_index_key(SavePlanFieldNames.BARCODE_SAMPLE_DESCRIPTION, i)] = None

    def __update_sample_saved_fields_from_kits(self, i):
        self.savedFields[self.__get_index_key(SavePlanFieldNames.SAMPLE_EXTERNAL_ID, i)] = None
        self.savedFields[self.__get_index_key(SavePlanFieldNames.SAMPLE_NAME, i)] = None
        self.savedFields[self.__get_index_key(SavePlanFieldNames.SAMPLE_DESCRIPTION, i)] = None
        self.savedFields[self.__get_index_key(SavePlanFieldNames.TUBE_LABEL, i)] = None

    def __update_ir_saved_fields_from_kits(self, i):
        self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_GENDER, i)] = None
        self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_WORKFLOW, i)] = None
        self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_RELATION, i)] = None
        self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_RELATION_ROLE, i)] = None
        self.savedFields[self.__get_index_key(SavePlanFieldNames.IR_SET_ID, i)] = None

            
    def __update_from_kits(self, kits_step):
        new_barcode_id = kits_step.savedFields[KitsFieldNames.BARCODE_ID]
        if str(new_barcode_id) == str(self.prepopulatedFields[SavePlanFieldNames.PREV_BARCODE_ID]):
            return
        
        if new_barcode_id:
            self.prepopulatedFields[SavePlanFieldNames.PREV_BARCODE_ID] = new_barcode_id
            self.prepopulatedFields[KitsFieldNames.BARCODES] = list(dnaBarcode.objects.filter(name=new_barcode_id).order_by('name', 'index'))
            self.validationErrors.clear()

            for i in self.prepopulatedFields['chipRange']:
                self.__update_sample_saved_fields_from_kits(i)
                self.__update_ir_saved_fields_from_kits(i)

            for barcode in self.prepopulatedFields[KitsFieldNames.BARCODES]:
                self.__update_barcodesample_saved_fields_from_kits(barcode.pk)
                self.__update_ir_saved_fields_from_kits(barcode.pk)

        else:
            self.prepopulatedFields['prevBarcodeId'] = None
            self.validationErrors.clear()
            
            for barcode in self.prepopulatedFields[KitsFieldNames.BARCODES]:
                self.__update_barcodesample_saved_fields_from_kits(barcode.pk)
                self.__update_ir_saved_fields_from_kits(barcode.pk)

            self.prepopulatedFields['barcodes'] = []
        self.updateSavedObjectsFromSavedFields()