# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.models import dnaBarcode, Sample, SampleAnnotation_CV
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.page_plan.kits_step_data import KitsFieldNames
from iondb.rundb.plan.page_plan.save_plan_step_data import SavePlanFieldNames

import logging
logger = logging.getLogger(__name__)

class BarcodeBySampleFieldNames():

    BARCODES = 'barcodes'
    BARCODE_SETS = 'barcodesets'
    PLANNEDITEM_BARCODE = 'planneditem_barcode'
    PLANNEDITEM_NAME = 'planneditem_sampleitem_name'
    PLANNED_SAMPLESET_ITEMS = 'planned_sampleset_items'
    PLANNEDITEM_EXTERNALID = 'planneditem_externalid'
    IS_BARCODED = 'is_barcoded'
    PLANNED_DNABARCODES = 'planned_dnabarcodes'
    

    SAMPLESET_ITEMS = 'samplesetitems'
    BARCODE_TO_SAMPLE = 'barcodeToSample'
    CHIP_TO_SAMPLE = 'chipToSample'
    SAMPLE_TO_BARCODE = 'sampleToBarcode'
    SAMPLE_ITEM = 'sampleitem'
    NUMBER_OF_CHIPS = 'chipNumber'
    SAMPLE_NAME = 'sampleName'
    SAMPLE_EXTERNAL_ID = 'sampleExternalId'
    SAMPLE_DESCRIPTION = 'sampleDescription'
    SAMPLE_BARCODE = 'sampleBarcode'
    SAMPLE_SEQUENCE = 'sampleSequence'
    GENDER = 'gender'
    RELATION_ROLE = 'relationRole'
    WORKFLOW = 'workflow'
    TEMPLATE_WORKFLOW  = 'irworkflow'
    BARCODE_ID = 'barcodeId'
    SET_ID = 'irSetID'
    BAD_BARCODES = 'bad_barcodes'
    BAD_ITEM_NAMES = 'bad_item_names'
    NO_BARCODE = 'no_barcode'
    NO_SAMPLE_NAME = 'no_sample_name'
    BARCODED_IR_PLUGIN_ENTRIES = 'barcodedIrPluginEntires'
    SAMPLE_ANNOTATIONS = 'sampleAnnotations'


class BarcodeBySampleStepData(AbstractStepData):

    def __init__(self):
        super(BarcodeBySampleStepData, self).__init__()

        self.resourcePath = 'rundb/plan/page_plan/page_plan_by_sample_barcode.html'
        self.savedListFieldNames = [BarcodeBySampleFieldNames.PLANNEDITEM_BARCODE, BarcodeBySampleFieldNames.PLANNEDITEM_NAME, BarcodeBySampleFieldNames.PLANNEDITEM_EXTERNALID]
        
        self.savedFields = OrderedDict()
        self.savedFields[BarcodeBySampleFieldNames.PLANNEDITEM_NAME] = None
        self.savedFields[BarcodeBySampleFieldNames.PLANNEDITEM_BARCODE] = None
        self.savedFields[BarcodeBySampleFieldNames.IS_BARCODED] = None
        self.savedFields[BarcodeBySampleFieldNames.PLANNEDITEM_EXTERNALID] = None
        self.savedFields[BarcodeBySampleFieldNames.BARCODE_ID] = None
        self.savedFields['applicationType'] = ''

        self.prepopulatedFields[BarcodeBySampleFieldNames.SAMPLESET_ITEMS] = []
        self.prepopulatedFields[BarcodeBySampleFieldNames.SAMPLE_ANNOTATIONS] = list(SampleAnnotation_CV.objects.all())
        self.prepopulatedFields['fireValidation'] = "1"

        self.savedObjects[BarcodeBySampleFieldNames.BARCODED_IR_PLUGIN_ENTRIES] = []

        self.savedObjects[BarcodeBySampleFieldNames.BARCODE_TO_SAMPLE] = OrderedDict()
        self.savedObjects[BarcodeBySampleFieldNames.SAMPLE_TO_BARCODE] = OrderedDict()
        self.savedObjects[BarcodeBySampleFieldNames.CHIP_TO_SAMPLE] = OrderedDict()

        self.prepopulatedFields[BarcodeBySampleFieldNames.BARCODES] = list(dnaBarcode.objects.values('name').distinct().order_by('name'))

        self._dependsOn.append(StepNames.IONREPORTER)
        self._dependsOn.append(StepNames.SAVE_PLAN_BY_SAMPLE)
        

    def getStepName(self):
        return StepNames.BARCODE_BY_SAMPLE

    def updateFromStep(self, updated_step):
        if updated_step.getStepName() == StepNames.IONREPORTER:
            self.prepopulatedFields['irworkflow'] = updated_step.savedFields['irworkflow']
            self.prepopulatedFields['accountId'] = updated_step.savedFields['irAccountId']
            self.updateSavedObjectsFromSavedFields()
            
        elif updated_step.getStepName() == StepNames.SAVE_PLAN_BY_SAMPLE:
            error_messages = updated_step.savedFields['error_messages']
            warning_messages = updated_step.savedFields['warning_messages']
            self.validationErrors['error_messages'] = []
            self.validationErrors['warning_messages'] = []
            for msg in error_messages.split(';'):
                if msg and msg != '':
                    self.validationErrors['error_messages'].append(msg)
            for msg in warning_messages.split(';'):
                if msg and msg != '':
                    self.validationErrors['warning_messages'].append(msg)
            # raise RuntimeError(self.validationErrors['error_messages'])
            if not warning_messages:
                self.validationErrors.pop('warning_messages')
            if not error_messages:
                self.validationErrors.pop('error_messages')


    def create_barcode_ir_userinput_dict(self, index):
        return {
            KitsFieldNames.BARCODE_ID             : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_BARCODE, index)],
            'sample'                              : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_NAME, index)],
            SavePlanFieldNames.SAMPLE_EXTERNAL_ID : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_EXTERNAL_ID, index)],
            SavePlanFieldNames.SAMPLE_DESCRIPTION : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_DESCRIPTION, index)],
            SavePlanFieldNames.WORKFLOW           : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.WORKFLOW, index)],
            SavePlanFieldNames.GENDER             : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.GENDER, index)],
            SavePlanFieldNames.RELATION_ROLE      : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.RELATION_ROLE, index)] or "",
            SavePlanFieldNames.SET_ID             : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SET_ID, index)],
        }

    def create_non_barcoded_userinput_dict(self, index):
        irworkflow = self.prepopulatedFields['irworkflow']
        pe_irworkflow = self.prepopulatedFields['pe_irworkflow']
        try:
            samplename = self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_NAME, index)]
            sampleDescription = self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_DESCRIPTION, index)]
            sampleExternalId = self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_EXTERNAL_ID, index)]
            gender = self.savedFields['%s%d' % (BarcodeBySampleFieldNames.GENDER, index)]
            relationRole = self.savedFields['%s%d' % (BarcodeBySampleFieldNames.RELATION_ROLE, index)] or ""
            workflow = self.savedFields['%s%d' % (BarcodeBySampleFieldNames.WORKFLOW, index)] if self.savedFields['%s%d' % (BarcodeBySampleFieldNames.WORKFLOW, index)] != pe_irworkflow or pe_irworkflow == irworkflow else irworkflow
            setId = self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SET_ID, index)]
            relation = self.savedFields['%s%d' % ('relation', index)]
        except:    
            samplename = ''
            sampleDescription = ''
            sampleExternalId = ''
            gender = ''
            relationRole = ''
            workflow = ''
            setId = ''
            relation = ''
        return {
                BarcodeBySampleFieldNames.NUMBER_OF_CHIPS : index + 1,
                BarcodeBySampleFieldNames.SAMPLE_NAME            : samplename,
                BarcodeBySampleFieldNames.SAMPLE_DESCRIPTION  : sampleDescription,
                BarcodeBySampleFieldNames.SAMPLE_EXTERNAL_ID     : sampleExternalId,
                BarcodeBySampleFieldNames.GENDER         : gender,
                BarcodeBySampleFieldNames.RELATION_ROLE  : relationRole,
                BarcodeBySampleFieldNames.WORKFLOW  : workflow,
                BarcodeBySampleFieldNames.SET_ID             : setId,
                'relation'                                       : relation,
            }

    def validateStep(self):

        is_barcoded = bool(int(self.savedFields[BarcodeBySampleFieldNames.IS_BARCODED]))

        self.validationErrors[BarcodeBySampleFieldNames.BAD_BARCODES] = []
        self.validationErrors[BarcodeBySampleFieldNames.NO_BARCODE] = []
        self.validationErrors[BarcodeBySampleFieldNames.BAD_ITEM_NAMES] = []
        self.validationErrors[BarcodeBySampleFieldNames.NO_SAMPLE_NAME] = []

        samplesetitems = self.prepopulatedFields[BarcodeBySampleFieldNames.SAMPLESET_ITEMS]
        user_samplesetitems = []
        user_barcodes = []

        for i in range(20):
            try:
                dnabarcode_id_str = self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_BARCODE, i)]
                sample_name = self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_NAME, i)]

            # if dnabarcode_id_str == '' and is_barcoded:
            #     if not self.validationErrors[BarcodeBySampleFieldNames.NO_BARCODE]:
            #         self.validationErrors[BarcodeBySampleFieldNames.NO_BARCODE] = "Please select a barcode for each sample set item"

            # if sample_name == '':
            #     if not self.validationErrors[BarcodeBySampleFieldNames.NO_SAMPLE_NAME]:
            #         self.validationErrors[BarcodeBySampleFieldNames.NO_SAMPLE_NAME] = "Please select a name for each sample item"

            # if sample_name and  sample_name in user_samplesetitems:
            #     if not self.validationErrors[BarcodeBySampleFieldNames.BAD_ITEM_NAMES]:
            #         self.validationErrors[BarcodeBySampleFieldNames.BAD_ITEM_NAMES] = "Sample item name selections have to be unique"
            # else:
            #     user_samplesetitems.append(sample_name)

                if dnabarcode_id_str and dnabarcode_id_str in user_barcodes and is_barcoded and sample_name:
                    if not self.validationErrors[BarcodeBySampleFieldNames.BAD_BARCODES]:
                        self.validationErrors[BarcodeBySampleFieldNames.BAD_BARCODES].append("Barcode %s selections have to be unique" % (dnabarcode_id_str))
                else:
                    user_barcodes.append(dnabarcode_id_str)
            except Exception, e:
                continue

        if not self.validationErrors[BarcodeBySampleFieldNames.NO_BARCODE]:
            self.validationErrors.pop(BarcodeBySampleFieldNames.NO_BARCODE, None)
            
        self.prepopulatedFields['fireValidation'] = "0"
            

        if not self.validationErrors[BarcodeBySampleFieldNames.NO_SAMPLE_NAME]:
            self.validationErrors.pop(BarcodeBySampleFieldNames.NO_SAMPLE_NAME, None)

        if not self.validationErrors[BarcodeBySampleFieldNames.BAD_ITEM_NAMES]:
            self.validationErrors.pop(BarcodeBySampleFieldNames.BAD_ITEM_NAMES, None)

        if not self.validationErrors[BarcodeBySampleFieldNames.BAD_BARCODES]:
            self.validationErrors.pop(BarcodeBySampleFieldNames.BAD_BARCODES)

        self.validationErrors.pop('error_messages', None)
        self.validationErrors.pop('warning_messages', None)


    def updateSavedObjectsFromSavedFields(self):

        self.savedObjects[BarcodeBySampleFieldNames.BARCODED_IR_PLUGIN_ENTRIES] = []

        samplesetitems = self.prepopulatedFields[BarcodeBySampleFieldNames.SAMPLESET_ITEMS]

        for i in range(20):
            try:
                sample_name = self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_NAME, i)]
                sample_external_id = self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_EXTERNAL_ID, i)]
                sample = Sample.objects.get(displayedName=sample_name, externalId=sample_external_id)

                self.savedObjects[BarcodeBySampleFieldNames.BARCODED_IR_PLUGIN_ENTRIES].append(self.create_barcode_ir_userinput_dict(i))
            except Exception, e:                
                sample_name = None
        
        irworkflow = self.prepopulatedFields['irworkflow']
        pe_irworkflow = self.prepopulatedFields['pe_irworkflow']
        if self.savedFields[BarcodeBySampleFieldNames.BARCODE_ID]:
        
            planned_dnabarcodes = list(dnaBarcode.objects.filter(name=self.savedFields[BarcodeBySampleFieldNames.BARCODE_ID]).order_by('id_str'))
            self.prepopulatedFields[BarcodeBySampleFieldNames.PLANNED_DNABARCODES] = planned_dnabarcodes

            barcode_to_sample = self.savedObjects[BarcodeBySampleFieldNames.BARCODE_TO_SAMPLE]
            sample_to_barcode = self.savedObjects[BarcodeBySampleFieldNames.SAMPLE_TO_BARCODE]

            barcode_to_sample.clear()
            sample_to_barcode.clear()

            for i in range(len(self.prepopulatedFields[BarcodeBySampleFieldNames.PLANNED_DNABARCODES])):
                sample_name = self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_NAME, i)]
                if sample_name:
                    sample_external_id = self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_EXTERNAL_ID, i)]
                    dnabarcode_id_str = self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_BARCODE, i)]
                    
                    barcode_to_sample['%s%d' % (dnabarcode_id_str, i)] = {
                        BarcodeBySampleFieldNames.NUMBER_OF_CHIPS        : '1',
                        BarcodeBySampleFieldNames.SAMPLE_NAME            : sample_name,
                        BarcodeBySampleFieldNames.SAMPLE_DESCRIPTION     : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_DESCRIPTION, i)],
                        BarcodeBySampleFieldNames.SAMPLE_EXTERNAL_ID     : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_EXTERNAL_ID, i)],
                        BarcodeBySampleFieldNames.GENDER                 : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.GENDER, i)],
                        BarcodeBySampleFieldNames.RELATION_ROLE          : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.RELATION_ROLE, i)],
                        BarcodeBySampleFieldNames.WORKFLOW               : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.WORKFLOW, i)] if self.savedFields['%s%d' % (BarcodeBySampleFieldNames.WORKFLOW, i)] != pe_irworkflow and self.savedFields['%s%d' % (BarcodeBySampleFieldNames.WORKFLOW, i)] != irworkflow else irworkflow,
                        BarcodeBySampleFieldNames.BARCODE_ID             : dnabarcode_id_str,
                        BarcodeBySampleFieldNames.SET_ID                 : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SET_ID, i)],
                        'relation'                                       : self.savedFields['%s%d' % ('relation', i)],
                        }
                    barcodes = [dnabarcode_id_str]
                    for j in range(len(self.prepopulatedFields[BarcodeBySampleFieldNames.PLANNED_DNABARCODES])):
                        if i != j:
                            other_sample_name = self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_NAME, j)]
                            if sample_name and other_sample_name and sample_name == other_sample_name:
                                other_dnabarcode_id_str = self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_BARCODE, j)]
                                if other_dnabarcode_id_str not in barcodes:
                                    barcodes.append(other_dnabarcode_id_str)

                    if sample_name and sample_name not in sample_to_barcode:
                        sample_to_barcode[sample_name] = {
                            'barcodeSampleInfo' : { 
                                    dnabarcode_id_str: {
                                                    'externalId'   : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_EXTERNAL_ID, i)],
                                                    'description'  : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_DESCRIPTION, i)],

                                    }
                                },
                            'barcodes'          : barcodes
                            }
                else:
                    dnabarcode = self.prepopulatedFields[BarcodeBySampleFieldNames.PLANNED_DNABARCODES][i]
                    barcode_to_sample['%s%d' % (dnabarcode.id_str, i)] = {
                        BarcodeBySampleFieldNames.NUMBER_OF_CHIPS        : '1',
                        BarcodeBySampleFieldNames.SAMPLE_NAME            : sample_name,
                        BarcodeBySampleFieldNames.SAMPLE_DESCRIPTION     : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_DESCRIPTION, i)],
                        BarcodeBySampleFieldNames.SAMPLE_EXTERNAL_ID     : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_EXTERNAL_ID, i)],
                        BarcodeBySampleFieldNames.GENDER                 : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.GENDER, i)],
                        BarcodeBySampleFieldNames.RELATION_ROLE          : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.RELATION_ROLE, i)],
                        BarcodeBySampleFieldNames.WORKFLOW               : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.WORKFLOW, i)] if self.savedFields['%s%d' % (BarcodeBySampleFieldNames.WORKFLOW, i)] != pe_irworkflow or pe_irworkflow == irworkflow else irworkflow,
                        BarcodeBySampleFieldNames.BARCODE_ID             : dnabarcode.id_str,
                        BarcodeBySampleFieldNames.SET_ID                 : self.savedFields['%s%d' % (BarcodeBySampleFieldNames.SET_ID, i)],
                        }
        else:
            self.savedObjects[BarcodeBySampleFieldNames.CHIP_TO_SAMPLE].clear()
            _range = 20 if self.prepopulatedFields['accountId'] == "0" else len(samplesetitems)
            for i in range(_range):
                self.savedObjects[BarcodeBySampleFieldNames.CHIP_TO_SAMPLE][i] = self.create_non_barcoded_userinput_dict(i)

