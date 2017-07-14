# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import logging
import json
from django.db.models import Q
from django.core.urlresolvers import reverse
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.models import KitInfo, Chip, dnaBarcode, LibraryKey,\
    ThreePrimeadapter, GlobalConfig, FlowOrder, common_CV
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.page_plan.application_step_data import ApplicationFieldNames
from iondb.utils import validation
from iondb.rundb.plan.plan_validator import validate_flows, validate_libraryReadLength, validate_templatingSize

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict


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
    FLOW_ORDERS = "flowOrders"
    FLOW_ORDER = "flowOrder"
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
    INSTRUMENT_TYPES = 'instrumentTypes'
    BARCODE_ID = 'barcodeId'
    BARCODES = 'barcodes'
    BARCODES_SUBSET = 'barcodes_subset'
    IS_DUPLICATED_READS = 'isDuplicateReads'
    BASE_RECALIBRATE = 'base_recalibrate'
    BASE_RECALIBRATION_MODES = 'base_recalibration_modes'
    REALIGN = 'realign'
    FLOWS = 'flows'

    TEMPLATE_KITS = "templateKits"

    IS_BARCODE_KIT_SELECTION_REQUIRED = "isBarcodeKitSelectionRequired"
    IS_CHIP_TYPE_REQUIRED = "is_chipType_required"
    BARCODE_KIT_NAME = "barcodeId"
    LIBRARY_READ_LENGTH = "libraryReadLength"
    READ_LENGTH = "readLength"
    TEMPLATING_SIZE_CHOICES = "templatingSizeChoices"
    TEMPLATING_SIZE = "templatingSize"
    READ_LENGTH_CHOICES = "readLengthChoices"
    FLOWS_FROM_CATEGORY_RULES = "defaultFlowsFromCategoryRules"
    SAMPLE_PREP_PROTOCOL = "samplePrepProtocol"
    SAMPLE_PREP_PROTOCOLS = "samplePrepProtocols"
    PLAN_CATEGORIES = "planCategories"


class KitsStepData(AbstractStepData):

    def __init__(self, sh_type):
        super(KitsStepData, self).__init__(sh_type)
        self.resourcePath = 'rundb/plan/page_plan/page_plan_kits.html'
        self.prev_step_url = reverse("page_plan_application")
        self.next_step_url = reverse("page_plan_plugins")

        # 20130827-test
        # self._dependsOn.append(StepNames.IONREPORTER)

        self._dependsOn.append(StepNames.APPLICATION)
        self._dependsOn.append(StepNames.BARCODE_BY_SAMPLE)

        self.savedFields[KitsFieldNames.SAMPLE_PREPARATION_KIT] = None
        self.prepopulatedFields[KitsFieldNames.SAMPLE_PREP_KITS] = KitInfo.objects.filter(kitType='SamplePrepKit', isActive=True).order_by('description')

        self.savedFields[KitsFieldNames.LIBRARY_KIT_NAME] = None
        self.prepopulatedFields[KitsFieldNames.LIB_KITS] = KitInfo.objects.filter(kitType__in=["LibraryKit", "LibraryPrepKit"], isActive=True).order_by("description")

        self.savedFields[KitsFieldNames.LIBRARY_KEY] = None
        self.prepopulatedFields[KitsFieldNames.FORWARD_LIB_KEYS] = LibraryKey.objects.filter(direction='Forward', runMode='single').order_by('-isDefault', 'name')
        self.savedFields[KitsFieldNames.LIBRARY_KEY] = self.prepopulatedFields[KitsFieldNames.FORWARD_LIB_KEYS][0].sequence

        self.savedFields[KitsFieldNames.TF_KEY] = GlobalConfig.get().default_test_fragment_key

        self.savedFields[KitsFieldNames.FORWARD_3_PRIME_ADAPTER] = None
        self.prepopulatedFields[KitsFieldNames.FORWARD_3_ADAPTERS] = ThreePrimeadapter.objects.filter(direction='Forward', runMode='single').order_by('-isDefault', 'chemistryType', 'name')
        self.savedFields[KitsFieldNames.FORWARD_3_PRIME_ADAPTER] = self.prepopulatedFields[KitsFieldNames.FORWARD_3_ADAPTERS][0].sequence

        self.savedFields[KitsFieldNames.FLOW_ORDER] = None
        self.prepopulatedFields[KitsFieldNames.FLOW_ORDERS] = FlowOrder.objects.filter(isActive=True).order_by('-isDefault', 'description')
        self.savedFields[KitsFieldNames.FLOW_ORDER] = None

        self.savedFields[KitsFieldNames.TEMPLATE_KIT_NAME] = None
        # no longer default to OneTouch
        # self.savedFields[KitsFieldNames.TEMPLATE_KIT_TYPE] = KitsFieldNames.ONE_TOUCH
        self.savedFields[KitsFieldNames.TEMPLATE_KIT_TYPE] = None

        oneTouchDict = {
            KitsFieldNames.KIT_VALUES: KitInfo.objects.filter(kitType__in=['TemplatingKit', 'AvalancheTemplateKit'], isActive=True).order_by("description"),
            KitsFieldNames.APPLICATION_DEFAULT: None
            }

        ionChefDict = {
            KitsFieldNames.KIT_VALUES: KitInfo.objects.filter(kitType='IonChefPrepKit', isActive=True).order_by("description"),
            KitsFieldNames.APPLICATION_DEFAULT: None
            }

        self.prepopulatedFields[KitsFieldNames.TEMPLATE_KIT_TYPES] = {
            KitsFieldNames.ONE_TOUCH: oneTouchDict,
            KitsFieldNames.ION_CHEF: ionChefDict,
            }

        self.savedFields[KitsFieldNames.SEQUENCE_KIT_NAME] = None
        self.prepopulatedFields[KitsFieldNames.SEQ_KITS] = KitInfo.objects.filter(kitType='SequencingKit', isActive=True).order_by("description")

        self.savedFields[KitsFieldNames.TEMPLATE_KIT_NAME] = None
        self.prepopulatedFields[KitsFieldNames.TEMPLATE_KITS] = KitInfo.objects.filter(kitType__in=['TemplatingKit', 'AvalancheTemplateKit', 'IonChefPrepKit'], isActive=True).order_by("description")

        self.savedFields[KitsFieldNames.CONTROL_SEQUENCE] = None
        self.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', isActive=True).order_by("description")

        self.savedFields[KitsFieldNames.CHIP_TYPE] = None
        self.prepopulatedFields[KitsFieldNames.INSTRUMENT_TYPES] = [(k, v) for k, v in Chip.ALLOWED_INSTRUMENT_TYPES if k]
        self.prepopulatedFields[KitsFieldNames.CHIP_TYPES] = list(Chip.objects.filter(isActive=True).order_by('description', 'name').distinct('description'))

        self.savedFields[KitsFieldNames.BARCODE_ID] = None
        self.prepopulatedFields[KitsFieldNames.BARCODES] = list(dnaBarcode.objects.filter(active=True).values('name').distinct().order_by('name'))

        gc = GlobalConfig.get()
        self.savedFields[KitsFieldNames.IS_DUPLICATED_READS] = gc.mark_duplicates

        self.savedFields[KitsFieldNames.BASE_RECALIBRATE] = gc.base_recalibration_mode

        self.prepopulatedFields[KitsFieldNames.BASE_RECALIBRATION_MODES] = OrderedDict()
        self.prepopulatedFields[KitsFieldNames.BASE_RECALIBRATION_MODES]["standard_recal"] = "Default Calibration"
        self.prepopulatedFields[KitsFieldNames.BASE_RECALIBRATION_MODES]["panel_recal"] = "Enable Calibration Standard"
        self.prepopulatedFields[KitsFieldNames.BASE_RECALIBRATION_MODES]["blind_recal"] = "Blind Calibration"         
        self.prepopulatedFields[KitsFieldNames.BASE_RECALIBRATION_MODES]["no_recal"] = "No Calibration"

        self.savedFields[KitsFieldNames.REALIGN] = gc.realign

        self.savedFields[KitsFieldNames.FLOWS] = 0
        self.savedFields[KitsFieldNames.LIBRARY_READ_LENGTH] = 0
        self.savedFields[KitsFieldNames.READ_LENGTH] = 0

        self.prepopulatedFields[KitsFieldNames.IS_BARCODE_KIT_SELECTION_REQUIRED] = False

        self.prepopulatedFields[KitsFieldNames.TEMPLATING_SIZE_CHOICES] = ["200", "400"]
        self.savedFields[KitsFieldNames.TEMPLATING_SIZE] = ""
        # For raptor templating kits, templating size cannot be used to drive UI behavior or db persistence.  Use read length instead.
        self.prepopulatedFields[KitsFieldNames.READ_LENGTH_CHOICES] = ["200", "400"]
        self.prepopulatedFields[KitsFieldNames.FLOWS_FROM_CATEGORY_RULES] = json.dumps(KitInfo._category_flowCount_rules)

        self.savedFields[KitsFieldNames.SAMPLE_PREP_PROTOCOL] = None
        self.prepopulatedFields[KitsFieldNames.SAMPLE_PREP_PROTOCOLS] = common_CV.objects.filter(isActive=True, cv_type = "samplePrepProtocol").order_by('uid')
        self.prepopulatedFields[KitsFieldNames.PLAN_CATEGORIES] = ""

        self.sh_type = sh_type

    def getStepName(self):
        return StepNames.KITS

    def updateSavedObjectsFromSavedFields(self):
        pass

    def alternateUpdateFromStep(self, updated_step):
        """
        update a step or section with alternate logic based on the step it is depending on.
        when editing a post-sequencing plan, if user changes application, we want to update the minimum set of info without
        altering what have previously been selected
        """

        # logger.debug("ENTER kits_step_data.alternateUpdateFromStep() updated_step.stepName=%s" %(updated_step.getStepName()))

        if updated_step.getStepName() == StepNames.APPLICATION and updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]:
            applProduct = updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]
            logger.debug("kits_step_data.alternateUpdateFromStep() Updating kits for applproduct %s" % applProduct.productCode)

            if applProduct.applType.runType in ["AMPS", "AMPS_EXOME"]:
                self.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType__in=["", "DNA", "AMPS_ANY"], isActive=True).order_by("name")
            elif applProduct.applType.runType in ["AMPS_RNA"]:
                self.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType__in=["", "RNA", "AMPS_ANY"], isActive=True).order_by("name")
            elif applProduct.applType.runType in ["RNA"]:
                self.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType="RNA", isActive=True).order_by("name")
            elif applProduct.applType.runType in ["AMPS_DNA_RNA"]:
                self.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType__in=["", "DNA", "RNA", "AMPS_ANY"], isActive=True).order_by("name")
            else:
                self.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType__in=["", "DNA"], isActive=True).order_by("name")

            available_dnaBarcodes = dnaBarcode.objects.filter(Q(active=True) | Q(name=self.savedFields[KitsFieldNames.BARCODE_ID]))
            self.prepopulatedFields[KitsFieldNames.BARCODES] = list(available_dnaBarcodes.values('name').distinct().order_by('name'))
            self.prepopulatedFields[KitsFieldNames.BARCODES_SUBSET] = \
                list(available_dnaBarcodes.filter(type__in=applProduct.barcodeKitSelectableTypes_list).values('name').distinct().order_by('name'))

            self.prepopulatedFields[KitsFieldNames.IS_BARCODE_KIT_SELECTION_REQUIRED] = applProduct.isBarcodeKitSelectionRequired

    def updateFromStep(self, updated_step):

        # logger.debug("ENTER kits_step_data.updateFromStep() updated_step.stepName=%s" %(updated_step.getStepName()))

        if updated_step.getStepName() == StepNames.BARCODE_BY_SAMPLE:
            self.savedFields[KitsFieldNames.BARCODE_ID] = updated_step.savedFields['barcodeSet']  # cannot use SavePlanFieldNames because of circular import

        # if user has not clicked on the Application chevron, we need to try to do some catch up
        if updated_step.getStepName() == StepNames.APPLICATION and updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]:
            applProduct = updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]
            logger.debug("kits_step_data.updateFromStep() Updating kits for applproduct %s" % applProduct.productCode)

            self.alternateUpdateFromStep(updated_step)

            if applProduct.defaultTemplateKit:
                self.prepopulatedFields[KitsFieldNames.TEMPLATE_KIT_TYPES][KitsFieldNames.ONE_TOUCH][KitsFieldNames.APPLICATION_DEFAULT] = applProduct.defaultTemplateKit

            if applProduct.defaultIonChefPrepKit:
                self.prepopulatedFields[KitsFieldNames.TEMPLATE_KIT_TYPES][KitsFieldNames.ION_CHEF][KitsFieldNames.APPLICATION_DEFAULT] = applProduct.defaultIonChefPrepKit

            if updated_step.savedObjects[ApplicationFieldNames.UPDATE_KITS_DEFAULTS]:
                self.updateFieldsFromDefaults(applProduct)

    def updateFieldsFromDefaults(self, applProduct):

        if not applProduct.defaultChipType:
            self.savedFields[KitsFieldNames.CHIP_TYPE] = None
        else:
            self.savedFields[KitsFieldNames.CHIP_TYPE] = applProduct.defaultChipType

        if applProduct.defaultFlowOrder:
            self.savedFields[KitsFieldNames.FLOW_ORDER] = applProduct.defaultFlowOrder.flowOrder
            logger.debug("applProduct kits_step_data FROM APPLPRODUCT savedFields[flowOrder]=%s" % (self.savedFields[KitsFieldNames.FLOW_ORDER]))
        else:
            if applProduct.defaultSequencingKit and applProduct.defaultSequencingKit.defaultFlowOrder:
                self.savedFields[KitsFieldNames.FLOW_ORDER] = applProduct.defaultSequencingKit.defaultFlowOrder.flowOrder
                logger.debug("applProduct kits_step_data FROM SEQUENCING KIT savedFields[flowOrder]=%s" % (self.savedFields[KitsFieldNames.FLOW_ORDER]))

        if applProduct.defaultSamplePrepKit:
            self.savedFields[KitsFieldNames.SAMPLE_PREPARATION_KIT] = applProduct.defaultSamplePrepKit.name
        else:
            self.savedFields[KitsFieldNames.SAMPLE_PREPARATION_KIT] = None

        if applProduct.defaultLibraryKit:
            self.savedFields[KitsFieldNames.LIBRARY_KIT_NAME] = applProduct.defaultLibraryKit.name

        if applProduct.defaultTemplateKit:
            self.savedFields[KitsFieldNames.TEMPLATE_KIT_NAME] = applProduct.defaultTemplateKit.name
            if applProduct.defaultTemplateKit.kitType in ['TemplatingKit']:
                self.savedFields[KitsFieldNames.TEMPLATE_KIT_TYPE] = KitsFieldNames.ONE_TOUCH
            elif applProduct.defaultTemplateKit.kitType == 'IonChefPrepKit':
                self.savedFields[KitsFieldNames.TEMPLATE_KIT_TYPE] = KitsFieldNames.ION_CHEF

        if applProduct.defaultSequencingKit:
            self.savedFields[KitsFieldNames.SEQUENCE_KIT_NAME] = applProduct.defaultSequencingKit.name

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
            logger.debug("kits_step_data.updateFieldsFromDefaults() USE APPLPRODUCT - flowCount=%s" % (str(self.savedFields[KitsFieldNames.FLOWS])))
        else:
            if applProduct.defaultSequencingKit:
                self.savedFields[KitsFieldNames.FLOWS] = applProduct.defaultSequencingKit.flowCount
            logger.debug("kits_step_data.updateFieldsFromDefaults() USE SEQ KIT- flowCount=%s" % (str(self.savedFields[KitsFieldNames.FLOWS])))

        
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

        if field_name == KitsFieldNames.CHIP_TYPE and self.prepopulatedFields.get('is_chipType_required', True):
            if validation.has_value(new_field_value):
                self.validationErrors.pop(field_name, None)
            else:
                self.validationErrors[field_name] = validation.required_error("Chip Type")

        if field_name == KitsFieldNames.TEMPLATE_KIT_NAME:
            if validation.has_value(new_field_value):
                self.validationErrors.pop(field_name, None)
            else:
                self.validationErrors[field_name] = validation.required_error("Template Kit")

        if field_name == KitsFieldNames.LIBRARY_READ_LENGTH:
            errors = validate_libraryReadLength(new_field_value)
            if errors:
                self.validationErrors[field_name] = ' '.join(errors)
            else:
                self.validationErrors.pop(field_name, None)

        if field_name == KitsFieldNames.READ_LENGTH:
            if new_field_value:
                errors = validate_libraryReadLength(new_field_value)
                if errors:
                    self.validationErrors[field_name] = ' '.join(errors)
                else:
                    self.validationErrors.pop(field_name, None)
                    self.savedFields[KitsFieldNames.LIBRARY_READ_LENGTH] = new_field_value

        if field_name == KitsFieldNames.TEMPLATING_SIZE:
            errors = validate_templatingSize(new_field_value)
            if errors:
                self.validationErrors[field_name] = ' '.join(errors)
            else:
                self.validationErrors.pop(field_name, None)

    def validateField_crossField_dependencies(self, fieldNames, fieldValues):
        if KitsFieldNames.LIBRARY_KIT_NAME in fieldNames and KitsFieldNames.BARCODE_KIT_NAME:
            newFieldValue = fieldValues.get(KitsFieldNames.LIBRARY_KIT_NAME, "")
            dependentFieldValue = fieldValues.get(KitsFieldNames.BARCODE_KIT_NAME, "")

            self.validateField_crossField_dependency(KitsFieldNames.LIBRARY_KIT_NAME, newFieldValue, KitsFieldNames.BARCODE_KIT_NAME, dependentFieldValue)

    def validateField_crossField_dependency(self, field_name, new_field_value, dependent_field_name, dependent_field_value):
        isBarcodeKitRequired = False

        if field_name == KitsFieldNames.LIBRARY_KIT_NAME:
            if new_field_value:
                libKit_objs = KitInfo.objects.filter(kitType__in=["LibraryKit", "LibraryPrepKit"], name=new_field_value).order_by("-isActive")
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
