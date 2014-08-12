# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.contrib.auth.models import User

from iondb.rundb.models import PlannedExperiment, Experiment, RunType, ApplProduct, \
    ReferenceGenome, Content, KitInfo, VariantFrequencies, dnaBarcode, \
    LibraryKey, ThreePrimeadapter, Chip, QCType, Project, Plugin, \
    PlannedExperimentQC

from iondb.rundb.plan.views_helper import dict_bed_hotspot, get_default_or_first_IR_account, get_internal_name_for_displayed_name, \
    get_ir_set_id, is_operation_supported_by_obj
from iondb.rundb.plan.plan_validator import validate_plan_name, validate_notes, validate_sample_name, validate_flows, \
    validate_QC, validate_projects, validate_sample_tube_label, validate_sample_id, validate_barcoded_sample_info

from traceback import format_exc

import plan_csv_writer
import iondb.rundb.plan.views

import copy

import logging
logger = logging.getLogger(__name__)

import simplejson


class MyPlan:
    def __init__(self, selectedTemplate, selectedExperiment, selectedEAS, userName):        
        if not selectedTemplate:
            self.planObj = None
            self.expObj = None
            self.easObj = None
        else:                    
            self.planObj = copy.copy(selectedTemplate)
            self.planObj.pk = None
            self.planObj.planGUID = None
            self.planObj.planShortID = None
            self.planObj.isReusable = False
            self.planObj.isSystem = False
            self.planObj.isSystemDefault = False
            self.planObj.expName = ""
            self.planObj.planName = ""
            self.planObj.planExecuted = False   
            self.planObj.latestEAS = None 
                      
            if userName:
                self.planObj.username = userName             

            self.expObj = copy.copy(selectedExperiment)
            self.expObj.pk = None
            self.expObj.unique = None
            self.expObj.plan = None

            # copy EAS
            self.easObj = copy.copy(selectedEAS)
            self.easObj.pk = None
            self.easObj.experiment = None
            self.easObj.isEditable = True
            
        self.sampleList = []
        self.sampleIdList = []
        self.nucleotideTypeList = []
        
       
    def get_planObj(self):
        return self.planObj
    
    def get_expObj(self):
        return self.expObj
    
    def get_easObj(self):
        return self.easObj
    
    def get_sampleList(self):
        return self.sampleList

    def get_sampleIdList(self):
        return self.sampleIdList

    def get_nucleotideTypeList(self):
        return self.nucleotideTypeList    
    

def validate_csv_plan(csvPlanDict, request):
    """ validate csv contents and convert user input to raw data to prepare for plan persistence
    returns: a collection of error messages if errors found, a dictionary of raw data values
    """
    
    logger.debug("ENTER plan_csv_validator.validate_csv_plan() csvPlanDict=%s; " %(csvPlanDict))
        
    failed = []
    rawPlanDict = {}
    planObj = None
    
    planDict = {}

    isToSkipRow = False

    selectedTemplate = None
    selectedExperiment = None
    selectedEAS = None
    
    #skip this row if no values found (will not prohibit the rest of the files from upload
    skipIfEmptyList = [plan_csv_writer.COLUMN_TEMPLATE_NAME]
    
    for skipIfEmpty in skipIfEmptyList:
        if skipIfEmpty in csvPlanDict:
            if not csvPlanDict[skipIfEmpty]:
                #required column is empty
                isToSkipRow = True
            
    if isToSkipRow:
        return failed, planDict, rawPlanDict, isToSkipRow
    
    #check if mandatory fields are present
    requiredList = [plan_csv_writer.COLUMN_TEMPLATE_NAME, plan_csv_writer.COLUMN_PLAN_NAME]
    
    for required in requiredList:
        if required in csvPlanDict:
            if not csvPlanDict[required]:
                failed.append((required, "Required column is empty"))
        else:
            failed.append((required, "Required column is missing"))

    templateName = csvPlanDict.get(plan_csv_writer.COLUMN_TEMPLATE_NAME)

    if templateName:
        selectedTemplate, errorMsg = _get_template(templateName)

        if selectedTemplate:
            isSupported = is_operation_supported_by_obj(selectedTemplate)
            
            logger.debug("plan_csv_validator.validate_csv_plan() selectedTemplate.pk=%d; selectedTemplate.planDisplayedName=%s; isSupported=%s" %(selectedTemplate.id, selectedTemplate.planDisplayedName, isSupported))            

            if (not isSupported):                                           
                errorMsg = "Template name: " + templateName + " is not supported to create plans from"
                failed.append((plan_csv_writer.COLUMN_TEMPLATE_NAME, errorMsg))  
                return failed, planDict, rawPlanDict, isToSkipRow
                        
            selectedExperiment = selectedTemplate.experiment
 
            selectedEAS = selectedTemplate.latestEAS
            if not selectedEAS:
                logger.debug("plan_csv_validator.validate_csv_plan() NO latestEAS FOUND for selectedTemplate.pk=%d; selectedTemplate.planDisplayedName=%s" %(selectedTemplate.id, selectedTemplate.planDisplayedName))
                selectedEAS = selectedTemplate.experiment.get_EAS()
            
            #logger.debug("plan_csv_validator.validate_csv_plan() selectedTemplate.pk=%d; selectedTemplate.planDisplayedName=%s; EAS.pk=%d" %(selectedTemplate.id, selectedTemplate.planDisplayedName, selectedEAS.id))            
        
        if errorMsg:
            failed.append((plan_csv_writer.COLUMN_TEMPLATE_NAME, errorMsg))  
            return failed, planDict, rawPlanDict, isToSkipRow

        planObj = _init_plan(selectedTemplate, selectedExperiment, selectedEAS, request)

    else:        
        return failed, planDict, rawPlanDict, isToSkipRow

    errorMsg = _validate_sample_prep_kit(csvPlanDict.get(plan_csv_writer.COLUMN_SAMPLE_PREP_KIT), selectedTemplate, planObj)  
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_SAMPLE_PREP_KIT, errorMsg))
    
    errorMsg = _validate_lib_kit(csvPlanDict.get(plan_csv_writer.COLUMN_LIBRARY_KIT), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_LIBRARY_KIT, errorMsg))
         
    errorMsg = _validate_template_kit(csvPlanDict.get(plan_csv_writer.COLUMN_TEMPLATING_KIT), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_TEMPLATING_KIT, errorMsg))
    
    errorMsg = _validate_control_seq_kit(csvPlanDict.get(plan_csv_writer.COLUMN_CONTROL_SEQ_KIT), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_CONTROL_SEQ_KIT, errorMsg))
    
    errorMsg = _validate_seq_kit(csvPlanDict.get(plan_csv_writer.COLUMN_SEQ_KIT), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_SEQ_KIT, errorMsg))
    
    errorMsg = _validate_chip_type(csvPlanDict.get(plan_csv_writer.COLUMN_CHIP_TYPE), selectedTemplate, planObj, selectedExperiment)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_CHIP_TYPE, errorMsg))
    
    errorMsg = _validate_flows(csvPlanDict.get(plan_csv_writer.COLUMN_FLOW_COUNT), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_FLOW_COUNT, errorMsg))
    
    errorMsg = _validate_sample_tube_label(csvPlanDict.get(plan_csv_writer.COLUMN_SAMPLE_TUBE_LABEL), selectedTemplate, planObj)
    
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_SAMPLE_TUBE_LABEL, errorMsg))
   
    errorMsg, beadLoadQCValue = _validate_qc_pct(csvPlanDict.get(plan_csv_writer.COLUMN_BEAD_LOAD_PCT), selectedTemplate, planObj, "Bead loading")
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_BEAD_LOAD_PCT, errorMsg))
    
    rawPlanDict["Bead Loading (%)"] = beadLoadQCValue
    
    errorMsg, keySignalQCValue = _validate_qc_pct(csvPlanDict.get(plan_csv_writer.COLUMN_KEY_SIGNAL_PCT), selectedTemplate, planObj, "Key signal")
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_KEY_SIGNAL_PCT, errorMsg))
    
    rawPlanDict["Key Signal (1-100)"] = keySignalQCValue
    
    errorMsg, usableSeqQCValue = _validate_qc_pct(csvPlanDict.get(plan_csv_writer.COLUMN_USABLE_SEQ_PCT), selectedTemplate, planObj, "Usable sequence")
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_USABLE_SEQ_PCT, errorMsg))
    
    rawPlanDict["Usable Sequence (%)"] = usableSeqQCValue
    
    errorMsg = _validate_ref(csvPlanDict.get(plan_csv_writer.COLUMN_REF), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_REF, errorMsg))
    
    errorMsg = _validate_target_bed(csvPlanDict.get(plan_csv_writer.COLUMN_TARGET_BED), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_TARGET_BED, errorMsg))
    
    errorMsg = _validate_hotspot_bed(csvPlanDict.get(plan_csv_writer.COLUMN_HOTSPOT_BED), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_HOTSPOT_BED, errorMsg))                        
    
    errorMsg,plugins = _validate_plugins(csvPlanDict.get(plan_csv_writer.COLUMN_PLUGINS), selectedTemplate, selectedEAS, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_PLUGINS, errorMsg))
        
    errorMsg, projects = _validate_projects(csvPlanDict.get(plan_csv_writer.COLUMN_PROJECTS), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_PROJECTS, errorMsg))
    
    rawPlanDict["newProjects"] = projects                 

    errorMsg = _validate_plan_name(csvPlanDict.get(plan_csv_writer.COLUMN_PLAN_NAME), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_PLAN_NAME, errorMsg))
    
    errorMsg = _validate_notes(csvPlanDict.get(plan_csv_writer.COLUMN_NOTES), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_NOTES, errorMsg))                        

    barcodedSampleJson = None
    sampleDisplayedName = None
    sampleId = None
    
    barcodeKitName = selectedEAS.barcodeKitName
    if barcodeKitName:
        errorMsg, barcodedSampleJson = _validate_barcodedSamples(csvPlanDict, selectedTemplate, barcodeKitName, planObj)

        if errorMsg:
            failed.append(("barcodedSample", errorMsg))
        else:
            #logger.debug("plan_csv_validator barcodedSampleJson.keys=%s" %(barcodedSampleJson.keys()))
            planObj.get_sampleList().extend(barcodedSampleJson.keys())

        errorMsg = _validate_barcodedSamples_collectively(barcodedSampleJson, selectedTemplate, planObj)
        if errorMsg:
            failed.append(("barcodedSample", errorMsg))
        
    else:    
        errorMsg, sampleDisplayedName = _validate_sample(csvPlanDict.get(plan_csv_writer.COLUMN_SAMPLE), selectedTemplate, planObj)
        if errorMsg:
            failed.append((plan_csv_writer.COLUMN_SAMPLE, errorMsg))                        
        else:
            if sampleDisplayedName:
                planObj.get_sampleList().append(sampleDisplayedName)

        errorMsg, sampleId = _validate_sample_id(csvPlanDict.get(plan_csv_writer.COLUMN_SAMPLE_ID), selectedTemplate, planObj)
        if errorMsg:
            failed.append((plan_csv_writer.COLUMN_SAMPLE_ID, errorMsg))                        
        else:
            if sampleDisplayedName:
                planObj.get_sampleIdList().append(sampleId if sampleId else "")
                planObj.get_nucleotideTypeList().append("")
                

    errorMsg, uploaders, has_ir_v1_0 = _validate_export(csvPlanDict.get(plan_csv_writer.COLUMN_EXPORT), selectedTemplate, selectedEAS, planObj, sampleDisplayedName, sampleId, barcodedSampleJson, request)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_EXPORT, errorMsg))                        
        
    #if uploaderJson:
    #   errorMsg = _validate_IR_workflow_v1_x(csvPlanDict.get(plan_csv_writer.COLUMN_IR_V1_X_WORKFLOW), selectedTemplate, planObj, uploaderJson)
    #   if errorMsg:
    #       failed.append((plan_csv_writer.COLUMN_IR_V1_X_WORKFLOW, errorMsg))                        

    try:
        if uploaders and has_ir_v1_0:
            errorMsg = _validate_IR_workflow_v1_0(csvPlanDict.get(plan_csv_writer.COLUMN_IR_V1_0_WORKFLOW), selectedTemplate, planObj, uploaders)
            if errorMsg:
                failed.append((plan_csv_writer.COLUMN_IR_V1_0_WORKFLOW, errorMsg))                        

        selectedPlugins = dict(plugins)
        selectedPlugins.update(uploaders)

        rawPlanDict["selectedPlugins"] = selectedPlugins

        planObj.get_easObj().selectedPlugins = selectedPlugins
    
    except:
        logger.exception(format_exc())
        errorMsg = "Internal error while processing selected plugins info. " + format_exc()
        failed.append(("selectedPlugins", errorMsg))
    
    logger.debug("EXIT plan_csv_validator.validate_csv_plan() rawPlanDict=%s; " %(rawPlanDict))
    
    planDict = {
                "plan" : planObj.get_planObj(),
                "exp" : planObj.get_expObj(),
                "eas" : planObj.get_easObj(),
                "samples" : planObj.get_sampleList(),
                "sampleIds" :  planObj.get_sampleIdList(),
                "sampleNucleotideTypes" : planObj.get_nucleotideTypeList()
                }
    
    return failed, planDict, rawPlanDict, isToSkipRow


def _get_template(templateName):    
    templates = PlannedExperiment.objects.filter(planDisplayedName = templateName.strip(), isReusable=True).order_by("-date")
    
    if templates:
        ##logger.debug("plan_csv_valiadtor._get_template() selectedTemplate=%d" %(templates[0].pk))
         
        return templates[0], None
    else:
        #original template name could have unicode (e.g., trademark)
        templates = PlannedExperiment.objects.filter(isReusable=True, isSystemDefault = False).order_by("-date")
        if templates:
            inputTemplateName = templateName.strip()
            for template in templates:
                templateDisplayedName = template.planDisplayedName.encode("ascii", "ignore")
                if templateDisplayedName == inputTemplateName:
                    return template, None

        logger.debug("plan_csv_validator._get_template() NO template found. ")       
        return None, "Template name: " + templateName + " cannot be found to create plans from"


def _init_plan(selectedTemplate, selectedExperiment, selectedEAS, request):    
    userName = request.user.username if request and request.user else ""

    return MyPlan(selectedTemplate, selectedExperiment, selectedEAS, userName)


def _validate_sample_prep_kit(input, selectedTemplate, planObj):
    """
    validate sample prep kit case-insensitively and ignore leading/trailing blanks in the input
    """

    errorMsg = None
    selectedKit = None
            
    if input:
        try:
            selectedKits = KitInfo.objects.filter(kitType = "SamplePrepKit", isActive = True, description__iexact = input.strip())
            if not selectedKits:
                selectedKit = KitInfo.objects.filter(kitType = "SamplePrepKit", isActive = True, name__iexact = input.strip())[0]
            else:
                selectedKit = selectedKits[0]
            
            planObj.get_planObj().samplePrepKitName = selectedKit.name
        except:
            errorMsg = input + " not found."
    else:
        planObj.get_planObj().samplePrepKtName = ""
        
    return errorMsg

def _validate_lib_kit(input, selectedTemplate, planObj):
    """
    validate library kit case-insensitively and ignore leading/trailing blanks in the input
    """
      
    errorMsg = None
    selectedKit = None
        
    if input:
        try:
            selectedKits = KitInfo.objects.filter(kitType = "LibraryKit", isActive = True, description__iexact = input.strip())
            if not selectedKits:
                selectedKit = KitInfo.objects.filter(kitType = "LibraryKit", isActive = True, name__iexact = input.strip())[0]
            else:
                selectedKit = selectedKits[0]
                
            planObj.get_easObj().libraryKitName = selectedKit.name
        except:
            errorMsg = input + " not found."
    else:
        planObj.get_easObj().libraryKitName = ""
          
    return errorMsg
    

def _validate_template_kit(input, selectedTemplate, planObj):
    """
    validate tempplating kit case-insensitively and ignore leading/trailing blanks in the input
    """
  
    errorMsg = None
    selectedKit = None
    
    if input:
        try:
            selectedKits = KitInfo.objects.filter(kitType__in = ["TemplatingKit", "IonChefPrepKit"] , isActive = True, description__iexact = input.strip())
            if not selectedKits:
                selectedKit = KitInfo.objects.filter(kitType__in = ["TemplatingKit", "IonChefPrepKit"], isActive = True, name__iexact = input.strip())[0]
            else:
                selectedKit = selectedKits[0]

            planObj.get_planObj().templatingKitName = selectedKit.name
            
            if selectedKit.kitType == "IonChefPrepKit":
                planObj.get_planObj().planStatus = "pending"
        except:
            errorMsg = input + " not found."
    else:
        ##planObj.get_planObj().templatingKitName = ""
        errorMsg = "Required column is empty. "
          
    return errorMsg



def _validate_control_seq_kit(input, selectedTemplate, planObj):
    """
    validate control sequencing kit case-insensitively and ignore leading/trailing blanks in the input
    """
        
    errorMsg = None
    selectedKit = None
    
    if input:
        try:
            selectedKits = KitInfo.objects.filter(kitType = "ControlSequenceKit", isActive = True, description__iexact = input.strip())
            if not selectedKits:
                selectedKit = KitInfo.objects.filter(kitType = "ControlSequenceKit" , isActive = True, name__iexact = input.strip())[0]
            else:
                selectedKit = selectedKits[0]

            planObj.get_planObj().controlSequencekitname = selectedKit.name
        except:
            errorMsg = input + " not found."
    else:
        planObj.get_planObj().controlSequencekitname = ""

    return errorMsg
    
def _validate_seq_kit(input, selectedTemplate, planObj):
    """
    validate sequencing kit case-insensitively and ignore leading/trailing blanks in the input
    """
    
    errorMsg = None
    selectedKit = None
        
    if input:
        try:
            selectedKits = KitInfo.objects.filter(kitType = "SequencingKit", isActive = True, description__iexact = input.strip())
            if not selectedKits:
                selectedKit = KitInfo.objects.filter(kitType = "SequencingKit", isActive = True, name__iexact = input.strip())[0]
            else:
                selectedKit = selectedKits[0]
            
            planObj.get_expObj().sequencekitname = selectedKit.name
        except:
            errorMsg = input + " not found."
    else:
        planObj.get_expObj().sequencekitname = ""
        
    return errorMsg

def _validate_chip_type(input, selectedTemplate, planObj, selectedExperiment):
    """
    validate chip type case-insensitively and ignore leading/trailing blanks in the input
    """
    errorMsg = None
    if input:
        try:
            selectedChips = Chip.objects.filter(description__iexact = input.strip(), isActive = True)

            #if selected chipType is ambiguous, try to go with the template's. If that doesn't help, settle with the 1st one 
            if len(selectedChips) == 1:
                planObj.get_expObj().chipType = selectedChips[0].name
            elif len(selectedChips) > 1:
                ##template_chipType = selectedTemplate.get_chipType()
                template_chipType = selectedExperiment.chipType
                                
                if template_chipType:
                    template_chipType_objs = Chip.objects.filter(name = template_chipType)
                    
                    if template_chipType_objs:
                        template_chipType_obj = template_chipType_objs[0]
                        if template_chipType_obj.description == input.strip():
                            planObj.get_expObj().chipType = template_chipType_obj.name
                        else:
                            planObj.get_expObj().chipType = selectedChips[0].name
            else:            
                planObj.get_expObj().chipType = selectedChips[0].name
        except:
            logger.exception(format_exc())
            errorMsg = input + " not found."
    else:
        planObj.get_expObj().chipType = ""
        
    return errorMsg

def _validate_flows(input, selectedTemplate, planObj):
    """
    validate flow value with leading/trailing blanks in the input ignored
    """
    errorMsg = None
    if input:
        value = input.strip()
        errors = validate_flows(value)
        if errors:
            errorMsg = '  '.join(errors)
        else:
            planObj.get_expObj().flows = int(value)
    
    return errorMsg


def _validate_sample_tube_label(input, selectedTemplate, planObj):
    """
    validate sample tube label with leading/trailing blanks in the input ignored
    """
    errorMsg = None
    if input:
        value = input.strip()
        errors = validate_sample_tube_label(value)
        if errors:
            errorMsg = '  '.join(errors)
        else:                
            planObj.get_planObj().sampleTubeLabel = value
    else:
        planObj.get_planObj().sampleTubeLabel = ""
        
    return errorMsg

def _validate_qc_pct(input, selectedTemplate, planObj, displayedName):
    """
    validate QC threshold with leading/trailing blanks in the input ignored
    """
    errorMsg = None
    qcValue = None
    if input:
        value = input.strip()
        errors = validate_QC(value, displayedName)
        if errors:
            errorMsg = '  '.join(errors)
        else:
            qcValue = int(value)
    else:
        qcValue = 1

    return errorMsg, qcValue

    
def _validate_ref(input, selectedTemplate, planObj):
    """
    validate genome reference case-insensitively with leading/trailing blanks in the input ignored
    """    
    errorMsg = None
    if input:
        value = input.strip()
        try:
            selectedRef= ReferenceGenome.objects.filter(name = value)[0]
            planObj.get_easObj().reference = selectedRef.short_name 
        except:
            try:
                selectedRef= ReferenceGenome.objects.filter(short_name = value)[0]
                planObj.get_easObj().reference = selectedRef.short_name 
            except:
                logger.exception(format_exc())      
                errorMsg = input + " not found."
    else:
        planObj.get_easObj().reference = ""
         
    return errorMsg

                                            
def _validate_target_bed(input, selectedTemplate, planObj):
    """
    validate target region BED file case-insensitively with leading/trailing blanks in the input ignored
    """        
    errorMsg = None
    if input:
        bedFileDict = dict_bed_hotspot()
        value = input.strip()
        
        isValidated = False
        for bedFile in bedFileDict.get("bedFiles"):
            if value == bedFile.file or value == bedFile.path:
                isValidated = True                
                planObj.get_easObj().targetRegionBedFile = bedFile.file
#            elif value.lower() == bedFile.file.lower() or value.lower() == bedFile.path.lower():
#                isValidated = True   
#                planObj.get_easObj().targetRegionBedFile = bedFile.file

        if not isValidated:
            logger.exception(format_exc())
            errorMsg = input + " not found."
    else:
        planObj.get_easObj().targetRegionBedFile = ""
        
    return errorMsg
    
def _validate_hotspot_bed(input, selectedTemplate, planObj):
    """
    validate hotSpot BED file case-insensitively with leading/trailing blanks in the input ignored
    """            
    errorMsg = None
    if input:
        bedFileDict = dict_bed_hotspot()
        value = input.strip()
        
        isValidated = False
        for bedFile in bedFileDict.get("hotspotFiles"):
            if value == bedFile.file or value == bedFile.path:
                isValidated = True                
                planObj.get_easObj().hotSpotRegionBedFile  = bedFile.file
#            elif value.lower() == bedFile.file.lower() or value.lower() == bedFile.path.lower():
#                isValidated = True   
#                planObj.get_easObj().hotSpotRegionBedFile  = bedFile.file

        if not isValidated:
            logger.exception(format_exc())
            errorMsg = input + " not found."
    else:
        planObj.get_easObj().hotSpotRegionBedFile = ""
    return errorMsg

def _validate_plugins(input, selectedTemplate, selectedEAS, planObj):
    """
    validate plugin case-insensitively with leading/trailing blanks in the input ignored
    """            

    errorMsg = ""
    plugins = {}

    if input:
        for plugin in input.split(";"):
            if plugin:
                value = plugin.strip()
                try:
                    selectedPlugin = Plugin.objects.filter(name = value, selected = True, active = True)[0]

                    pluginUserInput = {}

                    template_selectedPlugins = selectedEAS.selectedPlugins

                    if selectedPlugin.name in template_selectedPlugins:
                        #logger.info("_validate_plugins() FOUND plugin in selectedTemplate....=%s" %(template_selectedPlugins[plugin.strip()]))
                        pluginUserInput = template_selectedPlugins[selectedPlugin.name]["userInput"]

                    pluginDict = {
                                  "id" : selectedPlugin.id,
                                  "name" : selectedPlugin.name,
                                  "version" : selectedPlugin.version,
                                  "userInput" : pluginUserInput,
                                  "features": []
                                  }
                    
                    plugins[selectedPlugin.name] = pluginDict
                except:
                    logger.exception(format_exc())            
                    errorMsg += plugin + " not found. "
    else:
        planObj.get_easObj().selectedPlugins = ""
        
    return errorMsg, plugins


def _validate_projects(input, selectedTemplate, planObj):
    """
    validate projects case-insensitively with leading/trailing blanks in the input ignored
    """            
    
    errorMsg = None
    projects = ''
    
    if input:
        value = input.strip()
        errors, trimmed_projects = validate_projects(value, delim=";")
        if errors:
            errorMsg = '  '.join(errors)
        else:
            projects = trimmed_projects.replace(";",",")

    return errorMsg, projects


def _validate_export(input, selectedTemplate, selectedEAS, planObj, sampleDisplayedName, sampleId, barcodedSampleJson, request):
    """
    validate export case-insensitively with leading/trailing blanks in the input ignored
    """              
    errorMsg = ""
    
    plugins = {}
    has_ir_v1_0 = False
    
    try:
        if input:
            for plugin in input.split(";"):
                if plugin:
                    value = plugin.strip()
                    try:
                        #20121212-TODO: can we query and filter by EXPORT feature fast?
                        selectedPlugin = Plugin.objects.filter(name = value, selected = True, active = True)[0]
    
                        if selectedPlugin.name == "IonReporterUploader_V1_0":
                            has_ir_v1_0 = True

                        if selectedEAS:
                            templateSelectedPlugins = selectedEAS.selectedPlugins
                            if selectedPlugin.name == "IonReporterUploader":
                                isToUseDefaultIRAccount = False
                                
                                if "IonReporterUploader" in templateSelectedPlugins:
                                    templateIRUConfig = templateSelectedPlugins.get("IonReporterUploader", {})
                                    pluginDict = templateIRUConfig

                                    templateUserInput = pluginDict.get("userInput", "")
                                    if templateUserInput:
                                        accountId = templateUserInput.get("accountId", "")
                                        accountName = templateUserInput.get("accountName", "")
                                        
                                        userInputList = {
                                                     "accountId" : accountId,
                                                     "accountName" : accountName,
                                                     "userInputInfo" : _get_IR_userInputInfo_obj(selectedTemplate.irworkflow, sampleDisplayedName, sampleId, barcodedSampleJson)
                                                     } 
                                        pluginDict["userInput"] = userInputList
                            
                                        plugins[selectedPlugin.name] = pluginDict
                                        
                                    else:
                                        isToUseDefaultIRAccount = True
                                else:
                                    isToUseDefaultIRAccount = True
                                    
                                if isToUseDefaultIRAccount:
                                    userIRConfig = get_default_or_first_IR_account(request)
                                    if userIRConfig:
                                        pluginDict = {
                                                      "id" : selectedPlugin.id,
                                                      "name" : selectedPlugin.name,
                                                      "version" : selectedPlugin.version,
                                                      "features": ['export']
                                                      }
                                        
                                        userInputList = {
                                                         "accountId" : userIRConfig["id"],
                                                         "accountName" : userIRConfig["name"],
                                                         "userInputInfo" : _get_IR_userInputInfo_obj(selectedTemplate.irworkflow, sampleDisplayedName, sampleId, barcodedSampleJson)
                                                         } 
                                        pluginDict["userInput"] = userInputList
                            
                                        plugins[selectedPlugin.name] = pluginDict
                    except:
                        logger.exception(format_exc())            
                        errorMsg += plugin + " not found. "
        else:
            planObj.get_easObj().selectedPlugins = ""
    except:
        logger.exception(format_exc())  
        errorMsg = "Internal error during IRU processing"

    return errorMsg, plugins, has_ir_v1_0


def _get_IR_userInputInfo_obj(selectedIrWorkflow, sampleDisplayedName, sampleId, barcodedSampleJson):

    userInputInfo_obj =  []
    if sampleDisplayedName:
        userInputInfo_obj.append(_create_IR_sample_userInputInfo(selectedIrWorkflow, sampleDisplayedName, sampleId, ""))
    else:
        barcodedSamples = barcodedSampleJson.keys()
        for barcodedSample in barcodedSamples:
            value = barcodedSampleJson[barcodedSample]

            barcodeSampleInfo_list = value.get("barcodeSampleInfo", {})
            barcodes = value.get("barcodes", [])
            for barcode in barcodes:
                barcodeSampleInfo = barcodeSampleInfo_list.get(barcode, {})
                #logger.debug("_get_IR_userInputInfo_obj() barcodedSample=%s; sampleDisplayedName=%s; barcode=%s; barcodeSampleInfo=%s" %(barcodedSample, sampleDisplayedName, barcode, barcodeSampleInfo))
                
                userInputInfo_obj.append(_create_IR_barcoded_sample_userInputInfo(selectedIrWorkflow, barcodedSample, barcodeSampleInfo, barcode))
                
    return userInputInfo_obj
    


def _create_IR_sample_userInputInfo(selectedIrWorkflow, sampleDisplayedName, sampleId, sampleNucleotideType):
    #we don't have complete info to construct userInputInfo for IRU. Set workflow to blank
    irWorkflow = ""
    
    if sampleDisplayedName:
        info_dict = {
                     "ApplicationType" : "",
                     "Gender" : "",
                     "Relation" : "",
                     "RelationRole" : "",
                     "Workflow" : irWorkflow,
                     "sample" : sampleDisplayedName,
                     "sampleDescription" : "",
                     "sampleExternalId" : "" if sampleId == None else sampleId,
                     "sampleName" : get_internal_name_for_displayed_name(sampleDisplayedName),
                     "NucleotideType" : "" if sampleNucleotideType == None else sampleNucleotideType,
                     "setid" : get_ir_set_id()
                     
        }
        return info_dict
    
    return {}


def _create_IR_barcoded_sample_userInputInfo(selectedIrWorkflow, sampleDisplayedName, barcodeSampleInfo, barcodeName):
    #we don't have complete info to construct userInputInfo for IRU. Set workflow to blank
    irWorkflow = ""
        
    if sampleDisplayedName and barcodeName:
        info_dict = {
                     "ApplicationType" : "",
                     "Gender" : "",
                     "Relation" : "",
                     "RelationRole" : "",
                     "Workflow" : irWorkflow,
                     "barcodeId" : barcodeName,
                     "sample" : sampleDisplayedName,
                     "sampleDescription" : "",
                     "sampleExternalId" : barcodeSampleInfo.get("externalId", ""),
                     "sampleName" : get_internal_name_for_displayed_name(sampleDisplayedName),
                     "NucleotideType" : barcodeSampleInfo.get("nucleotideType", ""),                     
                     "setid" : get_ir_set_id()
                     
        }
        return info_dict
    
    return {}


def _validate_plan_name(input, selectedTemplate, planObj):
    """
    validate plan name with leading/trailing blanks in the input ignored
    """    
    errorMsg = None
    errors = validate_plan_name(input)
    if errors:
        errorMsg = '  '.join(errors)
    else:
        value = input.strip()
        planObj.get_planObj().planDisplayedName = value
        planObj.get_planObj().planName = value.replace(' ', '_')
    
    return errorMsg
    
def _validate_notes(input, selectedTemplate, planObj):
    """
    validate notes with leading/trailing blanks in the input ignored
    """    
    errorMsg = None
    if input:
        errors = validate_notes(input)
        if errors:
            errorMsg = '  '.join(errors)
        else:                
            planObj.get_expObj().notes = input.strip()
    else:
        planObj.get_expObj().notes = ""
        
    return errorMsg


def _validate_sample(input, selectedTemplate, planObj):
    """
    validate sample name with leading/trailing blanks in the input ignored
    """    
    
    errorMsg = None
    sampleDisplayedName = ""
    
    if not input:
        errorMsg = "Required column is empty"
    else:
        errors = validate_sample_name(input)
        if errors:
            errorMsg = '  '.join(errors)
        else:
            sampleDisplayedName = input.strip()
                    
    return errorMsg, sampleDisplayedName


def _validate_sample_id(input, selectedTemplate, planObj):
    """
    validate sample id with leading/trailing blanks in the input ignored
    """    
    
    errorMsg = None
    sampleId = ""

    if input:
        errors = validate_sample_id(input)
        if errors:
            errorMsg = '  '.join(errors)
        else:
            sampleId = input.strip()
                    
    return errorMsg, sampleId


def _validate_barcodedSamples_collectively(barcodedSampleJson, selectedTemplate, planObj):  
    """ 
    validation rules: 
    For DNA+RNA plans:
    - only allow 2 barcodes with sample
    - sample DOES NOT need to be the same name and id
    - only 1 DNA and 1 RNA nucleotideType
    For non-DNA+RNA plans:
    - nucleotideTypes can be either DNA or RNA but not both
    """
    errorMsg = ""
    
    runType = selectedTemplate.runType
    applicationGroup = selectedTemplate.applicationGroup.name if selectedTemplate.applicationGroup else ""
    logger.debug("plan_csv_validator._validate_barcodedSamples_collectively() runType=%s; applicationGroup=%s" %(runType, applicationGroup))

    sampleName_count = len(planObj.get_sampleList())
    sampleId_count = len(planObj.get_sampleIdList())
    nucleotideType_count = len(planObj.get_nucleotideTypeList())
    
    #logger.debug("plan_csv_validator._validate_barcodedSamples_collectively() sampleNames=%s; sampleIds=%s; nucleotideTypes=%s" %(planObj.get_sampleList(), planObj.get_sampleIdList(), planObj.get_nucleotideTypeList()))    
    #logger.debug("plan_csv_validator._validate_barcodedSamples_collectively() sampleName_count=%d; sampleId_count=%d; nucleotideType_count=%d" %(sampleName_count, sampleId_count, nucleotideType_count))
    
    unique_sampleName_count = len(set(planObj.get_sampleList()))
    unique_sampleId_count = len(set(planObj.get_sampleIdList()))
    unique_nucleotideType_count = len(set(planObj.get_nucleotideTypeList()))
        
    ##if (runType == "AMPS_DNA_RNA" and applicationGroup == "DNA + RNA"):
    if (runType == "AMPS_DNA_RNA"):        
        if (unique_sampleName_count > 2 or unique_sampleId_count > 2):
            errorMsg = "Only up to two samples are allowed for this plan creation. "
        if (unique_nucleotideType_count != 2):
            errorMsg = errorMsg + "Both DNA and RNA nucleotide types are needed for this plan creation"

    else:
        if (unique_nucleotideType_count > 2):
            errorMsg = "Mixed nucleotide types are not supported for this plan creation"

    if errorMsg:
        logger.debug("plan_csv_validator.. ERRORS _validate_barcodedSamples_collectively() errorMsg=%s" %(errorMsg))

    return errorMsg
    

def _validate_barcodedSamples(input, selectedTemplate, barcodeKitName, planObj):    
    errorMsg = None
    barcodedSampleJson = {}
        
    barcodes = list(dnaBarcode.objects.filter(name = barcodeKitName).values('id', 'id_str').order_by('id_str'))
    
    if len(barcodes) == 0:
        errorMsg = "Barcode "+ barcodeKitName + " cannot be found. " 
        return errorMsg, barcodedSampleJson
    
    errorMsgDict = {}



#20131211 example:
#barcodedSamples": {
#
#    "s 1": {
#        "barcodeSampleInfo": {
#            "MuSeek_001": {
#                "description": "",
#                "externalId": ""
#            },
#            "MuSeek_011": {
#                "description": "",
#                "externalId": ""
#            }
#        },
#        "barcodes": [
#            "MuSeek_001"
#            "MuSeek_011"
#        ]
#    },
#    "s 2": {
#        "barcodeSampleInfo": {
#            "MuSeek_002": {
#                "description": "",
#                "externalId": ""
#            }
#        },
#        "barcodes": [
#            "MuSeek_002"
#        ]
#    }
#
#},

#20140325 example for DNA + RNA:
#barcodedSamples": {
#
#    "s 1": {
#        "barcodeSampleInfo": {
#            "IonXpress_001": {
#                "controlSequenceType": "",
#                "description": "test desc",
#                "externalId": "ext 1",
#                "hotSpotRegionBedFile": "/results/uploads/BED/8/polio/unmerged/detail/polio_hotspot.bed",
#                "nucleotideType": "DNA",
#                "reference": "polio",
#                "targetRegionBedFile": "/results/uploads/BED/7/polio/unmerged/detail/polio.bed"
#            },
#            "IonXpress_002": {
#                "controlSequenceType": "",
#                "description": "test desc",
#                "externalId": "ext 1",
#                "hotSpotRegionBedFile": "",
#                "nucleotideType": "RNA",
#                "reference": "",
#                "targetRegionBedFile": ""
#            }
#        },
#        "barcodes": [
#            "IonXpress_001",
#            "IonXpress_002"
#        ]
#    }
#
#},


    sampleName = ""
    sampleId = ""
    sampleNucleotideType = ""
    
    runType = selectedTemplate.runType         
    ##applicationGroup = selectedTemplate.applicationGroup.name if selectedTemplate.applicationGroup else ""
    
    try:
        for barcode in barcodes:
            #reset sample info per barcode
            sampleName = ""
            sampleId = ""
            sampleNucleotideType = ""
            sampleRnaReference = ""
            
            barcodeName = barcode["id_str"]
            key = barcodeName + plan_csv_writer.COLUMN_BC_SAMPLE_KEY

            sampleInfo = input.get(key, "") 
            if sampleInfo:
                for sampleToken in sampleInfo.split(";"):
                    sampleToken = sampleToken.strip()
                    
                    if sampleToken:
                        if sampleToken.startswith("ID:"):
                            sampleId = sampleToken[3:].strip()
                        elif sampleToken.startswith("TYPE:"):
                            sampleNucleotideType = sampleToken[5:].strip().upper()
                        elif sampleToken.startswith("RNA REF:"):
                            sampleRnaReference = sampleToken[8:].strip()
                        else:
                            sampleName = sampleToken

                if (sampleId or sampleNucleotideType) and (not sampleName):
                    errorMsgDict[key] = '  '.join(["Sample name is required "])
                    
                if sampleName:
                    errors, rna_ref_short_name, sample_nucleotideType = validate_barcoded_sample_info(sampleName, sampleId, sampleNucleotideType, runType, sampleRnaReference)

                    planObj.get_nucleotideTypeList().append(sample_nucleotideType if sample_nucleotideType else "")
                                            
                    if errors:
                        ##logger.debug("plan_csv_validator.. ERRORS validate_barcode_sample_info. key=%s; errors=%s" %(key, errors))
                        errorMsgDict[key] = '  '.join(errors)
                        
                    else:
                        planObj.get_sampleList().append(sampleName)
                        planObj.get_sampleIdList().append(sampleId if sampleId else "")

                        sampleReference = planObj.get_easObj().reference
                        sampleHotSpotBedFile = planObj.get_easObj().hotSpotRegionBedFile
                        sampleTargetRegionBedFile = planObj.get_easObj().targetRegionBedFile
                        
                        if runType == "AMPS_DNA_RNA" and sampleNucleotideType.upper() == "RNA":
                            sampleReference = rna_ref_short_name
                            sampleHotSpotBedFile = ""
                            sampleTargetRegionBedFile = ""
                            
                        ##logger.debug("plan_csv_validator._validate_barcodedSamples() sampleNucleotideType=%s; sampleRnaReference=%s; rna_ref_short_name=%s; sampleReference=%s" %(sampleNucleotideType, sampleRnaReference, rna_ref_short_name, sampleReference))

                        barcodedSampleData = barcodedSampleJson.get(sampleName, {})

                        if barcodedSampleData:
                            barcodeList = barcodedSampleData.get("barcodes", [])
                            
                            if barcodeList:
                                barcodeList.append(barcodeName)
                                barcodeDict = {
                                               "barcodes" : barcodeList
                                               }                                    
                            else:
                                barcodeDict = {
                                               "barcodes" : [barcodeName]
                                               }
    
                            barcodeSampleInfoDict = barcodedSampleData.get("barcodeSampleInfo", {})
                            barcodeSampleInfoDict[barcodeName] = {
                                                    'externalId'   : sampleId,
                                                    'description'  : "",
                                                    'nucleotideType' : sampleNucleotideType.upper(),
                                                    'controlSequenceType' : '',
                                                    'reference' : sampleReference,
                                                    'hotSpotRegionBedFile' : sampleHotSpotBedFile,
                                                    'targetRegionBedFile' : sampleTargetRegionBedFile
    
                            }
                            barcodedSampleJson[sampleName.strip()] = {
                            'barcodeSampleInfo' : barcodeSampleInfoDict,
                            'barcodes'          : barcodeDict["barcodes"]
                            }
                                   
                        else:              
                            barcodeDict = {
                                           "barcodes" : [barcodeName]
                                           }
    
                            barcodedSampleJson[sampleName.strip()] = {
                            'barcodeSampleInfo' : { 
                                    barcodeName: {
                                                    'externalId'   : sampleId,
                                                    'description'  : "",
                                                    'nucleotideType' : sampleNucleotideType,
                                                    'controlSequenceType' : '',
                                                    'reference' : sampleReference,
                                                    'hotSpotRegionBedFile' : sampleHotSpotBedFile,
                                                    'targetRegionBedFile' : sampleTargetRegionBedFile
    
                                    }
                                },
                            'barcodes'          : barcodeDict["barcodes"]
                            }

    except:
        logger.exception(format_exc())  
        errorMsg = "Internal error during barcoded sample processing"
   
    if errorMsgDict:
        return simplejson.dumps(errorMsgDict), barcodedSampleJson
    
    if not barcodedSampleJson:
        errorMsg = "Required column is empty. At least one barcoded sample is required. "
    else:
        planObj.get_easObj().barcodedSamples = barcodedSampleJson

    return errorMsg, barcodedSampleJson
                

def _validate_IR_workflow_v1_0(input, selectedTemplate, planObj, uploaderList):
    errorMsg = None
                  
    if not input:
        return errorMsg

    irUploader = None
    for uploader in uploaderList:
        for key, value in uploader.items():
            
            if key == "name" and value == "IonReporterUploader_V1_0":  
                irUploader = uploader
    
    for key, value in irUploader.items():            
        if key == "userInput":
            userInputDict = value[0]
            userInputDict["Workflow"] = input.strip()
                
    #logger.info("EXIT _validate_IR_workflow_v1_0() workflow_v1_0 uploaderList=%s" %(uploaderList))        
    return errorMsg

#def _validate_IR_workflow_v1_x(input, selectedTemplate, planObj, uploaderJson):
 #   return None    

