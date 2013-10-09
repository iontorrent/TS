# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.contrib.auth.models import User

from iondb.rundb.models import PlannedExperiment, RunType, ApplProduct, \
    ReferenceGenome, Content, KitInfo, VariantFrequencies, dnaBarcode, \
    LibraryKey, ThreePrimeadapter, Chip, QCType, Project, Plugin, \
    PlannedExperimentQC

from iondb.rundb.plan.views_helper import dict_bed_hotspot, is_valid_chars, is_invalid_leading_chars, is_valid_length

from traceback import format_exc

import plan_csv_writer
import iondb.rundb.plan.views

import copy

import logging
logger = logging.getLogger(__name__)

import simplejson


class MyPlan:
    def __init__(self, selectedTemplate, selectedExperiment, selectedEAS):        
        if not selectedTemplate:
            self.planObj = None
            self.expObj = None
            self.easObj = None
            self.sampleList = []
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
        
       
    def get_planObj(self):
        return self.planObj
    
    def get_expObj(self):
        return self.expObj
    
    def get_easObj(self):
        return self.easObj
    
    def get_sampleList(self):
        return self.sampleList
    

def validate_csv_plan(csvPlanDict):
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
            selectedExperiment = selectedTemplate.experiment
            selectedEAS = selectedTemplate.experiment.get_EAS()
        
        if errorMsg:
            failed.append((plan_csv_writer.COLUMN_TEMPLATE_NAME, errorMsg))  
            return failed, planDict, rawPlanDict, isToSkipRow

        planObj = _init_plan(selectedTemplate, selectedExperiment, selectedEAS)

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
    
    errorMsg = _validate_chip_type(csvPlanDict.get(plan_csv_writer.COLUMN_CHIP_TYPE), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_CHIP_TYPE, errorMsg))
    
    errorMsg = _validate_flows(csvPlanDict.get(plan_csv_writer.COLUMN_FLOW_COUNT), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_FLOW_COUNT, errorMsg))
    
    errorMsg, beadLoadQCValue = _validate_qc_pct(csvPlanDict.get(plan_csv_writer.COLUMN_BEAD_LOAD_PCT), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_BEAD_LOAD_PCT, errorMsg))
    
    rawPlanDict["Bead Loading (%)"] = beadLoadQCValue
    
    errorMsg, keySignalQCValue = _validate_qc_pct(csvPlanDict.get(plan_csv_writer.COLUMN_KEY_SIGNAL_PCT), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_KEY_SIGNAL_PCT, errorMsg))
    
    rawPlanDict["Key Signal (1-100)"] = keySignalQCValue
    
    errorMsg, usableSeqQCValue = _validate_qc_pct(csvPlanDict.get(plan_csv_writer.COLUMN_USABLE_SEQ_PCT), selectedTemplate, planObj)
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
    
    errorMsg,plugins = _validate_plugins(csvPlanDict.get(plan_csv_writer.COLUMN_PLUGINS), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_PLUGINS, errorMsg))
        
    errorMsg, projects = _validate_projects(csvPlanDict.get(plan_csv_writer.COLUMN_PROJECTS), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_PROJECTS, errorMsg))
    
    rawPlanDict["newProjects"] = projects
        
    errorMsg, uploaders, has_ir_v1_0 = _validate_export(csvPlanDict.get(plan_csv_writer.COLUMN_EXPORT), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_EXPORT, errorMsg))                        

    errorMsg = _validate_plan_name(csvPlanDict.get(plan_csv_writer.COLUMN_PLAN_NAME), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_PLAN_NAME, errorMsg))
    
    errorMsg = _validate_notes(csvPlanDict.get(plan_csv_writer.COLUMN_NOTES), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_NOTES, errorMsg))                        

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

    barcodeKitName = selectedEAS.barcodeKitName
    if barcodeKitName:
        errorMsg, barcodedSampleJson = _validate_barcodedSamples(csvPlanDict, selectedTemplate, barcodeKitName, planObj)

        if errorMsg:
            failed.append(("barcodedSample", errorMsg))
        else:            
            planObj.get_sampleList().extend(barcodedSampleJson.keys())
    else:    
        errorMsg, sampleDisplayedName = _validate_sample(csvPlanDict.get(plan_csv_writer.COLUMN_SAMPLE), selectedTemplate, planObj)
        if errorMsg:
            failed.append((plan_csv_writer.COLUMN_SAMPLE, errorMsg))                        
        else:
            if sampleDisplayedName:
                planObj.get_sampleList().append(sampleDisplayedName)

    #if uploaderJson:
    #   errorMsg = _validate_IR_workflow_v1_x(csvPlanDict.get(plan_csv_writer.COLUMN_IR_V1_X_WORKFLOW), selectedTemplate, planObj, uploaderJson)
    #   if errorMsg:
    #       failed.append((plan_csv_writer.COLUMN_IR_V1_X_WORKFLOW, errorMsg))                        
    
    logger.debug("EXIT plan_csv_validator.validate_csv_plan() rawPlanDict=%s; " %(rawPlanDict))
    
    planDict = {
                "plan" : planObj.get_planObj(),
                "exp" : planObj.get_expObj(),
                "eas" : planObj.get_easObj(),
                "samples" : planObj.get_sampleList()
                }
    
    return failed, planDict, rawPlanDict, isToSkipRow


def _get_template(templateName):    
    templates = PlannedExperiment.objects.filter(planDisplayedName = templateName.strip(), isReusable=True).order_by("-date")
    
    if templates:
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


def _init_plan(selectedTemplate, selectedExperiment, selectedEAS):
    return MyPlan(selectedTemplate, selectedExperiment, selectedEAS)


def _validate_sample_prep_kit(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        try:
            selectedKit = KitInfo.objects.filter(kitType = "SamplePrepKit", description = input.strip())[0]
            planObj.get_planObj().samplePrepKitName = selectedKit.name
        except:
            errorMsg = input + " not found."
    else:
        planObj.get_planObj().samplePrepKtName = ""
        
    return errorMsg

def _validate_lib_kit(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        try:
            selectedKit = KitInfo.objects.filter(kitType = "LibraryKit", description = input.strip())[0]
            planObj.get_easObj().libraryKitName = selectedKit.name
        except:
            errorMsg = input + " not found."
    else:
        planObj.get_easObj().libraryKitName = ""
          
    return errorMsg
    

def _validate_template_kit(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        try:
            selectedKit = KitInfo.objects.filter(kitType = "TemplatingKit", description = input.strip())[0]
            planObj.get_planObj().templatingKitName = selectedKit.name
        except:
            errorMsg = input + " not found."
    else:
        planObj.get_planObj().templatingKitName = ""
          
    return errorMsg

def _validate_control_seq_kit(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        try:
            selectedKit = KitInfo.objects.filter(kitType = "ControlSequenceKit", description = input.strip())[0]
            planObj.get_planObj().controlSequencekitname = selectedKit.name
        except:
            errorMsg = input + " not found."
    else:
        planObj.get_planObj().controlSequencekitname = ""

    return errorMsg
    
def _validate_seq_kit(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        try:
            selectedKit = KitInfo.objects.filter(kitType = "SequencingKit", description = input.strip())[0]
            planObj.get_expObj().sequencekitname = selectedKit.name
        except:
            errorMsg = input + " not found."
    else:
        planObj.get_expObj().sequencekitname = ""
        
    return errorMsg

def _validate_chip_type(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        try:
            selectedChip = Chip.objects.filter(description = input.strip())[0]
            planObj.get_expObj().chipType = selectedChip.name
        except:
            logger.exception(format_exc())
            errorMsg = input + " not found."
    else:
        planObj.get_expObj().chipType = ""
        
    return errorMsg

def _validate_flows(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        try:
            flowCount = int(input)
            if (flowCount <= 0 or flowCount > 2000):
                errorMsg = flowCount + " should be a positive whole number within range [1, 2000)."
            else:
                planObj.get_expObj().flows = flowCount
        except:
            logger.exception(format_exc())
            errorMsg = input + " should be a positive whole number."
        
    return errorMsg
    
def _validate_qc_pct(input, selectedTemplate, planObj):
    errorMsg = None
    qcValue = None
    if input:
        try:
            pct = int(input)
            if (pct <= 0 or pct > 100):
                errorMsg = pct + " should be a positive whole number within range [1, 100)."
            else:
                qcValue = pct
        except:
            logger.exception(format_exc())
            errorMsg = input + " should be a positive whole number within range [1, 100)."
    else:
        qcValue = 1
    return errorMsg, qcValue

    
def _validate_ref(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        try:
            selectedRef= ReferenceGenome.objects.filter(name = input.strip())[0]
            planObj.get_easObj().reference = selectedRef.short_name 
        except:
            try:
                selectedRef= ReferenceGenome.objects.filter(short_name = input.strip())[0]
                planObj.get_easObj().reference = selectedRef.short_name 
            except:
                logger.exception(format_exc())      
                errorMsg = input + " not found."
    else:
        planObj.get_easObj().reference = ""
         
    return errorMsg

                                            
def _validate_target_bed(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        bedFileDict = dict_bed_hotspot()
        value = input.strip()
        
        if value in bedFileDict.get("bedFilePaths") or value in bedFileDict.get("bedFileFullPaths"):        
            for bedFile in bedFileDict.get("bedFiles"):
                if value == bedFile.file or value == bedFile.path:
                    planObj.get_easObj().targetRegionBedFile = bedFile.file
        else:
            logger.exception(format_exc())
            errorMsg = input + " not found."
    else:
        planObj.get_easObj().targetRegionBedFile = ""
        
    return errorMsg
    
def _validate_hotspot_bed(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        bedFileDict = dict_bed_hotspot()
        value = input.strip()
        
        if value in bedFileDict.get("hotspotPaths") or value in bedFileDict.get("hotspotFullPaths"):        
            for bedFile in bedFileDict.get("hotspotFiles"):
                if value == bedFile.file or value == bedFile.path:
                    planObj.get_easObj().hotSpotRegionBedFile = bedFile.file
        else: 
            logger.exception(format_exc())            
            errorMsg = input + " not found. "
    else:
        planObj.get_easObj().hotSpotRegionBedFile = ""
    return errorMsg

def _validate_plugins(input, selectedTemplate, planObj):
    errorMsg = ""
    plugins = {}

    if input:
        for plugin in input.split(";"):
            if plugin:
                try:
                    selectedPlugin = Plugin.objects.filter(name = plugin.strip(), selected = True, active = True)[0]

                    pluginUserInput = {}
                    template_selectedPlugins = selectedTemplate.get_selectedPlugins()

                    if plugin.strip() in template_selectedPlugins:
                        #logger.info("_validate_plugins() FOUND plugin in selectedTemplate....=%s" %(template_selectedPlugins[plugin.strip()]))
                        pluginUserInput = template_selectedPlugins[plugin.strip()]["userInput"]

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
    errorMsg = None
    projects = ''
    
    if input:
        for project in input.split(";"):
            if not is_valid_chars(project.strip()):
                errorMsg = "Project should contain only numbers, letters, spaces, and the following: . - _"
            else:
                value = input.strip()
                if value:
                    if not is_valid_length(value, iondb.rundb.plan.views.MAX_LENGTH_PROJECT_NAME):
                        errorMsg = "Project name length should be " + str(iondb.rundb.plan.views.MAX_LENGTH_PROJECT_NAME) + " characters maximum."
                    else:                                
                        projects += project.strip()
                        projects += ','

    return errorMsg, projects

    
def _validate_export(input, selectedTemplate, planObj):
    errorMsg = ""
    
    plugins = {}
    has_ir_v1_0 = False
    
    if input:
        for plugin in input.split(";"):
            if plugin:
                try:
                    #20121212-TODO: can we query and filter by EXPORT feature fast?
                    selectedPlugin = Plugin.objects.filter(name = plugin.strip(), selected = True, active = True)[0]

                    if selectedPlugin.name == "IonReporterUploader_V1_0":
                        has_ir_v1_0 = True

                    pluginDict = {
                                  "id" : selectedPlugin.id,
                                  "name" : selectedPlugin.name,
                                  "version" : selectedPlugin.version,
                                  "features": ['export']
                                  }

                    workflowDict = {
                                    'Workflow' : ""
                                    }
                    
                    userInputList = []
                    userInputList.append(workflowDict)
                    pluginDict["userInput"] = userInputList
        
                    plugins[selectedPlugin.name] = pluginDict
                except:
                    logger.exception(format_exc())            
                    errorMsg += plugin + " not found. "
    else:
        planObj.get_easObj().selectedPlugins = ""

    return errorMsg, plugins, has_ir_v1_0

def _validate_plan_name(input, selectedTemplate, planObj):
    errorMsg = None
    if not is_valid_chars(input.strip()):
        errorMsg = "Plan name should contain only numbers, letters, spaces, and the following: . - _"
    else:
        value = input.strip()
        if value:
            if not is_valid_length(value, iondb.rundb.plan.views.MAX_LENGTH_PLAN_NAME):
                errorMsg = "Plan name" + iondb.rundb.plan.views.ERROR_MSG_INVALID_LENGTH  %(str(iondb.rundb.plan.views.MAX_LENGTH_PLAN_NAME))
            else:
                planObj.get_planObj().planDisplayedName = value
                planObj.get_planObj().planName = value.replace(' ', '_')
    return errorMsg
    
def _validate_notes(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        if not is_valid_chars(input):
            errorMsg = "Notes"  + iondb.rundb.plan.views.ERROR_MSG_INVALID_CHARS
        else:
            value = input.strip()
            if value:
                if not is_valid_length(value, iondb.rundb.plan.views.MAX_LENGTH_NOTES):
                    errorMsg = "Notes" + iondb.rundb.plan.views.ERROR_MSG_INVALID_LENGTH  %(str(iondb.rundb.plan.views.MAX_LENGTH_NOTES))
                else:                
                    planObj.get_expObj().notes = value
    else:
        planObj.get_expObj().notes = ""
        
    return errorMsg


def _validate_sample(input, selectedTemplate, planObj):
    errorMsg = None
    sampleDisplayedName = ""
    
    if not input:
        errorMsg = "Required column is empty"
    else:
        if not is_valid_chars(input):
            errorMsg = "Sample name" + iondb.rundb.plan.views.ERROR_MSG_INVALID_CHARS        
        elif is_invalid_leading_chars(input):
            errorMsg = "Sample name" + iondb.rundb.plan.views.ERROR_MSG_INVALID_LEADING_CHARS                       
        else:
            value = input.strip()
            if value:
                if not is_valid_length(value, iondb.rundb.plan.views.MAX_LENGTH_SAMPLE_NAME):
                    errorMsg = "Sample name" +  iondb.rundb.plan.views.ERROR_MSG_INVALID_LENGTH  %(str(iondb.rundb.plan.views.MAX_LENGTH_SAMPLE_NAME))
                else:                            
                    sampleDisplayedName = value
                    sample = value.replace(' ', '_')
                    
    return errorMsg, sampleDisplayedName


def _validate_barcodedSamples(input, selectedTemplate, barcodeKitName, planObj):
    errorMsg = None
    barcodedSampleJson = {}
        
    #{"bc10_noPE_sample3":{"26":"IonXpress_010"},"bc04_noPE_sample1":{"20":"IonXpress_004"},"bc08_noPE_sample2":{"24":"IonXpress_008"}}
    #20121122-new JSON format
    #{"bcSample1":{"barcodes":["IonXpress_001","IonXpress_002"]},"bcSample2":{"barcodes":["IonXpress_003"]}}
        
    barcodes = list(dnaBarcode.objects.filter(name = barcodeKitName).values('id', 'id_str').order_by('id_str'))
    
    if len(barcodes) == 0:
        errorMsg = "Barcode "+ barcodeKitName + " cannot be found. " 
        return errorMsg, barcodedSampleJson
    
    errorMsgDict = {}
    try:
        for barcode in barcodes:            
            key = barcode["id_str"] + plan_csv_writer.COLUMN_BC_SAMPLE_KEY
            sample = input.get(key, "")        

            if sample:           
                if not is_valid_chars(sample):
                    errorMsgDict[key] = "Sample name" + iondb.rundb.plan.views.ERROR_MSG_INVALID_CHARS
                elif is_invalid_leading_chars(sample):
                    errorMsgDict[key] = "Sample name" + iondb.rundb.plan.views.ERROR_MSG_INVALID_LEADING_CHARS        
                else:
                    value = sample.strip()

                    if value:
                        if not is_valid_length(value, iondb.rundb.plan.views.MAX_LENGTH_SAMPLE_NAME):
                            errorMsgDict[key] = "Sample name" +  iondb.rundb.plan.views.ERROR_MSG_INVALID_LENGTH  %(str(iondb.rundb.plan.views.MAX_LENGTH_SAMPLE_NAME))                           
                        else:     
                            barcodedSample = barcodedSampleJson.get(value, {})
                    
                            if barcodedSample:
                                barcodeList = barcodedSample.get("barcodes", [])
                                if barcodeList:
                                    barcodeList.append(barcode["id_str"])
                                else:
                                    barcodeDict = {
                                                   "barcodes" : [barcode["id_str"]]
                                                   }

                                barcodedSampleJson[sample.strip()] = barcodeDict                                          
                            else:              
                                barcodeDict = {
                                               "barcodes" : [barcode["id_str"]]
                                               }

                                barcodedSampleJson[sample.strip()] = barcodeDict                                
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

