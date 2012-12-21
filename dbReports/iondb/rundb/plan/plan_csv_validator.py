# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.models import Project, PlannedExperiment
from django.contrib.auth.models import User

from iondb.rundb.models import PlannedExperiment, RunType, ApplProduct, \
    ReferenceGenome, Content, KitInfo, VariantFrequencies, dnaBarcode, \
    LibraryKey, ThreePrimeadapter, Chip, QCType, Project, Plugin, \
    PlannedExperimentQC

from iondb.rundb.plan.views_helper import dict_bed_hotspot

from traceback import format_exc

import plan_csv_writer

import copy
import re

import logging
logger = logging.getLogger(__name__)


def validate_csv_plan(csvPlanDict):
    """ validate csv contents and convert user input to raw data to prepare for plan persistence
    returns: a collection of error messages if errors found, a dictionary of raw data values
    """
    
    logger.debug("ENTER plan_csv_validator.validate_csv_plan() csvPlanDict=%s; " %(csvPlanDict))
    failed = []
    rawPlanDict = {}
    planObj = None
    
    isToSkipRow = False
    
    #skip this row if no values found (will not prohibit the rest of the files from upload
    skipIfEmptyList = [plan_csv_writer.COLUMN_TEMPLATE_NAME]
    
    for skipIfEmpty in skipIfEmptyList:
        if skipIfEmpty in csvPlanDict:
            if not csvPlanDict[skipIfEmpty]:
                #required column is empty
                isToSkipRow = True
            
    if isToSkipRow:
        return failed, planObj, rawPlanDict, isToSkipRow
    
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
        if errorMsg:  
            failed.append((plan_csv_writer.COLUMN_TEMPLATE_NAME, errorMsg))  
            return failed, planObj, rawPlanDict, isToSkipRow
        
        planObj = _init_plan(selectedTemplate)
    else:
        return failed, planObj, rawPlanDict, isToSkipRow
        
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

    errorMsg = _validate_variant_freq(csvPlanDict.get(plan_csv_writer.COLUMN_VARIANT_FREQ), selectedTemplate, planObj)
    if errorMsg:
        failed.append((plan_csv_writer.COLUMN_VARIANT_FREQ, errorMsg))                        
        
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

        selectedPlugins = {
                           'planplugins' : plugins,
                           'planuploaders' : uploaders
                            }

        rawPlanDict["selectedPlugins"] = selectedPlugins

        planObj.selectedPlugins = selectedPlugins
    
    except:
        logger.exception(format_exc())
        errorMsg = "Internal error while processing selected plugins info. " 
        failed.append(("selectedPlugins", errorMsg))        

        
    #todo: validate barcoded plan
    if selectedTemplate.barcodeId:
        errorMsg, barcodedSampleJson = _validate_barcodedSamples(csvPlanDict, selectedTemplate, planObj)
        if errorMsg:
            failed.append(("barcodedSample", errorMsg))
                            
    else:
        errorMsg = _validate_sample(csvPlanDict.get(plan_csv_writer.COLUMN_SAMPLE), selectedTemplate, planObj)

        if errorMsg:
            failed.append((plan_csv_writer.COLUMN_SAMPLE, errorMsg))                        

    #if uploaderJson:
    #   errorMsg = _validate_IR_workflow_v1_x(csvPlanDict.get(plan_csv_writer.COLUMN_IR_V1_X_WORKFLOW), selectedTemplate, planObj, uploaderJson)
    #   if errorMsg:
    #       failed.append((plan_csv_writer.COLUMN_IR_V1_X_WORKFLOW, errorMsg))                        

    
    logger.debug("EXIT plan_csv_validator.validate_csv_plan() rawPlanDict=%s; " %(rawPlanDict))
    
    return failed, planObj, rawPlanDict, isToSkipRow


def _get_template(templateName):
    templates = PlannedExperiment.objects.filter(planDisplayedName = templateName.strip(), isReusable=True).order_by("-date")
    
    if templates:
        return templates[0], None
    else:
        logger.debug("plan_csv_validator._get_template() NO template found. ")       
        return None, "Template name: " + templateName + " cannot be found to create plans from"


def _init_plan(selectedTemplate):
    if not selectedTemplate:
        return None
    
    planObj = copy.copy(selectedTemplate)
    planObj.pk = None
    planObj.planGUID = None
    planObj.planShortID = None
    planObj.isReusable = False
    planObj.isSystem = False
    planObj.isSystemDefault = False
    planObj.expName = ""
    planObj.planName = ""
    planObj.planExecuted = False
    
    #PDD
    #planObj.planStatus = "planned"

    return planObj


def isValidChars(value, validChars=r'^[a-zA-Z0-9-_\.\s\,]+$'):
    ''' Determines if value is valid: letters, numbers, spaces, dashes, underscores only '''
    return bool(re.compile(validChars).match(value))


def _validate_sample_prep_kit(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        try:
            selectedKit = KitInfo.objects.filter(kitType = "SamplePrepKit", description = input.strip())[0]
            planObj.samplePrepKitName = selectedKit.name
        except:
            errorMsg = input + " not found."
    else:
        planObj.samplePrepKtName = ""
        
    return errorMsg

def _validate_lib_kit(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        try:
            selectedKit = KitInfo.objects.filter(kitType = "LibraryKit", description = input.strip())[0]
            planObj.librarykitname = selectedKit.name
        except:
            errorMsg = input + " not found."
    else:
        planObj.librarykitname = ""
          
    return errorMsg
    

def _validate_template_kit(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        try:
            selectedKit = KitInfo.objects.filter(kitType = "TemplatingKit", description = input.strip())[0]
            planObj.templatingKitName = selectedKit.name
        except:
            errorMsg = input + " not found."
    else:
        planObj.templatingKitName = ""
          
    return errorMsg

def _validate_control_seq_kit(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        try:
            selectedKit = KitInfo.objects.filter(kitType = "ControlSequenceKit", description = input.strip())[0]
            planObj.controlSequencekitname = selectedKit.name
        except:
            errorMsg = input + " not found."
    else:
        planObj.controlSequencekitname = ""

    return errorMsg
    
def _validate_seq_kit(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        try:
            selectedKit = KitInfo.objects.filter(kitType = "SequencingKit", description = input.strip())[0]
            planObj.sequencekitname = selectedKit.name
        except:
            errorMsg = input + " not found."
    else:
        planObj.sequencekitname = ""
        
    return errorMsg

def _validate_chip_type(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        try:
            selectedChip = Chip.objects.filter(description = input.strip())[0]
            planObj.chipType = selectedChip.name
        except:
            logger.exception(format_exc())
            errorMsg = input + " not found."
    else:
        planObj.chipType = ""
        
    return errorMsg

def _validate_flows(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        try:
            flowCount = int(input)
            if (flowCount <= 0 or flowCount > 1000):
                errorMsg = flowCount + " should be a positive whole number within range [1, 1000)."
            else:
                planObj.flows = flowCount
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
            planObj.library = selectedRef.short_name 
        except:
            try:
                selectedRef= ReferenceGenome.objects.filter(short_name = input.strip())[0]
                planObj.library = selectedRef.short_name 
            except:
                logger.exception(format_exc())      
                errorMsg = input + " not found."
    else:
        planObj.library = ""
         
    return errorMsg

                                            
def _validate_target_bed(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        bedFileDict = dict_bed_hotspot()
        value = input.strip()
        
        if value in bedFileDict.get("bedFilePaths") or value in bedFileDict.get("bedFileFullPaths"):        
            for bedFile in bedFileDict.get("bedFiles"):
                if value == bedFile.file or value == bedFile.path:
                    planObj.bedfile = bedFile.file
        else:
            logger.exception(format_exc())
            errorMsg = input + " not found."
    else:
        planObj.bedfile = ""
        
    return errorMsg
    
def _validate_hotspot_bed(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        bedFileDict = dict_bed_hotspot()
        value = input.strip()
        
        if value in bedFileDict.get("hotspotPaths") or value in bedFileDict.get("hotspotFullPaths"):        
            for bedFile in bedFileDict.get("hotspotFiles"):
                if value == bedFile.file or value == bedFile.path:
                    planObj.regionfile = bedFile.file
        else: 
            logger.exception(format_exc())            
            errorMsg = input + " not found. "
    else:
        planObj.regionfile = ""
    return errorMsg

def _validate_plugins(input, selectedTemplate, planObj):
    errorMsg = ""
    plugins = []

    if input:
        for plugin in input.split(";"):
            if plugin:
                try:
                    selectedPlugin = Plugin.objects.filter(name = plugin.strip(), selected = True, active = True)[0]
                    
                    pluginDict = {
                                  "id" : selectedPlugin.id,
                                  "name" : selectedPlugin.name,
                                  "version" : selectedPlugin.version
                                  }
                    
                    plugins.append(pluginDict)
                except:
                    logger.exception(format_exc())            
                    errorMsg += plugin + " not found. "
    else:
        planObj.selectedPlugins = ""
        
    return errorMsg, plugins
   
def _validate_variant_freq(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        try:
            variantFreq = VariantFrequencies.objects.filter(name = input.strip())[0]
            planObj.variantfrequency = variantFreq.name
        except:
            logger.exception(format_exc())            
            errorMsg = input + " not found."
    else:
        planObj.variantfrequency = ""
        
    return errorMsg                  
    
def _validate_projects(input, selectedTemplate, planObj):
    errorMsg = None
    projects = ''
    
    if input:
        for project in input.split(";"):
            if not isValidChars(project.strip()):
                errorMsg = "Project should contain only numbers, letters, spaces, and the following: . - _"
            else:
                projects += project.strip()
                projects += ','

    return errorMsg, projects

    
def _validate_export(input, selectedTemplate, planObj):
    errorMsg = ""
    
    plugins = []
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
                                  "version" : selectedPlugin.version
                                  }

                    workflowDict = {
                                    'Workflow' : ""
                                    }
                    
                    userInputList = []
                    userInputList.append(workflowDict)
                    pluginDict["userInput"] = userInputList
        
                    plugins.append(pluginDict)
                except:
                    logger.exception(format_exc())            
                    errorMsg += plugin + " not found. "
    else:
        planObj.selectedPlugins = ""

    return errorMsg, plugins, has_ir_v1_0

def _validate_plan_name(input, selectedTemplate, planObj):
    errorMsg = None
    if not isValidChars(input.strip()):
        errorMsg = "Plan name should contain only numbers, letters, spaces, and the following: . - _"
    else:
        planObj.planDisplayedName = input.strip()
        planObj.planName = input.strip().replace(' ', '_')
    return errorMsg
    
def _validate_notes(input, selectedTemplate, planObj):
    errorMsg = None
    if input:
        if not isValidChars(input):
            errorMsg = "Notes should contain only numbers, letters, spaces, and the following: . - _"
        else:
            planObj.notes = input.strip()
    else:
        planObj.notes = ""
        
    return errorMsg
    
def _validate_sample(input, selectedTemplate, planObj):
    errorMsg = None
    if not input:
        errorMsg = "Required column is empty"
    else:
        if not isValidChars(input):
            errorMsg = "Sample should contain only numbers, letters, spaces, and the following: . - _"
        else:
            planObj.sampleDisplayedName = input.strip()
            planObj.sample = input.strip().replace(' ', '_')
                    
    return errorMsg


def _validate_barcodedSamples(input, selectedTemplate, planObj):
    errorMsg = None
    barcodedSampleJson = {}

    #{"bc10_noPE_sample3":{"26":"IonXpress_010"},"bc04_noPE_sample1":{"20":"IonXpress_004"},"bc08_noPE_sample2":{"24":"IonXpress_008"}}
    #20121122-new JSON format
    #{"bcSample1":{"barcodes":["IonXpress_001","IonXpress_002"]},"bcSample2":{"barcodes":["IonXpress_003"]}}
    
    barcodes = list(dnaBarcode.objects.filter(name = selectedTemplate.barcodeId).values('id', 'id_str').order_by('id_str'))
    
    if len(barcodes) == 0:
        errorMsg = "Barcode "+ selectedTemplate.bardcodeId + " cannot be found. " 
        return errorMsg, barcodedSampleJson
    
    errorMsgDict = {}
    try:
        for barcode in barcodes:
            key = barcode["id_str"] + plan_csv_writer.COLUMN_BC_SAMPLE_KEY
            sample = input.get(key, "")        

            if sample:           
                if not isValidChars(sample):
                    errorMsgDict[key] = "Sample should contain only numbers, letters, spaces, and the following: . - _"
                else:
                    barcodedSample = barcodedSampleJson.get(sample.strip(), {})
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
    
    if not barcodedSampleJson:
        errorMsg = "Required column is empty. At least one barcoded sample is required. "
    else:
        planObj.barcodedSamples = barcodedSampleJson
       
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

