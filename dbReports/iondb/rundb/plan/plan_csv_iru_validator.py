# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved

from iondb.rundb.plan.views_helper import get_ir_set_id

from iondb.rundb.plan.views_helper import get_default_or_first_IR_account_by_userName
from plan_csv_writer import PlanCSVcolumns, get_irSettings
from iondb.rundb.models import Plugin
from traceback import format_exc
import requests
import json
import uuid

import logging
logger = logging.getLogger(__name__)

"""
 This is main helper function which calls IRU api and process the response
    - Validate the iru configuration settings in the CSV plan upload.
    - construct the UserInputInfo for plan->selectedPlugins
"""

def get_userInfoDict(row,workflowObj, rowCount, setid_suffix):
    userInput_setid = row.get(PlanCSVcolumns.COLUMN_SAMPLE_IR_SET_ID).strip() + '__' + setid_suffix

    userInputInfoDict = {
        "ApplicationType": workflowObj.get("ApplicationType"),
        "sample": row.get(PlanCSVcolumns.COLUMN_SAMPLE_NAME, '').strip(),
        "Relation": workflowObj.get("RelationshipType"),
        "RelationRole": row.get(PlanCSVcolumns.COLUMN_SAMPLE_IR_RELATION),
        "Gender": row.get(PlanCSVcolumns.COLUMN_SAMPLE_IR_GENDER),
        "setid": userInput_setid,
        "cancerType": row.get(PlanCSVcolumns.COLUMN_SAMPLE_CANCER_TYPE),
        "cellularityPct": row.get(PlanCSVcolumns.COLUMN_SAMPLE_CELLULARITY),
        "biopsyDays": row.get(PlanCSVcolumns.COLUMN_SAMPLE_BIOSPY_DAYS),
        "barcodeId": row.get(PlanCSVcolumns.COLUMN_BARCODE),
        "coupleID": row.get(PlanCSVcolumns.COLUMN_SAMPLE_COUPLE_ID),
        "embryoID": row.get(PlanCSVcolumns.COLUMN_SAMPLE_EMBRYO_ID),
        "Workflow": row.get(PlanCSVcolumns.COLUMN_SAMPLE_IR_WORKFLOW),
        "NucleotideType": row.get(PlanCSVcolumns.COLUMN_NUCLEOTIDE_TYPE),
        "controlType": row.get(PlanCSVcolumns.COLUMN_SAMPLE_CONTROLTYPE),
        "tag_isFactoryProvidedWorkflow" : workflowObj.get("tag_isFactoryProvidedWorkflow"),
        "row": str(rowCount)
    }

    return userInputInfoDict

def getWorkflowObj(workflow, USERINPUT):
    validWorkflowObj = None
    allWorkflowsObj = USERINPUT["workflows"]

    for obj in allWorkflowsObj:
        if workflow in obj["Workflow"]:
            validWorkflowObj = obj
            break

    return validWorkflowObj

def irWorkflowNotValid(workflow, USERINPUT):
    notValid = False
    workflowObj = getWorkflowObj(workflow, USERINPUT)

    if not workflowObj:
        notValid = True
    return notValid, workflowObj

def get_samples_content_single_csv(csvPlanDict):
    samples_contents = []
    single_csv_samplesDict = {
        PlanCSVcolumns.COLUMN_SAMPLE : csvPlanDict.get(PlanCSVcolumns.COLUMN_SAMPLE),
        PlanCSVcolumns.COLUMN_SAMPLE_DESCRIPTION : csvPlanDict.get(PlanCSVcolumns.COLUMN_SAMPLE_DESCRIPTION),
        PlanCSVcolumns.COLUMN_REF : csvPlanDict.get(PlanCSVcolumns.COLUMN_REF),
        PlanCSVcolumns.COLUMN_TARGET_BED : csvPlanDict.get(PlanCSVcolumns.COLUMN_TARGET_BED),
        PlanCSVcolumns.COLUMN_HOTSPOT_BED : csvPlanDict.get(PlanCSVcolumns.COLUMN_HOTSPOT_BED)
    }

    irSetings = get_irSettings()
    for param in irSetings:
        single_csv_samplesDict[param] = csvPlanDict.get(param)

    samples_contents.append(single_csv_samplesDict)

    return samples_contents

def check_selected_values(planObj, samples_contents, csvPlanDict):
    userInput = []
    errorMsg = []
    errorMsgDict = {}
    USERINPUT = planObj.get_USERINPUT()
    errorDict = {
        "E001" : "No samples available. Please check your input",
        "E002": "Selected Workflow is not compatible or invalid: %s",
        "E003" : "Selected Cellularity % is not valid, should be 1 to 100: {0}"
    }

    #process sample_contents for non barcoded samples
    isSingleCSV = False
    if csvPlanDict.get(PlanCSVcolumns.COLUMN_SAMPLE):
        isSingleCSV = True
        sampleName = csvPlanDict.get(PlanCSVcolumns.COLUMN_SAMPLE)
        samples_contents = get_samples_content_single_csv(csvPlanDict)
    else:
        #Validate the main plan csv IR chevron workflow
        ir_chevron_workflow = csvPlanDict.get(PlanCSVcolumns.COLUMN_SAMPLE_IR_WORKFLOW)
        notValid, workflowObj = irWorkflowNotValid(ir_chevron_workflow, USERINPUT)
        if (notValid):
            msg = errorDict["E002"] % (ir_chevron_workflow)
            errorMsgDict[PlanCSVcolumns.COLUMN_SAMPLE_IR_WORKFLOW] = msg

    if not samples_contents:
        msg = errorDict["E001"]
        errorMsgDict[PlanCSVcolumns.COLUMN_SAMPLE] = msg

    if samples_contents:
        for index, row in enumerate(samples_contents):
            setid_suffix = str(uuid.uuid4())
            errors = []
            rowCount = index + 1 if isSingleCSV else index + 3

            if not isSingleCSV:
                sampleName = row.get(PlanCSVcolumns.COLUMN_SAMPLE_NAME)
            if not sampleName:
                continue
            workflow = row.get(PlanCSVcolumns.COLUMN_SAMPLE_IR_WORKFLOW)
            notValid, workflowObj = irWorkflowNotValid(workflow, USERINPUT)

            # perform basic validation from TS side, rest of the validation will be executed from IRU
            # this would avoid the heavy lifting validation of IRU API call.
            if notValid:
                if isSingleCSV:
                    msg = errorDict["E002"] % (row.get(PlanCSVcolumns.COLUMN_SAMPLE_IR_WORKFLOW))
                    errorMsgDict[PlanCSVcolumns.COLUMN_SAMPLE_IR_WORKFLOW] = msg
                else:
                    msg = errorDict["E002"] % (row.get(PlanCSVcolumns.COLUMN_SAMPLE_IR_WORKFLOW))
                    errors.append(msg)

            #validate cellularity %
            cellularityPct = row.get(PlanCSVcolumns.COLUMN_SAMPLE_CELLULARITY)
            if cellularityPct:
                if not cellularityPct.isdigit():
                    errors.append(errorDict["E003"].format(cellularityPct))
                elif int(cellularityPct) not in range(1,101):
                    errors.append(errorDict["E003"].format(cellularityPct))

            # Do not process get_userInfoDict if the workflow is invalid
            if not isSingleCSV and errors:
                errorMsgDict[rowCount] = errors

            if not errorMsgDict:
                userInput.append(get_userInfoDict(row, workflowObj, rowCount, setid_suffix))

        if errorMsgDict:
            if isSingleCSV:
                errorMsg = json.dumps(errorMsgDict)

            else:
                #csvFile = csvPlanDict.get(PlanCSVcolumns.COLUMN_SAMPLE_FILE_HEADER)
                errorMsg = json.dumps(errorMsgDict)

    return errorMsg, userInput


def populate_userinput_from_response(planObj, httpHost, ir_account_id):
    # This function is mainly used get the workflow's application Type and the tag_isFactoryProvided meta data
    # NOTE:  IRU and TS terminology differs slightly.  In IRU, the column "Relation" is equivalent to TS' "RelationRole",
    # and IRU's column RelationshipType is TS' "Relation" in the JSON blob that is saved to the selectedPlugins BLOB.

    USERINPUT = {
        "user_input_url" : "/rundb/api/v1/plugin/IonReporterUploader/extend/userInput/",
        "workflows" : []
    }

    iru_RelationShipURL = USERINPUT["user_input_url"] + "?format=json&id=" + ir_account_id
    base_url = "http://" + "localhost" + iru_RelationShipURL
    response = requests.get(base_url)
    data = response.json()

    sampleRelationshipsTableInfo = data.get("sampleRelationshipsTableInfo",None)
    if sampleRelationshipsTableInfo:
        column_map = sampleRelationshipsTableInfo.get("column-map", None)
        for cm in column_map:
            workflow = cm.get("Workflow","")
            tag_isFactoryProvidedWorkflow = cm.get("tag_isFactoryProvidedWorkflow","")
            applicationType = cm.get("ApplicationType", "")
            relationshipType = cm.get("RelationshipType", "")

            USERINPUT["workflows"].append({
                "Workflow": workflow,
                "tag_isFactoryProvidedWorkflow": tag_isFactoryProvidedWorkflow,
                "ApplicationType": applicationType,
                "RelationshipType" : relationshipType
            });

    planObj.USERINPUT = USERINPUT

def call_iru_validation_api(host, iruSelectedPlugins_output, csvFile):
    # perform all the required validation by IRU API call
    errorMsgDict = {}
    iru_validation_errMsg = []

    url = "/rundb/api/v1/plugin/IonReporterUploader/extend/wValidateUserInput/"
    accountId = iruSelectedPlugins_output["userInput"]["accountId"]

    base_url = "http://" + "localhost" + url + "?id=" + accountId

    userInputInfo = iruSelectedPlugins_output["userInput"]
    response = requests.post(base_url,data=json.dumps(userInputInfo))
    response = response.json()
    iruValidationResults = response.get("validationResults","")
    if response and "validationResults" not in response:
        logger.debug(response)
        iru_validation_errMsg = "Internal error during IRU processing"
    else:
        for result in iruValidationResults:
            error_key = "%s" % (result["row"])
            if result["errors"]:
                errorMsgDict[error_key] = result["errors"]
            if result["warnings"]:
                errorMsgDict[error_key] = result["warnings"]
    if errorMsgDict:
        iru_validation_errMsg = json.dumps(errorMsgDict)

    return iru_validation_errMsg

def get_user_input_info_from_csv(samples_contents, csvPlanDict, planObj, httpHost, ir_account_id):
    populate_userinput_from_response(planObj, httpHost, ir_account_id)

    errorMsg, userInput = check_selected_values(planObj, samples_contents, csvPlanDict)

    return userInput, errorMsg

def validate_iruConfig_process_userInputInfo(csvPlanDict, username, samples_contents, planObj, httpHost, selectedPlugins=None):
    IR_server_in_csv = csvPlanDict.get(PlanCSVcolumns.COLUMN_IR_ACCOUNT)

    is_vcSelected = False
    if IR_server_in_csv:
        value = "IonReporterUploader"

    if selectedPlugins and "variantCaller" in selectedPlugins.keys():
        is_vcSelected = True
        print selectedPlugins

    errorMsg = None

    plugins = {}

    try:
        selectedPlugin = Plugin.objects.filter(name=value, selected=True, active=True)[0]
        if selectedPlugin.name == "IonReporterUploader":
            userIRConfig = get_default_or_first_IR_account_by_userName(username, IR_server=IR_server_in_csv)

        if userIRConfig:
            userInputInfo, errorMsg = get_user_input_info_from_csv(samples_contents,
                                                                   csvPlanDict,
                                                                   planObj,
                                                                   httpHost,
                                                                   userIRConfig["id"])

            if not errorMsg:
                pluginDict = {
                    "id": selectedPlugin.id,
                    "name": selectedPlugin.name,
                    "version": selectedPlugin.version,
                    "features": ['export']
                }
                userInputList = {
                    "accountId": userIRConfig["id"],
                    "accountName": userIRConfig["name"],
                    "isVariantCallerSelected": is_vcSelected,
                    "isVariantCallerConfigured": False,
                    "userInputInfo": userInputInfo
                }
                pluginDict["userInput"] = userInputList

                plugins[selectedPlugin.name] = pluginDict
        else:
            errorMsg = json.dumps({PlanCSVcolumns.COLUMN_IR_ACCOUNT : "%s is not reachable or not configured." % IR_server_in_csv})
    except:
        logger.exception(format_exc())
        errorMsg = json.dumps({"unknown" : "Internal error during IRU processing"})

    if errorMsg:
        iru_validationErrors = errorMsg
    else:
        if selectedPlugins:
            selectedPlugins.update(plugins)
        else:
            selectedPlugins = plugins

        iru_validationErrors = call_iru_validation_api(httpHost,
                                                       selectedPlugins["IonReporterUploader"],
                                                       csvPlanDict.get(PlanCSVcolumns.COLUMN_SAMPLE_FILE_HEADER, ""))

    return iru_validationErrors, selectedPlugins