#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# vim: tabstop=4 shiftwidth=4 softtabstop=4 noexpandtab
# Ion Plugin - Ion Reporter Uploader

import httplib
import urllib
import urllib2
import requests
import json

pluginName = 'IonReporterUploader'
pluginDir = ""

def IRULibraryTestPrint(x):
    print "IRULibrary: ", x
    return

def getProductionURLS(bucket):
        versions = {"urls": []}

        IR40 = {"IRVersion" : "IR40", "server" : "40.dataloader.ionreporter.iontorrent.com",
                "port" : "443", "protocol" : "https"
        }

        IR1X = {"IRVersion" : "IR1X", "server" : "dataloader.ionreporter.iontorrent.com",
                "port" : "443", "protocol" : "https"
        }

        versions["urls"].append(IR40)
        versions["urls"].append(IR1X)

        return versions

def versions(bucket):
    if "request_post" in bucket:
        inputJson = {"irAccount": bucket["request_post"]}
        return get_versions(inputJson)
    else:
        return bucket

def details(bucket):
    if "request_post" in bucket:
        inputJson = {"irAccount": bucket["request_post"]}
        return getUserDetails(inputJson)
    else:
        return bucket


def configs(bucket):
    user = str(bucket["user"])
    if "request_get" in bucket:
        #grab the config blob
        config = bucket.get("config", False)
        all_userconfigs = config.get("userconfigs", False)
        userconfigs = all_userconfigs.get(user, False)
        #remove the _version_cache cruft
        for userconfig in userconfigs:
            if userconfig.get("_version_cache", False):
                del userconfig["_version_cache"]
        return userconfigs


        #if we got all the way down here it failed
        return False


def workflows(bucket):
    user = str(bucket["user"])
    if "request_get" in bucket:
        #get the id from the querystring
        config_id = bucket["request_get"].get("id", False)

        #grab the config blob
        config = bucket.get("config", False)
        all_userconfigs = config.get("userconfigs", False)
        userconfigs = all_userconfigs.get(user)

        #now search for the config with the id given
        for userconfig in userconfigs:
            if userconfig.get("id", False):
                if userconfig.get("id") == config_id:
                    selected = {}
                    selected["irAccount"] = userconfig
                    return getWorkflowList(selected)

        #if we got all the way down here it failed
        return False

def wValidateUserInput(bucket):
    """
    Takes in the config id as a querystring parameter, and the HTTP POST body and passes those to
    validateUserInput
    """
    user = str(bucket["user"])
    if "request_get" in bucket:
        #get the id from the querystring
        config_id = bucket["request_get"].get("id", False)

        #grab the config blob
        config = bucket.get("config", False)
        all_userconfigs = config.get("userconfigs", False)
        userconfigs = all_userconfigs.get(user)

        #now search for the config with the id given
        for userconfig in userconfigs:
            if userconfig.get("id", False):
                if userconfig.get("id") == config_id:
                    selected = {}
                    selected["irAccount"] = userconfig
                    #get the http post body (form data)
                    selected["userInput"] = bucket["request_post"]
                    return validateUserInput(selected)

        #if we got all the way down here it failed
        return False

def newWorkflow(bucket):
    user = str(bucket["user"])
    if "request_get" in bucket:
        #get the id from the querystring
        config_id = bucket["request_get"].get("id", False)

        #grab the config blob
        config = bucket.get("config", False)
        all_userconfigs = config.get("userconfigs", False)
        userconfigs = all_userconfigs.get(user)

        #now search for the config with the id given
        for userconfig in userconfigs:
            if userconfig.get("id", False):
                if userconfig.get("id") == config_id:
                    selected = {}
                    selected["irAccount"] = userconfig
                    return getWorkflowCreationLandingPageURL(selected)

        #if we got all the way down here it failed
        return False

def userInput(bucket):
    user = str(bucket["user"])
    if "request_get" in bucket:
        #get the id from the querystring
        config_id = bucket["request_get"].get("id", False)

        #grab the config blob
        config = bucket.get("config", False)
        all_userconfigs = config.get("userconfigs", False)
        userconfigs = all_userconfigs.get(user)

        #now search for the config with the id given
        for userconfig in userconfigs:
            if userconfig.get("id", False):
                if userconfig.get("id") == config_id:
                    selected = {}
                    selected["irAccount"] = userconfig
                    return getUserInput(selected)

        #if we got all the way down here it failed
        return False


def get_versions(inputJson):
    irAccountJson = inputJson["irAccount"]
    protocol = irAccountJson["protocol"]
    server = irAccountJson["server"]
    port = irAccountJson["port"]
    token = irAccountJson["token"]
    #version = irAccountJson["version"]
    #version = version.split("IR")[1]
    grwsPath = "grws_1_2"
    #if version == "40" :
    #   grwsPath="grws"
    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/versionList/"
        hdrs = {'Authorization': token}
        resp = requests.get(url, verify=False, headers=hdrs)
        result = {}
        if resp.status_code == requests.codes.ok:
            result = resp.json()
        else:
        #raise Exception ("IR WebService Error Code " + str(resp.status_code))
            raise Exception("IR WebService Error Code " + str(resp.status_code))
    except requests.exceptions.ConnectionError, e:
        raise Exception("Error Code " + str(e))
    except requests.exceptions.HTTPError, e:
        raise Exception("Error Code " + str(e))
    except requests.exceptions.RequestException, e:
        raise Exception("Error Code " + str(e))
    except Exception, e:
        raise Exception("Error Code " + str(e))
    return result



def getSampleTabulationRules_1_6(workflowFullDetail):
    sampleRelationshipDict = {}
    sampleRelationshipDict["column-map"] = workflowFullDetail
    sampleRelationshipDict["columns"] = []

    workflowDict = {"Name": "Workflow", "Order": "1", "Type": "list", "ValueType": "String"}
    relationshipTypeDict = {"Name": "RelationshipType", "Order": "2", "Type": "list", "ValueType": "String",
                            "Values": ["Self", "Tumor_Normal", "Sample_Control", "Trio"]}
    relationDict = {"Name": "Relation", "Order": "3", "Type": "list", "ValueType": "String",
                    "Values": ["Sample", "Control", "Tumor", "Normal", "Father", "Mother", "Self"]}
    genderDict = {"Name": "Gender", "Order": "4", "Type": "list", "ValueType": "String",
                  "Values": ["Male", "Female", "Unknown"]}
    setIDDict = {"Name": "SetID", "Order": "5", "Type": "input", "ValueType": "Integer"}

    workflowDictValues = []
    for entry in workflowFullDetail :
        workflowName = entry["Workflow"]
        workflowDictValues.append(workflowName)
    workflowDict["Values"] = workflowDictValues

    sampleRelationshipDict["columns"].append(genderDict)
    sampleRelationshipDict["columns"].append(workflowDict)
    sampleRelationshipDict["columns"].append(relationshipTypeDict)
    sampleRelationshipDict["columns"].append(setIDDict)
    sampleRelationshipDict["columns"].append(relationDict)

    restrictionRulesList = []
    #restrictionRulesList.append({"ruleNumber":"1", "validationType":"error",
	#                             "For":{"Name": "RelationShipType", "Value":"Self"}, "Disabled":{"Name":"SetID"}})
    restrictionRulesList.append({"ruleNumber":"2", "validationType":"error",
	                             "For": {"Name": "RelationShipType", "Value": "Self"}, "Disabled": {"Name": "Relation"}})
    restrictionRulesList.append({"ruleNumber":"3", "validationType":"error",
	                             "For": {"Name": "RelationshipType", "Value": "Tumor_Normal"},
                                 "Valid": {"Name": "Relation", "Values": ["Tumor", "Normal"]}})
    restrictionRulesList.append({"ruleNumber":"4", "validationType":"error",
	                             "For": {"Name": "RelationshipType", "Value": "Sample_Control"},
                                 "Valid": {"Name": "Relation", "Values": ["Sample", "Control"]}})
    restrictionRulesList.append({"ruleNumber":"5", "validationType":"error",
	                             "For": {"Name": "RelationshipType", "Value": "Trio"},
                                 "Valid": {"Name": "Relation", "Values": ["Father", "Mother", "Self"]}})

    restrictionRulesList.append({"ruleNumber":"6", "validationType":"error",
	                             "For": {"Name": "ApplicationType", "Value": "Tumor Normal Sequencing"},
                                 "Valid": {"Name": "RelationshipType", "Values": ["Tumor_Normal"]}})
    restrictionRulesList.append({"ruleNumber":"7", "validationType":"error",
	                             "For": {"Name": "ApplicationType", "Value": "Paired Sample Ampliseq Sequencing"},
                                 "Valid": {"Name": "RelationshipType", "Values": ["Sample_Control"]}})
    restrictionRulesList.append({"ruleNumber":"8", "validationType":"error",
	                             "For": {"Name": "ApplicationType", "Value": "Genetic Disease Screening"},
                                 "Valid": {"Name": "RelationshipType", "Values": ["Trio"]}})

    restrictionRulesList.append({"ruleNumber":"9", "validationType":"error",
	                             "For": {"Name": "Relation", "Value": "Father"},
                                 "Valid": {"Name": "Gender", "Values": ["Male"]}})
    restrictionRulesList.append({"ruleNumber":"10", "validationType":"error",
	                             "For": {"Name": "Relation", "Value": "Mother"},
                                 "Valid": {"Name": "Gender", "Values": ["Female"]}})

    sampleRelationshipDict["restrictionRules"] = restrictionRulesList
    #return sampleRelationshipDict
    return {"status": "true", "error": "none", "sampleRelationshipsTableInfo": sampleRelationshipDict}


def getSampleTabulationRules_4_0(workflowFullDetail):
    sampleRelationshipDict = {}
    sampleRelationshipDict["column-map"] = workflowFullDetail
    sampleRelationshipDict["columns"] = []

    workflowDict = {"Name": "Workflow", "Order": "1", "Type": "list", "ValueType": "String"}
#    relationshipTypeDict = {"Name": "RelationshipType", "Order": "3", "Type": "list", "ValueType": "String",
#                            "Values": ["Self", "Tumor_Normal", "Sample_Control", "Trio"]}
    relationDict = {"Name": "Relation", "Order": "2", "Type": "list", "ValueType": "String",
                    "Values": ["Sample", "Control", "Tumor", "Normal", "Father", "Mother", "Proband", "Self"]}
    genderDict = {"Name": "Gender", "Order": "3", "Type": "list", "ValueType": "String",
                  "Values": ["Male", "Female", "Unknown"]}
    setIDDict = {"Name": "SetID", "Order": "4", "Type": "input", "ValueType": "Integer"}

    workflowDictValues = []
    for entry in workflowFullDetail :
        workflowName = entry["Workflow"]
        workflowDictValues.append(workflowName)
    workflowDict["Values"] = workflowDictValues

    sampleRelationshipDict["columns"].append(genderDict)
    sampleRelationshipDict["columns"].append(workflowDict)
    #sampleRelationshipDict["columns"].append(relationshipTypeDict)
    sampleRelationshipDict["columns"].append(setIDDict)
    sampleRelationshipDict["columns"].append(relationDict)

    restrictionRulesList = []
    #restrictionRulesList.append({"ruleNumber":"1", "validationType":"error","For":{"Name": "RelationShipType", "Value":"Self"}, "Disabled":{"Name":"SetID"}})
    #restrictionRulesList.append({"ruleNumber":"2", "validationType":"error","For": {"Name": "RelationShipType", "Value": "Self"}, "Disabled": {"Name": "Relation"}})
    restrictionRulesList.append({"ruleNumber":"3", "validationType":"error",
	                             "For": {"Name": "RelationshipType", "Value": "Self"},
                                 "Valid": {"Name": "Relation", "Values": ["Self"]}})
    restrictionRulesList.append({"ruleNumber":"4", "validationType":"error",
	                             "For": {"Name": "RelationshipType", "Value": "Tumor_Normal"},
                                 "Valid": {"Name": "Relation", "Values": ["Tumor", "Normal"]}})
    restrictionRulesList.append({"ruleNumber":"5", "validationType":"error",
	                             "For": {"Name": "RelationshipType", "Value": "Sample_Control"},
                                 "Valid": {"Name": "Relation", "Values": ["Sample", "Control"]}})
    restrictionRulesList.append({"ruleNumber":"6", "validationType":"error",
	                             "For": {"Name": "RelationshipType", "Value": "Trio"},
                                 "Valid": {"Name": "Relation", "Values": ["Father", "Mother", "Proband"]}})

    restrictionRulesList.append({"ruleNumber":"7", "validationType":"error",
	                             "For": {"Name": "Relation", "Value": "Father"},
                                 "Valid": {"Name": "Gender", "Values": ["Male"]}})
    restrictionRulesList.append({"ruleNumber":"8", "validationType":"error",
	                             "For": {"Name": "Relation", "Value": "Mother"},
                                 "Valid": {"Name": "Gender", "Values": ["Female"]}})

    restrictionRulesList.append({"ruleNumber":"9", "validationType":"error",
	                             "For": {"Name": "ApplicationType", "Value": "METAGENOMICS"},
                                 "Valid": {"Name": "Gender", "Values": ["Unknown"]}})
    sampleRelationshipDict["restrictionRules"] = restrictionRulesList
    #return sampleRelationshipDict
    return {"status": "true", "error": "none", "sampleRelationshipsTableInfo": sampleRelationshipDict}




def getUserInput(inputJson):
    irAccountJson = inputJson["irAccount"]
    server = irAccountJson["server"]
    token = irAccountJson["token"]
    protocol = irAccountJson["protocol"]
    port = irAccountJson["port"]
    version = irAccountJson["version"]
    version = version.split("IR")[1]
    grwsPath = "grws_1_2"
    #if version == "40" :
    #   grwsPath="grws_4_0"

    workflowsCallResult = getWorkflowList(inputJson)
    if workflowsCallResult.get("status") == "false":
        return {"status": "false", "error": workflowsCallResult.get("error")}
    workflowFullDetail = workflowsCallResult.get("userWorkflows")

    if version == "40":
        return getSampleTabulationRules_4_0(workflowFullDetail)
    else:
        return getSampleTabulationRules_1_6(workflowFullDetail)



def authCheck(inputJson):
    irAccountJson = inputJson["irAccount"]
    protocol = irAccountJson["protocol"]
    server = irAccountJson["server"]
    port = irAccountJson["port"]
    token = irAccountJson["token"]
    version = irAccountJson["version"]
    version = version.split("IR")[1]
    grwsPath = "grws_1_2"
    #if version == "40" :
    #   grwsPath="grws"


    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/usr/authcheck/"
        hdrs = {'Authorization': token}
        resp = requests.get(url, verify=False, headers=hdrs)
        result = ""
        if resp.status_code == requests.codes.ok:          # status_code returns an int
            result = resp.text
        else:
        #raise Exception ("IR WebService Error Code " + str(resp.status_code))
            return {"status": "false", "error": "IR WebService Error Code " + str(resp.status_code)}
    except requests.exceptions.ConnectionError, e:
        #raise Exception("Error Code " + str(e.code))
        return {"status": "false", "error": str(e)}
    except requests.exceptions.HTTPError, e:
        #raise Exception("Error Code " + str(e.code))
        return {"status": "false", "error": str(e)}
    except requests.exceptions.RequestException, e:
        #raise Exception("Error Code " + str(e.code))
        return {"status": "false", "error": str(e)}
    except Exception, e:
        #raise Exception("Error Code " + str(e.code))
        #return {"status":"false", "error":str(e.message)}
        return {"status": "false", "error": str(e)}
    if result == "SUCCESS":
        return {"status": "true", "error": "none"}
    return {"status": "false", "error": "none"}


def getWorkflowList(inputJson):
    irAccountJson = inputJson["irAccount"]
    protocol = irAccountJson["protocol"]
    server = irAccountJson["server"]
    port = irAccountJson["port"]
    token = irAccountJson["token"]
    version = irAccountJson["version"]
    version = version.split("IR")[1]
    grwsPath = "grws_1_2"
    #if version == "40" :
    #   grwsPath="grws"

    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/workflowList/"
        hdrs = {'Authorization': token, 'Version': version}
        resp = requests.get(url, verify=False, headers=hdrs)
        result = {}
        if resp.status_code == requests.codes.ok:
            result = json.loads(resp.text)
            try:
              for workflowBlob in result:
                appType = str (workflowBlob.get("ApplicationType"))
                if appType.find("Genetic Disease") != -1  :
                    workflowBlob["RelationshipType"] = "Trio"
                elif appType.find("Tumor Normal") != -1 :
                    workflowBlob["RelationshipType"] = "Tumor_Normal"
                elif appType.find("Paired Sample") != -1 :
                    workflowBlob["RelationshipType"] = "Sample_Control"
                else:
                    workflowBlob["RelationshipType"] = "Self"
            except Exception, a:
               return {"status": "false", "error": str(a)}
        else:
        #raise Exception ("IR WebService Error Code " + str(resp.status_code))
            raise Exception("IR WebService Error Code " + str(resp.status_code))
    except requests.exceptions.ConnectionError, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    except requests.exceptions.HTTPError, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    except requests.exceptions.RequestException, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    except Exception, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    return {"status": "true", "error": "none", "userWorkflows": result}


def getUserDataUploadPath(inputJson):
    irAccountJson = inputJson["irAccount"]
    protocol = irAccountJson["protocol"]
    server = irAccountJson["server"]
    port = irAccountJson["port"]
    token = irAccountJson["token"]
    version = irAccountJson["version"]
    version = version.split("IR")[1]
    grwsPath = "grws_1_2"
    #if version == "40" :
    #   grwsPath="grws"

    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/uploadpath/"
        hdrs = {'Authorization': token}
        resp = requests.get(url, verify=False, headers=hdrs)
        result = ""
        if resp.status_code == requests.codes.ok:
            result = resp.text
        else:
            #raise Exception ("IR WebService Error Code " + str(resp.status_code))
            raise Exception("IR WebService Error Code " + str(resp.status_code))
    except requests.exceptions.ConnectionError, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    except requests.exceptions.HTTPError, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    except requests.exceptions.RequestException, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    except Exception, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    return {"status": "true", "error": "none", "userDataUploadPath": result}


def sampleExistsOnIR(inputJson):
    sampleName = inputJson["sampleName"]
    irAccountJson = inputJson["irAccount"]

    protocol = irAccountJson["protocol"]
    server = irAccountJson["server"]
    port = irAccountJson["port"]
    token = irAccountJson["token"]
    version = irAccountJson["version"]
    version = version.split("IR")[1]
    grwsPath = "grws_1_2"
    #if version == "40" :
    #   grwsPath="grws"

    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/sampleExists/"
        hdrs = {'Authorization': token}
        queryArgs = {"sampleName": sampleName}
        resp = requests.post(url, params=queryArgs, verify=False, headers=hdrs)
        result = ""
        if resp.status_code == requests.codes.ok:
            result = resp.text
        else:
            #raise Exception ("IR WebService Error Code " + str(resp.status_code))
            raise Exception("IR WebService Error Code " + str(resp.status_code))
    except requests.exceptions.ConnectionError, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    except requests.exceptions.HTTPError, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    except requests.exceptions.RequestException, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    except Exception, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    if result == "true":
        return {"status": "true", "error": "none"}
    else:
        return {"status": "false", "error": "none"}


def getUserDetails(inputJson):
    """give it the account dict in, get the token and other info out"""
    irAccountJson = inputJson["irAccount"]
    userId = irAccountJson["userid"]
    password = irAccountJson["password"]

    protocol = irAccountJson["protocol"]
    server = irAccountJson["server"]
    port = irAccountJson["port"]
    #token = irAccountJson["token"]
    version = irAccountJson["version"]
    version = version.split("IR")[1]
    grwsPath = "grws_1_2"
    #if version == "40" :
    #   grwsPath="grws"
    unSupportedIRVersionsForThisFunction = ['10', '12', '14', '16', '18', '20']
    if version in unSupportedIRVersionsForThisFunction:
        return {"status": "false", "error": "User Details Query not supported for this version of IR " + version,
                "details": {}}


    #for 4.0, now return a hard coded result  grws layer inside ir 4.0  is not implemented yet.
    #return {"status":"true", "error":"none","details":{"firstName" : "vipinchandran","lastName":"nair", "orgName":"lifetech","token":"NRhYyl2xpnRGtItIzFXnMkX52QIJ1x6Popf+hIWwvEW71TtHAvh7hLtFiY7jLKZrULcNTRFycBmetTudyMSIA+Fz2ZvUfO8E1G0z7VlS93w="}}
    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/getUserDetails/"
        formParams = {"userName": userId, "password": password}
        #hdrs = {'Authorization':token}
        #resp = requests.post(url,data=formParams,verify=False, headers=hdrs)
        resp = requests.post(url, verify=False, data=formParams)
        result = {}
        if resp.status_code == requests.codes.ok:
            result = json.loads(resp.text)
        else:
            #raise Exception ("IR WebService Error Code " + str(resp.status_code))
            raise Exception("IR WebService Error Code " + str(resp.status_code))
    except requests.exceptions.ConnectionError, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    except requests.exceptions.HTTPError, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    except requests.exceptions.RequestException, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    except Exception, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    return {"status": "true", "error": "none", "details": result}


def validateUserInput(inputJson):
    userInput = inputJson["userInput"]
    irAccountJson = inputJson["irAccount"]

    protocol = irAccountJson["protocol"]
    server = irAccountJson["server"]
    port = irAccountJson["port"]
    token = irAccountJson["token"]
    version = irAccountJson["version"]
    version = version.split("IR")[1]
    grwsPath = "grws_1_2"
    #if version == "40" :
    #   grwsPath="grws"
    unSupportedIRVersionsForThisFunction = ['10', '12', '14', '16', '18', '20']
    if version in unSupportedIRVersionsForThisFunction:
        return {"status": "true",
                "error": "UserInput Validation Query not supported for this version of IR " + version,
                "validationResults": []}

    # re-arrange the rules and workflow information in a frequently usable tree structure.
    getUserInputCallResult = getUserInput(inputJson)
    if getUserInputCallResult.get("status") == "false":
        return {"status": "false", "error": getUserInputCallResult.get("error")}
    currentRules = getUserInputCallResult.get("sampleRelationshipsTableInfo")
    currentlyAvaliableWorkflows={}
    for cmap in currentRules["column-map"]:
       currentlyAvaliableWorkflows[cmap["Workflow"]]=cmap
    orderedColumns={}
    for col in currentRules["columns"]:
       orderedColumns[col["Order"]] = col

    """ some debugging prints for the dev phase
    print "Current Rules"
    print currentRules
    print ""
    print "ordered Columns"
    print orderedColumns
    print ""
    print ""
    print "Order 1"
    if getElementWithKeyValueLD("Order","8", currentRules["columns"]) != None:
        print getElementWithKeyValueLD("Order","8", currentRules["columns"])
    print ""
    print ""
    print "Name Gender"
    print getElementWithKeyValueDD("Name","Gender", orderedColumns)
    print ""
    print ""
    """ 



    userInputInfo = userInput["userInputInfo"]
    validationResults = []
    mockLogic = 0
    if mockLogic == 1:
        #for 4.0, now return a mock logic .
        row = 1
        for uip in userInputInfo:
            if "row" in uip:
                resultRow={"row": uip["row"]}
            else:
               resultRow={"row":str(row)}
            if uip["Gender"] == "Unknown" :
                resultRow["errorMessage"]="For the time being... ERROR:  Gender cannot be Unknown"
            if uip["setid"].find("0_") != -1  :
                resultRow["warningMessage"]="For the time being... WARNING:  setid is still zero .. did you forget to set it correctly?"
            validationResults.append(resultRow)
            row = row + 1
        return {"status": "true", "error": "none", "validationResults": validationResults}



    setidHash={}
    rowErrors={}
    rowWarnings={}
    uniqueSamples={}
    analysisCost={}
    analysisCost["workflowCosts"]=[]


    row = 1
    for uip in userInputInfo:
        # make a row number if not provided
        if "row" not in uip:
           uip["row"]=str(row)
        rowStr = uip["row"]
        if  rowStr not in rowErrors:
           rowErrors[rowStr]=[]
        if  rowStr not in rowWarnings:
           rowWarnings[rowStr]=[]

        # all given sampleNames should be unique
        if uip["sample"] not in uniqueSamples:
            uip["sample"] = rowStr
        else:
            msg="sample name "+uip["sample"] + " on row("+ rowNum+") appears to be used in some other row (row "+rowStr+"). Please change the sample name"
            inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)

        # if workflow is empty then dont validate and dont include this row in setid for further validations.
        if uip["Workflow"] =="":
            continue

        # some known translations
        if  "setid" in uip :
            uip["SetID"]= uip["setid"]
        if  "RelationshpType" not in uip :
            if  "Relation" in uip :
                uip["RelationshipType"]= uip["Relation"]
            if  "RelationRole" in uip :
                uip["Relation"]= uip["RelationRole"]
        if uip["Workflow"] in currentlyAvaliableWorkflows:
            uip["ApplicationType"] = currentlyAvaliableWorkflows[uip["Workflow"]]["ApplicationType"]
        else:
            uip["ApplicationType"] = "unknown"

        # save the record on the setID
        setid = uip["SetID"]
        if  setid not in setidHash:
            setidHash[setid] = {}
            setidHash[setid]["records"] =[] 
            setidHash[setid]["firstWorkflow"]=uip["Workflow"]
            setidHash[setid]["firstRecordRow"]=uip["row"]
        else:
            expectedWorkflow = setidHash[setid]["firstWorkflow"]
            previousRow = setidHash[setid]["firstRecordRow"]
            if expectedWorkflow != uip["Workflow"]:
                msg="Selected workflow "+ uip["Workflow"] + " does not match a previous sample for the same setid, with workflow "+ expectedWorkflow +" on row "+ previousRow+ ". Either change this workflow to match the previous workflow selection for the this setid, or please change the setid to a new value if you intend this sample to be used in a different IR analysis."
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
        setidHash[setid]["records"].append(uip)



        # check if workflow is still active.
        if uip["Workflow"] not in currentlyAvaliableWorkflows:
            msg="selected workflow "+ uip["Workflow"] + " is not available for this user account at this time"
            inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)

        # check if sample already exists on IR at this time and give a warning..
        inputJson["sampleName"]= uip["sample"]
        sampleExistsCallResults = sampleExistsOnIR(inputJson)
        if sampleExistsCallResults.get("error") != "":
            if sampleExistsCallResults.get("status") == "true":
                msg="given sample name "+ uip["sample"] + " already exists on IonReporter "
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)

        # check the rules
        validateAllRulesOnRecord(currentRules["restrictionRules"], uip, setidHash, rowErrors, rowWarnings)

        row = row + 1


    # after validations of basic rules look for errors all role requirements, uniqueness in roles, excess number of
    # roles, insufficient number of roles, etc.
    for setid in setidHash:
        # first check all the required roles are there in the corresponding records
        rowsLooked = ""
        for validRole in setidHash[setid]["validRelationRoles"]:
            foundRole=0
            rowsLooked = ""
            for record in setidHash[setid]["records"]:
                if rowsLooked != "":
                    rowsLooked = rowsLooked + "," + record["row"]
                else:
                    rowsLooked = record["row"]
                if validRole == record["Relation"]:   #or RelationRole
                    foundRole = 1
            if foundRole == 0 :
                msg="For the workflow " + setidHash[setid]["firstWorkflow"] +", a required RelationRole "+ validRole + " is not found. Row(s) concerned with the same setid are " + rowsLooked
                inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)

        # check the number of records expected and number of records provided.
        sizeOfRequiredRoles = len(setidHash[setid]["validRelationRoles"])
        sizeOfAvailableRoles = len(setidHash[setid]["records"])
        if (sizeOfAvailableRoles > sizeOfRequiredRoles):
            msg="For the workflow " + setidHash[setid]["firstWorkflow"] +", more than the required number of RelationRoles are found. Expected number of roles is "+ str(sizeOfRequiredRoles) + ". Row(s) concerned with the same setid are " + rowsLooked
            inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)


        # calculate the cost of the analysis
        cost={}
        cost["row"]=setidHash[setid]["firstRecordRow"]
        cost["workflow"]=setidHash[setid]["firstWorkflow"]
        cost["cost"]="50.00"
        analysisCost["workflowCosts"].append(cost)

    analysisCost["totalCost"]="2739.99"
    analysisCost["text1"]="The following are the details of the analysis planned on the IonReporter, and their associated cost estimates."
    analysisCost["text2"]="Press OK if you have reviewed and agree to the estimated costs and wish to continue with the planned IR analysis, or press CANCEL to make modifications."





    """
    print ""
    print ""
    print "userInputInfo"
    print userInputInfo
    print ""
    print ""
    """
    #print ""
    #print ""
    #print "setidHash"
    #print setidHash
    #print ""
    #print ""

    # consolidate the  errors and warnings per row and return the results
    foundAtLeastOneError = 0
    for uip in userInputInfo:
        rowstr=uip["row"]
        emsg=""
        wmsg=""
        if rowstr in rowErrors:
           for e in  rowErrors[rowstr]:
              foundAtLeastOneError =1
              emsg = emsg + e + " ; "
        if rowstr in rowWarnings:
           for w in  rowWarnings[rowstr]:
              wmsg = wmsg + w + " ; "
        k={"row":rowstr, "errorMessage":emsg, "warningMessage":wmsg, "errors": rowErrors[rowstr], "warnings": rowWarnings[rowstr]}
        validationResults.append(k)

    # forumulate a few constant advices for use on certain conditions, to TS users
    advices={}
    #advices["onTooManyErrors"]= "Looks like there are some errors on this page. If you are not sure of the workflow requirements, you can opt to only upload the samples to IR and not run any IR analysis on those samples at this time, by not selecting any workflow on the Workflow column of this tabulation. You can later find the sample on the IR, and launch IR analysis on it later, by logging into the IR application."
    advices["onTooManyErrors"]= "There are errors on this page. If you only want to upload samples to Ion Reporter and not perform an Ion Reporter analysis at this time, you do not need to select a Workflow. When you are ready to launch an Ion Reporter analysis, you must log into Ion Reporter and select the samples to analyze."

    #true/false return code is reserved for error in executing the functionality itself, and not the condition of the results itself.
    # say if there are networking errors, talking to IR, etc will return false. otherwise, return pure results. The results internally
    # may contain errors, which is to be interpretted by the caller. If there are other helpful error info regarding the results itsef,
    # then additional variables may be used to reflect metadata about the results. the status/error flags may be used to reflect the
    # status of the call itself.
    #if (foundAtLeastOneError == 1):
    #    return {"status": "false", "error": "none", "validationResults": validationResults, "cost":analysisCost}
    #else:
    #    return {"status": "true", "error": "none", "validationResults": validationResults, "cost":analysisCost}
    return {"status": "true", "error": "none", "validationResults": validationResults, "cost":analysisCost, "advices": advices}

    """
    # if we want to implement this logic in grws, then here is the interface code.  But currently it is not yet implemented there.
    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/TSUserInputValidate/"
        hdrs = {'Authorization': token}
        resp = requests.post(url, verify=False, headers=hdrs)
        result = {}
        if resp.status_code == requests.codes.ok:
            result = json.loads(resp.text)
        else:
            #raise Exception ("IR WebService Error Code " + str(resp.status_code))
            raise Exception("IR WebService Error Code " + str(resp.status_code))
    except requests.exceptions.ConnectionError, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    except requests.exceptions.HTTPError, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    except requests.exceptions.RequestException, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    except Exception, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    return {"status": "true", "error": "none", "validationResults": result}
    """

def validateAllRulesOnRecord(rules, uip, setidHash, rowErrors, rowWarnings):
    row=uip["row"]
    setid=uip["SetID"]
    for  rule in rules:
        # find the rule Number
        if "ruleNumber" not in rule:
            msg="INTERNAL ERROR  Incompatible validation rules for this version of IRU. ruleNumber not specified in one of the rules."
            inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)
            ruleNum="unkhown"
        else:
            ruleNum=rule["ruleNumber"]

        # find the validation type
        if "validationType" in rule:
            validationType = rule["validationType"]
        else:
            validationType = "error"
        if validationType not in ["error", "warn"]:
            msg="INTERNAL ERROR ON RULE # "+ruleNum+"   Incompatible validation rules for this version of IRU. unrecognized validationType \"" + validationType + "\""
            inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)

        # execute all the rules
        if "For" in rule:
            if rule["For"]["Name"] not in uip:
                msg="INTERNAL ERROR ON RULE # "+ruleNum+"   Incompatible validation rules for this version of TSS. No such \"For\" field \"" + rule["For"]["Name"] + "\""
                inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)
            if "Valid" in rule:
                if rule["Valid"]["Name"] not in uip:
                    msg="INTERNAL ERROR ON RULE # "+ruleNum+"   Incompatible validation rules for this version of TSS. No such \"Valid\"field \"" + rule["Valid"]["Name"] + "\""
                    inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)
                else:
                    kFor=rule["For"]["Name"]
                    vFor=rule["For"]["Value"]
                    kValid=rule["Valid"]["Name"]
                    vValid=rule["Valid"]["Values"]
                    #print  "validating   kfor " +kFor  + " vFor "+ vFor + "  kValid "+ kValid
                    if uip[kFor] == vFor :
                        if uip[kValid] not in vValid :
                            msg="Incorrect value \"" + uip[kValid] + "\" found for " + kValid + " When "+ kFor + " is \"" + vFor +"\"   rule # "+ ruleNum
                            inputValidationErrorHandle(row, validationType, msg, rowErrors, rowWarnings)
                        # a small hardcoded update into the setidHash for later evaluation of the role uniqueness
                        if kValid == "Relation":
                            if setid  in setidHash:
                                if "validRelationRoles" not in setidHash[setid]:
                                    #print  "saving   row " +row + "  setid " + setid + "  kfor " +kFor  + " vFor "+ vFor + "  kValid "+ kValid
                                    setidHash[setid]["validRelationRoles"] = vValid   # this is actually roles
            elif "Invalid" in rule:
                if rule["Invalid"]["Name"] not in uip:
                    msg="INTERNAL ERROR ON RULE # "+ruleNum+"   Incompatible validation rules for this version of TSS. No such \"Invalid\" field \"" + rule["Invalid"]["Name"] + "\""
                    inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)
                else:
                    kFor=rule["For"]["Name"]
                    vFor=rule["For"]["Value"]
                    kInvalid=rule["Invalid"]["Name"]
                    vInvalid=rule["Invalid"]["Values"]
                    #print  "validating   kfor " +kFor  + " vFor "+ vFor + "  kInvalid "+ kInvalid
                    if uip[kFor] == vFor :
                        if uip[kInvalid] in vInvalid :
                            msg="Incorrect value \"" + uip[kInvalid] + "\" found for " + kInvalid + " When "+ kFor + " is \"" + vFor +"\"   rule # "+ ruleNum
                            inputValidationErrorHandle(row, validationType, msg, rowErrors, rowWarnings)
            elif "Disabled" in rule:
                pass
            else:
                 msg="INTERNAL ERROR ON RULE # "+ruleNum+"   Incompatible validation rules for this version of IRU. \"For\" specified without a \"Valid\" or \"Invalid\" tag."
                 inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)
        else:
            msg="INTERNAL ERROR ON RULE # "+ruleNum+"   Incompatible validation rules for this version of IRU. No action provided on this rule."
            inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)



def inputValidationErrorHandle(row, validationType, msg, rowErrors, rowWarnings):
    if validationType == "error":
        rowErrors[row].append(msg)
    elif validationType == "warn":
        rowWarnings[row].append(msg)

def getWorkflowCreationLandingPageURL(inputJson):
    irAccountJson = inputJson["irAccount"]

    protocol = irAccountJson["protocol"]
    server = irAccountJson["server"]
    port = irAccountJson["port"]
    token = irAccountJson["token"]
    version = irAccountJson["version"]
    version = version.split("IR")[1]
    grwsPath = "grws_1_2"
    #if version == "40" :
    #   grwsPath="grws"
    unSupportedIRVersionsForThisFunction = ['10', '12', '14', '16', '18', '20']
    if version in unSupportedIRVersionsForThisFunction:
        return {"status": "false",
                "error": "Workflow Creation UI redirection to IR is not supported for this version of IR " + version,
                "workflowCreationLandingPageURL": []}

    #for 4.0, now return a hard coded result  grws layer inside ir 4.0  is not implemented yet.
    queryParams = {'authToken': token}
    urlEncodedQueryParams = urllib.urlencode(queryParams)
    #url2 = protocol + "://" + server + ":" + port + "/ir/secure/workflow.html?" + urlEncodedQueryParams
    #urlPart1 = protocol + "://" + server + ":" + port
    urlPart2 = "/ir/secure/workflow.html?" + urlEncodedQueryParams
    #returnUrl = urlPart1+urlPart2
    #return {"status": "true", "error": "none", "workflowCreationLandingPageURL": returnUrl}

    #actually get the correct ui server address, port and protocol from the grws and use that one instead of using iru-server's address.
    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/getIrGUIUrl"
        print url 
        hdrs = {'Authorization': token}
        resp = requests.get(url, verify=False, headers=hdrs)
        #result = {}
        if resp.status_code == requests.codes.ok:
            #result = json.loads(resp.text)
            urlPart1 =str(resp.text)
        else:
            #raise Exception ("IR WebService Error Code " + str(resp.status_code))
            return {"status": "false", "error":"IR WebService Error Code " + str(resp.status_code)}
    except requests.exceptions.ConnectionError, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    except requests.exceptions.HTTPError, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    except requests.exceptions.RequestException, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    except Exception, e:
        #raise Exception("Error Code " + str(e))
        return {"status": "false", "error": str(e)}
    returnUrl = urlPart1+urlPart2
    return {"status": "true", "error": "none", "workflowCreationLandingPageURL": returnUrl}










def getElementWithKeyValueDD(k,v, dictOfDict):
    for k1 in dictOfDict:
        ele = dictOfDict[k1]
        if k in ele:
            if ele[k] == v:
                return ele
    return None

def getElementWithKeyValueLD(k,v, listOfDict):
    for ele in listOfDict:
        if k in ele:
            if ele[k] == v:
                return ele
    return None



# set to "IonReporterUploader" by default
def getPluginName():
    return pluginName


def setPluginName(x):
    pluginName = x
    return


def getPluginDir():
    return pluginDir


def setPluginDir(x):
    pluginDir = x
    return


if __name__ == "__main__":
    j = {'port': '8080',
         'protocol': 'http',
         'server': 'plum.itw',
         'token': 'wVcoTeYGfKxItiaWo2lngsV/r0jukG2pLKbZBkAFnlPbjKfPTXLbIhPb47YA9u78'}
    b = {}
    b["request_post"] = j
    #print IRUTest(b)

    k = {'port': '443',
         'protocol': 'https',
         'server': 'think2.itw',
         'version': 'IR40',
         'token': 'wVcoTeYGfKxItiaWo2lngsV/r0jukG2pLKbZBkAFnlPbjKfPTXLbIhPb47YA9u78'}
    c = {}
    c["irAccount"] = k

    p={"userInputInfo":[
        {
          "row": "6",
          "Workflow": "",
          "Gender": "Female",
          "barcodeId": "IonXpress_011",
          "sample": "pgm-s11",
          "Relation": "Self",
          "RelationRole": "Self",
          "setid": "0__837663e7-f7f8-4334-b14b-dea091dd353b"
        },
        {
          "row": "96",
          "Workflow": "TumorNormal",
          "Gender": "Unknown",
          "barcodeId": "IonXpress_012",
          "sample": "pgm-s12",
          "Relation": "Tumor_Normal",
          "RelationRole": "Normal",
          "setid": "2__7179df4c-c6bb-4cbe-97a4-bb48951a4acd"
        },
        {
          "row": "5",
          "Workflow": "TumorNormal",
          "Gender": "Male",
          "barcodeId": "IonXpress_013",
          "sample": "pgm-s12",
          "Relation": "Tumor_Normal",
          "RelationRole": "Normal",
          "setid": "2__7179df4c-c6bb-4cbe-97a4-bb48951a4acd"
        },
        {
          "row": "9",
          "Workflow": "TumorNormal",
          "Gender": "Male",
          "barcodeId": "IonXpress_012",
          "sample": "pgm-s12",
          "Relation": "Tumor_Normal",
          "RelationRole": "Normal",
          "setid": "2__7179df4c-c6bb-4cbe-97a4-bb48951a4acd"
        }
      ],
	  "accountId":"planned_irAccount_id_blahblahblah",
	  "accountName":"planned_irAccount_name_blahblahblah"
	 }
    c["userInput"]=p

    print ""
    print ""
    print ""
    print c
    print ""
    print ""
    print ""
    print ""
    print ""
    print validateUserInput(c)
    print ""
    print ""
    print ""
    print ""
    print ""
    print getWorkflowCreationLandingPageURL(c)
    print ""
    print ""
    print ""
    print ""
    print get_versions(c)

