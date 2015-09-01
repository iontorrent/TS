#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# vim: tabstop=4 shiftwidth=4 softtabstop=4 noexpandtab
# Ion Plugin - Ion Reporter Uploader

import glob
import json
import os
import requests
import urllib
import subprocess
import base64

pluginName = 'IonReporterUploader'
pluginDir = ""
debugMode = 0     # this needs a file /tmp/a.txt   existing and 777 permissions

def IRULibraryTestPrint(x):
    print "IRULibrary: ", x
    return

def testBucket(bucket):
    return bucket


# Writes to debug file
def write_debug_log(text):     # this needs a file /tmp/a.txt   existing and 777 permissions, and dont forget to switch on the global variable debugMode
    if (debugMode==0):
        return
    log_file = "/tmp/a.txt"
    file = open(log_file, "a")
    file.write(text)
    file.write("\n")
    return log_file

def get_plugin_dir():
    return os.path.dirname(__file__)

def set_classpath():
    plugin_dir=get_plugin_dir()

    jarscmd="find " + plugin_dir + "/lib/java/shared" + "  |grep \"jar$\" |xargs |sed 's/ /:/g'"
    #write_debug_log("jarscmd="+ jarscmd)
    proc = subprocess.Popen(jarscmd, shell=True, stdout=subprocess.PIPE)
    (jarsout, jarserr)= proc.communicate()
    #exitCode = proc.returncode
    #write_debug_log("jarsout="+ jarsout)
    if (jarserr):
        write_debug_log("jarserr="+ jarserr)
    classpath_str = plugin_dir + "/lib/java/shared:" + jarsout
    #write_debug_log("classpath="+ classpath_str)
    os.environ["CLASSPATH"] = classpath_str
    #write_debug_log("classpath from os ="+ os.getenv('CLASSPATH'))
    if (os.getenv("LD_LIBRARY_PATH")):
        ld_str = plugin_dir + "/lib:" + os.getenv("LD_LIBRARY_PATH")
    else:
        ld_str = plugin_dir + "/lib"
        os.environ["LD_LIBRARY_PATH"] = ld_str
    return classpath_str

def get_httpResponseFromSystemTools(cmd):
    write_debug_log("systemtools  cmd = "+ cmd)
    proc = subprocess.Popen( cmd, stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    exitCode = proc.returncode
    write_debug_log("systemtools  out = "+ out)
    if (err):
	    write_debug_log("systemtools  err = "+ err)
    else:
       err=""
    write_debug_log("systemtools  exitCode = "+ str(exitCode))
    if (exitCode != 0):
        return {"status": "false", "error": err, "exitCode":exitCode, "stdout": out}
    else:
        return {"status": "true", "error": err, "exitCode":exitCode, "stdout": out}
    return {"status": "false", "error": err, "exitCode":exitCode, "stdout": out}

def get_httpResponseFromSystemToolsAsJson(cmd):
    result = get_httpResponseFromSystemTools(cmd)
    if (result["status"] != "true"):
        return result
    j = {}
    try:
        j=json.loads(result["stdout"])
        result["json"]=j
        return result
    except ValueError:
        result["status"]="false"
        result["error"]="error decoding output as json "+ result["stdout"]
        result["json"]=""
        return result

def get_httpResponseFromIRUJavaAsJson(cmd):
    plugin_dir=get_plugin_dir()

    #TBD find with cmd = find java/ |grep bin |grep "bin\/java$"
    #javaBin=plugin_dir+ "/"+ "java/jre/jre1.8.0_45/bin/java"
    javaBin=plugin_dir+ "/"+ "java/jre/openjdk-7-jre-headless/usr/lib/jvm/java-7-openjdk-amd64/jre/bin/java"
    #javaBin="java"

    javaMemOptionsForJavaBelow_1_8=" -XX:MaxPermSize=256m"
    javaMemOptions=javaMemOptionsForJavaBelow_1_8
    #javaMemOptions=""       # perm size not required when using internally embedded java 1.8 and above.

    set_classpath()

    result={}
    result=get_httpResponseFromSystemToolsAsJson(javaBin + " -Xms3g -Xmx3g"+ javaMemOptions + " -Dlog.home=/tmp com.lifetechnologies.ionreporter.clients.irutorrentplugin.Launcher " + cmd)
    return result


def configs(bucket):
    user = str(bucket["user"])
    if "request_get" in bucket:
        #grab the config blob
        config = bucket.get("config", False)
        all_userconfigs = config.get("userconfigs", False)
        userconfigs = all_userconfigs.get(user, False)

        active_configs = []

        for userconfig in userconfigs:
            #remove the _version_cache cruft
            if userconfig.get("_version_cache", False):
                del userconfig["_version_cache"]

            #only if the version is not 1 add it to the list to return
            if userconfig.get("version", False):
                if userconfig["version"][-2] != "1":
                    active_configs.append(userconfig)

        return active_configs


def getSelectedIRAccountFromBucket(bucket):
    user = str(bucket["user"])
    if "request_get" in bucket:
        #get the id from the querystring
        config_id = bucket["request_get"].get("id", False)

        #grab the config blob
        config = bucket.get("config", False)
        all_userconfigs = config.get("userconfigs", False)
        userconfigs = all_userconfigs.get(user)

        if (userconfigs == None):
            return {"status": "false", "error": "Error getting list of IR accounts from plugin configuration"}

        #now search for the config with the id given
        for userconfig in userconfigs:
            if userconfig.get("id", False):
                if userconfig.get("id") == config_id:
                    selected = {}
                    selected["irAccount"] = userconfig
                    return {"status": "true", "error": "none","selectedAccount": selected}

        #if we got all the way down here it failed
        return {"status": "false", "error": "No such IR account"}
    return {"status": "false", "error": "request was not a GET "}




#example access http://10.43.24.24/rundb/api/v1/plugin/IonReporterUploader/extend/getProductionURLS/?format=json&id=mt6irqz9rtrc4i6y34fio9
def getProductionURLS(bucket):
        versions = {"urls": []}

        IR50 = {"IRVersion" : "IR50", "server" : "40.dataloader.ionreporter.lifetechnologies.com",
                "port" : "443", "protocol" : "https"
        }
        IR46 = {"IRVersion" : "IR46", "server" : "40.dataloader.ionreporter.lifetechnologies.com",
                "port" : "443", "protocol" : "https"
        }
        IR44 = {"IRVersion" : "IR44", "server" : "40.dataloader.ionreporter.lifetechnologies.com",
                "port" : "443", "protocol" : "https"
        }
        IR42 = {"IRVersion" : "IR42", "server" : "40.dataloader.ionreporter.lifetechnologies.com",
                "port" : "443", "protocol" : "https"
        }
        IR40 = {"IRVersion" : "IR40", "server" : "40.dataloader.ionreporter.lifetechnologies.com",
                "port" : "443", "protocol" : "https"
        }
        IR1X = {"IRVersion" : "IR1X", "server" : "dataloader.ionreporter.lifetechnologies.com",
                "port" : "443", "protocol" : "https"
        }

        versions["urls"].append(IR50)
        versions["urls"].append(IR46)
        versions["urls"].append(IR44)
        versions["urls"].append(IR42)
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

def auth(bucket):
    if "request_post" in bucket:
        inputJson = {"irAccount": bucket["request_post"]}
        return authCheck(inputJson)
    else:
        return bucket

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
                    if "filterKey" in bucket["request_get"]:
                        selected["filterKey"] = bucket["request_get"]["filterKey"]
                    if "filterValue" in bucket["request_get"]:
                        selected["filterValue"] = bucket["request_get"]["filterValue"]
                    if "andFilterKey2" in bucket["request_get"]:
                        selected["andFilterKey2"] = bucket["request_get"]["andFilterKey2"]
                    if "andFilterValue2" in bucket["request_get"]:
                        selected["andFilterValue2"] = bucket["request_get"]["andFilterValue2"]
                    return getWorkflowList(selected)

        #if we got all the way down here it failed
        return False


def workflowsWithOncomine(bucket):      # not likely to be used anymore due to TS UI changes 
    if "request_get" in bucket:
        selectedAccountResult = getSelectedIRAccountFromBucket(bucket)
        if (selectedAccountResult["status"] != "true") :
		    return selectedAccountResult
        inputJson = selectedAccountResult["selectedAccount"]
        return getWorkflowListWithOncomine(inputJson)
    return {"status": "false", "error": "request was not a GET"}

def workflowsWithoutOncomine(bucket):
    if "request_get" in bucket:
        selectedAccountResult = getSelectedIRAccountFromBucket(bucket)
        if (selectedAccountResult["status"] != "true") :
		    return selectedAccountResult
        inputJson = selectedAccountResult["selectedAccount"]
        return getWorkflowListWithoutOncomine(inputJson)
    return {"status": "false", "error": "request was not a GET"}



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
                    if "filterKey" in bucket["request_get"]:
                        selected["filterKey"] = bucket["request_get"]["filterKey"]
                    if "filterValue" in bucket["request_get"]:
                        selected["filterValue"] = bucket["request_get"]["filterValue"]
                    if "andFilterKey2" in bucket["request_get"]:
                        selected["andFilterKey2"] = bucket["request_get"]["andFilterKey2"]
                    if "andFilterValue2" in bucket["request_get"]:
                        selected["andFilterValue2"] = bucket["request_get"]["andFilterValue2"]
                    #get the http post body (form data)
                    selected["userInput"] = bucket["request_post"]
                    return validateUserInput(selected)

        #if we got all the way down here it failed
        return False

def newWorkflow(bucket):
    user = str(bucket["user"])
    if "request_get" in bucket:
        response = {"status": "false"}

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
                    version = userconfig["version"]
                    version = version.split("IR")[1]
                    selected["irAccount"] = userconfig
                    if version == '40':
                        response = getWorkflowCreationLandingPageURL(selected)     # token is embedded in the query params
                        response["method"] = "get"
                    else:
                        response = getWorkflowCreationLandingPageURLBase(selected) # no token.. just the base url. the caller has to embed the token in the post data
                        response["method"] = "post"

        #if we got all the way down here it failed
        return response

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
                    if "filterKey" in bucket["request_get"]:
                        selected["filterKey"] = bucket["request_get"]["filterKey"]
                    if "filterValue" in bucket["request_get"]:
                        selected["filterValue"] = bucket["request_get"]["filterValue"]
                    if "andFilterKey2" in bucket["request_get"]:
                        selected["andFilterKey2"] = bucket["request_get"]["andFilterKey2"]
                    if "andFilterValue2" in bucket["request_get"]:
                        selected["andFilterValue2"] = bucket["request_get"]["andFilterValue2"]
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

    #curl -ks -H Authorization:rwVcoTeYGfKxItiaWo2lngsV/r0jukG2pLKbZBkAFnlPbjKfPTXLbIhPb47YA9u78 https://xyz.com:443/grws_1_2/data/versionList
    url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/versionList/"
    cmd="curl -ks -H Authorization:"+token+ " " +url
    result = get_httpResponseFromSystemToolsAsJson(cmd)
    if (result["status"] =="true"):
        return result["json"]
    else:
        return result

    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/versionList/"
        hdrs = {'Authorization': token}
        resp = requests.get(url, verify=False, headers=hdrs,timeout=30)  #timeout is in seconds
        result = {}
        if resp.status_code == requests.codes.ok:
            result = resp.json()
        else:
        #raise Exception ("IR WebService Error Code " + str(resp.status_code))
            raise Exception("IR WebService Error Code " + str(resp.status_code))
    except requests.exceptions.Timeout, e:
        return {"status": "false", "error": "Timeout"}
    except requests.exceptions.ConnectionError, e:
        raise Exception("Error Code " + str(e))
    except requests.exceptions.HTTPError, e:
        raise Exception("Error Code " + str(e))
    except requests.exceptions.RequestException, e:
        raise Exception("Error Code " + str(e))
    except Exception, e:
        raise Exception("Error Code " + str(e))
    return result

def getIRCancerTypesList(inputJson):
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

    #curl -ks --request POST -H Authorization:rwVcoTeYGfKxItiaWo2lngsV/r0jukG2pLKbZBkAFnlPbjKfPTXLbIhPb47YA9u78 https://xyz.com:443/grws_1_2/data/getAvailableCancerType
    url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/getAvailableCancerType"
    cmd="curl -ks --request POST -H Authorization:"+token+ " " +url
    result = get_httpResponseFromSystemToolsAsJson(cmd)
    if (result["status"] =="true"):
        return {"status":"true", "error":"none", "cancerTypes":result["json"]}
    else:
        return result


    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/getAvailableCancerType/"
        hdrs = {'Authorization': token}
        resp = requests.post(url, verify=False, headers=hdrs,timeout=30)  #timeout is in seconds
        result = {}
        if resp.status_code == requests.codes.ok:
            #result = resp.json()
            #result = json.load(resp.text)
            result = resp.text
        else:
        #raise Exception ("IR WebService Error Code " + str(resp.status_code))
            #raise Exception("IR WebService Error Code " + str(resp.status_code))
            return {"status": "false", "error": str(resp.status_code)}
    except requests.exceptions.Timeout, e:
        return {"status": "false", "error": "Timeout"}
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

    list=[ "Bladder Cancer", "Breast Cancer", "Glioblastoma", "Colorectal Cancer", "Endometrial Cancer", "Esophageal Cancer", "Gastric Cancer", "Gastrointestinal Stromal Tumor", "Head and Neck Cancer", "Liver Cancer", "Non-Small Cell Lung Cancer", "Small Cell Lung", "Melanoma", "Mesothelioma", "Osteosarcoma", "Ovarian Cancer", "Pancreatic Cancer", "Prostate Cancer", "Renal Cancer", "Basal Cell Carcinoma", "Soft Tissue Sarcoma", "Testicular Cancer", "Thyroid Cancer" ]
    #return {"status":"true", "error":"none", "cancerTypes":result}
    return {"status":"true", "error":"none", "cancerTypes":result}




def getSampleTabulationRules_1_6(inputJson, workflowFullDetail):
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


def getSampleTabulationRules_4_0(inputJson, workflowFullDetail):
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



def getSampleTabulationRules_4_2(inputJson, workflowFullDetail):
    sampleRelationshipDict = {}
    sampleRelationshipDict["column-map"] = workflowFullDetail
    sampleRelationshipDict["columns"] = []
    cancerTypesListResult = getIRCancerTypesList(inputJson)
    if (cancerTypesListResult["status"] != "true"):
        return cancerTypesListResult
    cancerTypesList = cancerTypesListResult["cancerTypes"]

    workflowDict = {"Name": "Workflow", "FullName": "Workflow", "Order": "1", "key":"Workflow", "Type": "list", "ValueType": "String"}
#    relationshipTypeDict = {"Name": "RelationshipType", "Order": "3", "key":"Relation", "Type": "list", "ValueType": "String",
#                            "Values": ["Self", "Tumor_Normal", "Sample_Control", "Trio"]}
    relationDict = {"Name": "Relation", "FullName": "Relation Role", "Order": "2", "key":"RelationRole", "Type": "list", "ValueType": "String",
                    "Values": ["Sample", "Control", "Tumor", "Normal", "Father", "Mother", "Proband", "Self"]}
    genderDict =   {"Name": "Gender", "FullName": "Gender","Order": "3", "key":"Gender", "Type": "list", "ValueType": "String",
                    "Values": ["Male", "Female", "Unknown"]}
    nucleoDict =   {"Name": "NucleotideType", "FullName": "Nucleotide Type", "Order": "4","key":"NucleotideType",  "Type": "list", "ValueType": "String",
                    "Values": ["DNA", "RNA"]}
    cellPctDict =  {"Name": "CellularityPct", "FullName": "Cellularity Percentage", "Order": "5","key":"cellularityPct",  "Type": "input", 
	            "ValueType": "Integer", "Integer.Low":"0", "Integer.High":"100",
                    "ValueDefault":"0"}
    cancerDict =   {"Name": "CancerType", "FullName": "Cancer Type", "Order": "6", "key":"cancerType", "Type": "list", "ValueType": "String",
                    "Values": cancerTypesList}
    setIDDict =    {"Name": "SetID", "FullName": "IR Analysis Set ID", "Order": "7", "key":"setid", "Type": "input", "ValueType": "Integer"}


    workflowDictValues = []
    for entry in workflowFullDetail :
        workflowName = entry["Workflow"]
        workflowDictValues.append(workflowName)
    workflowDict["Values"] = workflowDictValues

    sampleRelationshipDict["columns"].append(workflowDict)
    #sampleRelationshipDict["columns"].append(relationshipTypeDict)
    sampleRelationshipDict["columns"].append(relationDict)
    sampleRelationshipDict["columns"].append(genderDict)
    sampleRelationshipDict["columns"].append(nucleoDict)
    sampleRelationshipDict["columns"].append(cellPctDict)
    sampleRelationshipDict["columns"].append(cancerDict)
    sampleRelationshipDict["columns"].append(setIDDict)

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
    restrictionRulesList.append({"ruleNumber":"99", "validationType":"error",            # a tempporary rule that is going to go away.
	                           "For": {"Name": "RelationshipType", "Value": "DNA_RNA"},
                                 "Valid": {"Name": "Relation", "Values": ["Self"]}})
    restrictionRulesList.append({"ruleNumber":"98", "validationType":"error",            # a tempporary rule that is going to go away.
	                           "For": {"Name": "RelationshipType", "Value": "SINGLE_RNA_FUSION"},
                                 "Valid": {"Name": "Relation", "Values": ["Self"]}})

    restrictionRulesList.append({"ruleNumber":"7", "validationType":"error",
	                           "For": {"Name": "Relation", "Value": "Father"},
                                 "Valid": {"Name": "Gender", "Values": ["Male", "Unknown"]}})
    restrictionRulesList.append({"ruleNumber":"8", "validationType":"error",
	                           "For": {"Name": "Relation", "Value": "Mother"},
                                 "Valid": {"Name": "Gender", "Values": ["Female", "Unknown"]}})
    restrictionRulesList.append({"ruleNumber":"9", "validationType":"error",
	                           "For": {"Name": "ApplicationType", "Value": "METAGENOMICS"},
                                 "Valid": {"Name": "Gender", "Values": ["Unknown"]}})
    restrictionRulesList.append({"ruleNumber":"10", "validationType":"error",
	                           "For": {"Name": "DNA_RNA_Workflow", "Value": "DNA_RNA"},
                                 "Valid": {"Name": "NucleotideType", "Values": ["DNA","RNA"]}})
    restrictionRulesList.append({"ruleNumber":"11", "validationType":"error",
	                           "For": {"Name": "DNA_RNA_Workflow", "Value":"RNA"},
                                 "Valid": {"Name": "NucleotideType", "Values": ["RNA"]}})
    restrictionRulesList.append({"ruleNumber":"12", "validationType":"error",
	                              "For": {"Name": "DNA_RNA_Workflow", "Value":"DNA"},
                                 "Disabled": {"Name": "NucleotideType"}})
    restrictionRulesList.append({"ruleNumber":"13", "validationType":"error",
	                              "For": {"Name": "CELLULARITY_PCT_REQUIRED", "Value":"0"},
                                 "Disabled": {"Name": "CellularityPct"}})
    restrictionRulesList.append({"ruleNumber":"14", "validationType":"error",
	                              "For": {"Name": "CANCER_TYPE_REQUIRED", "Value":"0"},
                                 "Disabled": {"Name": "CancerType"}})
    restrictionRulesList.append({"ruleNumber":"15", "validationType":"error",
	                              "For": {"Name": "CELLULARITY_PCT_REQUIRED", "Value":"1"},
                                 "NonEmpty": {"Name": "CellularityPct"}})
    restrictionRulesList.append({"ruleNumber":"16", "validationType":"error",
	                              "For": {"Name": "CANCER_TYPE_REQUIRED", "Value":"1"},
                                 "NonEmpty": {"Name": "CancerType"}})
    sampleRelationshipDict["restrictionRules"] = restrictionRulesList
    #return sampleRelationshipDict
    return {"status": "true", "error": "none", "sampleRelationshipsTableInfo": sampleRelationshipDict}


def getSampleTabulationRules_4_4(inputJson, workflowFullDetail):
    sampleRelationshipDict = {}
    sampleRelationshipDict["column-map"] = workflowFullDetail
    sampleRelationshipDict["columns"] = []
    cancerTypesListResult = getIRCancerTypesList(inputJson)
    if (cancerTypesListResult["status"] != "true"):
        return cancerTypesListResult
    cancerTypesList = cancerTypesListResult["cancerTypes"]

    workflowDict = {"Name": "Workflow", "FullName": "Workflow", "Order": "1", "key":"Workflow", "Type": "list", "ValueType": "String"}
#    relationshipTypeDict = {"Name": "RelationshipType", "Order": "3", "key":"Relation", "Type": "list", "ValueType": "String",
#                            "Values": ["Self", "Tumor_Normal", "Sample_Control", "Trio"]}
    relationDict = {"Name": "Relation", "FullName": "Relation Role", "Order": "2", "key":"RelationRole", "Type": "list", "ValueType": "String",
                    "Values": ["Sample", "Control", "Tumor", "Normal", "Father", "Mother", "Proband", "Self"]}
    genderDict =   {"Name": "Gender", "FullName": "Gender","Order": "3", "key":"Gender", "Type": "list", "ValueType": "String",
                    "Values": ["Male", "Female", "Unknown"]}
    nucleoDict =   {"Name": "NucleotideType", "FullName": "Nucleotide Type", "Order": "4","key":"NucleotideType",  "Type": "list", "ValueType": "String",
                    "Values": ["DNA", "RNA"]}
    cellPctDict =  {"Name": "CellularityPct", "FullName": "Cellularity Percentage", "Order": "5","key":"cellularityPct",  "Type": "input", 
	            "ValueType": "Integer", "Integer.Low":"0", "Integer.High":"100",
                    "ValueDefault":"0"}
    cancerDict =   {"Name": "CancerType", "FullName": "Cancer Type", "Order": "6", "key":"cancerType", "Type": "list", "ValueType": "String",
                    "Values": cancerTypesList}
    setIDDict =    {"Name": "SetID", "FullName": "IR Analysis Set ID", "Order": "7", "key":"setid", "Type": "input", "ValueType": "Integer"}


    workflowDictValues = []
    for entry in workflowFullDetail :
        workflowName = entry["Workflow"]
        workflowDictValues.append(workflowName)
    workflowDict["Values"] = workflowDictValues

    sampleRelationshipDict["columns"].append(workflowDict)
    #sampleRelationshipDict["columns"].append(relationshipTypeDict)
    sampleRelationshipDict["columns"].append(relationDict)
    sampleRelationshipDict["columns"].append(genderDict)
    sampleRelationshipDict["columns"].append(nucleoDict)
    sampleRelationshipDict["columns"].append(cellPctDict)
    sampleRelationshipDict["columns"].append(cancerDict)
    sampleRelationshipDict["columns"].append(setIDDict)

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
    restrictionRulesList.append({"ruleNumber":"99", "validationType":"error",            # a tempporary rule that is going to go away.
	                           "For": {"Name": "RelationshipType", "Value": "DNA_RNA"},
                                 "Valid": {"Name": "Relation", "Values": ["Self"]}})
    restrictionRulesList.append({"ruleNumber":"98", "validationType":"error",            # a tempporary rule that is going to go away.
	                           "For": {"Name": "RelationshipType", "Value": "SINGLE_RNA_FUSION"},
                                 "Valid": {"Name": "Relation", "Values": ["Self"]}})

    restrictionRulesList.append({"ruleNumber":"7", "validationType":"error",
	                           "For": {"Name": "Relation", "Value": "Father"},
                                 "Valid": {"Name": "Gender", "Values": ["Male", "Unknown"]}})
    restrictionRulesList.append({"ruleNumber":"8", "validationType":"error",
	                           "For": {"Name": "Relation", "Value": "Mother"},
                                 "Valid": {"Name": "Gender", "Values": ["Female", "Unknown"]}})
    restrictionRulesList.append({"ruleNumber":"9", "validationType":"error",
	                           "For": {"Name": "ApplicationType", "Value": "METAGENOMICS"},
                                 "Valid": {"Name": "Gender", "Values": ["Unknown"]}})
    restrictionRulesList.append({"ruleNumber":"10", "validationType":"error",
	                           "For": {"Name": "DNA_RNA_Workflow", "Value": "DNA_RNA"},
                                 "Valid": {"Name": "NucleotideType", "Values": ["DNA","RNA"]}})
    restrictionRulesList.append({"ruleNumber":"11", "validationType":"error",
	                           "For": {"Name": "DNA_RNA_Workflow", "Value":"RNA"},
                                 "Valid": {"Name": "NucleotideType", "Values": ["RNA"]}})
    restrictionRulesList.append({"ruleNumber":"12", "validationType":"error",
	                              "For": {"Name": "DNA_RNA_Workflow", "Value":"DNA"},
                                 "Disabled": {"Name": "NucleotideType"}})
    restrictionRulesList.append({"ruleNumber":"13", "validationType":"error",
	                              "For": {"Name": "CELLULARITY_PCT_REQUIRED", "Value":"0"},
                                 "Disabled": {"Name": "CellularityPct"}})
    restrictionRulesList.append({"ruleNumber":"14", "validationType":"error",
	                              "For": {"Name": "CANCER_TYPE_REQUIRED", "Value":"0"},
                                 "Disabled": {"Name": "CancerType"}})
    restrictionRulesList.append({"ruleNumber":"15", "validationType":"error",
	                              "For": {"Name": "CELLULARITY_PCT_REQUIRED", "Value":"1"},
                                 "NonEmpty": {"Name": "CellularityPct"}})
    restrictionRulesList.append({"ruleNumber":"16", "validationType":"error",
	                              "For": {"Name": "CANCER_TYPE_REQUIRED", "Value":"1"},
                                 "NonEmpty": {"Name": "CancerType"}})
    sampleRelationshipDict["restrictionRules"] = restrictionRulesList
    #return sampleRelationshipDict
    return {"status": "true", "error": "none", "sampleRelationshipsTableInfo": sampleRelationshipDict}


def getSampleTabulationRules_4_6(inputJson, workflowFullDetail):
    sampleRelationshipDict = {}
    sampleRelationshipDict["column-map"] = workflowFullDetail
    sampleRelationshipDict["columns"] = []
    cancerTypesListResult = getIRCancerTypesList(inputJson)
    if (cancerTypesListResult["status"] != "true"):
        return cancerTypesListResult
    cancerTypesList = cancerTypesListResult["cancerTypes"]

    workflowDict = {"Name": "Workflow", "FullName": "Workflow", "Order": "1", "key":"Workflow", "Type": "list", "ValueType": "String"}
#    relationshipTypeDict = {"Name": "RelationshipType", "Order": "3", "key":"Relation", "Type": "list", "ValueType": "String",
#                            "Values": ["Self", "Tumor_Normal", "Sample_Control", "Trio"]}
    relationDict = {"Name": "Relation", "FullName": "Relation Role", "Order": "2", "key":"RelationRole", "Type": "list", "ValueType": "String",
                    "Values": ["Sample", "Control", "Tumor", "Normal", "Father", "Mother", "Proband", "Self"]}
    genderDict =   {"Name": "Gender", "FullName": "Gender","Order": "3", "key":"Gender", "Type": "list", "ValueType": "String",
                    "Values": ["Male", "Female", "Unknown"]}
    nucleoDict =   {"Name": "NucleotideType", "FullName": "Nucleotide Type", "Order": "4","key":"NucleotideType",  "Type": "list", "ValueType": "String",
                    "Values": ["DNA", "RNA"]}
    cellPctDict =  {"Name": "CellularityPct", "FullName": "Cellularity Percentage", "Order": "5","key":"cellularityPct",  "Type": "input", 
	            "ValueType": "Integer", "Integer.Low":"0", "Integer.High":"100",
                    "ValueDefault":"0"}
    cancerDict =   {"Name": "CancerType", "FullName": "Cancer Type", "Order": "6", "key":"cancerType", "Type": "list", "ValueType": "String",
                    "Values": cancerTypesList}
    setIDDict =    {"Name": "SetID", "FullName": "IR Analysis Set ID", "Order": "7", "key":"setid", "Type": "input", "ValueType": "Integer"}


    workflowDictValues = []
    for entry in workflowFullDetail :
        workflowName = entry["Workflow"]
        workflowDictValues.append(workflowName)
    workflowDict["Values"] = workflowDictValues

    sampleRelationshipDict["columns"].append(workflowDict)
    #sampleRelationshipDict["columns"].append(relationshipTypeDict)
    sampleRelationshipDict["columns"].append(relationDict)
    sampleRelationshipDict["columns"].append(genderDict)
    sampleRelationshipDict["columns"].append(nucleoDict)
    sampleRelationshipDict["columns"].append(cellPctDict)
    sampleRelationshipDict["columns"].append(cancerDict)
    sampleRelationshipDict["columns"].append(setIDDict)

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
    restrictionRulesList.append({"ruleNumber":"99", "validationType":"error",            # a tempporary rule that is going to go away.
	                           "For": {"Name": "RelationshipType", "Value": "DNA_RNA"},
                                 "Valid": {"Name": "Relation", "Values": ["Self"]}})
    restrictionRulesList.append({"ruleNumber":"98", "validationType":"error",            # a tempporary rule that is going to go away.
	                           "For": {"Name": "RelationshipType", "Value": "SINGLE_RNA_FUSION"},
                                 "Valid": {"Name": "Relation", "Values": ["Self"]}})

    restrictionRulesList.append({"ruleNumber":"7", "validationType":"error",
	                           "For": {"Name": "Relation", "Value": "Father"},
                                 "Valid": {"Name": "Gender", "Values": ["Male", "Unknown"]}})
    restrictionRulesList.append({"ruleNumber":"8", "validationType":"error",
	                           "For": {"Name": "Relation", "Value": "Mother"},
                                 "Valid": {"Name": "Gender", "Values": ["Female", "Unknown"]}})
    restrictionRulesList.append({"ruleNumber":"9", "validationType":"error",
	                           "For": {"Name": "ApplicationType", "Value": "METAGENOMICS"},
                                 "Valid": {"Name": "Gender", "Values": ["Unknown"]}})
    restrictionRulesList.append({"ruleNumber":"10", "validationType":"error",
	                           "For": {"Name": "DNA_RNA_Workflow", "Value": "DNA_RNA"},
                                 "Valid": {"Name": "NucleotideType", "Values": ["DNA","RNA"]}})
    restrictionRulesList.append({"ruleNumber":"11", "validationType":"error",
	                           "For": {"Name": "DNA_RNA_Workflow", "Value":"RNA"},
                                 "Valid": {"Name": "NucleotideType", "Values": ["RNA"]}})
    restrictionRulesList.append({"ruleNumber":"12", "validationType":"error",
	                              "For": {"Name": "DNA_RNA_Workflow", "Value":"DNA"},
                                 "Disabled": {"Name": "NucleotideType"}})
    restrictionRulesList.append({"ruleNumber":"13", "validationType":"error",
	                              "For": {"Name": "CELLULARITY_PCT_REQUIRED", "Value":"0"},
                                 "Disabled": {"Name": "CellularityPct"}})
    restrictionRulesList.append({"ruleNumber":"14", "validationType":"error",
	                              "For": {"Name": "CANCER_TYPE_REQUIRED", "Value":"0"},
                                 "Disabled": {"Name": "CancerType"}})
    restrictionRulesList.append({"ruleNumber":"15", "validationType":"error",
	                              "For": {"Name": "CELLULARITY_PCT_REQUIRED", "Value":"1"},
                                 "NonEmpty": {"Name": "CellularityPct"}})
    restrictionRulesList.append({"ruleNumber":"16", "validationType":"error",
	                              "For": {"Name": "CANCER_TYPE_REQUIRED", "Value":"1"},
                                 "NonEmpty": {"Name": "CancerType"}})
    sampleRelationshipDict["restrictionRules"] = restrictionRulesList
    #return sampleRelationshipDict
    return {"status": "true", "error": "none", "sampleRelationshipsTableInfo": sampleRelationshipDict}


def getSampleTabulationRules_5_0(inputJson, workflowFullDetail):
    sampleRelationshipDict = {}
    sampleRelationshipDict["column-map"] = workflowFullDetail
    sampleRelationshipDict["columns"] = []
    cancerTypesListResult = getIRCancerTypesList(inputJson)
    if (cancerTypesListResult["status"] != "true"):
        return cancerTypesListResult
    cancerTypesList = cancerTypesListResult["cancerTypes"]

    workflowDict = {"Name": "Workflow", "FullName": "Workflow", "Order": "1", "key":"Workflow", "Type": "list", "ValueType": "String"}
#    relationshipTypeDict = {"Name": "RelationshipType", "Order": "3", "key":"Relation", "Type": "list", "ValueType": "String",
#                            "Values": ["Self", "Tumor_Normal", "Sample_Control", "Trio"]}
    relationDict = {"Name": "Relation", "FullName": "Relation Role", "Order": "2", "key":"RelationRole", "Type": "list", "ValueType": "String",
                    "Values": ["Sample", "Control", "Tumor", "Normal", "Father", "Mother", "Proband", "Self"]}
    genderDict =   {"Name": "Gender", "FullName": "Gender","Order": "3", "key":"Gender", "Type": "list", "ValueType": "String",
                    "Values": ["Male", "Female", "Unknown"]}
    nucleoDict =   {"Name": "NucleotideType", "FullName": "Nucleotide Type", "Order": "4","key":"NucleotideType",  "Type": "list", "ValueType": "String",
                    "Values": ["DNA", "RNA"]}
    cellPctDict =  {"Name": "CellularityPct", "FullName": "Cellularity Percentage", "Order": "5","key":"cellularityPct",  "Type": "input", 
	            "ValueType": "Integer", "Integer.Low":"0", "Integer.High":"100",
                    "ValueDefault":"0"}
    cancerDict =   {"Name": "CancerType", "FullName": "Cancer Type", "Order": "6", "key":"cancerType", "Type": "list", "ValueType": "String",
                    "Values": cancerTypesList}
    setIDDict =    {"Name": "SetID", "FullName": "IR Analysis Set ID", "Order": "7", "key":"setid", "Type": "input", "ValueType": "Integer"}


    workflowDictValues = []
    for entry in workflowFullDetail :
        workflowName = entry["Workflow"]
        workflowDictValues.append(workflowName)
    workflowDict["Values"] = workflowDictValues

    sampleRelationshipDict["columns"].append(workflowDict)
    #sampleRelationshipDict["columns"].append(relationshipTypeDict)
    sampleRelationshipDict["columns"].append(relationDict)
    sampleRelationshipDict["columns"].append(genderDict)
    sampleRelationshipDict["columns"].append(nucleoDict)
    sampleRelationshipDict["columns"].append(cellPctDict)
    sampleRelationshipDict["columns"].append(cancerDict)
    sampleRelationshipDict["columns"].append(setIDDict)

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
    restrictionRulesList.append({"ruleNumber":"99", "validationType":"error",            # a tempporary rule that is going to go away.
	                           "For": {"Name": "RelationshipType", "Value": "DNA_RNA"},
                                 "Valid": {"Name": "Relation", "Values": ["Self"]}})
    restrictionRulesList.append({"ruleNumber":"98", "validationType":"error",            # a tempporary rule that is going to go away.
	                           "For": {"Name": "RelationshipType", "Value": "SINGLE_RNA_FUSION"},
                                 "Valid": {"Name": "Relation", "Values": ["Self"]}})

    restrictionRulesList.append({"ruleNumber":"7", "validationType":"error",
	                           "For": {"Name": "Relation", "Value": "Father"},
                                 "Valid": {"Name": "Gender", "Values": ["Male", "Unknown"]}})
    restrictionRulesList.append({"ruleNumber":"8", "validationType":"error",
	                           "For": {"Name": "Relation", "Value": "Mother"},
                                 "Valid": {"Name": "Gender", "Values": ["Female", "Unknown"]}})
    restrictionRulesList.append({"ruleNumber":"9", "validationType":"error",
	                           "For": {"Name": "ApplicationType", "Value": "METAGENOMICS"},
                                 "Valid": {"Name": "Gender", "Values": ["Unknown"]}})
    restrictionRulesList.append({"ruleNumber":"10", "validationType":"error",
	                           "For": {"Name": "DNA_RNA_Workflow", "Value": "DNA_RNA"},
                                 "Valid": {"Name": "NucleotideType", "Values": ["DNA","RNA"]}})
    restrictionRulesList.append({"ruleNumber":"11", "validationType":"error",
	                           "For": {"Name": "DNA_RNA_Workflow", "Value":"RNA"},
                                 "Valid": {"Name": "NucleotideType", "Values": ["RNA"]}})
    restrictionRulesList.append({"ruleNumber":"12", "validationType":"error",
	                              "For": {"Name": "DNA_RNA_Workflow", "Value":"DNA"},
                                 "Disabled": {"Name": "NucleotideType"}})
    restrictionRulesList.append({"ruleNumber":"13", "validationType":"error",
	                              "For": {"Name": "CELLULARITY_PCT_REQUIRED", "Value":"0"},
                                 "Disabled": {"Name": "CellularityPct"}})
    restrictionRulesList.append({"ruleNumber":"14", "validationType":"error",
	                              "For": {"Name": "CANCER_TYPE_REQUIRED", "Value":"0"},
                                 "Disabled": {"Name": "CancerType"}})
    restrictionRulesList.append({"ruleNumber":"15", "validationType":"error",
	                              "For": {"Name": "CELLULARITY_PCT_REQUIRED", "Value":"1"},
                                 "NonEmpty": {"Name": "CellularityPct"}})
    restrictionRulesList.append({"ruleNumber":"16", "validationType":"error",
	                              "For": {"Name": "CANCER_TYPE_REQUIRED", "Value":"1"},
                                 "NonEmpty": {"Name": "CancerType"}})
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

    # add Upload Only option
    workflowFullDetail.insert(0,{'ApplicationType':'UploadOnly', 'Workflow':'Upload Only', 'RelationshipType': 'Self'})

    if version == "50":
        return getSampleTabulationRules_5_0(inputJson, workflowFullDetail)
    elif version == "46":
        return getSampleTabulationRules_4_6(inputJson, workflowFullDetail)
    elif version == "44":
        return getSampleTabulationRules_4_4(inputJson, workflowFullDetail)
    elif version == "42":
        return getSampleTabulationRules_4_2(inputJson, workflowFullDetail)
    elif version == "40":
        return getSampleTabulationRules_4_0(inputJson, workflowFullDetail)   # TBD Jose, this need to go away when IR fixes the version list
    else:
        return getSampleTabulationRules_5_0(inputJson, workflowFullDetail)



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

    #curl -ks -H Authorization:rwVcoTeYGfKxItiaWo2lngsV/r0jukG2pLKbZBkAFnlPbjKfPTXLbIhPb47YA9u78 https://xyz.com:443/grws_1_2/usr/authcheck
    url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/usr/authcheck"
    cmd="curl -ks -H Authorization:"+token+ " " +url
    result = get_httpResponseFromSystemTools(cmd)
    if (   (result["status"] =="true")  and (result["stdout"] == "SUCCESS")   ):
        return {"status": "true", "error": "none"}
    else:
        return result


    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/usr/authcheck/"
        hdrs = {'Authorization': token}
        resp = requests.get(url, verify=False, headers=hdrs, timeout=30)  #timeout is in seconds
        result = ""
        if resp.status_code == requests.codes.ok:          # status_code returns an int
            result = resp.text
        else:
        #raise Exception ("IR WebService Error Code " + str(resp.status_code))
            return {"status": "false", "error": "IR WebService Error Code " + str(resp.status_code)}
    except requests.exceptions.Timeout, e:
        return {"status": "false", "error": "Timeout"}
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
    #return inputJson  #debug only

    try:
        #url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/workflowList/"
        #hdrs = {'Authorization': token, 'Version': version}
        #resp = requests.get(url, verify=False, headers=hdrs,timeout=30)  #timeout is in seconds
        #result = {}
        #returnJson = []
        #if resp.status_code == requests.codes.ok:
            #result = json.loads(resp.text)


        #curl -ks -H Authorization:rwVcoTeYGfKxItiaWo2lngsV/r0jukG2pLKbZBkAFnlPbjKfPTXLbIhPb47YA9u78 -H Version:42 https://xyz.com:443/grws_1_2/data/workflowList
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/workflowList"
        cmd="curl -ks    -H Authorization:"+token  +   " -H Version:"+version   +   " "+url
        cmdResult = get_httpResponseFromSystemToolsAsJson(cmd)
        result = {}
        returnJson = []
        if (cmdResult["status"] !="true"):
            return cmdResult
        else:
            result = cmdResult["json"]



            try:
              for workflowBlob in result:
                appType = str (workflowBlob.get("ApplicationType"))

                # populate the relation roles type based on specific patterns of application type
                if appType.find("Genetic Disease") != -1  :
                    workflowBlob["RelationshipType"] = "Trio"
                elif appType.find("Tumor Normal") != -1 :
                    workflowBlob["RelationshipType"] = "Tumor_Normal"
                elif appType.find("Paired Sample") != -1 :
                    workflowBlob["RelationshipType"] = "Sample_Control"
                else:
                    workflowBlob["RelationshipType"] = "Self"

                # populate the ocp enabled workflow or not
                if "OCP_Workflow" not in workflowBlob :
                    workflowBlob["OCP_Workflow"] = "false"
                    # covers IR46
                    if "tag_Oncomine" in workflowBlob :
                        workflowBlob["OCP_Workflow"] = "true"
                    else:
                        # covers IR42, IR44, IR46       for custom workflows containing oncomine plugin
                        for k in workflowBlob :
                            if k.startswith("wfl_plugin_Oncomine"):
                                workflowBlob["OCP_Workflow"] = "true"
                                break
                    # covers IR42 IR44    DNA_RNA and RNA
                    if workflowBlob["OCP_Workflow"] == "false":
                        #if appType.find("Oncomine") != -1 :
                        if (   (  version in ["40","42","44"]  )   and    (appType.find("Oncomine") != -1)   ) :
                            workflowBlob["OCP_Workflow"] = "true"
                        # covers IR42 IR44    DNA     A bad way, but no other way.
                        elif (  (  version in ["40","42","44"]   )  and   ((appType == "Amplicon Low Frequency Sequencing")  or  (appType == "Annotation"))    ):
                            workflowBlob["OCP_Workflow"] = "true"
                    # safely remove the other type from the list, becuase both of these types are mutually exclusive in nature. 
                    if "tag_ColonLung" in workflowBlob :
                        workflowBlob["OCP_Workflow"] = "false"

                # populate the onconet enabled workflow or not
                if "Onconet_Workflow" not in workflowBlob :
                    workflowBlob["Onconet_Workflow"] = "false"
                    # no way to recognize before IR46
                    # recognizable only from IR46
                    if "tag_ColonLung" in workflowBlob :
                        workflowBlob["Onconet_Workflow"] = "true"
                    #elif (  (version in ["40","42","44"] )   and    (appType.find("Oncomine") != -1)   ) :
                    elif appType.find("Oncomine") != -1 :
                        workflowBlob["Onconet_Workflow"] = "true"
                    #elif (  (version in ["40","42","44"])  and   ((appType == "Amplicon Low Frequency Sequencing")   )    ):
                    elif appType == "Amplicon Low Frequency Sequencing":
                        workflowBlob["Onconet_Workflow"] = "true"
                    # safely remove the other type from the list, becuase both of these types are mutually exclusive in nature. 
                    if "tag_Oncomine" in workflowBlob :
                        workflowBlob["Onconet_Workflow"] = "false"
                    for k in workflowBlob :
                        if k.startswith("wfl_plugin_Oncomine"):
                            workflowBlob["Onconet_Workflow"] = "false"


                # populate the whether DNA/RNA type workflow
                if "DNA_RNA_Workflow" not in workflowBlob :
                    if appType.find("DNA_RNA") != -1 :
                        workflowBlob["DNA_RNA_Workflow"] = "DNA_RNA"
                    elif appType.find("RNA") != -1 :
                        workflowBlob["DNA_RNA_Workflow"] = "RNA"
                    else:
                        workflowBlob["DNA_RNA_Workflow"] = "DNA"


                # A temporary overriding for 4.2. Should go away.
                # The original relationship type should be preserved.
                # Should go away when the TS planning page dynamism is
                # correctly built based on the restriction rules.
                if workflowBlob["DNA_RNA_Workflow"] == "DNA_RNA" :
                    workflowBlob["RelationshipType"] = "DNA_RNA"
                if workflowBlob["DNA_RNA_Workflow"] == "RNA" :
                    workflowBlob["RelationshipType"] = "SINGLE_RNA_FUSION"

                keyPairExists=False
                andKeyPair2Exists=False
                atleastOneKeyPairExists=False
                if (  ("filterKey" in inputJson ) and ("filterValue" in inputJson )  ):
                    keyPairExists=True
                    atleastOneKeyPairExists=True
                if (  ("andFilterKey2" in inputJson ) and ("andFilterValue2" in inputJson )  ):
                    andKeyPair2Exists=True
                    atleastOneKeyPairExists=True

                if  atleastOneKeyPairExists==True:   # should go through the filteration. else return all..
                    qualifyFilteration=True
                    if keyPairExists:
                        if (  workflowBlob[inputJson["filterKey"]] !=  inputJson["filterValue"])  :
                            qualifyFilteration=False
                    if andKeyPair2Exists:
                        if (  workflowBlob[inputJson["andFilterKey2"]] !=  inputJson["andFilterValue2"])  :
                            qualifyFilteration=False
                    if qualifyFilteration==True:
                        returnJson.append (workflowBlob)
                else:
                    returnJson.append (workflowBlob)
            except Exception, a:
               return {"status": "false", "error": str(a)}
    except requests.exceptions.Timeout, e:
        return {"status": "false", "error": "Timeout"}
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
    return {"status": "true", "error": "none", "userWorkflows": returnJson}


def getWorkflowListWithOncomine(inputJson):
    allWorkflowListResult = getWorkflowList(inputJson)
    if (allWorkflowListResult["status"] != "true"):
        return allWorkflowListResult
    allWorkflowList = allWorkflowListResult["userWorkflows"]
    result = []
    for w in allWorkflowList:
        if (w["OCP_Workflow"] == "true" ):
            result.append(w)
    return {"status": "true", "error": "none", "userWorkflows": result}


def getWorkflowListWithoutOncomine(inputJson):
    allWorkflowListResult = getWorkflowList(inputJson)
    if (allWorkflowListResult["status"] != "true"):
        return allWorkflowListResult
    allWorkflowList = allWorkflowListResult["userWorkflows"]
    result = []
    for w in allWorkflowList:
        if (w["OCP_Workflow"] == "false" ):
            result.append(w)
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

    #curl -ks -H Authorization:rwVcoTeYGfKxItiaWo2lngsV/r0jukG2pLKbZBkAFnlPbjKfPTXLbIhPb47YA9u78 https://xyz.com:443/grws_1_2/data/uploadpath
    url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/uploadpath/"
    cmd="curl -ks    -H Authorization:"+token  +   " -H Version:"+version   +   " "+url
    result = get_httpResponseFromSystemTools(cmd)
    if (result["status"] =="true"):
        return {"status": "true", "error": "none", "userDataUploadPath": result["stdout"]}
    else:
        return result



    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/uploadpath/"
        hdrs = {'Authorization': token}
        resp = requests.get(url, verify=False, headers=hdrs,timeout=30)  #timeout is in seconds
        result = ""
        if resp.status_code == requests.codes.ok:
            result = resp.text
        else:
            #raise Exception ("IR WebService Error Code " + str(resp.status_code))
            raise Exception("IR WebService Error Code " + str(resp.status_code))
    except requests.exceptions.Timeout, e:
        return {"status": "false", "error": "Timeout"}
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


    #curl -ks --request POST -H Authorization:rwVcoTeYGfKxItiaWo2lngsV/r0jukG2pLKbZBkAFnlPbjKfPTXLbIhPb47YA9u78 -H Version:42 https://xyz.com:443/grws_1_2/data/sampleExists?sampleName=xyz
    url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/sampleExists"
    cmd="curl -ks --request POST -H Authorization:"+token  +   " -H Version:"+version   +   " "+url  + "?sampleName="+sampleName
    result = get_httpResponseFromSystemTools(cmd)
    if (result["status"] =="true"):
        if (result["stdout"] =="true"):
            return {"status": "true", "error": "none"}
        else:
            return {"status": "false", "error": "none"}
    else:
        return result


    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/sampleExists/"
        hdrs = {'Authorization': token}
        queryArgs = {"sampleName": sampleName}
        resp = requests.post(url, params=queryArgs, verify=False, headers=hdrs,timeout=30)  #timeout is in seconds
        result = ""
        if resp.status_code == requests.codes.ok:
            result = resp.text
        else:
            #raise Exception ("IR WebService Error Code " + str(resp.status_code))
            raise Exception("IR WebService Error Code " + str(resp.status_code))
    except requests.exceptions.Timeout, e:
        return {"status": "false", "error": "Timeout"}
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
    write_debug_log("getting details  "+ __file__ + "    " + os.path.dirname(__file__))
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
    unSupportedIRVersionsForThisFunction = ['10', '12', '14', '16', '18', '20']
    if version in unSupportedIRVersionsForThisFunction:
        return {"status": "false", "error": "User Details Query not supported for this version of IR " + version,
                "details": {}}

    encodedPassword=base64.b64encode(password)
    #write_debug_log("encoded password is "+encodedPassword)

    result= get_httpResponseFromIRUJavaAsJson("-u " + userId + " -w " + encodedPassword + " -p "+ protocol + " -a " + server + " -x " + port + " -v " + version + " -o userDetails")
    if "status" in result:
        if (result["status"] == "true") :
            if "json" in result:
                return {"status": "true", "error": "none", "details": result["json"]}
    #return {"status": "false", "error":  result["error"]}
    return result

    ## to be deleted later, if all goes well... 
    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/getUserDetails/"
        formParams = {"userName": userId, "password": password}
        #hdrs = {'Authorization':token}
        #resp = requests.post(url,data=formParams,verify=False, headers=hdrs)
        resp = requests.post(url, verify=False, data=formParams, timeout=4)
        result = {}
        if resp.status_code == requests.codes.ok:
            #result = json.loads(resp.text)
            result = resp.json()
        else:
            #raise Exception("IR WebService Error Code " + str(resp.status_code))
            return {"status": "false", "error": "IonReporter Error Status " + str(resp.status_code)}
    except requests.exceptions.ConnectionError, e:
        protocol_error = False
        if "BadStatusLine" in str(e):
            protocol_error = True
        return {"status": "false", "error": "Connection", "protocol_error" : protocol_error}
    except requests.exceptions.Timeout, e:
        return {"status": "false", "error": "Timeout"}
    except requests.exceptions.HTTPError, e:
        return {"status": "false", "error": str(e)}
    except requests.exceptions.RequestException, e:
        return {"status": "false", "error": str(e)}
    except ValueError, e:                                 #json conversion error. just send text as such, even if empty
        return {"status": "false", "error": resp.text}
    except Exception, e:
        return {"status": "false", "error": str(e)}
    if isinstance(result, basestring):
        return {"status": "false", "error": result}
    if "status" in result:
        if result["status"] == "false":
            if "error" in result:
                return {"status": "false", "error": result["error"]}
            else :
                return {"status": "false", "error": "unknown error in getting user info"}
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

    #write_debug_log(userInput)

    if version == "40":
        return validateUserInput_4_0(inputJson)
    elif version == "42":
        return validateUserInput_4_2(inputJson)
    elif version == "44":
        return validateUserInput_4_4(inputJson)
    elif version == "46":
        return validateUserInput_4_6(inputJson)
    elif version == "50":
        return validateUserInput_5_0(inputJson)


def validateUserInput_4_0(inputJson):
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
    userInputInfo = userInput["userInputInfo"]
    validationResults = []
    # create a hash currentlyAvailableWorkflows with workflow name as key and value as a hash of all props of workflows from column-map
    currentlyAvaliableWorkflows={}
    for cmap in currentRules["column-map"]:
       currentlyAvaliableWorkflows[cmap["Workflow"]]=cmap
    # create a hash orderedColumns with column order number as key and value as a hash of all properties of each column from columns
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


    ###################################### This is a mock logic. This is not the real validation code. This is for test only 
    ###################################### This can be enabled or disabled using the control variable just below.
    mockLogic = 0
    if mockLogic == 1:
        #for 4.0, for now, return validation results based on a mock logic .
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
    ######################################
    ######################################



    setidHash={}
    rowErrors={}
    rowWarnings={}
    uniqueSamples={}
    analysisCost={}
    analysisCost["workflowCosts"]=[]

    requiresVariantCallerPlugin = False
    row = 1
    for uip in userInputInfo:
        # make a row number if not provided, else use whats provided as rownumber.
        if "row" not in uip:
           uip["row"]=str(row)
        rowStr = uip["row"]
        # register such a row in the error bucket  and warning buckets.. basically create two holder arrays in those buckets
        if  rowStr not in rowErrors:
           rowErrors[rowStr]=[]
        if  rowStr not in rowWarnings:
           rowWarnings[rowStr]=[]

        # some known key translations on the uip, before uip can be used for validations
        if  "setid" in uip :
            uip["SetID"]= uip["setid"]
        if  "RelationshipType" not in uip :
            if  "Relation" in uip :
                uip["RelationshipType"]= uip["Relation"]
            if  "RelationRole" in uip :
                uip["Relation"]= uip["RelationRole"]
        if uip["Workflow"] == "Upload Only":
            uip["Workflow"] = ""
        if  uip["Workflow"] != "":
            if uip["Workflow"] in currentlyAvaliableWorkflows:
                uip["ApplicationType"] = currentlyAvaliableWorkflows[uip["Workflow"]]["ApplicationType"]
            else:
                uip["ApplicationType"] = "unknown"
        if  "nucleotideType" in uip :
            uip["NucleotideType"] = uip["nucleotideType"]
        if  "NucleotideType" not in uip :
            uip["NucleotideType"] = ""

        if  uip["NucleotideType"] == "RNA":
            msg="NucleotideType "+ uip["NucleotideType"] + " is not supported for IR 4.0 user accounts"
            inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
            continue

        # all given sampleNames should be unique   TBD Jose   this requirement is going away.. need to safely remove this part. First, IRU plugin should be corrected before correcting this rule.
        if uip["sample"] not in uniqueSamples:
            uniqueSamples[uip["sample"]] = rowStr
        else:
            existingRowStr= uniqueSamples[uip["sample"]]
            msg="sample name "+uip["sample"] + " in row "+ rowStr+" is also in row "+existingRowStr+". Please change the sample name"
            inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)


        # if workflow is empty then dont validate and dont include this row in setid for further validations.
        if uip["Workflow"] == "":
            continue

        # see whether variant Caller plugin is required or not.
        if  (   ("ApplicationType" in uip)  and  (uip["ApplicationType"] == "Annotation")   ) :
            requiresVariantCallerPlugin = True


        # if setid is empty or it starts with underscore , then dont validate and dont include this row in setid hash for further validations.
        if (   (uip["SetID"].startswith("_")) or   (uip["SetID"]=="")  ):
            msg ="SetID in row("+ rowStr+") should not be empty or start with an underscore character. Please update the SetID."
            inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
            continue
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
                msg="Selected workflow "+ uip["Workflow"] + " does not match a previous sample with the same SetID, with workflow "+ expectedWorkflow +" in row "+ previousRow+ ". Either change this workflow to match the previous workflow selection for the this SetID, or change the SetiD to a new value if you intend this sample to be used in a different IR analysis."
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
        setidHash[setid]["records"].append(uip)



        # check if workflow is still active.
        if uip["Workflow"] not in currentlyAvaliableWorkflows:
            msg="selected workflow "+ uip["Workflow"] + " is not available for this IR user account at this time"
            inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
            continue

        # check if sample already exists on IR at this time and give a warning..
        inputJson["sampleName"]= uip["sample"]
        sampleExistsCallResults = sampleExistsOnIR(inputJson)
        if sampleExistsCallResults.get("error") != "":
            if sampleExistsCallResults.get("status") == "true":
                msg="sample name "+ uip["sample"] + " already exists in Ion Reporter "
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)

        # check the rules.. the results of the check goes into the hashes provided as arguments.
        validateAllRulesOnRecord_4_0(currentRules["restrictionRules"], uip, setidHash, rowErrors, rowWarnings)

        row = row + 1


    # after validations of basic rules look for errors all role requirements, uniqueness in roles, excess number of
    # roles, insufficient number of roles, etc.
    for setid in setidHash:
        # first check all the required roles are there in the corresponding records
        rowsLooked = ""
        if "validRelationRoles" in setidHash[setid]:
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
                    msg="For workflow " + setidHash[setid]["firstWorkflow"] +", a required RelationRole "+ validRole + " is not found. Please check row(s) " + rowsLooked
                    inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)

            # check the number of records expected and number of records provided.
            sizeOfRequiredRoles = len(setidHash[setid]["validRelationRoles"])
            sizeOfAvailableRoles = len(setidHash[setid]["records"])
            if (sizeOfAvailableRoles > sizeOfRequiredRoles):
                msg="For workflow " + setidHash[setid]["firstWorkflow"] +", more than the required number of RelationRoles are found. Expected number of roles is "+ str(sizeOfRequiredRoles) + ". Please check row(s) " + rowsLooked
                inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)


        # calculate the cost of the analysis
        cost={}
        cost["row"]=setidHash[setid]["firstRecordRow"]
        cost["workflow"]=setidHash[setid]["firstWorkflow"]
        cost["cost"]="50.00"     # TBD  actually, get it from IR. There are now APIs available.. TS is not yet popping this to user before plan submission.
        analysisCost["workflowCosts"].append(cost)

    analysisCost["totalCost"]="2739.99"   # TBD need to have a few lines to add the individual cost... TS is not yet popping this to user before plan submission.
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
    #advices["onTooManyErrors"]= "There are errors on this page. If you only want to upload samples to Ion Reporter and not perform an Ion Reporter analysis at this time, you do not need to select a Workflow. When you are ready to launch an Ion Reporter analysis, you must log into Ion Reporter and select the samples to analyze."
    advices["onTooManyErrors"]= "<html> <body> There are errors on this page. To remove them, either: <br> &nbsp;&nbsp;1) Change the Workflow to &quot;Upload Only&quot; for affected samples. Analyses will not be automatically launched in Ion Reporter.<br> &nbsp;&nbsp;2) Correct all errors to ensure autolaunch of correct analyses in Ion Reporter.<br> Visit the Torrent Suite documentation at <a href=/ion-docs/Home.html > docs </a>  for examples. </body> </html>"

    # forumulate a few conditions, which may be required beyond this validation.
    conditions={}
    conditions["requiresVariantCallerPlugin"]=requiresVariantCallerPlugin


    #true/false return code is reserved for error in executing the functionality itself, and not the condition of the results itself.
    # say if there are networking errors, talking to IR, etc will return false. otherwise, return pure results. The results internally
    # may contain errors, which is to be interpretted by the caller. If there are other helpful error info regarding the results itsef,
    # then additional variables may be used to reflect metadata about the results. the status/error flags may be used to reflect the
    # status of the call itself.
    #if (foundAtLeastOneError == 1):
    #    return {"status": "false", "error": "none", "validationResults": validationResults, "cost":analysisCost}
    #else:
    #    return {"status": "true", "error": "none", "validationResults": validationResults, "cost":analysisCost}
    return {"status": "true", "error": "none", "validationResults": validationResults, "cost":analysisCost, "advices": advices,
           "conditions": conditions
           }

    """
    # if we want to implement this logic in grws, then here is the interface code.  But currently it is not yet implemented there.
    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/TSUserInputValidate/"
        hdrs = {'Authorization': token}
        resp = requests.post(url, verify=False, headers=hdrs,timeout=30)  #timeout is in seconds
        result = {}
        if resp.status_code == requests.codes.ok:
            result = json.loads(resp.text)
        else:
            #raise Exception ("IR WebService Error Code " + str(resp.status_code))
            raise Exception("IR WebService Error Code " + str(resp.status_code))
    except requests.exceptions.Timeout, e:
        return {"status": "false", "error": "Timeout"}
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

def validateAllRulesOnRecord_4_0(rules, uip, setidHash, rowErrors, rowWarnings):
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
                            #msg="Incorrect value \"" + uip[kValid] + "\" found for " + kValid + " When "+ kFor + " is \"" + vFor +"\"   rule # "+ ruleNum
                            msg="Incorrect value \"" + uip[kValid] + "\" found for " + kValid + " When "+ kFor + " is \"" + vFor +"\"."
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
                            #msg="Incorrect value \"" + uip[kInvalid] + "\" found for " + kInvalid + " When "+ kFor + " is \"" + vFor +"\"   rule # "+ ruleNum
                            msg="Incorrect value \"" + uip[kInvalid] + "\" found for " + kInvalid + " When "+ kFor + " is \"" + vFor +"\"."
                            inputValidationErrorHandle(row, validationType, msg, rowErrors, rowWarnings)
            elif "Disabled" in rule:
                pass
            else:
                 msg="INTERNAL ERROR ON RULE # "+ruleNum+"   Incompatible validation rules for this version of IRU. \"For\" specified without a \"Valid\" or \"Invalid\" tag."
                 inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)
        else:
            msg="INTERNAL ERROR ON RULE # "+ruleNum+"   Incompatible validation rules for this version of IRU. No action provided on this rule."
            inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)


def validateUserInput_4_2(inputJson):
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
    userInputInfo = userInput["userInputInfo"]
    validationResults = []
    # create a hash currentlyAvailableWorkflows with workflow name as key and value as a hash of all props of workflows from column-map
    currentlyAvaliableWorkflows={}
    for cmap in currentRules["column-map"]:
       currentlyAvaliableWorkflows[cmap["Workflow"]]=cmap
    # create a hash orderedColumns with column order number as key and value as a hash of all properties of each column from columns
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


    ###################################### This is a mock logic. This is not the real validation code. This is for test only 
    ###################################### This can be enabled or disabled using the control variable just below.
    mockLogic = 0
    if mockLogic == 1:
        #for 4.0, for now, return validation results based on a mock logic .
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
    ######################################
    ######################################



    setidHash={}
    rowErrors={}
    rowWarnings={}
    uniqueSamples={}
    analysisCost={}
    analysisCost["workflowCosts"]=[]

    requiresVariantCallerPlugin = False
    row = 1
    for uip in userInputInfo:
        # make a row number if not provided, else use whats provided as rownumber.
        if "row" not in uip:
           uip["row"]=str(row)
        rowStr = uip["row"]
        # register such a row in the error bucket  and warning buckets.. basically create two holder arrays in those buckets
        if  rowStr not in rowErrors:
           rowErrors[rowStr]=[]
        if  rowStr not in rowWarnings:
           rowWarnings[rowStr]=[]

        # some known key translations on the uip, before uip can be used for validations
        if  "setid" in uip :
            uip["SetID"] = uip["setid"]
        if  "RelationshipType" not in uip :
            if  "Relation" in uip :
                uip["RelationshipType"] = uip["Relation"]
            if  "RelationRole" in uip :
                uip["Relation"] = uip["RelationRole"]
        if uip["Workflow"] == "Upload Only":
            uip["Workflow"] = ""
        if uip["Workflow"] !="":
            if uip["Workflow"] in currentlyAvaliableWorkflows:
                #uip["ApplicationType"] = currentlyAvaliableWorkflows[uip["Workflow"]]["ApplicationType"]
                #uip["DNA_RNA_Workflow"] = currentlyAvaliableWorkflows[uip["Workflow"]]["DNA_RNA_Workflow"]
                #uip["OCP_Workflow"] = currentlyAvaliableWorkflows[uip["Workflow"]]["OCP_Workflow"]

                # another temporary check which is not required if all the parameters of workflow  were  properly handed off from TS 
                if "RelationshipType" not in uip :
                    msg="INTERNAL ERROR:  For selected workflow "+ uip["Workflow"] + ", an internal key  RelationshipType is missing for row " + rowStr
                    inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
                    continue

                #bring in all the workflow parameter so far available, into the uip hash.
                for k in  currentlyAvaliableWorkflows[uip["Workflow"]] : 
                    uip[k] = currentlyAvaliableWorkflows[uip["Workflow"]][k]
            else:
                uip["ApplicationType"] = "unknown"
                msg="selected workflow "+ uip["Workflow"] + " is not available for this IR user account at this time"
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
                continue
        if  "nucleotideType" in uip :
            uip["NucleotideType"] = uip["nucleotideType"]
        if  "NucleotideType" not in uip :
            uip["NucleotideType"] = ""
        if  "cellularityPct" in uip :
            uip["CellularityPct"] = uip["cellularityPct"]
        if  "CellularityPct" not in uip :
            uip["CellularityPct"] = ""
        if  "cancerType" in uip :
            uip["CancerType"] = uip["cancerType"]
        if  "CancerType" not in uip :
            uip["CancerType"] = ""


        # all given sampleNames should be unique   TBD Jose   this requirement is going away.. need to safely remove this part. First, IRU plugin should be corrected before correcting this rule.
        if uip["sample"] not in uniqueSamples:
            uniqueSamples[uip["sample"]] = uip  #later if its a three level then make it into an array of uips
        else:
            duplicateSamplesExists = True
            theOtherUip = uniqueSamples[uip["sample"]]
            theOtherRowStr = theOtherUip["row"]
            theOtherSetid = theOtherUip["setid"]
            theOtherDNA_RNA_Workflow = ""
            if "DNA_RNA_Workflow" in theOtherUip:
			    theOtherDNA_RNA_Workflow = theOtherUip["DNA_RNA_Workflow"]
            thisDNA_RNA_Workflow = ""
            if "DNA_RNA_Workflow" in uip:
			    thisDNA_RNA_Workflow = uip["DNA_RNA_Workflow"]
            theOtherNucleotideType = ""
            if "NucleotideType" in theOtherUip:
                theOtherNucleotideType = theOtherUip["NucleotideType"]
            thisNucleotideType = ""
            if "NucleotideType" in uip:
                thisNucleotideType = uip["NucleotideType"]
            # if the rows are for DNA_RNA workflow, then dont complain .. just pass it along..
            #debug print  uip["row"] +" == " + theOtherRowStr + " samplename similarity  " + uip["DNA_RNA_Workflow"] + " == "+ theOtherDNA_RNA_Workflow
            if (       ((uip["Workflow"]=="Upload Only")or(uip["Workflow"] == "")) and (thisNucleotideType != theOtherNucleotideType)     ):
                duplicateSamplesExists = False
            if (       (uip["setid"] == theOtherSetid) and (thisDNA_RNA_Workflow == theOtherDNA_RNA_Workflow ) and (thisDNA_RNA_Workflow == "DNA_RNA")      ):
                duplicateSamplesExists = False
            if duplicateSamplesExists :
                msg ="sample name "+uip["sample"] + " in row "+ rowStr+" is also in row "+theOtherRowStr+". Please change the sample name"
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
            # else dont flag an error

        # if workflow is empty then dont validate and dont include this row in setid for further validations.
        if uip["Workflow"] =="":
            continue

        # see whether variant Caller plugin is required or not.
        if  (   ("ApplicationType" in uip)  and  (uip["ApplicationType"] == "Annotation")   ) :
            requiresVariantCallerPlugin = True


        # if setid is empty or it starts with underscore , then dont validate and dont include this row in setid hash for further validations.
        if (   (uip["SetID"].startswith("_")) or   (uip["SetID"]=="")  ):
            msg ="SetID in row("+ rowStr+") should not be empty or start with an underscore character. Please update the SetID."
            inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
            continue
        # save the workflow information of the record on the setID.. also check workflow mismatch with previous row of the same setid
        setid = uip["SetID"]
        if  setid not in setidHash:
            setidHash[setid] = {}
            setidHash[setid]["records"] =[]
            setidHash[setid]["firstRecordRow"]=uip["row"]
            setidHash[setid]["firstWorkflow"]=uip["Workflow"]
            setidHash[setid]["firstRelationshipType"]=uip["RelationshipType"]
            setidHash[setid]["firstRecordDNA_RNA"]=uip["DNA_RNA_Workflow"]
        else:
            previousRow = setidHash[setid]["firstRecordRow"]
            expectedWorkflow = setidHash[setid]["firstWorkflow"]
            expectedRelationshipType = setidHash[setid]["firstRelationshipType"]
            #print  uip["row"] +" == " + previousRow + " set id similarity  " + uip["RelationshipType"] + " == "+ expectedRelationshipType
            if expectedWorkflow != uip["Workflow"]:
                msg="Selected workflow "+ uip["Workflow"] + " does not match a previous sample with the same SetID, with workflow "+ expectedWorkflow +" in row "+ previousRow+ ". Either change this workflow to match the previous workflow selection for the this SetID, or change the SetiD to a new value if you intend this sample to be used in a different IR analysis."
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
            elif expectedRelationshipType != uip["RelationshipType"]:
                #print  "error on " + uip["row"] +" == " + previousRow + " set id similarity  " + uip["RelationshipType"] + " == "+ expectedRelationshipType
                msg="INTERNAL ERROR:  RelationshipType "+ uip["RelationshipType"] + " of the selected workflow, does not match a previous sample with the same SetID, with RelationshipType "+ expectedRelationshipType +" in row "+ previousRow+ "."
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
        setidHash[setid]["records"].append(uip)

        # check if sample already exists on IR at this time and give a warning..
        inputJson["sampleName"] = uip["sample"]
        if uip["sample"] not in uniqueSamples:    # no need to repeat if the check has been done for the same sample name on an earlier row.
            sampleExistsCallResults = sampleExistsOnIR(inputJson)
            if sampleExistsCallResults.get("error") != "":
                if sampleExistsCallResults.get("status") == "true":
                    msg="sample name "+ uip["sample"] + " already exists in Ion Reporter "
                    inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)

        # check all the generic rules for this uip .. the results of the check goes into the hashes provided as arguments.
        validateAllRulesOnRecord_4_2(currentRules["restrictionRules"], uip, setidHash, rowErrors, rowWarnings)

        row = row + 1


    # after validations of basic rules look for errors all role requirements, uniqueness in roles, excess number of
    # roles, insufficient number of roles, etc.
    for setid in setidHash:
        # first check all the required roles are there in the given set of records of the set
        rowsLooked = ""
        if "validRelationRoles" in setidHash[setid]:
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
                    msg="For workflow " + setidHash[setid]["firstWorkflow"] +", a required RelationRole "+ validRole + " is not found. "
                    if   rowsLooked != "" :
                        if rowsLooked.find(",") != -1  :
                            msg = msg + "Please check the rows " + rowsLooked
                        else:
                            msg = msg + "Please check the row " + rowsLooked
                    inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)
            # check if any extra roles exists.  Given that the above test exists for lack of roles, it is sufficient if we 
            # verify the total number of roles expected and number of records got, for this setid. If there is a mismatch,
            # it means there are more than the number of roles required.
            #    Use the value of the rowsLooked,  populated from the above loop.
            sizeOfRequiredRoles = len(setidHash[setid]["validRelationRoles"])
            numRecordsForThisSetId = len(setidHash[setid]["records"])
            if (numRecordsForThisSetId > sizeOfRequiredRoles):
                complainAboutTooManyRoles = True

                if setidHash[setid]["firstRecordDNA_RNA"] == "DNA_RNA":
                    complainAboutTooManyRoles = False

                if complainAboutTooManyRoles:
                    msg="For workflow " + setidHash[setid]["firstWorkflow"] + ", more than the required number of RelationRoles is found. Expected number of roles is " + str(sizeOfRequiredRoles) + ". "
                    if   rowsLooked != "" :
                        if rowsLooked.find(",") != -1  :
                            msg = msg + "Please check the rows " + rowsLooked
                        else:
                            msg = msg + "Please check the row " + rowsLooked
                    inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)

        ##
        # validate the nucleotidetypes, similar to the roles.
        # first check all the required nucleotides are there in the given set of records of the set
        #    Use the value of the rowsLooked,  populated from the above loop.
        if (   (setidHash[setid]["firstRecordDNA_RNA"] == "DNA_RNA")  or  (setidHash[setid]["firstRecordDNA_RNA"] == "RNA")   ): 
            for validNucloetide in setidHash[setid]["validNucleotideTypes"]:
                foundNucleotide=0
                for record in setidHash[setid]["records"]:
                    if validNucloetide == record["NucleotideType"]:   #or NucleotideType
                        foundNucleotide = 1
                if foundNucleotide == 0 :
                    msg="For workflow " + setidHash[setid]["firstWorkflow"] +", a required NucleotideType "+ validNucloetide + " is not found. "
                    if   rowsLooked != "" :
                        if rowsLooked.find(",") != -1  :
                            msg = msg + "Please check the rows " + rowsLooked
                        else:
                            msg = msg + "Please check the row " + rowsLooked
                    inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)
            # check if any extra nucleotides exists.  Given that the above test exists for missing nucleotides, it is sufficient if we 
            # verify the total number of nucleotides expected and number of records got, for this setid. If there is a mismatch,
            # it means there are more than the number of nucleotides required.
            #    Use the value of the rowsLooked,  populated from the above loop.
            sizeOfRequiredNucleotides = len(setidHash[setid]["validNucleotideTypes"])
            #numRecordsForThisSetId = len(setidHash[setid]["records"])   #already done as part of roles check
            if (numRecordsForThisSetId > sizeOfRequiredNucleotides):
                msg="For workflow " + setidHash[setid]["firstWorkflow"] + ", more than the required number of Nucleotides is found. Expected number of Nucleotides is " + str(sizeOfRequiredNucleotides) + ". "
                if   rowsLooked != "" :
                    if rowsLooked.find(",") != -1  :
                        msg = msg + "Please check the rows " + rowsLooked
                    else:
                        msg = msg + "Please check the row " + rowsLooked
                inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)


        # calculate the cost of the analysis
        cost={}
        cost["row"]=setidHash[setid]["firstRecordRow"]
        cost["workflow"]=setidHash[setid]["firstWorkflow"]
        cost["cost"]="50.00"     # TBD  actually, get it from IR. There are now APIs available.. TS is not yet popping this to user before plan submission.
        analysisCost["workflowCosts"].append(cost)

    analysisCost["totalCost"]="2739.99"   # TBD need to have a few lines to add the individual cost... TS is not yet popping this to user before plan submission.
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
    #advices["onTooManyErrors"]= "There are errors on this page. If you only want to upload samples to Ion Reporter and not perform an Ion Reporter analysis at this time, you do not need to select a Workflow. When you are ready to launch an Ion Reporter analysis, you must log into Ion Reporter and select the samples to analyze."
    advices["onTooManyErrors"]= "<html> <body> There are errors on this page. To remove them, either: <br> &nbsp;&nbsp;1) Change the Workflow to &quot;Upload Only&quot; for affected samples. Analyses will not be automatically launched in Ion Reporter.<br> &nbsp;&nbsp;2) Correct all errors to ensure autolaunch of correct analyses in Ion Reporter.<br> Visit the Torrent Suite documentation at <a href=/ion-docs/Home.html > docs </a>  for examples. </body> </html>"

    # forumulate a few conditions, which may be required beyond this validation.
    conditions={}
    conditions["requiresVariantCallerPlugin"]=requiresVariantCallerPlugin


    #true/false return code is reserved for error in executing the functionality itself, and not the condition of the results itself.
    # say if there are networking errors, talking to IR, etc will return false. otherwise, return pure results. The results internally
    # may contain errors, which is to be interpretted by the caller. If there are other helpful error info regarding the results itsef,
    # then additional variables may be used to reflect metadata about the results. the status/error flags may be used to reflect the
    # status of the call itself.
    #if (foundAtLeastOneError == 1):
    #    return {"status": "false", "error": "none", "validationResults": validationResults, "cost":analysisCost}
    #else:
    #    return {"status": "true", "error": "none", "validationResults": validationResults, "cost":analysisCost}
    return {"status": "true", "error": "none", "validationResults": validationResults, "cost":analysisCost, "advices": advices,
	        "conditions": conditions
           }

    """
    # if we want to implement this logic in grws, then here is the interface code.  But currently it is not yet implemented there.
    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/TSUserInputValidate/"
        hdrs = {'Authorization': token}
        resp = requests.post(url, verify=False, headers=hdrs,timeout=30)  #timeout is in seconds
        result = {}
        if resp.status_code == requests.codes.ok:
            result = json.loads(resp.text)
        else:
            #raise Exception ("IR WebService Error Code " + str(resp.status_code))
            raise Exception("IR WebService Error Code " + str(resp.status_code))
    except requests.exceptions.Timeout, e:
        return {"status": "false", "error": "Timeout"}
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

def validateAllRulesOnRecord_4_2(rules, uip, setidHash, rowErrors, rowWarnings):
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
                #msg="INTERNAL ERROR ON RULE # "+ruleNum+"   Incompatible validation rules for this version of TSS. No such \"For\" field \"" + rule["For"]["Name"] + "\""
                #inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)
                continue
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
                            #msg="Incorrect value \"" + uip[kValid] + "\" found for " + kValid + " When "+ kFor + " is \"" + vFor +"\"   rule # "+ ruleNum
                            msg="Incorrect value \"" + uip[kValid] + "\" found for " + kValid + " When "+ kFor + " is \"" + vFor +"\"."
                            inputValidationErrorHandle(row, validationType, msg, rowErrors, rowWarnings)
                        # a small hardcoded update into the setidHash for later evaluation of the role uniqueness
                        if kValid == "Relation":
                            if setid  in setidHash:
                                if "validRelationRoles" not in setidHash[setid]:
                                    #print  "saving   row " +row + "  setid " + setid + "  kfor " +kFor  + " vFor "+ vFor + "  kValid "+ kValid
                                    setidHash[setid]["validRelationRoles"] = vValid   # this is actually roles
                        # another small hardcoded update into the setidHash for later evaluation of the nucleotideType  uniqueness
                        if kValid == "NucleotideType":
                            if setid  in setidHash:
                                if "validNucleotideTypes" not in setidHash[setid]:
                                    #print  "saving   row " +row + "  setid " + setid + "  kfor " +kFor  + " vFor "+ vFor + "  kValid "+ kValid
                                    setidHash[setid]["validNucleotideTypes"] = vValid   # this is actually list of valid nucleotideTypes
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
                            #msg="Incorrect value \"" + uip[kInvalid] + "\" found for " + kInvalid + " When "+ kFor + " is \"" + vFor +"\"   rule # "+ ruleNum
                            msg="Incorrect value \"" + uip[kInvalid] + "\" found for " + kInvalid + " When "+ kFor + " is \"" + vFor +"\"."
                            inputValidationErrorHandle(row, validationType, msg, rowErrors, rowWarnings)
            elif "NonEmpty" in rule:
                kFor=rule["For"]["Name"]
                vFor=rule["For"]["Value"]
                kNonEmpty=rule["NonEmpty"]["Name"]
                print  "non empty validating   kfor " +kFor  + " vFor "+ vFor + "  kNonEmpty "+ kNonEmpty
                #if kFor not in uip :
                #    print  "kFor not in uip   "  + " kFor "+ kFor 
                #    continue
                if uip[kFor] == vFor :
                    if (   (kNonEmpty not in uip)   or  (uip[kNonEmpty] == "")   ):
                        msg="Empty value found for " + kNonEmpty + " When "+ kFor + " is \"" + vFor +"\"."
                        inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)
            elif "Disabled" in rule:
                pass
            else:
                 msg="INTERNAL ERROR ON RULE # "+ruleNum+"   Incompatible validation rules for this version of IRU. \"For\" specified without a \"Valid\" or \"Invalid\" tag."
                 inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)
        else:
            msg="INTERNAL ERROR ON RULE # "+ruleNum+"   Incompatible validation rules for this version of IRU. No action provided on this rule."
            inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)







def validateUserInput_4_4(inputJson):
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
    userInputInfo = userInput["userInputInfo"]
    validationResults = []
    # create a hash currentlyAvailableWorkflows with workflow name as key and value as a hash of all props of workflows from column-map
    currentlyAvaliableWorkflows={}
    for cmap in currentRules["column-map"]:
       currentlyAvaliableWorkflows[cmap["Workflow"]]=cmap
    # create a hash orderedColumns with column order number as key and value as a hash of all properties of each column from columns
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


    ###################################### This is a mock logic. This is not the real validation code. This is for test only 
    ###################################### This can be enabled or disabled using the control variable just below.
    mockLogic = 0
    if mockLogic == 1:
        #for 4.0, for now, return validation results based on a mock logic .
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
    ######################################
    ######################################



    setidHash={}
    rowErrors={}
    rowWarnings={}
    uniqueSamples={}
    analysisCost={}
    analysisCost["workflowCosts"]=[]

    requiresVariantCallerPlugin = False
    row = 1
    for uip in userInputInfo:
        # make a row number if not provided, else use whats provided as rownumber.
        if "row" not in uip:
           uip["row"]=str(row)
        rowStr = uip["row"]
        # register such a row in the error bucket  and warning buckets.. basically create two holder arrays in those buckets
        if  rowStr not in rowErrors:
           rowErrors[rowStr]=[]
        if  rowStr not in rowWarnings:
           rowWarnings[rowStr]=[]

        # some known key translations on the uip, before uip can be used for validations
        if  "setid" in uip :
            uip["SetID"] = uip["setid"]
        if  "RelationshipType" not in uip :
            if  "Relation" in uip :
                uip["RelationshipType"] = uip["Relation"]
            if  "RelationRole" in uip :
                uip["Relation"] = uip["RelationRole"]
        if uip["Workflow"] == "Upload Only":
            uip["Workflow"] = ""
        if uip["Workflow"] !="":
            if uip["Workflow"] in currentlyAvaliableWorkflows:
                #uip["ApplicationType"] = currentlyAvaliableWorkflows[uip["Workflow"]]["ApplicationType"]
                #uip["DNA_RNA_Workflow"] = currentlyAvaliableWorkflows[uip["Workflow"]]["DNA_RNA_Workflow"]
                #uip["OCP_Workflow"] = currentlyAvaliableWorkflows[uip["Workflow"]]["OCP_Workflow"]

                # another temporary check which is not required if all the parameters of workflow  were  properly handed off from TS 
                if "RelationshipType" not in uip :
                    msg="INTERNAL ERROR:  For selected workflow "+ uip["Workflow"] + ", an internal key  RelationshipType is missing for row " + rowStr
                    inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
                    continue

                #bring in all the workflow parameter so far available, into the uip hash.
                for k in  currentlyAvaliableWorkflows[uip["Workflow"]] : 
                    uip[k] = currentlyAvaliableWorkflows[uip["Workflow"]][k]
            else:
                uip["ApplicationType"] = "unknown"
                msg="selected workflow "+ uip["Workflow"] + " is not available for this IR user account at this time"
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
                continue
        if  "nucleotideType" in uip :
            uip["NucleotideType"] = uip["nucleotideType"]
        if  "NucleotideType" not in uip :
            uip["NucleotideType"] = ""
        if  "cellularityPct" in uip :
            uip["CellularityPct"] = uip["cellularityPct"]
        if  "CellularityPct" not in uip :
            uip["CellularityPct"] = ""
        if  "cancerType" in uip :
            uip["CancerType"] = uip["cancerType"]
        if  "CancerType" not in uip :
            uip["CancerType"] = ""


        # all given sampleNames should be unique   TBD Jose   this requirement is going away.. need to safely remove this part. First, IRU plugin should be corrected before correcting this rule.
        if uip["sample"] not in uniqueSamples:
            uniqueSamples[uip["sample"]] = uip  #later if its a three level then make it into an array of uips
        else:
            duplicateSamplesExists = True
            theOtherUip = uniqueSamples[uip["sample"]]
            theOtherRowStr = theOtherUip["row"]
            theOtherSetid = theOtherUip["setid"]
            theOtherDNA_RNA_Workflow = ""
            if "DNA_RNA_Workflow" in theOtherUip:
			    theOtherDNA_RNA_Workflow = theOtherUip["DNA_RNA_Workflow"]
            thisDNA_RNA_Workflow = ""
            if "DNA_RNA_Workflow" in uip:
			    thisDNA_RNA_Workflow = uip["DNA_RNA_Workflow"]
            theOtherNucleotideType = ""
            if "NucleotideType" in theOtherUip:
                theOtherNucleotideType = theOtherUip["NucleotideType"]
            thisNucleotideType = ""
            if "NucleotideType" in uip:
                thisNucleotideType = uip["NucleotideType"]
            # if the rows are for DNA_RNA workflow, then dont complain .. just pass it along..
            #debug print  uip["row"] +" == " + theOtherRowStr + " samplename similarity  " + uip["DNA_RNA_Workflow"] + " == "+ theOtherDNA_RNA_Workflow
            if (       ((uip["Workflow"]=="Upload Only")or(uip["Workflow"] == "")) and (thisNucleotideType != theOtherNucleotideType)     ):
                duplicateSamplesExists = False
            if (       (uip["setid"] == theOtherSetid) and (thisDNA_RNA_Workflow == theOtherDNA_RNA_Workflow ) and (thisDNA_RNA_Workflow == "DNA_RNA")      ):
                duplicateSamplesExists = False
            if duplicateSamplesExists :
                msg ="sample name "+uip["sample"] + " in row "+ rowStr+" is also in row "+theOtherRowStr+". Please change the sample name"
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
            # else dont flag an error

        # if workflow is empty then dont validate and dont include this row in setid for further validations.
        if uip["Workflow"] =="":
            continue

        # see whether variant Caller plugin is required or not.
        if  (   ("ApplicationType" in uip)  and  (uip["ApplicationType"] == "Annotation")   ) :
            requiresVariantCallerPlugin = True


        # if setid is empty or it starts with underscore , then dont validate and dont include this row in setid hash for further validations.
        if (   (uip["SetID"].startswith("_")) or   (uip["SetID"]=="")  ):
            msg ="SetID in row("+ rowStr+") should not be empty or start with an underscore character. Please update the SetID."
            inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
            continue
        # save the workflow information of the record on the setID.. also check workflow mismatch with previous row of the same setid
        setid = uip["SetID"]
        if  setid not in setidHash:
            setidHash[setid] = {}
            setidHash[setid]["records"] =[]
            setidHash[setid]["firstRecordRow"]=uip["row"]
            setidHash[setid]["firstWorkflow"]=uip["Workflow"]
            setidHash[setid]["firstRelationshipType"]=uip["RelationshipType"]
            setidHash[setid]["firstRecordDNA_RNA"]=uip["DNA_RNA_Workflow"]
        else:
            previousRow = setidHash[setid]["firstRecordRow"]
            expectedWorkflow = setidHash[setid]["firstWorkflow"]
            expectedRelationshipType = setidHash[setid]["firstRelationshipType"]
            #print  uip["row"] +" == " + previousRow + " set id similarity  " + uip["RelationshipType"] + " == "+ expectedRelationshipType
            if expectedWorkflow != uip["Workflow"]:
                msg="Selected workflow "+ uip["Workflow"] + " does not match a previous sample with the same SetID, with workflow "+ expectedWorkflow +" in row "+ previousRow+ ". Either change this workflow to match the previous workflow selection for the this SetID, or change the SetiD to a new value if you intend this sample to be used in a different IR analysis."
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
            elif expectedRelationshipType != uip["RelationshipType"]:
                #print  "error on " + uip["row"] +" == " + previousRow + " set id similarity  " + uip["RelationshipType"] + " == "+ expectedRelationshipType
                msg="INTERNAL ERROR:  RelationshipType "+ uip["RelationshipType"] + " of the selected workflow, does not match a previous sample with the same SetID, with RelationshipType "+ expectedRelationshipType +" in row "+ previousRow+ "."
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
        setidHash[setid]["records"].append(uip)

        # check if sample already exists on IR at this time and give a warning..
        inputJson["sampleName"] = uip["sample"]
        if uip["sample"] not in uniqueSamples:    # no need to repeat if the check has been done for the same sample name on an earlier row.
            sampleExistsCallResults = sampleExistsOnIR(inputJson)
            if sampleExistsCallResults.get("error") != "":
                if sampleExistsCallResults.get("status") == "true":
                    msg="sample name "+ uip["sample"] + " already exists in Ion Reporter "
                    inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)

        # check all the generic rules for this uip .. the results of the check goes into the hashes provided as arguments.
        validateAllRulesOnRecord_4_4(currentRules["restrictionRules"], uip, setidHash, rowErrors, rowWarnings)

        row = row + 1


    # after validations of basic rules look for errors all role requirements, uniqueness in roles, excess number of
    # roles, insufficient number of roles, etc.
    for setid in setidHash:
        # first check all the required roles are there in the given set of records of the set
        rowsLooked = ""
        if "validRelationRoles" in setidHash[setid]:
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
                    msg="For workflow " + setidHash[setid]["firstWorkflow"] +", a required RelationRole "+ validRole + " is not found. "
                    if   rowsLooked != "" :
                        if rowsLooked.find(",") != -1  :
                            msg = msg + "Please check the rows " + rowsLooked
                        else:
                            msg = msg + "Please check the row " + rowsLooked
                    inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)
            # check if any extra roles exists.  Given that the above test exists for lack of roles, it is sufficient if we 
            # verify the total number of roles expected and number of records got, for this setid. If there is a mismatch,
            # it means there are more than the number of roles required.
            #    Use the value of the rowsLooked,  populated from the above loop.
            sizeOfRequiredRoles = len(setidHash[setid]["validRelationRoles"])
            numRecordsForThisSetId = len(setidHash[setid]["records"])
            if (numRecordsForThisSetId > sizeOfRequiredRoles):
                complainAboutTooManyRoles = True

                if setidHash[setid]["firstRecordDNA_RNA"] == "DNA_RNA":
                    complainAboutTooManyRoles = False

                if complainAboutTooManyRoles:
                    msg="For workflow " + setidHash[setid]["firstWorkflow"] + ", more than the required number of RelationRoles is found. Expected number of roles is " + str(sizeOfRequiredRoles) + ". "
                    if   rowsLooked != "" :
                        if rowsLooked.find(",") != -1  :
                            msg = msg + "Please check the rows " + rowsLooked
                        else:
                            msg = msg + "Please check the row " + rowsLooked
                    inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)

        ##
        # validate the nucleotidetypes, similar to the roles.
        # first check all the required nucleotides are there in the given set of records of the set
        #    Use the value of the rowsLooked,  populated from the above loop.
        if (   (setidHash[setid]["firstRecordDNA_RNA"] == "DNA_RNA")  or  (setidHash[setid]["firstRecordDNA_RNA"] == "RNA")   ): 
            for validNucloetide in setidHash[setid]["validNucleotideTypes"]:
                foundNucleotide=0
                for record in setidHash[setid]["records"]:
                    if validNucloetide == record["NucleotideType"]:   #or NucleotideType
                        foundNucleotide = 1
                if foundNucleotide == 0 :
                    msg="For workflow " + setidHash[setid]["firstWorkflow"] +", a required NucleotideType "+ validNucloetide + " is not found. "
                    if   rowsLooked != "" :
                        if rowsLooked.find(",") != -1  :
                            msg = msg + "Please check the rows " + rowsLooked
                        else:
                            msg = msg + "Please check the row " + rowsLooked
                    inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)
            # check if any extra nucleotides exists.  Given that the above test exists for missing nucleotides, it is sufficient if we 
            # verify the total number of nucleotides expected and number of records got, for this setid. If there is a mismatch,
            # it means there are more than the number of nucleotides required.
            #    Use the value of the rowsLooked,  populated from the above loop.
            sizeOfRequiredNucleotides = len(setidHash[setid]["validNucleotideTypes"])
            #numRecordsForThisSetId = len(setidHash[setid]["records"])   #already done as part of roles check
            if (numRecordsForThisSetId > sizeOfRequiredNucleotides):
                msg="For workflow " + setidHash[setid]["firstWorkflow"] + ", more than the required number of Nucleotides is found. Expected number of Nucleotides is " + str(sizeOfRequiredNucleotides) + ". "
                if   rowsLooked != "" :
                    if rowsLooked.find(",") != -1  :
                        msg = msg + "Please check the rows " + rowsLooked
                    else:
                        msg = msg + "Please check the row " + rowsLooked
                inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)


        # calculate the cost of the analysis
        cost={}
        cost["row"]=setidHash[setid]["firstRecordRow"]
        cost["workflow"]=setidHash[setid]["firstWorkflow"]
        cost["cost"]="50.00"     # TBD  actually, get it from IR. There are now APIs available.. TS is not yet popping this to user before plan submission.
        analysisCost["workflowCosts"].append(cost)

    analysisCost["totalCost"]="2739.99"   # TBD need to have a few lines to add the individual cost... TS is not yet popping this to user before plan submission.
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
    #advices["onTooManyErrors"]= "There are errors on this page. If you only want to upload samples to Ion Reporter and not perform an Ion Reporter analysis at this time, you do not need to select a Workflow. When you are ready to launch an Ion Reporter analysis, you must log into Ion Reporter and select the samples to analyze."
    advices["onTooManyErrors"]= "<html> <body> There are errors on this page. To remove them, either: <br> &nbsp;&nbsp;1) Change the Workflow to &quot;Upload Only&quot; for affected samples. Analyses will not be automatically launched in Ion Reporter.<br> &nbsp;&nbsp;2) Correct all errors to ensure autolaunch of correct analyses in Ion Reporter.<br> Visit the Torrent Suite documentation at <a href=/ion-docs/Home.html > docs </a>  for examples. </body> </html>"

    # forumulate a few conditions, which may be required beyond this validation.
    conditions={}
    conditions["requiresVariantCallerPlugin"]=requiresVariantCallerPlugin

    #true/false return code is reserved for error in executing the functionality itself, and not the condition of the results itself.
    # say if there are networking errors, talking to IR, etc will return false. otherwise, return pure results. The results internally
    # may contain errors, which is to be interpretted by the caller. If there are other helpful error info regarding the results itsef,
    # then additional variables may be used to reflect metadata about the results. the status/error flags may be used to reflect the
    # status of the call itself.
    #if (foundAtLeastOneError == 1):
    #    return {"status": "false", "error": "none", "validationResults": validationResults, "cost":analysisCost}
    #else:
    #    return {"status": "true", "error": "none", "validationResults": validationResults, "cost":analysisCost}
    return {"status": "true", "error": "none", "validationResults": validationResults, "cost":analysisCost, "advices": advices,
            "conditions": conditions
           }

    """
    # if we want to implement this logic in grws, then here is the interface code.  But currently it is not yet implemented there.
    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/TSUserInputValidate/"
        hdrs = {'Authorization': token}
        resp = requests.post(url, verify=False, headers=hdrs,timeout=30)  #timeout is in seconds
        result = {}
        if resp.status_code == requests.codes.ok:
            result = json.loads(resp.text)
        else:
            #raise Exception ("IR WebService Error Code " + str(resp.status_code))
            raise Exception("IR WebService Error Code " + str(resp.status_code))
    except requests.exceptions.Timeout, e:
        return {"status": "false", "error": "Timeout"}
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

def validateAllRulesOnRecord_4_4(rules, uip, setidHash, rowErrors, rowWarnings):
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
                #msg="INTERNAL ERROR ON RULE # "+ruleNum+"   Incompatible validation rules for this version of TSS. No such \"For\" field \"" + rule["For"]["Name"] + "\""
                #inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)
                continue
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
                            #msg="Incorrect value \"" + uip[kValid] + "\" found for " + kValid + " When "+ kFor + " is \"" + vFor +"\"   rule # "+ ruleNum
                            msg="Incorrect value \"" + uip[kValid] + "\" found for " + kValid + " When "+ kFor + " is \"" + vFor +"\"."
                            inputValidationErrorHandle(row, validationType, msg, rowErrors, rowWarnings)
                        # a small hardcoded update into the setidHash for later evaluation of the role uniqueness
                        if kValid == "Relation":
                            if setid  in setidHash:
                                if "validRelationRoles" not in setidHash[setid]:
                                    #print  "saving   row " +row + "  setid " + setid + "  kfor " +kFor  + " vFor "+ vFor + "  kValid "+ kValid
                                    setidHash[setid]["validRelationRoles"] = vValid   # this is actually roles
                        # another small hardcoded update into the setidHash for later evaluation of the nucleotideType  uniqueness
                        if kValid == "NucleotideType":
                            if setid  in setidHash:
                                if "validNucleotideTypes" not in setidHash[setid]:
                                    #print  "saving   row " +row + "  setid " + setid + "  kfor " +kFor  + " vFor "+ vFor + "  kValid "+ kValid
                                    setidHash[setid]["validNucleotideTypes"] = vValid   # this is actually list of valid nucleotideTypes
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
                            #msg="Incorrect value \"" + uip[kInvalid] + "\" found for " + kInvalid + " When "+ kFor + " is \"" + vFor +"\"   rule # "+ ruleNum
                            msg="Incorrect value \"" + uip[kInvalid] + "\" found for " + kInvalid + " When "+ kFor + " is \"" + vFor +"\"."
                            inputValidationErrorHandle(row, validationType, msg, rowErrors, rowWarnings)
            elif "NonEmpty" in rule:
                kFor=rule["For"]["Name"]
                vFor=rule["For"]["Value"]
                kNonEmpty=rule["NonEmpty"]["Name"]
                print  "non empty validating   kfor " +kFor  + " vFor "+ vFor + "  kNonEmpty "+ kNonEmpty
                #if kFor not in uip :
                #    print  "kFor not in uip   "  + " kFor "+ kFor 
                #    continue
                if uip[kFor] == vFor :
                    if (   (kNonEmpty not in uip)   or  (uip[kNonEmpty] == "")   ):
                        msg="Empty value found for " + kNonEmpty + " When "+ kFor + " is \"" + vFor +"\"."
                        inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)
            elif "Disabled" in rule:
                pass
            else:
                 msg="INTERNAL ERROR ON RULE # "+ruleNum+"   Incompatible validation rules for this version of IRU. \"For\" specified without a \"Valid\" or \"Invalid\" tag."
                 inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)
        else:
            msg="INTERNAL ERROR ON RULE # "+ruleNum+"   Incompatible validation rules for this version of IRU. No action provided on this rule."
            inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)


def validateUserInput_4_6(inputJson):
    userInput = inputJson["userInput"]
    irAccountJson = inputJson["irAccount"]

    protocol = irAccountJson["protocol"]
    server = irAccountJson["server"]
    port = irAccountJson["port"]
    token = irAccountJson["token"]
    version = irAccountJson["version"]
    version = version.split("IR")[1]
    grwsPath = "grws_1_2"

    #variantCaller check variables
    requiresVariantCallerPlugin = False
    isVariantCallerSelected = "Unknown"
    if "isVariantCallerSelected" in userInput:
        isVariantCallerSelected = userInput["isVariantCallerSelected"]
    isVariantCallerConfigured = "Unknown"
    if "isVariantCallerConfigured" in userInput:
        isVariantCallerConfigured = userInput["isVariantCallerConfigured"]

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
    userInputInfo = userInput["userInputInfo"]
    validationResults = []
    # create a hash currentlyAvailableWorkflows with workflow name as key and value as a hash of all props of workflows from column-map
    currentlyAvaliableWorkflows={}
    for cmap in currentRules["column-map"]:
       currentlyAvaliableWorkflows[cmap["Workflow"]]=cmap
    # create a hash orderedColumns with column order number as key and value as a hash of all properties of each column from columns
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


    ###################################### This is a mock logic. This is not the real validation code. This is for test only 
    ###################################### This can be enabled or disabled using the control variable just below.
    mockLogic = 0
    if mockLogic == 1:
        #for 4.0, for now, return validation results based on a mock logic .
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
    ######################################
    ######################################



    setidHash={}
    rowErrors={}
    rowWarnings={}
    uniqueSamples={}
    analysisCost={}
    analysisCost["workflowCosts"]=[]

    row = 1
    for uip in userInputInfo:
        # make a row number if not provided, else use whats provided as rownumber.
        if "row" not in uip:
           uip["row"]=str(row)
        rowStr = uip["row"]
        # register such a row in the error bucket  and warning buckets.. basically create two holder arrays in those buckets
        if  rowStr not in rowErrors:
           rowErrors[rowStr]=[]
        if  rowStr not in rowWarnings:
           rowWarnings[rowStr]=[]

        # some known key translations on the uip, before uip can be used for validations
        if  "setid" in uip :
            uip["SetID"] = uip["setid"]
        if  "RelationshipType" not in uip :
            if  "Relation" in uip :
                uip["RelationshipType"] = uip["Relation"]
            if  "RelationRole" in uip :
                uip["Relation"] = uip["RelationRole"]
        if uip["Workflow"] == "Upload Only":
            uip["Workflow"] = ""
        if uip["Workflow"] !="":
            if uip["Workflow"] in currentlyAvaliableWorkflows:
                #uip["ApplicationType"] = currentlyAvaliableWorkflows[uip["Workflow"]]["ApplicationType"]
                #uip["DNA_RNA_Workflow"] = currentlyAvaliableWorkflows[uip["Workflow"]]["DNA_RNA_Workflow"]
                #uip["OCP_Workflow"] = currentlyAvaliableWorkflows[uip["Workflow"]]["OCP_Workflow"]

                # another temporary check which is not required if all the parameters of workflow  were  properly handed off from TS 
                if "RelationshipType" not in uip :
                    msg="INTERNAL ERROR:  For selected workflow "+ uip["Workflow"] + ", an internal key  RelationshipType is missing for row " + rowStr
                    inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
                    continue

                #bring in all the workflow parameter so far available, into the uip hash.
                for k in  currentlyAvaliableWorkflows[uip["Workflow"]] : 
                    uip[k] = currentlyAvaliableWorkflows[uip["Workflow"]][k]
            else:
                uip["ApplicationType"] = "unknown"
                msg="selected workflow "+ uip["Workflow"] + " is not available for this IR user account at this time"
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
                continue
        if  "nucleotideType" in uip :
            uip["NucleotideType"] = uip["nucleotideType"]
        if  "NucleotideType" not in uip :
            uip["NucleotideType"] = ""
        if  "cellularityPct" in uip :
            uip["CellularityPct"] = uip["cellularityPct"]
        if  "CellularityPct" not in uip :
            uip["CellularityPct"] = ""
        if  "cancerType" in uip :
            uip["CancerType"] = uip["cancerType"]
        if  "CancerType" not in uip :
            uip["CancerType"] = ""


        # all given sampleNames should be unique   TBD Jose   this requirement is going away.. need to safely remove this part. First, IRU plugin should be corrected before correcting this rule.
        if uip["sample"] not in uniqueSamples:
            uniqueSamples[uip["sample"]] = uip  #later if its a three level then make it into an array of uips
        else:
            duplicateSamplesExists = True
            theOtherUip = uniqueSamples[uip["sample"]]
            theOtherRowStr = theOtherUip["row"]
            theOtherSetid = theOtherUip["setid"]
            theOtherDNA_RNA_Workflow = ""
            if "DNA_RNA_Workflow" in theOtherUip:
			    theOtherDNA_RNA_Workflow = theOtherUip["DNA_RNA_Workflow"]
            thisDNA_RNA_Workflow = ""
            if "DNA_RNA_Workflow" in uip:
			    thisDNA_RNA_Workflow = uip["DNA_RNA_Workflow"]
            theOtherNucleotideType = ""
            if "NucleotideType" in theOtherUip:
                theOtherNucleotideType = theOtherUip["NucleotideType"]
            thisNucleotideType = ""
            if "NucleotideType" in uip:
                thisNucleotideType = uip["NucleotideType"]
            # if the rows are for DNA_RNA workflow, then dont complain .. just pass it along..
            #debug print  uip["row"] +" == " + theOtherRowStr + " samplename similarity  " + uip["DNA_RNA_Workflow"] + " == "+ theOtherDNA_RNA_Workflow
            if (       ((uip["Workflow"]=="Upload Only")or(uip["Workflow"] == "")) and (thisNucleotideType != theOtherNucleotideType)     ):
                duplicateSamplesExists = False
            if (       (uip["setid"] == theOtherSetid) and (thisDNA_RNA_Workflow == theOtherDNA_RNA_Workflow ) and (thisDNA_RNA_Workflow == "DNA_RNA")      ):
                duplicateSamplesExists = False
            if duplicateSamplesExists :
                msg ="sample name "+uip["sample"] + " in row "+ rowStr+" is also in row "+theOtherRowStr+". Please change the sample name"
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
            # else dont flag an error

        # if workflow is empty then dont validate and dont include this row in setid for further validations.
        if uip["Workflow"] =="":
            continue

        # see whether variant Caller plugin is required or not.
        if  (   ("ApplicationType" in uip)  and  (uip["ApplicationType"] == "Annotation")   ) :
            requiresVariantCallerPlugin = True
            if (  (isVariantCallerSelected != "Unknown")  and (isVariantCallerConfigured != "Unknown")  ):
                if (isVariantCallerSelected != "True"):
                    msg ="Workflow "+ uip["Workflow"] +" in row("+ rowStr+") requires selecting and configuring Variant Caller plugin. Please select and configure Variant Caller plugin before using this workflow."
                    inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
                    continue
                if (isVariantCallerConfigured != "True"):
                    msg ="Workflow "+ uip["Workflow"] +" in row("+ rowStr+") requires selecting and configuring Variant Caller plugin. The Variant Caller plugin is selected, but not configured. Please configure Variant Caller plugin before using this workflow."
                    inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
                    continue


        # if setid is empty or it starts with underscore , then dont validate and dont include this row in setid hash for further validations.
        if (   (uip["SetID"].startswith("_")) or   (uip["SetID"]=="")  ):
            msg ="SetID in row("+ rowStr+") should not be empty or start with an underscore character. Please update the SetID."
            inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
            continue
        # save the workflow information of the record on the setID.. also check workflow mismatch with previous row of the same setid
        setid = uip["SetID"]
        if  setid not in setidHash:
            setidHash[setid] = {}
            setidHash[setid]["records"] =[]
            setidHash[setid]["firstRecordRow"]=uip["row"]
            setidHash[setid]["firstWorkflow"]=uip["Workflow"]
            setidHash[setid]["firstRelationshipType"]=uip["RelationshipType"]
            setidHash[setid]["firstRecordDNA_RNA"]=uip["DNA_RNA_Workflow"]
        else:
            previousRow = setidHash[setid]["firstRecordRow"]
            expectedWorkflow = setidHash[setid]["firstWorkflow"]
            expectedRelationshipType = setidHash[setid]["firstRelationshipType"]
            #print  uip["row"] +" == " + previousRow + " set id similarity  " + uip["RelationshipType"] + " == "+ expectedRelationshipType
            if expectedWorkflow != uip["Workflow"]:
                msg="Selected workflow "+ uip["Workflow"] + " does not match a previous sample with the same SetID, with workflow "+ expectedWorkflow +" in row "+ previousRow+ ". Either change this workflow to match the previous workflow selection for the this SetID, or change the SetiD to a new value if you intend this sample to be used in a different IR analysis."
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
            elif expectedRelationshipType != uip["RelationshipType"]:
                #print  "error on " + uip["row"] +" == " + previousRow + " set id similarity  " + uip["RelationshipType"] + " == "+ expectedRelationshipType
                msg="INTERNAL ERROR:  RelationshipType "+ uip["RelationshipType"] + " of the selected workflow, does not match a previous sample with the same SetID, with RelationshipType "+ expectedRelationshipType +" in row "+ previousRow+ "."
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
        setidHash[setid]["records"].append(uip)

        # check if sample already exists on IR at this time and give a warning..
        inputJson["sampleName"] = uip["sample"]
        if uip["sample"] not in uniqueSamples:    # no need to repeat if the check has been done for the same sample name on an earlier row.
            sampleExistsCallResults = sampleExistsOnIR(inputJson)
            if sampleExistsCallResults.get("error") != "":
                if sampleExistsCallResults.get("status") == "true":
                    msg="sample name "+ uip["sample"] + " already exists in Ion Reporter "
                    inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)

        # check all the generic rules for this uip .. the results of the check goes into the hashes provided as arguments.
        validateAllRulesOnRecord_4_6(currentRules["restrictionRules"], uip, setidHash, rowErrors, rowWarnings)

        row = row + 1


    # after validations of basic rules look for errors all role requirements, uniqueness in roles, excess number of
    # roles, insufficient number of roles, etc.
    for setid in setidHash:
        # first check all the required roles are there in the given set of records of the set
        rowsLooked = ""
        if "validRelationRoles" in setidHash[setid]:
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
                    msg="For workflow " + setidHash[setid]["firstWorkflow"] +", a required RelationRole "+ validRole + " is not found. "
                    if   rowsLooked != "" :
                        if rowsLooked.find(",") != -1  :
                            msg = msg + "Please check the rows " + rowsLooked
                        else:
                            msg = msg + "Please check the row " + rowsLooked
                    inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)
            # check if any extra roles exists.  Given that the above test exists for lack of roles, it is sufficient if we 
            # verify the total number of roles expected and number of records got, for this setid. If there is a mismatch,
            # it means there are more than the number of roles required.
            #    Use the value of the rowsLooked,  populated from the above loop.
            sizeOfRequiredRoles = len(setidHash[setid]["validRelationRoles"])
            numRecordsForThisSetId = len(setidHash[setid]["records"])
            if (numRecordsForThisSetId > sizeOfRequiredRoles):
                complainAboutTooManyRoles = True

                if setidHash[setid]["firstRecordDNA_RNA"] == "DNA_RNA":
                    complainAboutTooManyRoles = False

                if complainAboutTooManyRoles:
                    msg="For workflow " + setidHash[setid]["firstWorkflow"] + ", more than the required number of RelationRoles is found. Expected number of roles is " + str(sizeOfRequiredRoles) + ". "
                    if   rowsLooked != "" :
                        if rowsLooked.find(",") != -1  :
                            msg = msg + "Please check the rows " + rowsLooked
                        else:
                            msg = msg + "Please check the row " + rowsLooked
                    inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)

        ##
        # validate the nucleotidetypes, similar to the roles.
        # first check all the required nucleotides are there in the given set of records of the set
        #    Use the value of the rowsLooked,  populated from the above loop.
        if (   (setidHash[setid]["firstRecordDNA_RNA"] == "DNA_RNA")  or  (setidHash[setid]["firstRecordDNA_RNA"] == "RNA")   ): 
            for validNucloetide in setidHash[setid]["validNucleotideTypes"]:
                foundNucleotide=0
                for record in setidHash[setid]["records"]:
                    if validNucloetide == record["NucleotideType"]:   #or NucleotideType
                        foundNucleotide = 1
                if foundNucleotide == 0 :
                    msg="For workflow " + setidHash[setid]["firstWorkflow"] +", a required NucleotideType "+ validNucloetide + " is not found. "
                    if   rowsLooked != "" :
                        if rowsLooked.find(",") != -1  :
                            msg = msg + "Please check the rows " + rowsLooked
                        else:
                            msg = msg + "Please check the row " + rowsLooked
                    inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)
            # check if any extra nucleotides exists.  Given that the above test exists for missing nucleotides, it is sufficient if we 
            # verify the total number of nucleotides expected and number of records got, for this setid. If there is a mismatch,
            # it means there are more than the number of nucleotides required.
            #    Use the value of the rowsLooked,  populated from the above loop.
            sizeOfRequiredNucleotides = len(setidHash[setid]["validNucleotideTypes"])
            #numRecordsForThisSetId = len(setidHash[setid]["records"])   #already done as part of roles check
            if (numRecordsForThisSetId > sizeOfRequiredNucleotides):
                msg="For workflow " + setidHash[setid]["firstWorkflow"] + ", more than the required number of Nucleotides is found. Expected number of Nucleotides is " + str(sizeOfRequiredNucleotides) + ". "
                if   rowsLooked != "" :
                    if rowsLooked.find(",") != -1  :
                        msg = msg + "Please check the rows " + rowsLooked
                    else:
                        msg = msg + "Please check the row " + rowsLooked
                inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)


        # calculate the cost of the analysis
        cost={}
        cost["row"]=setidHash[setid]["firstRecordRow"]
        cost["workflow"]=setidHash[setid]["firstWorkflow"]
        cost["cost"]="50.00"     # TBD  actually, get it from IR. There are now APIs available.. TS is not yet popping this to user before plan submission.
        analysisCost["workflowCosts"].append(cost)

    analysisCost["totalCost"]="2739.99"   # TBD need to have a few lines to add the individual cost... TS is not yet popping this to user before plan submission.
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
    #advices["onTooManyErrors"]= "There are errors on this page. If you only want to upload samples to Ion Reporter and not perform an Ion Reporter analysis at this time, you do not need to select a Workflow. When you are ready to launch an Ion Reporter analysis, you must log into Ion Reporter and select the samples to analyze."
    advices["onTooManyErrors"]= "<html> <body> There are errors on this page. To remove them, either: <br> &nbsp;&nbsp;1) Change the Workflow to &quot;Upload Only&quot; for affected samples. Analyses will not be automatically launched in Ion Reporter.<br> &nbsp;&nbsp;2) Correct all errors to ensure autolaunch of correct analyses in Ion Reporter.<br> Visit the Torrent Suite documentation at <a href=/ion-docs/Home.html > docs </a>  for examples. </body> </html>"

    # forumulate a few conditions, which may be required beyond this validation.
    conditions={}
    conditions["requiresVariantCallerPlugin"]=requiresVariantCallerPlugin


    #true/false return code is reserved for error in executing the functionality itself, and not the condition of the results itself.
    # say if there are networking errors, talking to IR, etc will return false. otherwise, return pure results. The results internally
    # may contain errors, which is to be interpretted by the caller. If there are other helpful error info regarding the results itsef,
    # then additional variables may be used to reflect metadata about the results. the status/error flags may be used to reflect the
    # status of the call itself.
    #if (foundAtLeastOneError == 1):
    #    return {"status": "false", "error": "none", "validationResults": validationResults, "cost":analysisCost}
    #else:
    #    return {"status": "true", "error": "none", "validationResults": validationResults, "cost":analysisCost}
    return {"status": "true", "error": "none", "validationResults": validationResults, "cost":analysisCost, "advices": advices,
	       "conditions": conditions
	       }

    """
    # if we want to implement this logic in grws, then here is the interface code.  But currently it is not yet implemented there.
    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/TSUserInputValidate/"
        hdrs = {'Authorization': token}
        resp = requests.post(url, verify=False, headers=hdrs,timeout=30)  #timeout is in seconds
        result = {}
        if resp.status_code == requests.codes.ok:
            result = json.loads(resp.text)
        else:
            #raise Exception ("IR WebService Error Code " + str(resp.status_code))
            raise Exception("IR WebService Error Code " + str(resp.status_code))
    except requests.exceptions.Timeout, e:
        return {"status": "false", "error": "Timeout"}
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

def validateAllRulesOnRecord_4_6(rules, uip, setidHash, rowErrors, rowWarnings):
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
                #msg="INTERNAL ERROR ON RULE # "+ruleNum+"   Incompatible validation rules for this version of TSS. No such \"For\" field \"" + rule["For"]["Name"] + "\""
                #inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)
                continue
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
                            #msg="Incorrect value \"" + uip[kValid] + "\" found for " + kValid + " When "+ kFor + " is \"" + vFor +"\"   rule # "+ ruleNum
                            msg="Incorrect value \"" + uip[kValid] + "\" found for " + kValid + " When "+ kFor + " is \"" + vFor +"\"."
                            inputValidationErrorHandle(row, validationType, msg, rowErrors, rowWarnings)
                        # a small hardcoded update into the setidHash for later evaluation of the role uniqueness
                        if kValid == "Relation":
                            if setid  in setidHash:
                                if "validRelationRoles" not in setidHash[setid]:
                                    #print  "saving   row " +row + "  setid " + setid + "  kfor " +kFor  + " vFor "+ vFor + "  kValid "+ kValid
                                    setidHash[setid]["validRelationRoles"] = vValid   # this is actually roles
                        # another small hardcoded update into the setidHash for later evaluation of the nucleotideType  uniqueness
                        if kValid == "NucleotideType":
                            if setid  in setidHash:
                                if "validNucleotideTypes" not in setidHash[setid]:
                                    #print  "saving   row " +row + "  setid " + setid + "  kfor " +kFor  + " vFor "+ vFor + "  kValid "+ kValid
                                    setidHash[setid]["validNucleotideTypes"] = vValid   # this is actually list of valid nucleotideTypes
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
                            #msg="Incorrect value \"" + uip[kInvalid] + "\" found for " + kInvalid + " When "+ kFor + " is \"" + vFor +"\"   rule # "+ ruleNum
                            msg="Incorrect value \"" + uip[kInvalid] + "\" found for " + kInvalid + " When "+ kFor + " is \"" + vFor +"\"."
                            inputValidationErrorHandle(row, validationType, msg, rowErrors, rowWarnings)
            elif "NonEmpty" in rule:
                kFor=rule["For"]["Name"]
                vFor=rule["For"]["Value"]
                kNonEmpty=rule["NonEmpty"]["Name"]
                print  "non empty validating   kfor " +kFor  + " vFor "+ vFor + "  kNonEmpty "+ kNonEmpty
                #if kFor not in uip :
                #    print  "kFor not in uip   "  + " kFor "+ kFor 
                #    continue
                if uip[kFor] == vFor :
                    if (   (kNonEmpty not in uip)   or  (uip[kNonEmpty] == "")   ):
                        msg="Empty value found for " + kNonEmpty + " When "+ kFor + " is \"" + vFor +"\"."
                        inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)
            elif "Disabled" in rule:
                pass
            else:
                 msg="INTERNAL ERROR ON RULE # "+ruleNum+"   Incompatible validation rules for this version of IRU. \"For\" specified without a \"Valid\" or \"Invalid\" tag."
                 inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)
        else:
            msg="INTERNAL ERROR ON RULE # "+ruleNum+"   Incompatible validation rules for this version of IRU. No action provided on this rule."
            inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)




def validateUserInput_5_0(inputJson):
    userInput = inputJson["userInput"]
    irAccountJson = inputJson["irAccount"]

    protocol = irAccountJson["protocol"]
    server = irAccountJson["server"]
    port = irAccountJson["port"]
    token = irAccountJson["token"]
    version = irAccountJson["version"]
    version = version.split("IR")[1]
    grwsPath = "grws_1_2"

    #variantCaller check variables
    requiresVariantCallerPlugin = False
    isVariantCallerSelected = "Unknown"
    if "isVariantCallerSelected" in userInput:
        isVariantCallerSelected = userInput["isVariantCallerSelected"]
    isVariantCallerConfigured = "Unknown"
    if "isVariantCallerConfigured" in userInput:
        isVariantCallerConfigured = userInput["isVariantCallerConfigured"]

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
    userInputInfo = userInput["userInputInfo"]
    validationResults = []
    # create a hash currentlyAvailableWorkflows with workflow name as key and value as a hash of all props of workflows from column-map
    currentlyAvaliableWorkflows={}
    for cmap in currentRules["column-map"]:
       currentlyAvaliableWorkflows[cmap["Workflow"]]=cmap
    # create a hash orderedColumns with column order number as key and value as a hash of all properties of each column from columns
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


    ###################################### This is a mock logic. This is not the real validation code. This is for test only 
    ###################################### This can be enabled or disabled using the control variable just below.
    mockLogic = 0
    if mockLogic == 1:
        #for 4.0, for now, return validation results based on a mock logic .
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
    ######################################
    ######################################



    setidHash={}
    rowErrors={}
    rowWarnings={}
    uniqueSamples={}
    analysisCost={}
    analysisCost["workflowCosts"]=[]

    row = 1
    for uip in userInputInfo:
        # make a row number if not provided, else use whats provided as rownumber.
        if "row" not in uip:
           uip["row"]=str(row)
        rowStr = uip["row"]
        # register such a row in the error bucket  and warning buckets.. basically create two holder arrays in those buckets
        if  rowStr not in rowErrors:
           rowErrors[rowStr]=[]
        if  rowStr not in rowWarnings:
           rowWarnings[rowStr]=[]

        # some known key translations on the uip, before uip can be used for validations
        if  "setid" in uip :
            uip["SetID"] = uip["setid"]
        if  "RelationshipType" not in uip :
            if  "Relation" in uip :
                uip["RelationshipType"] = uip["Relation"]
            if  "RelationRole" in uip :
                uip["Relation"] = uip["RelationRole"]
        if uip["Workflow"] == "Upload Only":
            uip["Workflow"] = ""
        if uip["Workflow"] !="":
            if uip["Workflow"] in currentlyAvaliableWorkflows:
                #uip["ApplicationType"] = currentlyAvaliableWorkflows[uip["Workflow"]]["ApplicationType"]
                #uip["DNA_RNA_Workflow"] = currentlyAvaliableWorkflows[uip["Workflow"]]["DNA_RNA_Workflow"]
                #uip["OCP_Workflow"] = currentlyAvaliableWorkflows[uip["Workflow"]]["OCP_Workflow"]

                # another temporary check which is not required if all the parameters of workflow  were  properly handed off from TS 
                if "RelationshipType" not in uip :
                    msg="INTERNAL ERROR:  For selected workflow "+ uip["Workflow"] + ", an internal key  RelationshipType is missing for row " + rowStr
                    inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
                    continue

                #bring in all the workflow parameter so far available, into the uip hash.
                for k in  currentlyAvaliableWorkflows[uip["Workflow"]] : 
                    uip[k] = currentlyAvaliableWorkflows[uip["Workflow"]][k]
            else:
                uip["ApplicationType"] = "unknown"
                msg="selected workflow "+ uip["Workflow"] + " is not available for this IR user account at this time"
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
                continue
        if  "nucleotideType" in uip :
            uip["NucleotideType"] = uip["nucleotideType"]
        if  "NucleotideType" not in uip :
            uip["NucleotideType"] = ""
        if  "cellularityPct" in uip :
            uip["CellularityPct"] = uip["cellularityPct"]
        if  "CellularityPct" not in uip :
            uip["CellularityPct"] = ""
        if  "cancerType" in uip :
            uip["CancerType"] = uip["cancerType"]
        if  "CancerType" not in uip :
            uip["CancerType"] = ""


        # all given sampleNames should be unique   TBD Jose   this requirement is going away.. need to safely remove this part. First, IRU plugin should be corrected before correcting this rule.
        if uip["sample"] not in uniqueSamples:
            uniqueSamples[uip["sample"]] = uip  #later if its a three level then make it into an array of uips
        else:
            duplicateSamplesExists = True
            theOtherUip = uniqueSamples[uip["sample"]]
            theOtherRowStr = theOtherUip["row"]
            theOtherSetid = theOtherUip["setid"]
            theOtherDNA_RNA_Workflow = ""
            if "DNA_RNA_Workflow" in theOtherUip:
			    theOtherDNA_RNA_Workflow = theOtherUip["DNA_RNA_Workflow"]
            thisDNA_RNA_Workflow = ""
            if "DNA_RNA_Workflow" in uip:
			    thisDNA_RNA_Workflow = uip["DNA_RNA_Workflow"]
            theOtherNucleotideType = ""
            if "NucleotideType" in theOtherUip:
                theOtherNucleotideType = theOtherUip["NucleotideType"]
            thisNucleotideType = ""
            if "NucleotideType" in uip:
                thisNucleotideType = uip["NucleotideType"]
            # if the rows are for DNA_RNA workflow, then dont complain .. just pass it along..
            #debug print  uip["row"] +" == " + theOtherRowStr + " samplename similarity  " + uip["DNA_RNA_Workflow"] + " == "+ theOtherDNA_RNA_Workflow
            if (       ((uip["Workflow"]=="Upload Only")or(uip["Workflow"] == "")) and (thisNucleotideType != theOtherNucleotideType)     ):
                duplicateSamplesExists = False
            if (       (uip["setid"] == theOtherSetid) and (thisDNA_RNA_Workflow == theOtherDNA_RNA_Workflow ) and (thisDNA_RNA_Workflow == "DNA_RNA")      ):
                duplicateSamplesExists = False
            if duplicateSamplesExists :
                msg ="sample name "+uip["sample"] + " in row "+ rowStr+" is also in row "+theOtherRowStr+". Please change the sample name"
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
            # else dont flag an error

        # if workflow is empty then dont validate and dont include this row in setid for further validations.
        if uip["Workflow"] =="":
            continue

        # see whether variant Caller plugin is required or not.
        if  (   ("ApplicationType" in uip)  and  (uip["ApplicationType"] == "Annotation")   ) :
            requiresVariantCallerPlugin = True
            if (  (isVariantCallerSelected != "Unknown")  and (isVariantCallerConfigured != "Unknown")  ):
                if (isVariantCallerSelected != "True"):
                    msg ="Workflow "+ uip["Workflow"] +" in row("+ rowStr+") requires selecting and configuring Variant Caller plugin. Please select and configure Variant Caller plugin before using this workflow."
                    inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
                    continue
                if (isVariantCallerConfigured != "True"):
                    msg ="Workflow "+ uip["Workflow"] +" in row("+ rowStr+") requires selecting and configuring Variant Caller plugin. The Variant Caller plugin is selected, but not configured. Please configure Variant Caller plugin before using this workflow."
                    inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
                    continue


        # if setid is empty or it starts with underscore , then dont validate and dont include this row in setid hash for further validations.
        if (   (uip["SetID"].startswith("_")) or   (uip["SetID"]=="")  ):
            msg ="SetID in row("+ rowStr+") should not be empty or start with an underscore character. Please update the SetID."
            inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
            continue
        # save the workflow information of the record on the setID.. also check workflow mismatch with previous row of the same setid
        setid = uip["SetID"]
        if  setid not in setidHash:
            setidHash[setid] = {}
            setidHash[setid]["records"] =[]
            setidHash[setid]["firstRecordRow"]=uip["row"]
            setidHash[setid]["firstWorkflow"]=uip["Workflow"]
            setidHash[setid]["firstRelationshipType"]=uip["RelationshipType"]
            setidHash[setid]["firstRecordDNA_RNA"]=uip["DNA_RNA_Workflow"]
        else:
            previousRow = setidHash[setid]["firstRecordRow"]
            expectedWorkflow = setidHash[setid]["firstWorkflow"]
            expectedRelationshipType = setidHash[setid]["firstRelationshipType"]
            #print  uip["row"] +" == " + previousRow + " set id similarity  " + uip["RelationshipType"] + " == "+ expectedRelationshipType
            if expectedWorkflow != uip["Workflow"]:
                msg="Selected workflow "+ uip["Workflow"] + " does not match a previous sample with the same SetID, with workflow "+ expectedWorkflow +" in row "+ previousRow+ ". Either change this workflow to match the previous workflow selection for the this SetID, or change the SetiD to a new value if you intend this sample to be used in a different IR analysis."
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
            elif expectedRelationshipType != uip["RelationshipType"]:
                #print  "error on " + uip["row"] +" == " + previousRow + " set id similarity  " + uip["RelationshipType"] + " == "+ expectedRelationshipType
                msg="INTERNAL ERROR:  RelationshipType "+ uip["RelationshipType"] + " of the selected workflow, does not match a previous sample with the same SetID, with RelationshipType "+ expectedRelationshipType +" in row "+ previousRow+ "."
                inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)
        setidHash[setid]["records"].append(uip)

        # check if sample already exists on IR at this time and give a warning..
        inputJson["sampleName"] = uip["sample"]
        if uip["sample"] not in uniqueSamples:    # no need to repeat if the check has been done for the same sample name on an earlier row.
            sampleExistsCallResults = sampleExistsOnIR(inputJson)
            if sampleExistsCallResults.get("error") != "":
                if sampleExistsCallResults.get("status") == "true":
                    msg="sample name "+ uip["sample"] + " already exists in Ion Reporter "
                    inputValidationErrorHandle(rowStr, "error", msg, rowErrors, rowWarnings)

        # check all the generic rules for this uip .. the results of the check goes into the hashes provided as arguments.
        validateAllRulesOnRecord_5_0(currentRules["restrictionRules"], uip, setidHash, rowErrors, rowWarnings)

        row = row + 1


    # after validations of basic rules look for errors all role requirements, uniqueness in roles, excess number of
    # roles, insufficient number of roles, etc.
    for setid in setidHash:
        # first check all the required roles are there in the given set of records of the set
        rowsLooked = ""
        if "validRelationRoles" in setidHash[setid]:
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
                    msg="For workflow " + setidHash[setid]["firstWorkflow"] +", a required RelationRole "+ validRole + " is not found. "
                    if   rowsLooked != "" :
                        if rowsLooked.find(",") != -1  :
                            msg = msg + "Please check the rows " + rowsLooked
                        else:
                            msg = msg + "Please check the row " + rowsLooked
                    inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)
            # check if any extra roles exists.  Given that the above test exists for lack of roles, it is sufficient if we 
            # verify the total number of roles expected and number of records got, for this setid. If there is a mismatch,
            # it means there are more than the number of roles required.
            #    Use the value of the rowsLooked,  populated from the above loop.
            sizeOfRequiredRoles = len(setidHash[setid]["validRelationRoles"])
            numRecordsForThisSetId = len(setidHash[setid]["records"])
            if (numRecordsForThisSetId > sizeOfRequiredRoles):
                complainAboutTooManyRoles = True

                if setidHash[setid]["firstRecordDNA_RNA"] == "DNA_RNA":
                    complainAboutTooManyRoles = False

                if complainAboutTooManyRoles:
                    msg="For workflow " + setidHash[setid]["firstWorkflow"] + ", more than the required number of RelationRoles is found. Expected number of roles is " + str(sizeOfRequiredRoles) + ". "
                    if   rowsLooked != "" :
                        if rowsLooked.find(",") != -1  :
                            msg = msg + "Please check the rows " + rowsLooked
                        else:
                            msg = msg + "Please check the row " + rowsLooked
                    inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)

        ##
        # validate the nucleotidetypes, similar to the roles.
        # first check all the required nucleotides are there in the given set of records of the set
        #    Use the value of the rowsLooked,  populated from the above loop.
        if (   (setidHash[setid]["firstRecordDNA_RNA"] == "DNA_RNA")  or  (setidHash[setid]["firstRecordDNA_RNA"] == "RNA")   ): 
            for validNucloetide in setidHash[setid]["validNucleotideTypes"]:
                foundNucleotide=0
                for record in setidHash[setid]["records"]:
                    if validNucloetide == record["NucleotideType"]:   #or NucleotideType
                        foundNucleotide = 1
                if foundNucleotide == 0 :
                    msg="For workflow " + setidHash[setid]["firstWorkflow"] +", a required NucleotideType "+ validNucloetide + " is not found. "
                    if   rowsLooked != "" :
                        if rowsLooked.find(",") != -1  :
                            msg = msg + "Please check the rows " + rowsLooked
                        else:
                            msg = msg + "Please check the row " + rowsLooked
                    inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)
            # check if any extra nucleotides exists.  Given that the above test exists for missing nucleotides, it is sufficient if we 
            # verify the total number of nucleotides expected and number of records got, for this setid. If there is a mismatch,
            # it means there are more than the number of nucleotides required.
            #    Use the value of the rowsLooked,  populated from the above loop.
            sizeOfRequiredNucleotides = len(setidHash[setid]["validNucleotideTypes"])
            #numRecordsForThisSetId = len(setidHash[setid]["records"])   #already done as part of roles check
            if (numRecordsForThisSetId > sizeOfRequiredNucleotides):
                msg="For workflow " + setidHash[setid]["firstWorkflow"] + ", more than the required number of Nucleotides is found. Expected number of Nucleotides is " + str(sizeOfRequiredNucleotides) + ". "
                if   rowsLooked != "" :
                    if rowsLooked.find(",") != -1  :
                        msg = msg + "Please check the rows " + rowsLooked
                    else:
                        msg = msg + "Please check the row " + rowsLooked
                inputValidationErrorHandle(setidHash[setid]["firstRecordRow"], "error", msg, rowErrors, rowWarnings)


        # calculate the cost of the analysis
        cost={}
        cost["row"]=setidHash[setid]["firstRecordRow"]
        cost["workflow"]=setidHash[setid]["firstWorkflow"]
        cost["cost"]="50.00"     # TBD  actually, get it from IR. There are now APIs available.. TS is not yet popping this to user before plan submission.
        analysisCost["workflowCosts"].append(cost)

    analysisCost["totalCost"]="2739.99"   # TBD need to have a few lines to add the individual cost... TS is not yet popping this to user before plan submission.
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
    #advices["onTooManyErrors"]= "There are errors on this page. If you only want to upload samples to Ion Reporter and not perform an Ion Reporter analysis at this time, you do not need to select a Workflow. When you are ready to launch an Ion Reporter analysis, you must log into Ion Reporter and select the samples to analyze."
    advices["onTooManyErrors"]= "<html> <body> There are errors on this page. To remove them, either: <br> &nbsp;&nbsp;1) Change the Workflow to &quot;Upload Only&quot; for affected samples. Analyses will not be automatically launched in Ion Reporter.<br> &nbsp;&nbsp;2) Correct all errors to ensure autolaunch of correct analyses in Ion Reporter.<br> Visit the Torrent Suite documentation at <a href=/ion-docs/Home.html > docs </a>  for examples. </body> </html>"

    # forumulate a few conditions, which may be required beyond this validation.
    conditions={}
    conditions["requiresVariantCallerPlugin"]=requiresVariantCallerPlugin


    #true/false return code is reserved for error in executing the functionality itself, and not the condition of the results itself.
    # say if there are networking errors, talking to IR, etc will return false. otherwise, return pure results. The results internally
    # may contain errors, which is to be interpretted by the caller. If there are other helpful error info regarding the results itsef,
    # then additional variables may be used to reflect metadata about the results. the status/error flags may be used to reflect the
    # status of the call itself.
    #if (foundAtLeastOneError == 1):
    #    return {"status": "false", "error": "none", "validationResults": validationResults, "cost":analysisCost}
    #else:
    #    return {"status": "true", "error": "none", "validationResults": validationResults, "cost":analysisCost}
    return {"status": "true", "error": "none", "validationResults": validationResults, "cost":analysisCost, "advices": advices,
	       "conditions": conditions
	       }

    """
    # if we want to implement this logic in grws, then here is the interface code.  But currently it is not yet implemented there.
    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/TSUserInputValidate/"
        hdrs = {'Authorization': token}
        resp = requests.post(url, verify=False, headers=hdrs,timeout=30)  #timeout is in seconds
        result = {}
        if resp.status_code == requests.codes.ok:
            result = json.loads(resp.text)
        else:
            #raise Exception ("IR WebService Error Code " + str(resp.status_code))
            raise Exception("IR WebService Error Code " + str(resp.status_code))
    except requests.exceptions.Timeout, e:
        return {"status": "false", "error": "Timeout"}
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

def validateAllRulesOnRecord_5_0(rules, uip, setidHash, rowErrors, rowWarnings):
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
                #msg="INTERNAL ERROR ON RULE # "+ruleNum+"   Incompatible validation rules for this version of TSS. No such \"For\" field \"" + rule["For"]["Name"] + "\""
                #inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)
                continue
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
                            #msg="Incorrect value \"" + uip[kValid] + "\" found for " + kValid + " When "+ kFor + " is \"" + vFor +"\"   rule # "+ ruleNum
                            msg="Incorrect value \"" + uip[kValid] + "\" found for " + kValid + " When "+ kFor + " is \"" + vFor +"\"."
                            inputValidationErrorHandle(row, validationType, msg, rowErrors, rowWarnings)
                        # a small hardcoded update into the setidHash for later evaluation of the role uniqueness
                        if kValid == "Relation":
                            if setid  in setidHash:
                                if "validRelationRoles" not in setidHash[setid]:
                                    #print  "saving   row " +row + "  setid " + setid + "  kfor " +kFor  + " vFor "+ vFor + "  kValid "+ kValid
                                    setidHash[setid]["validRelationRoles"] = vValid   # this is actually roles
                        # another small hardcoded update into the setidHash for later evaluation of the nucleotideType  uniqueness
                        if kValid == "NucleotideType":
                            if setid  in setidHash:
                                if "validNucleotideTypes" not in setidHash[setid]:
                                    #print  "saving   row " +row + "  setid " + setid + "  kfor " +kFor  + " vFor "+ vFor + "  kValid "+ kValid
                                    setidHash[setid]["validNucleotideTypes"] = vValid   # this is actually list of valid nucleotideTypes
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
                            #msg="Incorrect value \"" + uip[kInvalid] + "\" found for " + kInvalid + " When "+ kFor + " is \"" + vFor +"\"   rule # "+ ruleNum
                            msg="Incorrect value \"" + uip[kInvalid] + "\" found for " + kInvalid + " When "+ kFor + " is \"" + vFor +"\"."
                            inputValidationErrorHandle(row, validationType, msg, rowErrors, rowWarnings)
            elif "NonEmpty" in rule:
                kFor=rule["For"]["Name"]
                vFor=rule["For"]["Value"]
                kNonEmpty=rule["NonEmpty"]["Name"]
                print  "non empty validating   kfor " +kFor  + " vFor "+ vFor + "  kNonEmpty "+ kNonEmpty
                #if kFor not in uip :
                #    print  "kFor not in uip   "  + " kFor "+ kFor 
                #    continue
                if uip[kFor] == vFor :
                    if (   (kNonEmpty not in uip)   or  (uip[kNonEmpty] == "")   ):
                        msg="Empty value found for " + kNonEmpty + " When "+ kFor + " is \"" + vFor +"\"."
                        inputValidationErrorHandle(row, "error", msg, rowErrors, rowWarnings)
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

def getIRGUIBaseURL(inputJson):
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
                "IRGUIBaseURL": ""}

    # for now, return a hardcoded version of the url for 40 production.  The url returned from IR is wrong.. 
    #if (   (version == "40") and (server == "40.dataloader.ionreporter.lifetechnologies.com")  ):
    #    returnUrl = protocol + "://" + "ionreporter.lifetechnologies.com" + ":" + port
    #    return {"status": "true", "error": "none", "IRGUIBaseURL": returnUrl,"version":version}

    # for now, return a hardcoded debug version of the url, because there are still configuraiton issues in the local IR servers.
    #returnUrl = protocol + "://" + server + ":" + port
    #return {"status": "true", "error": "none", "IRGUIBaseURL": returnUrl,"version":version}


    #curl ${CURLOPT}  ${protocol}://${server}:${port}/grws_1_2/data/getIrGUIUrl
    url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/getIrGUIUrl"
    #cmd="curl -ks    -H Authorization:"+token  +   " -H Version:"+version   +   " "+url
    cmd="curl -ks     "+url      #actually, this much will do for this webservice. token not required.
    result = get_httpResponseFromSystemTools(cmd)
    if (result["status"] =="true"):
        return {"status": "true", "error": "none", "IRGUIBaseURL": result["stdout"], "version":version}
    else:
        return result



    #get the correct ui server address, port and protocol from the grws and use that one instead of using iru-server's address.
    try:
        url = protocol + "://" + server + ":" + port + "/" + grwsPath + "/data/getIrGUIUrl"
        hdrs = {'Authorization': token}
        resp = requests.get(url, verify=False, headers=hdrs,timeout=30)  #timeout is in seconds
        #result = {}
        if resp.status_code == requests.codes.ok:
            #returnUrl = json.loads(resp.text)
            returnUrl =str(resp.text)
        else:
            #raise Exception ("IR WebService Error Code " + str(resp.status_code))
            return {"status": "false", "error":"IR WebService Error Code " + str(resp.status_code)}
    except requests.exceptions.Timeout, e:
        return {"status": "false", "error": "Timeout"}
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
    return {"status": "true", "error": "none", "IRGUIBaseURL": returnUrl, "version":version}

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
                "workflowCreationLandingPageURL": "",
                "version": version}

    baseURLResult = getIRGUIBaseURL(inputJson)
    if baseURLResult["status"] == "false":
         return baseURLResult
    urlPart1 = baseURLResult["IRGUIBaseURL"]

    queryParams = {'authToken': token}
    urlEncodedQueryParams = urllib.urlencode(queryParams)
    # a debug version of the url to return a hardcoded url
    #url2 = protocol + "://" + server + ":" + port + "/ir/secure/workflow.html?" + urlEncodedQueryParams
    #urlPart1 = protocol + "://" + server + ":" + port
    #returnUrl = urlPart1+urlPart2
    #return {"status": "true", "error": "none", "workflowCreationLandingPageURL": returnUrl}
    if version == "40":
        urlPart2 = "/ir/secure/workflow.html?" + urlEncodedQueryParams
    else: #if version == "42": or above
        urlPart2 = "/ir/postauth/workflow.html"

    returnUrl = urlPart1+urlPart2
    return {"status": "true", "error": "none", "workflowCreationLandingPageURL": returnUrl,
            "version": version}


def getWorkflowCreationLandingPageURLBase(inputJson):
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
                "workflowCreationLandingPageURL": "",
                "version": version}

    #queryParams = {'authToken': token}
    #urlEncodedQueryParams = urllib.urlencode(queryParams)
    # a debug version of the url to return a hardcoded url
    #url2 = protocol + "://" + server + ":" + port + "/ir/secure/workflow.html?" + urlEncodedQueryParams
    #urlPart1 = protocol + "://" + server + ":" + port
    #returnUrl = urlPart1+urlPart2
    #return {"status": "true", "error": "none", "workflowCreationLandingPageURL": returnUrl}
    #print "version = " + version

    if version == "40":
        urlPart2 = "/ir/secure/workflow.html"
    else: #if version == "42": or above
        urlPart2 = "/ir/postauth/workflow.html"

    baseURLResult = getIRGUIBaseURL(inputJson)
    if baseURLResult["status"] == "false":
         return baseURLResult
    urlPart1 = baseURLResult["IRGUIBaseURL"]
    returnUrl = urlPart1+urlPart2
    return {"status": "true", "error": "none", "workflowCreationLandingPageURL": returnUrl,
            "version": version, "token": token}

def uploadStatus(bucket):
    """finds all the IRU progress and globs it together
    """
    def proton_progress(plugin_result):
        payload = {
            "pre": {},
            "post": {}
        }
        pre_json_path = os.path.join(plugin_result["path"],"consolidatedStatus", "pre.json")
        post_json_path = os.path.join(plugin_result["path"],"consolidatedStatus", "post.json")
        composite_json = json.load(open(os.path.join(plugin_result["path"],"startplugin.json")))

        try:
            payload["pre"] = json.load(open(pre_json_path))
        except Exception as err:
            print("Failed to load Proton pre.json")

        try:
            post_json = json.load(open(post_json_path))
            payload["post"] = post_json
        except Exception as err:
            print("Failed to load Proton post.json")


        #find the composite block json files
        consolidatedBlockFiles = glob.glob(os.path.join(plugin_result["path"],"consolidatedStatus", "X*.json"))
        consolidatedBlocks = {}

        for consolidatedBlock in consolidatedBlockFiles:
            if os.path.exists(consolidatedBlock):
                localConsolidatedBlock = json.load(open(consolidatedBlock))
                consolidatedBlocks[localConsolidatedBlock["block.id"]] = localConsolidatedBlock

        #find all the blocks
        blockDirs = composite_json["runplugin"]["block_dirs"]

        progress = {}

        totalProgress = 0

        for block in blockDirs:
            #look for a startplugin.json for each block
            block_plugin = glob.glob(os.path.join(block, "plugin_out/*." + plugin_result_id + "/startplugin.json"))
            if block_plugin and os.path.exists(block_plugin[0]):
                #get the block id
                block_id = json.load(open(block_plugin[0]))["runplugin"]["blockId"]

                #now get the progress.json path
                progress_path = os.path.join(os.path.split(block_plugin[0])[0],"progress.json")


                if os.path.exists(progress_path):
                    #include the progress.json as part of the response
                    progress[block_id] = json.load(open(progress_path))
                    #if not don't add anything to the progress
                    totalProgress += float(progress[block_id].get("progress",0))

        payload["blockProgress"] = progress
        payload["numBlocks"] =  composite_json["runplugin"]["numBlocks"]
        payload["consolidatedBlockStatus"] = consolidatedBlocks
        payload["totalProgress"] = totalProgress / float(payload["numBlocks"])
        return payload

    def pgm_progress(plugin_result, progress_path):
        try:
            progress = json.load(open(progress_path))
        except (IOError, ValueError) as err:
            progress = {}
        payload = {
            "totalProgress": progress.get('progress', 0),
            "status": progress.get('status', "No Status")
        }
        return payload

    if "request_get" in bucket:

        plugin_result_id = bucket["request_get"].get("plugin_result_id", False)

        if plugin_result_id:
            #get the plugin results resource from the api
            plugin_result = requests.get("http://localhost/rundb/api/v1/pluginresult/" + plugin_result_id).json()
            progress_path = os.path.join(plugin_result["path"], "post", "progress.json")
            if os.path.exists(progress_path):
                payload = pgm_progress(plugin_result, progress_path)
            else:
                payload = proton_progress(plugin_result)

            return payload

        #if we got all the way down here it failed
        return False

def lastrun(bucket):
	# check whether previous instance of IRU is in-progress
	lockfile = 'iru_status.lock'

	pluginresult = bucket['request_post']['pluginresult']
	current_version = bucket['request_post'].get('version') or bucket['version']
	state = pluginresult['State']

	if current_version != pluginresult['Version']:
		in_progress = False
		msg = 'Previous plugin instance version %s does not match current version' % pluginresult['Version']
	elif state == 'Completed':
		lockpath = os.path.join(pluginresult['Path'],'post',lockfile)
		if os.path.exists(lockpath):
			in_progress = False
			msg = 'Previous plugin instance state is %s, plugin post-level lock exists %s' % (state, lockpath)
		else:
			in_progress = True
			msg = 'Previous plugin instance state is %s, but plugin post-level lock not found %s' % (state, lockpath)
	else:
		in_progress = True if state != 'Error' else False
		msg = 'Previous plugin instance state is %s.' % state

	return {'in_progress':in_progress, 'msg':msg }


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

    k = {'port': '443',    # this is the one that is used for most of the test cases below
         'protocol': 'https',
         'server': '40.dataloader.ionreporter.lifetechnologies.com',
         'version': 'IR50',
         'userid': 'ion.reporter@lifetech.com',
         'password': '123456',
         'token': 'rwVcoTeYGfKxItiaWo2lngsV/r0jukG2pLKbZBkAFnlPbjKfPTXLbIhPb47YA9u78'
        }
    c = {}
    c["irAccount"] = k



    l = {'port': '443',
         'protocol': 'https',
         'server': '40.dataloader.ionreporter.lifetechnologies.com',
         'version': 'IR42',
         'userid': 'ion.reporter@lifetech.com',
         'password': '123456',
         'token': 'rwVcoTeYGfKxItiaWo2lngsV/r0jukG2pLKbZBkAFnlPbjKfPTXLbIhPb47YA9u78'
        }
    d = {}
    d["irAccount"] = l



    p={"userInputInfo":[
        {
          "row": "6",
          "Workflow": "",
          "Gender": "Female",
          "barcodeId": "IonXpress_011",
          "sample": "pgm-s11",
          #"Relation": "Self",
          "RelationRole": "Self",
          "setid": "0__837663e7-f7f8-4334-b14b-dea091dd353b"
        },
        {
          "row": "95",
          "Workflow": "AmpliSeq Exome tumor-normal pair",
          "Gender": "Unknown",
          "barcodeId": "IonXpress_012",
          "sample": "pgm-s12T",
          #"Relation": "Tumor_Normal",
          "RelationRole": "Tumor",
          "setid": "1__7179df4c-c6bb-4cbe-97a4-bb48951a4acd"
        },
        {
          "row": "96",
          "Workflow": "AmpliSeq Exome tumor-normal pair",
          "Gender": "Unknown",
          "barcodeId": "IonXpress_012",
          "sample": "pgm-s12N",
          "Relation": "Tumor_Normal",
          "RelationRole": "Normal",
          "setid": "1__7179df4c-c6bb-4cbe-97a4-bb48951a4acd"
        },
        {
          "row": "5",
          "Workflow": "AmpliSeq Exome tumor-normal pair",
          "Gender": "Male",
          "barcodeId": "IonXpress_013",
          "sample": "pgm-s12",
          "Relation": "Tumor_Normal",
          "RelationRole": "Tumor",
          "setid": "2__7179df4c-c6bb-4cbe-97a4-bb48951a4acd"
        },
        {
          "row": "9",
          "Workflow": "AmpliSeq Exome tumor-normal pair",
          "Gender": "Male",
          "barcodeId": "IonXpress_012",
          "sample": "pgm-s12n",
          "Relation": "Tumor_Normal",
          "RelationRole": "Normal",
          "setid": "2__7179df4c-c6bb-4cbe-97a4-bb48951a4acd"
        },
        {
          "row": "21",
          "Workflow": "AmpliSeq OCP DNA RNA Fusions",
          #"Workflow": "Upload Only",
          "Gender": "Male",
          "barcodeId": "IonXpress_012",
          "sample": "pgm-s12_dna_rna",
          "Relation": "DNA_RNA",
          "RelationRole": "Self",
          "NucleotideType": "DNA",
          #"cellularityPct": "10",
          #"cancerType": "Liver Cancer",
          "setid": "4__7179df4c-c6bb-4cbe-97a4-bb48951a4acd"
        },
        {
          "row": "22",
          "Workflow": "AmpliSeq OCP DNA RNA Fusions",
          #"Workflow": "Upload Only",
          "Gender": "Male",
          "barcodeId": "IonXpress_012",
          "sample": "pgm-s12_dna_rna",
          "Relation": "DNA_RNA",
          "RelationRole": "Self",
          "NucleotideType": "RNA",
          "cellularityPct": "11",
          "cancerType": "Liver Cancer",
          "setid": "4__7179df4c-c6bb-4cbe-97a4-bb48951a4acd"
        },
        {
          "row": "23",
          "Workflow": "AmpliSeq OCP DNA RNA Fusions",
          "Gender": "Male",
          "barcodeId": "IonXpress_012",
          "sample": "pgm-s13_dna_rna",
          "Relation": "DNA_RNA",
          "RelationRole": "Self",
          "NucleotideType": "RNA",
          "cellularityPct": "10",
          "cancerType": "Liver Cancer",
          "setid": "5__7179df4c-c6bb-4cbe-97a4-bb48951a4acd"
        },
        {
          "row": "24",
          "Workflow": "AmpliSeq OCP DNA RNA Fusions",
          "Gender": "Male",
          "barcodeId": "IonXpress_012",
          "sample": "pgm-s13_dna_rna",
          "Relation": "DNA_RNA",
          "RelationRole": "Self",
          "NucleotideType": "DNA",
          "cellularityPct": "11",
          "cancerType": "Liver Cancer",
          "setid": "5__7179df4c-c6bb-4cbe-97a4-bb48951a4acd"
        },
        {
          "row": "25",
          "Workflow": "AmpliSeq Colon Lung v2 with RNA Lung Fusion paired sample",
          #"Workflow": "Upload Only",
          "Gender": "Male",
          "barcodeId": "IonXpress_012",
          "sample": "pgm-s12_dna_rna",
          "Relation": "DNA_RNA",
          "RelationRole": "Self",
          "NucleotideType": "DNA",
          #"cellularityPct": "10",
          #"cancerType": "Liver Cancer",
          "setid": "4__7179df4c-c6bb-4cbe-97a4-bb48951a4acd"
        },
        {
          "row": "26",
          "Workflow": "AmpliSeq Colon Lung v2 with RNA Lung Fusion paired sample",
          #"Workflow": "Upload Only",
          "Gender": "Male",
          "barcodeId": "IonXpress_012",
          "sample": "pgm-s12_dna_rna",
          "Relation": "DNA_RNA",
          "RelationRole": "Self",
          "NucleotideType": "RNA",
          "cellularityPct": "11",
          "cancerType": "Liver Cancer",
          "setid": "4__7179df4c-c6bb-4cbe-97a4-bb48951a4acd"
        },
        {
          "row": "27",
          "Workflow": "Annotate variants single sample",
          #"Workflow": "Upload Only",
          "Gender": "Male",
          "barcodeId": "IonXpress_013",
          "sample": "2015-03-24_025028_C",
          "Relation": "DNA",
          "RelationRole": "Self",
          "NucleotideType": "DNA",
          "cellularityPct": "11",
          "cancerType": "Liver Cancer",
          "setid": "7__7179df4c-c6bb-4cbe-97a4-bb48951a4acd"
        }
      ],
	  "accountId":"planned_irAccount_id_blahblahblah",
	  "accountName":"planned_irAccount_name_blahblahblah"
	 }
    c["userInput"]=p

    #print ""
    #print ""
    #print ""
    #print ""
    #print ""
    #print "get_plugin_dir() "
    #print get_plugin_dir()
    #print ""
    #print ""
    #print ""
    #print ""
    ##print ""
    #print "set_classpath() "
    #print set_classpath()
    #print ""
    #print ""
    #print ""
    #print ""
    #print ""
    ##print "java command outputs for "
    #print get_httpResponseFromIRUJavaAsJson("-u ion.reporter@lifetech.com -w 123456 -p https -a think1.itw -x 443 -v 46 -o userDetails")
    #print ""
    #print ""
    #print ""
    #print ""
    #print ""
    print "user details ==============================="
    print getUserDetails(c)
    #print ""
    #print ""
    #print ""
    #print ""
    #print ""
    #print "config"
    #print c
    #print ""
    #print ""
    #print ""
    #print ""
    #print ""
    #print "cancer types"
    #print getIRCancerTypesList(c)
    #print ""
    #print ""
    #print ""
    #print ""
    #print ""
    #print "landing page"
    #print getWorkflowCreationLandingPageURL(c)
    #print ""
    #print ""
    #print ""
    #print ""
    #print ""
    #print "versions list ==============================="
    #print get_versions(c)
    #print ""
    #print ""
    #print ""
    #print ""
    #print ""
    #print " workflow list"
    #print getWorkflowList(c)
    #print ""
    #print ""
    #print ""
    #print ""
    #print ""
    #print "user input table "
    #print getUserInput(c)
    #print ""
    #print ""
    #print ""
    #print ""
    #print ""
    ##print "validate user input"
    #print validateUserInput(c)
    #print ""
    #print ""
    #print ""
    #print ""
    #print ""
    #print "IR GUI url "
    #print getIRGUIBaseURL(c)
    #print ""
    #print ""
    #print ""
    #print ""
    #print ""
    #print "landing page"
    #print getWorkflowCreationLandingPageURL(c)
    #print ""
    #print ""
    #print ""
    #print ""
    #print ""
    #print "landing page base"
    #print getWorkflowCreationLandingPageURLBase(c)
    #print ""
    #print ""
    #print ""
    #print ""
    #print ""
    #print "user details 2"
    #print getUserDetails(d)
    #print ""
    #print ""
    #print ""




