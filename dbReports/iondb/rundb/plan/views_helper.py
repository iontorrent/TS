# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.models import Project, PlannedExperiment, Content, Plugin, GlobalConfig, Chip
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404

from traceback import format_exc

import re
import uuid

from iondb.utils.validation import is_valid_chars, is_valid_length

import collections
import logging
logger = logging.getLogger(__name__)


def get_projects_helper(projectIdAndNameList):
    found, missing = [], []
    if (isinstance(projectIdAndNameList, basestring)):
        projectIdAndNameList = projectIdAndNameList.split(',')

    for projectIdAndName in projectIdAndNameList:
        projectId = projectIdAndName.split('|')[0]
        try:
            found.append(Project.objects.get(id=int(projectId)))
        except:
            missing.append(projectIdAndName)

    return found, missing


def get_projects(username, json_data):
    # selected projects, projectIdAndNameList is a string if 1 entry; list otherwise
    projectIdAndNameList = json_data.get('projects', '')
    projectObjList = []
    if projectIdAndNameList:
        projectObjList, missings = get_projects_helper(projectIdAndNameList)
        for missing in missings:
            logger.debug("views.editplannedexperiment project= %s is no longer in db" % missing)
    # new projects added
    newProjectNames = json_data.get('newProjects', '')
    if newProjectNames:
        newProjectNames = newProjectNames.split(',')
        projectObjList.extend(Project.bulk_get_or_create(newProjectNames, User.objects.get(username=username)))
    return projectObjList


def dict_bed_hotspot():
    data = {}
    allFiles = Content.objects.filter(publisher__name="BED", path__contains="/unmerged/detail/").order_by('path')
    bedFiles, hotspotFiles = [], []
    bedFileFullPaths, bedFilePaths, hotspotFullPaths, hotspotPaths = [], [], [], []
    for _file in allFiles:
        if _file.meta.get("hotspot", False):
            hotspotFiles.append(_file)
            hotspotFullPaths.append(_file.file)
            hotspotPaths.append(_file.path)
        else:
            bedFiles.append(_file)
            bedFileFullPaths.append(_file.file)
            bedFilePaths.append(_file.path)
    data["bedFiles"] = bedFiles
    data["hotspotFiles"] = hotspotFiles
    data["bedFileFullPaths"] = bedFileFullPaths
    data["bedFilePaths"] = bedFilePaths
    data["hotspotFullPaths"] = hotspotFullPaths
    data["hotspotPaths"] = hotspotPaths
    return data


def get_available_plan_count():
    ''' Returns the number of plans (excluding templates) that are ready to use '''
    
    return PlannedExperiment.objects.filter(isReusable = False, planExecuted = False).exclude(planStatus = "voided").count()



def getPlanDisplayedName(plan):
    '''
    return the plan displayed name in ascii format (in case the original name has e.g. trademark unicode in it
    '''
    planDisplayedName = ""
    if plan:        
        planDisplayedName = plan.planDisplayedName.encode("ascii", "ignore")
        
    return planDisplayedName


def getPlanBarcodeCount(plan):
    count = 0
                
    experiment = plan.experiment
    if experiment:
        if plan.latest_eas and plan.latest_eas.barcodedSamples:
            barcodedSamples = plan.latest_eas.barcodedSamples
            
            for key, value in barcodedSamples.iteritems():
                if value.get("barcodes", ""):
                    count += len(value.get("barcodes"))

        #logger.debug("views_helper.getPlanBarcodeCount() plan.id=%d; count=%d" %(plan.id, count))                    
        
    return count
    

def get_IR_accounts_for_user(request):
    '''
    return the all the IR accounts for the user found in the request object. 
    '''
    
    if request and request.user:
        return get_IR_accounts_by_userName(request.user.username)
    return {}


def get_IR_accounts_by_userName(userName):
    '''
    return the all the IR accounts for the user. 
    '''
    
    empty_value = {}
    if userName:
        #logger.debug("views_helper.get_IR_accounts_by_user() userName=%s" %(userName))
        
        iru_plugins = Plugin.objects.filter(name__iexact = "IonReporterUploader", selected = True, active = True)
        if iru_plugins:
            iru_plugin = iru_plugins[0]

            if iru_plugin:
                userConfigs = iru_plugin.config.get('userconfigs').get(userName, {})
                return userConfigs
            
    return empty_value


def get_default_or_first_IR_account(request):
    '''
    return the default IR account for the user found in the request object. 
    If no default specified, just return the first IR account defined.
    '''
    
    if request and request.user:
        return get_default_or_first_IR_account_by_userName(request.user.username)
    return {}


def get_default_or_first_IR_account_by_userName(userName):
    '''
    return the default IR account for the user. 
    If no default specified, just return the first IR account defined.
    '''
    
    empty_value = {}    
    if userName:
        userConfigs = get_IR_accounts_by_userName(userName)

        if userConfigs:
            for userConfig in userConfigs:
                isDefault = userConfig.get("default", False)
                if isDefault:
                    return userConfig
            return userConfigs[0]
            
    return empty_value


def get_internal_name_for_displayed_name(displayedName):
    if displayedName:
        return displayedName.strip().replace(' ', '_')
    else:
        return ""
    

def get_ir_set_id(id_prefix = "1"):
    if id_prefix:
        suffix = str(uuid.uuid4())
        id = id_prefix.strip() + '__' + suffix
        return id
    else:
        return ""

    

def isOCP_enabled():
    return GlobalConfig.get().enable_compendia_OCP


def is_operation_supported(plan_or_template_pk):
    isSupported = isOCP_enabled()

    if isSupported:
        return isSupported
    
    planTemplate = get_object_or_404(PlannedExperiment, pk = plan_or_template_pk)
    
#    if (planTemplate.sampleGrouping and planTemplate.sampleGrouping.displayedName == "DNA_RNA"):
#        return isSupported

    if (planTemplate.categories and "Oncomine" in planTemplate.categories):
          return False      

    return True


def is_operation_supported_by_obj(plan_or_template):
    isSupported = isOCP_enabled()

#    if (plan_or_template and plan_or_template.sampleGrouping and plan_or_template.sampleGrouping.displayedName == "DNA_RNA"):
#        return isSupported

    if isSupported:
        return isSupported

    if (plan_or_template.categories and "Oncomine" in plan_or_template.categories):
          return False      
    
    return True

def getChipDisplayedVersion(chipDisplayedName):
    parts = chipDisplayedName.split("v", 1)
    return "v"+ parts[1]  if len(parts) > 1 else ""


def getChipDisplayedNamePrimaryPrefix(chipDisplayedName):
    """
    Returns all the primary chip displayed name for UI to display 
    e.g., for chip name "318 Select", return 318
    """
    isVersionInfoFound, prefixes = Chip.getChipDisplayedNameParts(chipDisplayedName)
    
    return prefixes[0]


def getChipDisplayedNameSecondaryPrefix(chipDisplayedName):
    """
    Returns all the second portion of the chip displayed name for UI to display 
    e.g., for chip name "318 Select", return "Select"
    """
    isVersionInfoFound, prefixes = Chip.getChipDisplayedNameParts(chipDisplayedName)
        
    return prefixes[-1] if len(prefixes) > 1 else ""



        