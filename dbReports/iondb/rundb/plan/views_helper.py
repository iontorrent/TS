# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.models import Project, PlannedExperiment, Content
from django.contrib.auth.models import User

from traceback import format_exc

import re

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


def is_valid_chars(value, validChars=r'^[a-zA-Z0-9-_\.\s]+$'):
    ''' Determines if value is valid: letters, numbers, spaces, dashes, underscores, dots only '''
    return bool(re.compile(validChars).match(value))


def is_invalid_leading_chars(value, invalidChars=r'[\.\-\_]'):
    ''' Determines if leading characters contain dashes, underscores or dots '''
    if value:
        return bool(re.compile(invalidChars).match(value.strip(), 0))
    else:
        True
    

def is_valid_length(value, maxLength = 0):
    ''' Determines if value length is within the maximum allowed '''
    if value:
        return len(value.strip()) <= maxLength
    return True


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