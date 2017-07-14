# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.models import Project, PlannedExperiment, Content, Plugin, GlobalConfig, Chip, ApplicationGroup, common_CV
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
    bedFiles, hotspotFiles, sseFiles = [], [], []
    bedFileFullPaths, bedFilePaths, hotspotFullPaths, hotspotPaths = [], [], [], []
    for _file in allFiles:
        if _file.meta.get("sse"):
            sseFiles.append(_file)
        elif _file.meta.get("hotspot", False):
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
    data["sseFiles"] = sseFiles
    return data


def get_available_plan_count():
    ''' Returns the number of plans (excluding templates) that are ready to use '''

    return PlannedExperiment.objects.filter(isReusable=False, planExecuted=False).exclude(planStatus="voided").count()


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

        # logger.debug("views_helper.getPlanBarcodeCount() plan.id=%d; count=%d" %(plan.id, count))

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
        # logger.debug("views_helper.get_IR_accounts_by_user() userName=%s" %(userName))

        iru_plugins = Plugin.objects.filter(name__iexact="IonReporterUploader", selected=True, active=True)
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


def get_default_or_first_IR_account_by_userName(userName, IR_server = None):
    '''
    return the default IR account for the user.
    If no default specified, just return the first IR account defined.
    '''

    empty_value = {}

    if userName:
        userConfigs = get_IR_accounts_by_userName(userName)

        if userConfigs:
            for userConfig in userConfigs:
                if IR_server:
                    existing_IR_name = userConfig.get("name").strip()
                    if IR_server.strip() == existing_IR_name:
                        return userConfig
                else:
                    isDefault = userConfig.get("default", False)
                    if isDefault:
                        return userConfig
            if not IR_server:
                return userConfigs[0]

    return empty_value


def get_internal_name_for_displayed_name(displayedName):
    if displayedName:
        return displayedName.strip().replace(' ', '_')
    else:
        return ""


def get_ir_set_id(id_prefix="1"):
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

    planTemplate = get_object_or_404(PlannedExperiment, pk=plan_or_template_pk)

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
    return "v" + parts[1] if len(parts) > 1 else ""


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


def get_template_categories():
    '''
    Generate categories for Templates web page, can be combinations of runTypes and applicationGroups.
    Define isActive as a database call to be able to hide/show category via off-cycle release (e.g. PGx example)
    '''
    applicationGroup_PGx = ApplicationGroup.objects.filter(name='PGx')
    applicationGroup_HID = ApplicationGroup.objects.filter(name='HID')
    applicationGroup_tagSeq = ApplicationGroup.objects.filter(name='onco_liquidBiopsy')    
    applicationGroup_immuneRepertoire = ApplicationGroup.objects.filter(name='immune_repertoire')

    category_solidTumor = common_CV.objects.filter(cv_type = "applicationCategory", value = "onco_solidTumor")
    category_immune = common_CV.objects.filter(cv_type = "applicationCategory", value = "onco_immune")
    category_repro = common_CV.objects.filter(cv_type = "applicationCategory", value = "repro")
    category_16s = common_CV.objects.filter(cv_type = "applicationCategory", value = "16s")
    category_inheritedDisease = common_CV.objects.filter(cv_type = "applicationCategory", value = "inheritedDisease")
    category_oncoHeme = common_CV.objects.filter(cv_type = "applicationCategory", value = "onco_heme")
                    
    categories = [
        # Favorites
        {
            'tag': 'favorites',
            'displayedName': 'Favorites',
            'api_filter': '&isFavorite=true',
            'img': 'resources/img/star-blue.png',
            'isActive': True,
            'ampliSeq_upload': True,
            'code': 0,
        },
        # Recently Created
        {
            'tag': 'recently_created',
            'displayedName': 'All',
            'api_filter': '',
            'img': 'resources/img/recentlyused.png',
            'isActive': True,
            'ampliSeq_upload': True,
            'code': 0,
        },
        # AmpliSeq DNA
        {
            'tag': 'ampliseq_dna',
            'displayedName': 'AmpliSeq DNA',
            'api_filter': '&runType__in=AMPS,AMPS_EXOME&applicationGroup__name__iexact=DNA',
            'img': 'resources/img/appl_ampliSeq.png',
            'isActive': True,
            'ampliSeq_upload': True,
            'code': 1,
        },
        # AmpliSeq RNA
        {
            'tag': 'ampliseq_rna',
            'displayedName': 'AmpliSeq RNA',
            'api_filter': '&runType=AMPS_RNA&applicationGroup__name__iexact=RNA',
            'img': 'resources/img/appl_ampliSeqRna.png',
            'isActive': True,
            'ampliSeq_upload': True,
            'code': 5,
        },
        # DNA and Fusions
        {
            'tag': 'fusions',
            'displayedName': 'DNA and Fusions',
            'api_filter': '&runType__in=AMPS,AMPS_EXOME,AMPS_RNA,AMPS_DNA_RNA&applicationGroup__uid__iexact=APPLGROUP_0005',
            'img': 'resources/img/appl_ampliSeqDNA_RNA.png',
            'isActive': True,
            'code': 8,
        },
        # Generic Sequencing
        {
            'tag': 'genericseq',
            'displayedName': 'Generic Sequencing',
            'api_filter': '&runType=GENS',
            'img': 'resources/img/appl_wildcard.png',
            'isActive': True,
            'code': 0,
        },
        # HID
        {
            'tag': 'hid',
            'displayedName': applicationGroup_HID[0].description,
            'api_filter':'&runType=AMPS&applicationGroup__name__iexact=HID',
            'img': 'resources/img/appl_hid.png',
            'isActive': applicationGroup_HID[0].isActive if applicationGroup_HID else False,
            'ampliSeq_upload': True,            
            'code': 11,
        },
        # Immune Repertoire
        {
            'tag': 'immune_repertoire',
            'displayedName': applicationGroup_immuneRepertoire[0].description,
            'api_filter':'&runType=AMPS_RNA&applicationGroup__name__iexact=immune_repertoire',
            'img': 'resources/img/appl_immuneRepertoire.png',
            'isActive': applicationGroup_immuneRepertoire[0].isActive if applicationGroup_immuneRepertoire else False,           
            'code': 12,
        },                  
        # Inherited disease
        {
            'tag': 'category_inherited_disease',
            'displayedName': category_inheritedDisease[0].displayedValue if category_inheritedDisease else "--",
            'api_filter': '&categories__icontains=inheritedDisease',
            'img': 'resources/img/appl_category_inherited_disease.png',
            'isActive': category_inheritedDisease[0].isActive if category_inheritedDisease else False,
            'ampliSeq_upload': True,   
            'code': 1,
        },
        # Oncology - hemeOnc
        {
            'tag': 'category_onco_hemeOnc',
            'displayedName': category_oncoHeme[0].displayedValue if category_oncoHeme else "--",
            'api_filter': '&categories__icontains=onco_heme',
            'img': 'resources/img/appl_category_onco_heme.png',
            'isActive': category_oncoHeme[0].isActive if category_oncoHeme else False,
            'ampliSeq_upload': True,            
            'code': 1,
        },
        # Oncology - immunology
        {
            'tag': 'category_onco_immune',
            'displayedName': category_immune[0].displayedValue if category_immune else "--",
            'api_filter': '&categories__icontains=onco_immune',
            'img': 'resources/img/appl_category_onco_immune.png',
            'isActive': category_immune[0].isActive if category_immune else False,
            'ampliSeq_upload': True,            
            'code': 1,
        },
        # Oncology - Liquid Biopsy
        {
            'tag': 'onco_liquidBiopsy',
            'displayedName': applicationGroup_tagSeq[0].description,
            'api_filter':'&runType=TAG_SEQUENCING&applicationGroup__name__iexact=onco_liquidBiopsy',
            'img': 'resources/img/appl_tagSequencing.png',
            'isActive': applicationGroup_tagSeq[0].isActive if applicationGroup_tagSeq else False,
            'code': 10,
        },
        # Oncology - solid tumor
        {
            'tag': 'category_onco_solidTumor',
            'displayedName': category_solidTumor[0].displayedValue if category_solidTumor else "--",
            'api_filter': '&categories__icontains=onco_solidTumor',
            'img': 'resources/img/appl_category_onco_solid_tumor.png',
            'isActive': category_solidTumor[0].isActive if category_solidTumor else False,
            'ampliSeq_upload': True,   
            'code': 8,
        },                  
        # Pharmacogenomics
        {
            'tag': 'pharmacogenomics',
            'displayedName':  applicationGroup_PGx[0].description,
            'api_filter': '&runType=AMPS&applicationGroup__name__iexact=PGx',
            'img': 'resources/img/appl_pgx.png',
            'isActive': applicationGroup_PGx[0].isActive if applicationGroup_PGx else False,
            'ampliSeq_upload': True,
            'code': 9,
        },
        # Reproductive
        {
            'tag': 'category_repro',
            'displayedName': category_repro[0].displayedValue if category_repro else "--",
            'api_filter': '&categories__icontains=repro',
            'img': 'resources/img/appl_category_reproSeq.png',
            'isActive': category_repro[0].isActive if category_repro else False,
            'ampliSeq_upload': True,            
            'code': 3,
        },                            
        # RNA Seq
        {
            'tag': 'rna_seq',
            'displayedName': 'RNA Seq',
            'api_filter': '&runType=RNA',
            'img': 'resources/img/appl_rnaSeq.png',
            'isActive': True,
            'code': 4,
        },
        # TargetSeq
        {
            'tag': 'targetseq',
            'displayedName': 'TargetSeq',
            'api_filter': '&runType=TARS',
            'img': 'resources/img/appl_targetSeq.png',
            'isActive': True,
            'code': 2,
        },
        # Whole Genome
        {
            'tag': 'whole_genome',
            'displayedName': 'Whole Genome',
            'api_filter': '&runType=WGNM',
            'img': 'resources/img/appl_wholeGenome.png',
            'isActive': True,
            'code': 3,
        },
        # 16S rRNA profiling
        {
            'tag': 'category_16s',
            'displayedName': category_16s[0].displayedValue if category_16s else "--",
            'api_filter': '&categories__icontains=16s',
            'img': 'resources/img/appl_category_16s_profile.png',
            'isActive': category_16s[0].isActive if category_16s else False,
            'code': 7,
        },                     
        # 16S Target Sequencing
        {
            'tag': '16s_targetseq',
            'displayedName': '16S Target Sequencing',
            'api_filter': '&runType=TARS_16S',
            'img': 'resources/img/appl_metagenomics.png',
            'isActive': True,
            'code': 7,
        },
    ]

    if not isOCP_enabled():
        for category in categories:
            category['api_filter'] += "&categories__regex=^((?!Oncomine))"

    return categories
