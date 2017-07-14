# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved

"""
Currently, TS software (release or patch) need be launched just because some kits or chips need to be launched.
This mechanism can be used to de-couple the software release/Patch from the actual consumable product launch
This feature can also used to update any DataBase models.
Note : This does not perform any Insert operation, only Update.
       Also, No Schema changes.
"""

import logging
import json
import httplib2
import urllib2
from ion import version as TS_version
from iondb.bin.add_or_update_systemPlanTemplates import add_or_updateSystemTemplate_OffCycleRelease
import re
import xmlrpclib
import iondb.utils
import traceback
import subprocess

from django.contrib.auth.decorators import login_required
from django.shortcuts import render_to_response, get_object_or_404
from django.http import HttpResponse, HttpResponsePermanentRedirect, HttpResponseRedirect
from django.template import RequestContext
from django.conf import settings
from django.core import urlresolvers
from django.contrib.auth.models import User

from iondb.rundb.ajax import render_to_json
from iondb.rundb.models import ReferenceGenome, FileMonitor, dnaBarcode, Plugin, GlobalConfig
from iondb.utils.utils import GetChangeLog, VersionChange, get_apt_cache
from django.db.models import get_model
from distutils.version import StrictVersion, LooseVersion

logger = logging.getLogger(__name__)


errorCode = {
    "E001": "TS version({0}) not supported",
    "E002": "Missing Product Info. Please try again later ({0})",
    "E003": "HTTPError {0}",
    "E004": "User ({0}) not authorized to update the Product. Please consult Torrent Suite Administrator",
    "E005": "Validation Error",
    #"E006" : "TBD. This error code should be overridden for future update".
    "E007": "Host not reachable. Please check your internet connectivity and try again ({0})",
    "E008": "{0}. Please check the network and try again",
    "E009": "System Template Add/Update Failed"
}


def get_productUpdateList(url=None):
    h = httplib2.Http()
    isValid = True
    productContents = []
    errorMsg = ""
    PRODUCT_UPDATE_LIST_URL = settings.PRODUCT_UPDATE_BASEURL + settings.PRODUCT_UPDATE_PATH
    try:
        response, content = h.request(PRODUCT_UPDATE_LIST_URL)
        if response['status'] == '200':
            productJson = json.loads(content)
            productContents = productJson['contents']
        if response['status'] == '404':
            isValid = False
            #errorMsg = "Product Information not available. Stay Tuned for New Products"
    except httplib2.ServerNotFoundError, err:
        isValid = False
        errorMsg = err
        logger.debug("httplib2.ServerNotFoundError: iondb.rundb.configure.updateProducts.get_productUpdateList %s", err)
    except Exception, err:
        logger.debug("urllib2.HTTPError: iondb.rundb.configure.updateProducts.get_productUpdateList %s", err)
    return (productContents, isValid, errorMsg)


def validate_product_fixture(productjsonURL):
    productInfo = None
    error = ""
    try:
        productInfo = urllib2.urlopen(productjsonURL)
    except urllib2.HTTPError, err:
        if err.code == 404:
            logger.debug("Missing Product Fixture{0}. Try again later".format(productjsonURL))
            error = errorCode['E002'].format(err)
        else:
            error = errorCode['E003'].format(err)
            logger.debug("urllib2.HTTPError: iondb.rundb.configure.updateProducts.validate_product_fixture %s", err)
    except urllib2.URLError, err:
        error = errorCode["E007"].format(err)
        logger.debug("urllib2.URLError: iondb.rundb.configure.updateProducts.validate_product_fixture %s", err)
    except Exception, err:
        error = errorCode['E008'].format(err)
        logger.debug("Error: iondb.rundb.configure.updateProducts.validate_product_fixture %s", err)

    return productInfo, error


def validate_product_updateVersion(productContents):
    ''' Updates and returns productContents:
        1) add product['done']=True if product was already updated (FileMonitor exists)
        2) remove product if it's version_req or version_max are not compatible with current TS version
    '''
    valid = []
    tsVersion = get_TSversion()
    downloads = FileMonitor.objects.filter(tags__contains="offCycleRel", status="Complete")

    for product in productContents:
        for download in downloads.filter(url=product['url']):
            try:
                updateVersion = download.tags.split("offCycleRel_")[1]
                if LooseVersion(updateVersion.strip()) >= LooseVersion(product['update_version'].strip()):
                    product['done'] = True
            except Exception, err:
                logger.debug("Error: iondb.rundb.configure.updateProducts.validate_product_updateVersion %s", err)

        version_match = TS_version_comparison(tsVersion, product['version_req'], product['version_max'])
        if version_match:
            """
              Validate and verify that the customer is allowed to view system Templates
              If "visible_to"  key does not exist in json or If value is "All":
                     products/sys template will be displayed for all customers
              For other values:
                  show/hide according to rundb_globalConfig settings Ex:enable_compendia_OCP
              if "visible_to" key present in json and it is empty/None -> do not show to any customers
            """
            if "visible_to" in product:
                isCustomerAllowedToView = verify_customer_access(product['visible_to'])
                if isCustomerAllowedToView:
                    valid.append(product)
            else:
                valid.append(product)
    return valid

def verify_customer_access(visible_to):
    logger.debug("Validate and verify that the customer is allowed to view system Templates")
    isCustEligible = False
    globalConfig = GlobalConfig.objects.get(name="Config")
    if visible_to:
        if visible_to.lower() == "all":
            isCustEligible = True
        else:
            try:
                isCustEligible = getattr(globalConfig, visible_to)
            except Exception,err:
                logger.debug("Error:configure.updateProducts.verify_customer_access %s" % err)

    return isCustEligible


def TS_version_comparison(tsVersion, version_req, version_max):
    # Compare local TS version with required and max TS Version, return True if passes
    checkReq = StrictVersion(tsVersion) >= StrictVersion(str(version_req.strip()))
    checkMax = StrictVersion(tsVersion) <= StrictVersion(str(version_max.strip()))
    return checkReq and checkMax


def get_TSversion():
    # return 3 digit TS version
    match = re.match(r'([0-9]+\.[0-9]+(\.[0-9]+)?)', str(TS_version))
    return (match.group(1));


def get_update_plugins():
    # look up user friendly names for the plugins
    pluginClient = xmlrpclib.ServerProxy(settings.IPLUGIN_STR)
    upgradeablePlugins = pluginClient.GetSupportedPlugins(list(), True, True)
    pluginPackages = upgradeablePlugins.keys()
    pluginUpdates = list()
    error = ""
    try:
        for pluginPackage in pluginPackages:
            # separately attempt to get the changelog in and ommit it in case it fails
            changeLog = VersionChange()
            try:
                changeLog = GetChangeLog(pluginPackage, upgradeablePlugins[pluginPackage]['AvailableVersions'][-1])
            except Exception as exc:
                logger.exception(exc)
                changeLog = VersionChange()

            currentVersion = upgradeablePlugins[pluginPackage]['CurrentVersion']
            plugin = Plugin.objects.get(packageName=pluginPackage, version=currentVersion)
            pluginUpdates.append({
                'name': plugin.name,
                'description': plugin.description,
                'currentVersion': currentVersion,
                'availableVersions': upgradeablePlugins[pluginPackage]['AvailableVersions'],
                'upgradable': upgradeablePlugins[pluginPackage]['UpgradeAvailable'],
                'changes': changeLog.Changes,
                'pk': plugin.pk
            })

    except Exception as err:
        logger.error(traceback.format_exc())
        error = err

    return {'pluginContents': pluginUpdates, 'error': error }


def get_update_products():
    productContents, isValid, error = get_productUpdateList() or []
    if isValid:
        productContents = validate_product_updateVersion(productContents)

    return {'productContents': productContents, 'error': error }


def update_product(name, update_version):
    ''' Update products via Off-Cycle Release path
    '''
    productContents, isValidNetwork, network_or_file_errorMsg = get_productUpdateList() or []
    if not isValidNetwork:
        raise Exception(network_or_file_errorMsg)

    product = [ p for p in productContents if name == p['name'] and update_version == p['update_version'] ]
    if not product:
        raise Exception('Invalid product name: %s version: %s' % (name, update_version))
    else:
        product = product[0]

    productjsonURL = product['url']
    productInfo, errMsg = validate_product_fixture(productjsonURL)
    if errMsg:
        logger.debug("Error: iondb.rundb.configure.updateProducts.update_product %s", errMsg)
        raise Exception(errMsg)

    data = json.loads(productInfo.read())
    productName = data['productName']
    
    if 'models_info' in data or 'sys_template_info' in data:
        monitor_pk = start_update_product(productName, productjsonURL, update_version)
    else:
        raise Exception('Did not find any objects to update')

    # create or update database objects
    modelsToUpdate = data.get('models_info')
    if modelsToUpdate:
        logger.debug("Going to update database objects via off-cycle release")
        # Validation for any invalid Model Name or Invalid PKs
        isValidModelInfo, errMsg = validate_modelObject(modelsToUpdate)
        if isValidModelInfo:
            for model in modelsToUpdate:
                pk = model.get('pk', None)
                modelName = model['model']
                fields = model['fields']
                modelObject = get_model('rundb', modelName)
                off_software_release_product_update(modelObject, pk, modelName, **fields)
        else:
            status = "Invalid Product/PK update."
            logger.debug("Error: iondb.rundb.configure.updateProducts.update_product %s", status)
            #delete the entry in File Monitor so that User can try again later
            updateFileMonitor(monitor_pk, status)
            raise Exception(errMsg)

    # add or update system templates
    sysTemplatesToUpdate = data.get('sys_template_info')
    if sysTemplatesToUpdate:
        logger.debug("Going to install system templates via off-cycle release")
        for sysTemp in sysTemplatesToUpdate:
            ctx_sys_temp_upd = add_or_updateSystemTemplate_OffCycleRelease(**sysTemp)
            if not ctx_sys_temp_upd['isValid']:
                status = errorCode['E009']
                #delete the entry in File Monitor so that User can try again later
                updateFileMonitor(monitor_pk, status)
                raise Exception(ctx_sys_temp_upd['msg'])

    updateFileMonitor(monitor_pk, 'Complete')


def validate_modelObject(modelsToUpdate):
    isValid = True
    errMsg = None
    errCnt = 0
    for model in modelsToUpdate:
        isModelValid = True
        modelName = model.get('model', None)
        pk = model.get('pk', None)
        try:
            modelObject = get_model('rundb', modelName)
        except Exception as err:
            isModelValid = False
            logger.debug("Unable to find the Model({0}) object to Update. {1}".format(modelName, err))
        if not modelObject or not isModelValid:
            errCnt = + 1
            isModelValid = False
            logger.debug("Unable to find the Model({0}) object to Update".format(modelName))
        if modelObject and isModelValid:
            try:
                if pk:
                    modelObject.objects.get(pk=pk)
            except Exception as err:
                errCnt = + 1
                logger.debug("Unable to Update the Model's({0}) PK ({1}). {2}".format(modelName, pk, err))
    if not errCnt:
        return isValid, errMsg
    else:
        isValid = False
        errMsg = "Invalid Product Update/Missing Product ID. Please consult Torrent Suite administrator."
        logger.debug("validation failure {0}".format(errMsg))
        return isValid, errMsg


def start_update_product(name, url, updateVersion, callback=None):
    tagsInfo = "offCycleRel_{0}".format(updateVersion)
    monitor = FileMonitor(name=name, url=url, tags=tagsInfo)
    monitor.status = "downloading"
    monitor.save()
    return monitor.id


def add_or_update_dnaBarcode(fields):
    name = fields.get("name")
    sequence = fields.get("sequence")
    if sequence == "ALL":
        updateObjs = dnaBarcode.objects.filter(name=name)
        if updateObjs.count() == 0:
            raise Exception("Barcode set %s is not installed" % name)
        else:
            fields.pop("sequence")
            updateObjs.update(**fields) 
    else:
        try:
            barcodeObj = dnaBarcode.objects.get(name=name, sequence=sequence)
        except dnaBarcode.DoesNotExist:
            barcodeObj = dnaBarcode()

        for key, value in fields.items():
            setattr(barcodeObj, key, value)
        barcodeObj.save()


def get_fk_model(model, fieldname):
    '''returns None if not foreignkey, otherswise the relevant model'''
    field_object, model, direct, m2m = model._meta.get_field_by_name(fieldname)
    if direct and field_object.get_internal_type() in ['ForeignKey', 'OneToOneField', 'ManyToManyField']:
        return field_object.rel.to
    return None

# process foreign key and store the object if exists
def process_model_fields(fields, modelObj):
    inValidFKs = {}
    processed_model_fields = {}

    for key, value in fields.items():
        isFKObj = None
        #check if field is a foreign key
        isFK = get_fk_model(modelObj, key)
        if isFK and value:
            if type(value) is list:
                value = value[0]

            qLists = ["id", "name", "uid", "runType"]
            for FK_field in qLists:
                try:
                    isFKObj = isFK.objects.get(**{FK_field: value})
                    processed_model_fields[key] = isFKObj
                    break
                except:
                    continue

            if not isFKObj:
                inValidFKs[key] = value
            continue
        processed_model_fields[key] = value

    return processed_model_fields, inValidFKs

def off_software_release_product_update(modelObj, pk, modelName=None, **fields):
    try:
        if modelName == "dnaBarcode":
            add_or_update_dnaBarcode(fields)
        else:
            # if any of the given field is a foreign key, update the field with foreign key object
            fields, inValidFKs = process_model_fields(fields, modelObj)
            if inValidFKs:
                errMsg = "Foreign key record does not exists for {0}".format(json.dumps(inValidFKs))
                logger.debug(errMsg)
                raise Exception(errMsg)
            elif pk:
                obj = modelObj.objects.get(pk=pk)
                for key, value in fields.items():
                    setattr(obj, key, value)
                    obj.save()
            else:
                obj = modelObj(**fields)
                obj.save()
    except Exception, e:
       logger.debug("Model Object creation failed, %s" % e)
       raise Exception("Model Object creation failed, %s" % e)


def updateFileMonitor(filemonitor_pk, status):
    if filemonitor_pk:
        fm_pk = FileMonitor.objects.get(pk=filemonitor_pk)
        if status == "Validation Error" or \
           status == "System Template Add/Update Failed":
            fm_pk.delete()
            logger.debug("Error: iondb.rundb.configure.updateProducts.updateFileMonitor: %s" % status)
        else:
            fm_pk.status = status
            fm_pk.save()


def get_update_packages():
    installPackages = [
        ('ion-chefupdates', 'Ion Chef scripts'),
    ]
    contents = []
    error = ""
    try:
        for name, description in installPackages:
            package, cache = get_apt_cache(name)
            if package.installed.version not in package.candidate.version:
                availableVersions = package.versions.keys()
                installableVersions = [version for version in availableVersions if StrictVersion(version) > StrictVersion(package.installed.version)]
                if installableVersions or package.is_upgradable:
                    contents.append({
                        'name': name,
                        'description': description,
                        'currentVersion': package.installed.version,
                        'candidateVersion': package.candidate.version,
                        'availableVersions': installableVersions,
                        'upgradable': package.is_upgradable
                    })

    except Exception as err:
        logger.error(traceback.format_exc())
        error = err
        
    return {"packageContents": contents, "error": error}


def update_package(name, version):
    # call external install script
    logger.debug('Install package %s version %s' % (name, version))
    cmd = ['sudo','/opt/ion/iondb/bin/sudo_utils.py', 'install_ion_package', name, version]
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # print error to stdout to avoid getting dpkg error messages
        error, _ = process.communicate()
    except Exception as err:
        logger.debug("Error: iondb.rundb.configure.updateProducts.update_package: Sub Process execution failed %s" % err)
        error = err

    if process.returncode:
        logger.debug("Error: iondb.rundb.configure.updateProducts.update_package: %s" % error)
        raise Exception(error)
