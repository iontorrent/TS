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
import os
import glob
import time
import shutil
import tempfile
from iondb.rundb import tasks

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
from iondb.plugins.manager import pluginmanager

logger = logging.getLogger(__name__)


errorCode = {
    "E001": "TS version({0}) not supported",
    "E002": "Missing Product Info. Please try again later ({0})",
    "E003": "HTTPError {0}",
    "E004": "User ({0}) not authorized to update the Product. Please consult Torrent Suite Administrator",
    "E005": "Validation Error",
    "E006" : "Invalid product file found. Please check",
    "E007": "Host not reachable. Please check your internet connectivity and try again ({0})",
    "E008": "{0}. Please check the network and try again",
    "E009": "System Template Add/Update Failed"
}

PRODUCT_UPDATE_PATH_LOCAL = os.path.join(settings.OFFCYCLE_UPDATE_PATH_LOCAL, "products")


def get_productUpdateList(url=None, offcycle_type = "online"):
    h = httplib2.Http()
    isValid = True
    productContents = []
    errorMsg = ""
    PRODUCT_UPDATE_LIST_URL = os.path.join(settings.PRODUCT_UPDATE_BASEURL, settings.PRODUCT_UPDATE_PATH)
    product_local_main = os.path.join(PRODUCT_UPDATE_PATH_LOCAL, "main.json")
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

    # Handle the users with and without internet connection
    # Generate the main.json only if user has already uploaded the product zip/json and only when product exists
    if os.path.exists(PRODUCT_UPDATE_PATH_LOCAL) and os.listdir(PRODUCT_UPDATE_PATH_LOCAL):
        isValid = True
        if offcycle_type == 'manual':
            isValid, errorMsg = generate_mainJson_local(productContents)
        if isValid and os.path.exists(product_local_main):
            try:
                errorMsg = ""
                productJson_local = json.loads(open(product_local_main).read())
                productContents_local = productJson_local["contents"]
                for product in productContents_local:
                    manual_update_version = product["update_version"].strip()
                    manual_product_name = product["name"].strip()
                    productDone = False
                    for p in productContents:
                        if (manual_product_name == p["name"].strip() and
                                LooseVersion(manual_update_version) <= LooseVersion(p["update_version"].strip())):
                                productDone = True
                    if not productDone:
                        productContents.append(product)
            except Exception, err:
                errorMsg = errorCode['E005'].format(err)
                logger.debug("Error: iondb.rundb.configure.updateProducts.validate_product_fixture %s", err)
    return (productContents, isValid, errorMsg)


def generate_mainJson_local(onlineMainContents=None):
    # Generate the main.json contents for the manual upload and store into /results/uploads/offcycle/products folder
    # Compare the uploaded products contents with the online contents and existing old manual main.json,
    # Save the unique product contents into local main.json
    offline_main = {}
    mainFileMeta = {}
    available_manualUploadProducts = []
    mainFileContents = []
    isValid = True
    error = None
    productDone = None
    local_mainFile = os.path.join(PRODUCT_UPDATE_PATH_LOCAL, 'main.json')


    # Get the product info from the main file if exists
    if os.path.exists(local_mainFile):
        mainFileData = json.loads(open(local_mainFile).read())
        mainFileContents = mainFileData["contents"]
        available_manualUploadProducts = [(p["name"].strip(), p["update_version"].strip()) for p in mainFileContents]
        mainFileMeta = mainFileData["meta"]

    # Get the unique product listing from manual and online offcycle update
    if onlineMainContents:
        onlineMainProducts = [(p["name"].strip(),p["update_version"]) for p in onlineMainContents]
        available_Products = list(set(available_manualUploadProducts + onlineMainProducts))
    else:
        available_Products = list(set(available_manualUploadProducts))

    # Construct the main.json on the fly using the uploaded product.json files
    # Ignore any online product which clashes with the offline products for those users with internet option
    # skip the product if update version is equal/less than the installed version

    for productFile in glob.glob(PRODUCT_UPDATE_PATH_LOCAL +'/*.json'):
        if os.path.basename(productFile) == "main.json":
            continue
        try:
            mainContentDict = {}
            productFileContent = json.loads(open(productFile).read())
            p_name = productFileContent.get("name") or productFileContent.get("productName")
            if p_name:
                p_name = p_name.strip()
            if available_Products:
                for available_Product in available_Products:
                    name, update_version = available_Product # extract product meta data from tuple
                    productDone = False
                    if (name == p_name):
                        if (LooseVersion(productFileContent["update_version"].strip()) <= LooseVersion(update_version.strip())):
                            productDone = True
                            break
                if productDone:
                    continue # proceed with the next productFile

            # backward compatibility
            mainContentDict["name"] = p_name
            mainContentDict["version_req"] = productFileContent.get("version_req") or productFileContent.get("version_required")
            mainContentDict["version_max"] = productFileContent["version_max"]
            mainContentDict["url"] = os.path.basename(productFile)
            mainContentDict["update_version"] = productFileContent["update_version"]
            mainContentDict["product_desc"] =  productFileContent.get("product_desc") or productFileContent.get("productDesc")
            mainContentDict["offcycle_type"] = "manual"
            mainFileContents.append(mainContentDict)
        except Exception,Err:
            logger.debug("Invalid product file uploaded {0}. Please check".format(Err))
            raise Exception("Invalid product file uploaded. Please check %s" % Err)

    offline_main["contents"] = mainFileContents
    # get the meta if exists or generate one if not
    if mainFileMeta:
        offline_main["meta"] = mainFileMeta
    else:
        mainFileMeta["dateCreated"] = time.strftime("%d/%m/%Y")
        mainFileMeta["lastUpdated"] = time.strftime("%d/%m/%Y")
        offline_main["meta"] = {
            "dateCreated" : time.strftime("%m/%d/%Y"),
            "lastUpdated" : time.strftime("%m/%d/%Y")
        }

    # Finally create the main.json and store the product info
    with open(local_mainFile, mode='w') as main:
        json.dump(offline_main, main)

    return (isValid, error)

def validate_product_fixture(productjsonURL):
    productInfo = None
    error = ""
    product_individual_local = os.path.join(PRODUCT_UPDATE_PATH_LOCAL, productjsonURL)
    if "http://" in productjsonURL:
        try:
            product = urllib2.urlopen(productjsonURL)
            productInfo = json.loads(product.read())
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
    elif os.path.exists(product_individual_local):        # validate for the offline product
        try:
            productInfo = json.loads(open(product_individual_local).read())
        except Exception, err:
            error = errorCode['E005'].format(err)
            logger.debug("Error: iondb.rundb.configure.updateProducts.validate_product_fixture %s", err)
    else:
        error = errorCode['E006']

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


def get_update_products(offcycle_type = "online"):
    productContents, isValid, error = get_productUpdateList(offcycle_type = offcycle_type) or []
    if isValid:
        productContents = validate_product_updateVersion(productContents)
        if productContents and offcycle_type == "manual":
            manual_product_install(productContents)

    return {'productContents': productContents, 'error': error }


def manual_product_install(productContents):
    for product in productContents:
        offcycleType = product.get("offcycle_type", None)
        productDone = product.get("done", False)
        # make sure that the online product is not updated via manual automatic update
        if not productDone and offcycleType == "manual":
            update_product(product["name"], product["update_version"])


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
    data, errMsg = validate_product_fixture(productjsonURL)
    if errMsg:
        logger.debug("Error: iondb.rundb.configure.updateProducts.update_product %s", errMsg)
        raise Exception(errMsg)

    productName = data.get('productName') or data.get('name')
    
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
    installPackages = settings.SUPPORTED_INSTALL_PACKAGES
    contents = []
    error = ""
    try:
        for name, description in installPackages:
            package, cache = get_apt_cache(name)
            current_version = None
            if package.installed:
                current_version = package.installed.version

            if not current_version or current_version not in package.candidate.version:
                availableVersions = package.versions.keys()

                if not current_version:
                    installableVersions = [version for version in availableVersions]
                else:
                    installableVersions = [version for version in availableVersions if StrictVersion(version) > StrictVersion(current_version)]

                if installableVersions or package.is_upgradable:
                    contents.append({
                        'name': name,
                        'description': description,
                        'currentVersion': current_version,
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


def InstallProducts(pathToProductFile, extension, fileName=None):
    """
    Installs a off-cycle bundle from a zip file
    :param pathToZip: A local file path to the zip file in question
    """

    logger.info("Starting install process for offline offcycle at " + pathToProductFile)

    # check that the file exists
    if not os.path.exists(pathToProductFile):
        raise Exception("Attempt to install offcycle bundle from zip failed because " + pathToProductFile + " does not exist.")

    zipSize = os.path.getsize(pathToProductFile)
    if zipSize == 0:
        raise Exception("The zip file " + pathToProductFile + " is of zero size and has no contents.")

    try:
        # single product update using a single json file
        if extension == "json":
            if not validate_productFile(pathToProductFile, fileName):
                raise Exception("Error: %s" % errorCode['E006'])
        else:
            # Proceed below if the product file is a zip bundle
            # create a temporary directory to extract the zip file to
            pathToExtracted = tempfile.mkdtemp()
            try:
                # extract the zip file
                tasks.extract_zip(pathToProductFile, pathToExtracted, logger=logger)

                listOfDirectories = [name for name in os.listdir(pathToExtracted) if
                                     os.path.isdir(os.path.join(pathToExtracted, name))]
                if len(listOfDirectories) > 1:
                    raise Exception("The zip file contained a number of directories where the specification only calls for one.  "
                                    "This has caused an ambiguous state where the product Update cannot be divined.")
                else:
                    if len(listOfDirectories) == 1:
                        extractedDir = listOfDirectories[0].strip()
                        grepFiles = os.path.join(pathToExtracted, extractedDir, "*.json")
                    else:
                        # handle when product file is zipped without directory
                        grepFiles = os.path.join(pathToExtracted, "*.json")

                    grepFiles_filtered = filter(os.path.isfile, glob.glob(grepFiles))
                    if not grepFiles_filtered:
                        raise Exception("The zip file " + fileName + " has no product contents to install")
                    for productFile in grepFiles_filtered:
                        if not validate_productFile(productFile):
                            raise Exception("Error: %s" % errorCode['E006'])
            finally:
                shutil.rmtree(pathToExtracted, True)
    except Exception, Err:
        raise Exception("The product file uploaded has some issues. %s" % Err)

def validate_productFile_params(productFileContent):
    isValid = True
    # handle backward compatability
    name = productFileContent.get("name") or productFileContent.get("productName")
    version_req = productFileContent.get("version_required") or productFileContent.get("version_req")
    product_desc = productFileContent.get("productDesc") or productFileContent.get("product_desc")

    if not name or not product_desc or not version_req:
        isValid = False

    if isValid:
        required_product_keys = ("version_max", "update_version")
        if not (set((required_product_keys)) <= set(productFileContent)):
            isValid = False

    return isValid, name

def validate_productFile(productFile, fileName = None):
    """
     - Validate the product file  : manual upload
     - Error out if any missing fields in uploaded product
    """
    isValid = True
    name = None
    offcycle_localPath = settings.OFFCYCLE_UPDATE_PATH_LOCAL
    offcycleProducts_localPath = os.path.join(offcycle_localPath, "products")

    if not os.path.exists(offcycleProducts_localPath):
        os.makedirs(offcycleProducts_localPath)
        os.chmod(offcycleProducts_localPath, 0777)

    if not fileName:
        fileName = os.path.basename(productFile)

    destinationFilePath = os.path.join(offcycleProducts_localPath, fileName)
    productFileContent = json.loads(open(productFile).read())

    # validate,
    #   - mandatory fields
    #   - version_required vs system version
    #   - name or Productname exists
    #   - no content/object provided in the product file

    isValid, name = validate_productFile_params(productFileContent)
    if isValid:
        tsVersion = get_TSversion()
        if not (TS_version_comparison(tsVersion,
                                        productFileContent.get('version_req') or
                                        productFileContent.get('version_required'),
                                        productFileContent['version_max'])):
            raise Exception("%s for %s." % (errorCode['E001'].format(tsVersion), name))

        if not (('models_info' in productFileContent and productFileContent['models_info']) or
                ('sys_template_info' in productFileContent and productFileContent["sys_template_info"])):
            logger.debug("Did not find any object to update. Missing either models_info or system_template_info fields")
            raise Exception('Missing product content, please consult Torrent Suite administrator.')

        if os.path.exists(destinationFilePath):
            existingProductData = json.loads(open(destinationFilePath).read())
            existingProductName = existingProductData.get("name") or existingProductData.get("productName")
            isProductOld = StrictVersion(str(productFileContent.get('update_version'))) <= StrictVersion(str(existingProductData.get('update_version')))
            if name == existingProductName and isProductOld:
                logger.info("Product already installed on this Torrent Server. Going to skip the product install: %s" % destinationFilePath)
                os.remove(productFile)
                return isValid
            else:
                logger.info("Product already installed on this Torrent Server but the product contents are new. Going to overwrite with the updated version: %s" % destinationFilePath)
            os.remove(destinationFilePath)
        shutil.move(productFile, destinationFilePath)
    else:
        os.remove(productFile)

    return isValid

def InstallDeb(pathToDeb, actualFileName = None):
    """
    Install a misc. package[Ex:ion-chefupdates] from a deb package file
    :param pathToDeb: A path to the deb file which will be installed.
    """

    # do some sanity checks on the package
    if not os.path.exists(pathToDeb):
        raise Exception("No file at " + pathToDeb)

    # call the ion plugin deb install script which handles the misc deb packages installation
    p = subprocess.Popen(["sudo", "/opt/ion/iondb/bin/ion_package_install_deb.py", pathToDeb, "miscDeb"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out, err = p.communicate()

    # raise if any error
    if p.returncode:
        logger.debug("System Error at iondb.rundb.configure.updateProduct.InstallDeb : %s" % err)
        if "SystemError" in err:
            err = ("Invalid file (%s) uploaded or file corrupted. Please check the file and try again." % actualFileName)
        elif "conflicts" in err or "not part of the offcycle" in err:
            err = err
        else:
            err = ("Something went wrong. Check the file (%s) uploaded and try again. "
                   "If problem exists again, please consult with your Torrent Suite administrator." % actualFileName)
        raise Exception(err)

    if 'ion-plugin' in actualFileName:
        pluginmanager.rescan()