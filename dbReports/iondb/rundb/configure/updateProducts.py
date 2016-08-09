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
import ion.utils.TSversion
import re

from django.contrib.auth.decorators import login_required
from django.shortcuts import render_to_response, get_object_or_404
from django.http import HttpResponse, HttpResponsePermanentRedirect, HttpResponseRedirect
from django.template import RequestContext
from django.conf import settings
from django.core import urlresolvers
from django.contrib.auth.models import User

from iondb.rundb.ajax import render_to_json
from iondb.rundb.models import ReferenceGenome, FileMonitor
from django.db.models import get_model
from distutils.version import StrictVersion

logger = logging.getLogger(__name__)


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
            errorMsg = "Product Information not available. Stay Tuned for New Products"
    except httplib2.ServerNotFoundError, err:
        isValid = False
        errorMsg = err
    return (productContents,isValid,errorMsg)

def TS_version_comparison(tsVersion, version_req, url, **ctx):
    isValid = True
    # Compare local TS version with Required TS Version for any update

    if StrictVersion(str(tsVersion.strip())) < StrictVersion(str(version_req.strip())):
        isValid = False
        ctx['versionMismatch'] = True
        ctx['versionMismatch_url'] = url
        ctx['errCode'] = 'E001'
        ctx['msg'] = "TS version({0}) not supported".format(tsVersion)
    ctx['isValid'] = isValid

    return (isValid, ctx)

def validate_product_fixture(productjsonURL, **ctx):
    isValid = True
    productInfo = None
    try:
        productInfo = urllib2.urlopen(productjsonURL)
    except urllib2.HTTPError, err:
        isValid = False
        ctx['versionMismatch_url'] = productjsonURL
        if err.code == 404:
            logger.debug("Missing Product Fixture{0}. Try again later".format(productjsonURL))
            ctx['errCode'] = 'E002'
            ctx['msg'] = "Missing Product Info. Please try again later ({0})".format(err)
        else:
            ctx['errCode'] = 'E003'
            ctx['msg'] = err
    except urllib2.URLError, err:
        isValid = False
        ctx['versionMismatch_url'] = productjsonURL
        ctx['errCode'] = 'E007'
        ctx['msg'] = "Host not reachable. Please check your internet connectivity and try again ({0})".format(err)
        logger.debug(ctx['msg'])
    except Exception, err:
        isValid = False
        ctx['versionMismatch_url'] = productjsonURL
        ctx['errCode'] = 'E008'
        ctx['msg'] = "{0}. Please check the network and try again".format("err")
        logger.debug(ctx['msg'])
    ctx['isValid'] = isValid
    return (isValid, productInfo, ctx)

def validate_user_isStaff(userName,**ctx):
    isValid = True
    try:
        user = User.objects.get(username=userName)
        userAcesssDenied = False
        isStaff = user.is_staff
    except:
        isStaff = None
        userAcesssDenied = True
    if not isStaff or userAcesssDenied:
        isValid = False
        ctx['errCode'] = 'E005'
        ctx['isAccessDenied'] = "disabled"
        ctx['msg'] = "User ({0}) not authorized to update the Product. Please consult Torrent Suite Administrator".format(userName)
    ctx['isValid'] = isValid

    return (isValid, ctx)

def validate_product_updateVersion(productContents, downloads, tsVersion):
    for product in productContents:
        for download in downloads:
            tagsInfo = download.tags
            try:
                updateVersion = tagsInfo.split("offCycleRel_")[1]
            except:
                updateVersion = None
            download.updateVersion = updateVersion
            if download.status == "Complete" and download.url == product['url'] and updateVersion >= product['update_version']:
                product['disable'] = "disabled"
        product['TSVersion'] = tsVersion
    return productContents

def getOnlyTwo_ThreeDigits_TSversion():
    #TS-11832: TS release comparison should extend to the patch release version if available
    versionsAll, versionTS = ion.utils.TSversion.findVersions()
    getOnlyTwoDigits = re.match(r'([0-9]+\.[0-9]+(\.[0-9]+)?)', str(versionTS))

    return (getOnlyTwoDigits.group(1));

@login_required
def update_product(request):
    tsVersion = getOnlyTwo_ThreeDigits_TSversion()
    productContents, isValidNetwork, network_or_file_errorMsg = get_productUpdateList() or []

    if not isValidNetwork:
        ctx = { 
               'isValidNetwork' : isValidNetwork, 
               'errorMsg' : network_or_file_errorMsg 
               }
        return render_to_response("rundb/configure/updateProducts.html", ctx,
                  context_instance=RequestContext(request))

    if not productContents:
        ctx = {
               'isValidNetwork' : isValidNetwork,
               'productContents' : "N/A",
               'errorMsg' : "Product information is not available. Please check this link later for New Products"
               }
        return render_to_response("rundb/configure/updateProducts.html", ctx,
                  context_instance=RequestContext(request))

    downloads = FileMonitor.objects.filter(tags__contains="offCycleRel").order_by('-updated')
    productContents = validate_product_updateVersion(productContents, downloads, tsVersion)

    ctx = {
            'isValidNetwork' : isValidNetwork,
       		'productContents': productContents,
       		'downloads' : downloads,
            'TSVersion' : tsVersion,
    	  }
    (isValid,ctx) = validate_user_isStaff(request.user, **ctx)
    if not isValid:
        return render_to_response("rundb/configure/updateProducts.html", ctx,
     	         context_instance=RequestContext(request))

    if request.method == "POST" and request.POST.get("offCycleUpdate"):
        url = request.POST.get("kitchip_url", None)
        version_req = request.POST.get("version_req", None)
        update_version = request.POST.get("update_version", None)
        (isValid, ctx) = TS_version_comparison(tsVersion, version_req, url, **ctx)
        if not isValid:
            return render_to_response("rundb/configure/updateProducts.html", ctx,
        		context_instance=RequestContext(request))
        logger.debug("TS version is valid, going to the next step {0} with Product Update {1}".format(url, tsVersion))

        if url and productContents:
            productjsonURL = url
            (isValid,productInfo, ctx) = validate_product_fixture(productjsonURL, **ctx)
            if not isValid:
                return render_to_response("rundb/configure/updateProducts.html", ctx,
                        context_instance=RequestContext(request))
            data = json.loads(productInfo.read())
            productName = data['productName']
            modelsToUpdate = data.get('models_info')
            if modelsToUpdate:
                monitor_pk = start_update_product(productName,url, update_version)
                # Validation for any invalid Model Name or Invalid PKs
                isValidModelInfo, errMsg = validate_modelObject(modelsToUpdate)

                if isValidModelInfo:
                    for model in modelsToUpdate:
                        pk = model['pk']
                        modelName = model['model']
                        fields = model['fields']
                        modelObject = get_model('rundb', modelName)
                        #"validate_modelObject" validates all the Models and PKs specified in the Product JSON Fixture
                        #Go ahead and update the models
                        off_software_release_product_update(modelObject,pk, modelName, **fields)
                    status = 'Complete'
                else:
                    logger.debug("Invalid Product/PK update")
                    status = "Validation Error"
                    ctx['isValid'] = False
                    ctx['errCode'] = 'E005'
                    ctx['msg'] = errMsg
                    ctx['invalidProductUrl'] = url
                    #delete the entry in File Monitor so that User can try again later
                    updateFileMonitor(monitor_pk, status)
                    return render_to_response("rundb/configure/updateProducts.html", ctx,
                        context_instance=RequestContext(request))
                updateFileMonitor(monitor_pk, status)
            else:
                ctx['isValid'] = False
                ctx['errCode'] = 'E004'
                ctx['msg'] = "You are not authorized to Update the Product. Please consult Torrent Suite administrator."
                return render_to_response("rundb/configure/updateProducts.html", ctx,
        				context_instance=RequestContext(request))    
        return HttpResponseRedirect(urlresolvers.reverse("update_product"))

    return render_to_response("rundb/configure/updateProducts.html", ctx,
        context_instance=RequestContext(request))

def validate_modelObject(modelsToUpdate):
    isValid = True
    errMsg = None
    errCnt = 0
    for model in modelsToUpdate:
        isModelValid = True
        modelName = model.get('model',None)
        pk = model.get('pk', None)
        try:
            modelObject = get_model('rundb', modelName)
        except Exception as err:
            isModelValid = False
            logger.debug("Unable to find the Model({0}) object to Update. {1}".format(modelName, err))
        if not modelObject or not isModelValid:
            errCnt =+ 1
            isModelValid = False
            logger.debug("Unable to find the Model({0}) object to Update".format(modelName))
        if modelObject and isModelValid:
            try:
                modelObject.objects.get(pk=pk)
            except Exception as err:
                errCnt =+ 1
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

def off_software_release_product_update(modelObject,pk, modelName = None, **fields):
    UpdatePk = modelObject.objects.get(pk=pk)
    if UpdatePk:
        for key, value in fields.items():
            setattr(UpdatePk, key, value)
        UpdatePk.save()
    
def updateFileMonitor(filemonitor_pk, status):
    if filemonitor_pk:
        fm_pk = FileMonitor.objects.get(pk=filemonitor_pk)
        if status == "Validation Error":
            fm_pk.delete()
        else:
            fm_pk.status  = status
            fm_pk.save()


