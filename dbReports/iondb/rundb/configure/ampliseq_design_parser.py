# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved
import json
import logging
import os
import re

import requests
import urlparse
from django.conf import settings

from iondb.rundb.models import FileMonitor, ContentUpload
from iondb.rundb.plan.ampliseq import AmpliSeqPanelImport
from iondb.utils.utils import convert

logger = logging.getLogger(__name__)

TIMEOUT_LIMIT_SEC = settings.REQUESTS_TIMEOUT_LIMIT_SEC

"""
 This parses all the existing designs, both user created(logged in User) panels
 and fixed designs panels available in AmpliSeq.com
 The related panels will be displayed on Grid with required filter to import
"""


def _get_all_ampliseq_designs(api_url, user, password):
    """Get all the ampliseq design panels including Ampliseq HD
    "ampliseq-hd=true" filter is used to get the HD panels
    """

    url = urlparse.urljoin(settings.AMPLISEQ_URL, api_url)
    response = {}
    errMsg = None
    all_ampseq_designs = []

    try:
        response = requests.get(url, auth=(user, password), timeout=TIMEOUT_LIMIT_SEC)
        response.raise_for_status()
        design_data = response.json()
        all_ampseq_designs = design_data.get("AssayDesigns", [])
    except ValueError as decodeError:
        errMsg = "Unable to parse the response from {url}: {err}".format(
            url=url, err=str(decodeError)
        )
    except (
        requests.ConnectionError,
        requests.Timeout,
        requests.HTTPError,
    ) as serverError:
        errMsg = "%s. Please check your network connection and try again." % str(
            serverError
        )
    except Exception as err:
        errMsg = (
            "There was an unknown error when contacting Ampliseq.com for Design Panels: %s"
            % str(err)
        )

    if errMsg:
        response = {"status": "500", "err_msg": errMsg}
        logger.error(errMsg)

    return response, all_ampseq_designs


def getASPanelImportStatus(panel_type, design_id, solution_id=None, sourceUrl=None):
    url = os.path.join(settings.AMPLISEQ_URL.strip("/"), sourceUrl.strip("/"))
    ampliseqPanels = FileMonitor.objects.filter(url=url, tags="ampliseq_template")
    downloadStatus_filemonitor = None
    if ampliseqPanels:
        for ampliseqPanel in ampliseqPanels:
            downloadStatus_filemonitor = ampliseqPanel
            if ampliseqPanel.status == "Complete":
                panelName = ampliseqPanel.name
                ampliseqPanels_downloded = ContentUpload.objects.filter(
                    upload_type="ampliseq", file_path__icontains=panelName
                )
                for panelDownloaded in ampliseqPanels_downloded:
                    if "completed" in panelDownloaded.status.lower():
                        return panelDownloaded

                if ampliseqPanels_downloded and len(ampliseqPanels_downloded) > 0:
                    contentUploadStatus = ampliseqPanels_downloded[0]
                    return contentUploadStatus

    return downloadStatus_filemonitor


def ampliseq_concurrent_api_call(user, password, api_url):
    if "tmpldesign" in api_url:
        return get_ampliseq_fixed_designs(user, password, api_url)

    return get_ampliseq_designs(user, password, api_url)


def get_ampliseq_designs(user, password, api_url):
    ctx = {}
    ampliseq_url = settings.AMPLISEQ_URL
    chemistryType = "Ampliseq"
    if "hd" in api_url:
        chemistryType = "AmpliseqHD"

    response, designs = _get_all_ampliseq_designs(api_url, user, password)
    if designs:
        ctx["unordered_solutions"] = []
        ctx["ordered_solutions"] = []
        unordered_tmpList = []
        ordered_tmpList = []
        for design in designs:
            for template in design.get("DesignSolutions", []):
                ampliSeq = AmpliSeqPanelImport(data=template)
                solution, meta = ampliSeq.handle_versioned_plans()
                solution_id = solution["id"]
                configurationChoices = solution["configuration_choices"]
                sourceUrl = solution["resultsUri"]
                uploadHistoryLink = None
                status = None
                downloadObj = getASPanelImportStatus(
                    "ordered", design["id"], solution_id, sourceUrl
                )
                if downloadObj:
                    try:
                        uploadHistoryLink = "/rundb/uploadstatus/{0}/".format(
                            downloadObj.id
                        )
                        status = downloadObj.status
                    except Exception as Err:
                        logger.error(
                            "Unknown error %s" % Err
                        )  # do not crash if any unknown issue

                if solution.get("ordered", False):
                    panelType = template.get("panelType") or "on-demand"
                    ordered_tmpList.append(
                        {
                            "configuration_choices": configurationChoices,
                            "id": solution_id,
                            "solution_id": solution_id,
                            "displayedID": template.get(
                                "request_id_and_solution_ordering_id", ""
                            ),
                            "design_id": design["id"],
                            "name": template["designName"],
                            "genome": template["genome"],
                            "pipeline": template["pipeline"],
                            "ampliseq_url": ampliseq_url,
                            "chemistryType": chemistryType,
                            "panelType": panelType,
                            "status": status,
                            "uploadHistory": uploadHistoryLink,
                            "recommended_application": "",
                        }
                    )
                else:
                    panelType = template.get("panelType") or "made-to-order"
                    unordered_tmpList.append(
                        {
                            "configuration_choices": configurationChoices,
                            "id": solution_id,
                            "displayedID": template.get(
                                "request_id_and_solution_ordering_id", ""
                            ),
                            "solution_id": solution_id,
                            "design_id": design["id"],
                            "name": template["designName"],
                            "genome": template["genome"],
                            "pipeline": template["pipeline"],
                            "ampliseq_url": ampliseq_url,
                            "chemistryType": chemistryType,
                            "panelType": panelType,
                            "status": status,
                            "uploadHistory": uploadHistoryLink,
                            "recommended_application": "",
                        }
                    )
        unordered_tmpList = convert(unordered_tmpList)
        ordered_tmpList = convert(ordered_tmpList)
        ctx["unordered_solutions"] = unordered_tmpList
        ctx["ordered_solutions"] = ordered_tmpList
        return ctx
    else:
        return {}


def get_fixed_designs_list(fixed_design_data):
    # creates fixed_solutions: a list of dictionaries with :
    #    Design id : Ready-to-Use ampliseq panel ID
    #    Type : Community or Fixed Panel - used to to link to Ampliseq website
    #    Genome : Supported genome by the specific panel.
    #    ChemistryType : Ampliseq or AmpliseqHD
    # final_fixed_soln_data : #Lists all the design panels available for all the pipeline
    # fixed_ids_choices : This is used to compare the parametized files available and show the warnings

    fixed = []
    chemistryType = "ampliseq"
    ampliseq_url = settings.AMPLISEQ_URL
    fixedDesigns = fixed_design_data.get("TemplateDesigns", [])
    for template in fixedDesigns:
        ampliSeq = AmpliSeqPanelImport(data=template)
        data, meta = ampliSeq.handle_versioned_plans()
        fixed.append(data)

    if fixed:
        ordered_solutions = []
        tmpList = []
        tmpFixedsolLists = []
        fixed_solutions = filter(lambda x: x["status"] == "ORDERABLE", fixed)
        for design in fixed_solutions:
            designID = design["id"]
            sourceUrl = design["resultsUri"]
            configurationChoices = design["configuration_choices"]
            uploadHistoryLink = None
            status = None
            downloadObj = getASPanelImportStatus("fixed", designID, sourceUrl=sourceUrl)

            if downloadObj:
                uploadHistoryLink = "/rundb/uploadstatus/{0}/".format(downloadObj.id)
                status = downloadObj.status
            tmpDict = {"id": designID, "configuration_choices": configurationChoices}
            description = design["description"]
            rec_app = None
            if description:
                desc = re.sub(r"\r\n", "", description.strip())
                pattern = (
                    "<td>\s+<strong>Recommended Application</strong>\s+(.*?)\s+</td>"
                )
                rec_app = re.search(pattern, desc)
                if rec_app:
                    rec_app = rec_app.group(1)
            else:
                description = (
                    "No description available for this panel from Ampliseq.com"
                )
            panelType = design.get("panelType") or "ready-to-use"
            tmpFixedSolDict = {
                "id": designID,
                "design_id": designID,
                "displayedID": designID,
                "solution_id": "",
                "genome": design["genome"],
                "name": design["name"],
                "type": design["type"],
                "pipeline": design["pipeline"],
                "ampliseq_url": ampliseq_url,
                "chemistryType": chemistryType,
                "panelType": panelType,
                "status": status,
                "uploadHistory": uploadHistoryLink,
                "description": description,
                "recommended_application": rec_app,
            }
            tmpList.append(tmpDict)
            tmpFixedsolLists.append(tmpFixedSolDict)
        # This is used to compare the parametized files available and show the warnings
        fixed_ids_choices = json.dumps(convert(tmpList))

        # Lists all the design panels available for the specific pipeline
        final_fixed_soln_data = tmpFixedsolLists

    return final_fixed_soln_data, ordered_solutions, fixed_ids_choices


def get_ampliseq_fixed_designs(user, password, api_url):
    ctx = {}
    try:
        url = urlparse.urljoin(settings.AMPLISEQ_URL, api_url)
        response = requests.get(url, auth=(user, password), timeout=TIMEOUT_LIMIT_SEC)
        response.raise_for_status()
        fixed_design_data = response.json()
        fixed_solutions, ordered_solutions, fixed_ids_choices = get_fixed_designs_list(
            fixed_design_data
        )
        ctx["fixed_ids_choices"] = fixed_ids_choices
        ctx["fixed_solutions"] = fixed_solutions
    except ValueError as decodeError:
        ctx["http_error"] = "Could not parse the responses from AmpliSeq.com."
        errMsg = "Unable to parse the response from {url}: {err}".format(
            url=url, err=str(decodeError)
        )
        logger.error(errMsg)
    except (
        requests.ConnectionError,
        requests.Timeout,
        requests.HTTPError,
    ) as serverError:
        ctx["http_error"] = "Could not connect to AmpliSeq.com."
        errMsg = "%s. Please check your network connection and try again." % str(
            serverError
        )
        logger.error(errMsg)
    except Exception as Err:
        ctx["http_error"] = "Could not connect to AmpliSeq.com."
        logger.error("There was a unknown error when contacting ampliseq: %s" % Err)

    return ctx
