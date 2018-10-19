# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
# Design doc @ https://confluence.amer.thermo.com/display/TS/Tech+design+proposal+%3A+TS-API+for+Flexible+Chef+Workflow+Design+Proposal
from iondb.rundb.models import Chip, KitInfo
import logging
from django.db.models import Q
logger = logging.getLogger(__name__)
from dateutil.parser import parse as parse_date
import requests
import math
import datetime
import logging
import multiprocessing
import re

logger = logging.getLogger(__name__)


# validation of each mesh nodes
def validate_remote_mesh(mesh_id):
    """ Used in a multiprocess pool to validate the remote TS mesh node """
    response = {}
    mesh_api = 'http://%s/rundb/api/v1/ionmeshnode/' % 'localhost'
    validated_mesh_data = requests.get(mesh_api + str(mesh_id) + "/validate/").json()
    error = validated_mesh_data.get('error', None)
    status = validated_mesh_data.get("status")
    hostname = validated_mesh_data.get("hostname")
    if error and "Good" not in status:
        logger.exception("plan.chef_flexible_workflow_validator : %s" % error)
        response["allowRunToContinue"] = True
        response["errorCodes"] = ["W400"]
        response["detailMeshMessages"] = error + "(" + hostname + ")"

    return response


class ChefFlexibleWorkflowValidator(object):

    def __init__(self):
        self.response = {
            "allowRunToContinue": True,
            "errorCodes": [],
            "numberReagentSerialUsage": 0,
            "numberSolutionSerialUsage": 0,
            "detailMessages": dict(),
            "hoursSinceReagentFirstUse": "",
            "hoursSinceSolutionFirstUse": ""
        }
        self.error_warning_codes = {
            "W100": "time between solutions cartridge usage has been exceeded ({0}) days, continuing run is not recommended",
            "W200": "time between reagents cartridge usage has been exceeded ({0}) days, continuing run is not recommended",
            "E300": "No. of reagent and solution usage do not match",
            "E301": "No. of reagents usage is >= {0}",
            "E302": "No. of Solutions usage is >= {0}",
            "E303": "Both the no. of reagents and solutions cartridge usages are >= {0}",
            "E304": "Invalid chef flexible workflow kit",
            "E305": "Missing chef inputs filter option {0}",
            "E306": "Specified chef current time does not sync up with system current time",
            "E307": "Invalid kit specified",
            "E308": "time between solutions usage has been exceeded the max limit of ({0}) days, continuing run is not allowed",
            "E309": "time between reagents usage has been exceeded the max limit of ({0}) days, continuing run is not allowed",
            "E310": "Invalid chef current time. Check the time format",
            "E311": "exceededs the GS1 standard (should not be greater than {0} chars long)",
            "W400": "Ion Mesh warning: Remote server has incompatible software version ({0})",
            "E401": "Invalid user credentials",
            "E402": "{0}", # captures unknown exceptions during composite experiment api call
            "W403": "No Ion Mesh configured currently",
            "E404": "Ion Mesh error: Could not fetch runs from one or more Torrent Server ({0})",
            "E405": "Ion Mesh error:  Could not make a connection to one or more remote server ({0})",
            "W406": "{0}",  # captures any other ion mesh warnings
            "W407": "Ion Mesh warning:  One or more Torrent Server ({0}) has too many results to search. Only experiments newer than {1} are considered"
        }

    def validate_inputParams(self, params=None):
        response = self.response
        reqInputs = ["chefReagentsSerialNum", "chefSolutionsSerialNum", "kitName"]

        keysNotExists = [input for input in reqInputs if input not in params]
        if keysNotExists:
            errCode = "E305"
            response["allowRunToContinue"] = False
            response["errorCodes"] = [errCode]
            response["detailMessages"][errCode] = self.error_warning_codes[errCode].format(keysNotExists)

        if response["allowRunToContinue"]:
            # Compare the chefCurrentTime with the system time to avoid negative time comparison
            now = datetime.datetime.utcnow()
            delta = datetime.timedelta(minutes=5)
            pastTimeBuffer = (now - delta)
            futureTimeBuffer = (now + delta)
            try:
                chefCurrentTime = parse_date(params["chefCurrentTime"])
            except:
                errCode = "E310"
                response["allowRunToContinue"] = False
                response["errorCodes"] = [errCode]
                response["detailMessages"][errCode] = self.error_warning_codes[errCode].format(keysNotExists)
                return response

            if chefCurrentTime < pastTimeBuffer or chefCurrentTime > futureTimeBuffer:
                errCode = "E306"
                response["allowRunToContinue"] = False
                response["errorCodes"] = [errCode]
                response["detailMessages"][errCode] = self.error_warning_codes[errCode].format(keysNotExists)

            #validate that the serial nos are adhered to the GS1 standards
            keys = ["chefReagentsSerialNum", "chefSolutionsSerialNum"]
            for key in keys:
                errMsg = self.validate_GS1_standards(params[key])
                if errMsg:
                    errCode = "E311"
                    if errCode in response["errorCodes"]:
                        response["detailMessages"][errCode] = ', '.join(keys) + " : " + errMsg
                    else:
                        response["detailMessages"][errCode] = key + " : " + errMsg
                    response["allowRunToContinue"] = False
                    response["errorCodes"] = [errCode]

        return response


    def validate_chefFlexibleWorkflowKit(self, params=None):
        response = self.response
        # validate the Chef Flexible workflow Kit
        chefFlexibleKit = params["kitName"]
        selectedKits = KitInfo.objects.filter((Q(kitType__in=["TemplatingKit", "IonChefPrepKit"]) & Q(isActive=True)) &
                                              Q(name__iexact=chefFlexibleKit))
        """
        Checking only chipType(550) for chef flexible workflow KIT may not be sufficient because
        the chip type could have cross over to different kits, so validating this specific flexible workflow parameter
        """
        if selectedKits:
            defaultCartridgeUsageCount = selectedKits[0].defaultCartridgeUsageCount or ""
            cartridgeExpirationDayLimit = selectedKits[0].cartridgeExpirationDayLimit or ""
            cartridgeBetweenUsageAbsoluteMaxDayLimit = selectedKits[0].cartridgeBetweenUsageAbsoluteMaxDayLimit or ""
            if not defaultCartridgeUsageCount or not cartridgeExpirationDayLimit:
                errCode = "E304"
                response["allowRunToContinue"] = False
                response["detailMessages"][errCode] = self.error_warning_codes[errCode]
                response["errorCodes"] = [errCode]

            response["defaultCartridgeUsageCount"] = defaultCartridgeUsageCount
            response["cartridgeExpirationDayLimit"] = cartridgeExpirationDayLimit
            response["cartridgeBetweenUsageAbsoluteMaxDayLimit"] = cartridgeBetweenUsageAbsoluteMaxDayLimit
        else:
            errCode = "E307"
            response["allowRunToContinue"] = False
            response["detailMessages"][errCode] = self.error_warning_codes[errCode]
            response["errorCodes"] = [errCode]
        return response

    def checkCartrideExpiration(self, oldestChefReagentStartTime, oldestChefSolutionStartTime, params, dbParams):
        """
        No alarming flag "allowRunToContinue = false" since chef team has decided to continue the run with
        the expired reagents and solutions cartridge
        """
        response = self.response
        response["allowRunToContinue"] = True
        detailMessage = []
        endDateTime = parse_date(params["chefCurrentTime"]).utcnow()
        """
        Find the hoursSinceReagentFirstUse
        Integer number of hours between (oldest associated experiment using Reagent) and (chefCurrentTime).
        Rounded down to the nearest hour is OK
        """
        if oldestChefReagentStartTime:
            timeDelta = (endDateTime - oldestChefReagentStartTime).total_seconds()
            timeDelta_hours = timeDelta/60/60
            response["hoursSinceReagentFirstUse"] = math.floor(timeDelta_hours)
        """
        Find hoursSinceSolutionFirstUse
        Integer number of hours between (oldest associated experiment using Solution) and (chefCurrentTime).
        Rounded down to the nearest hour is OK
        """
        if oldestChefSolutionStartTime:
            timeDelta = (endDateTime - oldestChefSolutionStartTime).total_seconds()
            timeDelta_hours = timeDelta/60/60
            response["hoursSinceSolutionFirstUse"] = math.floor(timeDelta_hours)

        cartridgeExpirationDayLimit = dbParams["cartridgeExpirationDayLimit"]
        cartridgeBetweenUsageAbsoluteMaxDayLimit = dbParams["cartridgeBetweenUsageAbsoluteMaxDayLimit"]

        # send error if it exceeds the cartridgeBetweenUsageAbsoluteMaxDayLimit days (hardstop)
        if oldestChefSolutionStartTime and ((endDateTime - oldestChefSolutionStartTime).days > cartridgeBetweenUsageAbsoluteMaxDayLimit):
            errCode = "E308"
            response["allowRunToContinue"] = False
            response["errorCodes"].append(errCode)
            response["detailMessages"][errCode] = self.error_warning_codes[errCode].format(cartridgeBetweenUsageAbsoluteMaxDayLimit)

        if oldestChefReagentStartTime and ((endDateTime - oldestChefReagentStartTime).days > cartridgeBetweenUsageAbsoluteMaxDayLimit):
            errCode = "E309"
            response["allowRunToContinue"] = False
            response["errorCodes"].append(errCode)
            response["detailMessages"][errCode] = self.error_warning_codes[errCode].format(cartridgeBetweenUsageAbsoluteMaxDayLimit)

        if response["allowRunToContinue"]:
            if oldestChefSolutionStartTime and ((endDateTime - oldestChefSolutionStartTime).days > cartridgeExpirationDayLimit):
                errCode = "W100"
                response["errorCodes"].append(errCode)
                response["detailMessages"][errCode] = self.error_warning_codes[errCode].format(cartridgeExpirationDayLimit)

            if oldestChefReagentStartTime and ((endDateTime - oldestChefReagentStartTime).days > cartridgeExpirationDayLimit):
                errCode = "W200"
                response["errorCodes"].append(errCode)
                response["detailMessages"][errCode] = self.error_warning_codes[errCode].format(cartridgeExpirationDayLimit)

        return response

    def validate_get_chefCartridge_info(self, data=None, params=None, dbParams = None):
        """
        parse the composite exp. data for the and validate below:
            - Solutions, Reagents cartridge usage expiration limit, Usage count
              should not exceed the flexible kit's configuration limit
            - mismatch in reagents and solutions cartridges usages
            - Any Mesh failures like connectivity/network issue, version incompatibility etc.,
        """
        response = self.response
        num_reagentsExps = 0
        num_solutionsExps = 0
        allReagentStartTimes = []
        allSolutionsStartTimes = []
        oldestReagentStartTime = None
        oldestSolutionStartTime = None
        detailMessage = []

        for exp in data["objects"]:
            chefStartTime = None
            exp_chefStartTime = exp.get("chefStartTime", None)
            exp_chefReagentsSerialNum = exp.get("chefReagentsSerialNum", None)
            exp_chefSolutionsSerialNum = exp.get("chefSolutionsSerialNum", None)
            if exp_chefStartTime:
                chefStartTime = parse_date(exp_chefStartTime).replace(tzinfo=None)
            if exp_chefReagentsSerialNum == params["chefReagentsSerialNum"]:
                num_reagentsExps += 1
                if chefStartTime:
                    allReagentStartTimes.append(chefStartTime)
            if exp_chefSolutionsSerialNum == params["chefSolutionsSerialNum"]:
                num_solutionsExps += 1
                if chefStartTime:
                    allSolutionsStartTimes.append(chefStartTime)

        if allReagentStartTimes:
            oldestReagentStartTime = min(allReagentStartTimes)
        if allSolutionsStartTimes:
            oldestSolutionStartTime = min(allSolutionsStartTimes)

        response["numberReagentSerialUsage"] = num_reagentsExps
        response["numberSolutionSerialUsage"] = num_solutionsExps
        cartridgeUsageLimit = dbParams["cartridgeUsageCount"]

        if response["allowRunToContinue"]:
            self.checkCartrideExpiration(oldestReagentStartTime, oldestSolutionStartTime, params, dbParams)

        if num_reagentsExps >= cartridgeUsageLimit and num_solutionsExps >= cartridgeUsageLimit:
            errCode = "E303"
            response["allowRunToContinue"] = False
            response["errorCodes"].append(errCode)
            response["detailMessages"][errCode] = self.error_warning_codes[errCode].format(cartridgeUsageLimit)
        elif num_reagentsExps >= cartridgeUsageLimit:
            errCode = "E301"
            response["allowRunToContinue"] = False
            response["errorCodes"].append(errCode)
            response["detailMessages"][errCode] = self.error_warning_codes[errCode].format(cartridgeUsageLimit)
        elif num_solutionsExps >= cartridgeUsageLimit:
            errCode = "E302"
            response["allowRunToContinue"] = False
            response["errorCodes"].append(errCode)
            response["detailMessages"][errCode] = self.error_warning_codes[errCode].format(cartridgeUsageLimit)

        if num_reagentsExps != num_solutionsExps:
            errCode = "E300"
            response["allowRunToContinue"] = False
            response["errorCodes"].append(errCode)
            response["detailMessages"][errCode] = self.error_warning_codes[errCode]
        if data["warnings"]:
            warnings = data["warnings"]
            fetchRunsIssueServers = []
            fetchRunsIssue = False
            otherIssue = False # prepare to capture any other mesh issue
            otherWarnings = []
            # do not break when there is any future changes in the error data structure in Mesh
            if type(warnings) is list:
                for warning in warnings:
                    if "Could not fetch runs" in warning:
                        fetchRunsIssue = True
                        fetchRunsIssueServers.append(re.search("\(s\) (.*)\!", warning).group(1))
                    elif "have too many results to display" in warning:
                        response["errorCodes"].append("W407")
                        m = re.search("\(s\) (.*) have too many results to display\. (.*) (\d+\-\d+\-\d+)", warning)
                        errMsg = self.error_warning_codes["W407"].format(m.group(1), m.group(3))
                        response["detailMessages"]["W407"] = errMsg
                    else:
                        otherIssue = True
                        #If any other uncategorized warning contains string (s), which is reserved for dynamic message generation for Chef, replace it as "s"
                        warning.replace('(s)','s')
                        otherWarnings.append(warning)
            else:
                otherIssue = True
                otherWarnings.append(warnings)

            if fetchRunsIssue:
                response["allowRunToContinue"] = False
                response["errorCodes"].append("E404")
                fetchRunsIssueServers = ", ".join(fetchRunsIssueServers)
                errMsg = self.error_warning_codes["E404"].format(fetchRunsIssueServers)
                response["detailMessages"]["E404"] = errMsg
            if otherIssue:
                response["errorCodes"].append("W406")
                response["detailMessages"]["W406"] = ", ".join(otherWarnings)

        return response

    def validate_ionMesh_nodes(self):
        response = self.response
        try:
            mesh_api = 'http://%s/rundb/api/v1/ionmeshnode/' % 'localhost'
            mesh_nodes = requests.get(mesh_api)
            if mesh_nodes.status_code == 401:
                errCode = "E401"
                response["allowRunToContinue"] = False
                response["errorCodes"].append(errCode)
                response["detailMessages"][errCode] = self.error_warning_codes[errCode]
            elif mesh_nodes.status_code != 200:
                errCode = "E402"
                response["allowRunToContinue"] = False
                response["errorCodes"].append(errCode)
                response["detailMessages"] = self.error_warning_codes[errCode].format(mesh_nodes.reason)

            if response["allowRunToContinue"]:
                data = mesh_nodes.json()
                mesh_ids = [node['id'] for node in data["objects"]]
                if len(mesh_ids):
                    # validate the each mesh nodes in parallel
                    pool = multiprocessing.Pool(processes=len(mesh_ids))
                    results = pool.map(validate_remote_mesh, mesh_ids)
                    all_error_codes = ["".join(res["errorCodes"]) for res in results if res.get("errorCodes")]
                    all_individual_error = [res["detailMeshMessages"] for res in results if res.get("detailMeshMessages")]
                    if all_individual_error:
                        # decided to leave this as just warning since it blocks the user to proceed for
                        # any known mesh issues
                        response["allowRunToContinue"] = True
                        isCompatibleSoftVer = False
                        incompatibleNodes = []
                        isConnectionIssue = False
                        connectionIssueNodes = []
                        othersIssues = False
                        othersIssuesMsgs = []
                        for err in all_individual_error:
                            if "incompatible software version" in err:
                                isCompatibleSoftVer = True
                                incompatibleNodes.append(re.search("\((.*)\)", err).group(1))
                            elif "Could not make a connection":
                                isConnectionIssue = True
                                connectionIssueNodes.append(re.search("\((.*)\)", err).group(1))
                            else:
                                othersIssues = True
                                othersIssuesMsgs.append(err)
                        if isCompatibleSoftVer:
                            errCode = "W400"
                            response["errorCodes"].append(errCode)
                            errMsg = self.error_warning_codes[errCode].format(", ".join(incompatibleNodes))
                            response["detailMessages"][errCode] = errMsg
                        if isConnectionIssue:
                            response["allowRunToContinue"] = False
                            errCode = "E405"
                            response["errorCodes"].append(errCode)
                            errMsg = self.error_warning_codes[errCode].format(", ".join(connectionIssueNodes))
                            response["detailMessages"][errCode] = errMsg
                        if othersIssues:
                            errCode = "W406"
                            response["errorCodes"].append(errCode)
                            errMsg = "Ion Mesh warning: " + self.error_warning_codes[errCode].format(", ".join(othersIssuesMsgs))
                            response["detailMessages"][errCode] = errMsg
                else:
                    errCode = "W403"
                    response["errorCodes"].append(errCode)
                    response["detailMessages"][errCode] = self.error_warning_codes[errCode]
        except Exception as exc:
            errCode = "E501"
            logger.exception("plan.chef_flexible_workflow_validator : %s" % str(exc))
            response["allowRunToContinue"] = False
            response["errorCodes"].append(errCode)
            response["detailMessages"][errCode] = str(exc)

        return response

    def validate_GS1_standards(self, value):
        #adhere to GS1 standards
        GS1 = 20 # GS1 standard max limit
        errMsg = None
        errCode = "E311"
        if len(str(value)) > GS1:
            errMsg = self.error_warning_codes[errCode].format(GS1)

        return errMsg
