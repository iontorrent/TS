#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved

"""
 - This takes the ampliseq plan.json and converts into plan which is compatatible to the Torrent suite
 - Uploads the template/Post the Plan info to TS via API call
"""

import traceback
from iondb.bin.djangoinit import *
from iondb.rundb import models


class processAmpliSeqPanel(object):
    _internalError = None
    ampliSeq_plan_metaData = {}
    errMessage = {
        "E001": "The Imported Panel is not supported for the selected Instrument Type : {0}",
        "E002": "The supported Instrument Types for this panel {0}",
    }

    def __init__(self):
        self.adapter = None
        self.applicationGroupDisplayedName = ""
        self.autoAnalyze = True
        self.autoName = None
        # Set if isBarcoded
        self.barcodeId = ""
        self.barcodedSamples = {}
        self.bedfile = ""
        self.regionfile = ""
        self.chipBarcode = None
        self.chipType = None
        self.controlSequencekitname = ""
        self.cycles = None
        self.date = ""
        self.expName = ""
        self.flows = ""
        self.flowsInOrder = ""
        self.forward3primeadapter = "ATCACCGACTGCCCATAGAGAGGCTGAGAC"
        self.platform = None
        self.irworkflow = ""
        self.isFavorite = False
        self.isPlanGroup = False
        self.isReusable = True
        self.isReverseRun = False
        self.isSystem = False
        self.isSystemDefault = False
        self.libkit = None
        self.library = ""
        self.libraryKey = "TCAG"
        self.chipTecDfltAmbient = None
        self.chipTecSlope = None
        self.chipTecMinThreshold = None
        self.manTecDfltAmbient = None
        self.manTecSlope = None
        self.manTecMinThreshold = None
        self.categories = None
        # Kit
        self.librarykitname = ""
        self.metaData = {}
        self.notes = ""
        self.origin = "ampliseq.com"
        self.pairedEndLibraryAdapterName = ""
        self.parentPlan = None
        self.planDisplayedName = None
        self.planExecuted = False
        self.planExecutedDate = None
        self.planName = None
        self.planPGM = None
        self.planStatus = ""
        self.preAnalysis = True
        self.reverse3primeadapter = None
        self.reverse_primer = None
        self.reverselibrarykey = None
        self.runMode = ""
        self.runType = None
        self.runname = None
        self.sample = ""
        self.sampleDisplayedName = ""
        self.samplePrepKitName = ""
        self.selectedPlugins = ""
        self.seqKitBarcode = None
        # Kit
        self.sequencekitname = ""
        self.sseBedFile = ""
        self.storageHost = None
        self.storage_options = "A"
        # Kit
        self.templatingKitName = ""
        self.usePostBeadfind = True
        self.usePreBeadfind = True
        self.username = "ionuser"

    def update(self, d):
        fields = list(self.__dict__.keys())
        for key, value in list(d.items()):
            if key in fields:
                setattr(self, key, value)
            else:
                raise Exception("Incorrect field key: %s" % key)

    def _set_error(self, errMsg):
        self._internalError = errMsg

    def get_error(self):
        return self._internalError

    def get_plan_stub(self):
        return self.__dict__

    def get_pluginDetails(self):
        plugin_details = self.ampliSeq_plan_metaData["plugin_details"]
        alignmentargs_override = None
        if plugin_details:
            if (
                "variantCaller" in plugin_details
                and "userInput" in plugin_details["variantCaller"]
            ):
                try:
                    if "meta" not in plugin_details["variantCaller"]["userInput"]:
                        plugin_details["variantCaller"]["userInput"]["meta"] = {}

                    plugin_details["variantCaller"]["userInput"]["meta"][
                        "built_in"
                    ] = True

                    plugin_details["variantCaller"]["userInput"]["meta"][
                        "compatibility"
                    ] = {
                        "panel": "/rundb/api/v1/contentupload/"
                        + str(self.ampliSeq_plan_metaData["upload_id"])
                        + "/"
                    }

                    if (
                        "configuration"
                        not in plugin_details["variantCaller"]["userInput"]["meta"]
                    ):
                        plugin_details["variantCaller"]["userInput"]["meta"][
                            "configuration"
                        ] = ""

                    if (
                        plugin_details["variantCaller"]["userInput"]["meta"][
                            "configuration"
                        ]
                        == "custom"
                    ):
                        plugin_details["variantCaller"]["userInput"]["meta"][
                            "configuration"
                        ] = ""

                    if (
                        "ts_version"
                        not in plugin_details["variantCaller"]["userInput"]["meta"]
                    ):
                        plugin_details["variantCaller"]["userInput"]["meta"][
                            "ts_version"
                        ] = "5.2"

                    if (
                        "name"
                        not in plugin_details["variantCaller"]["userInput"]["meta"]
                    ):
                        plugin_details["variantCaller"]["userInput"]["meta"]["name"] = (
                            "Panel-optimized - "
                            + self.ampliSeq_plan_metaData["design_name"]
                        )

                    if (
                        "repository_id"
                        not in plugin_details["variantCaller"]["userInput"]["meta"]
                    ):
                        plugin_details["variantCaller"]["userInput"]["meta"][
                            "repository_id"
                        ] = ""

                    if (
                        "tooltip"
                        not in plugin_details["variantCaller"]["userInput"]["meta"]
                    ):
                        plugin_details["variantCaller"]["userInput"]["meta"][
                            "tooltip"
                        ] = "Panel-optimized parameters from AmpliSeq.com"

                    plugin_details["variantCaller"]["userInput"]["meta"][
                        "user_selections"
                    ] = {
                        "chip": "s5",
                        "frequency": "germline",
                        "library": "ampliseq",
                        "panel": "/rundb/api/v1/contentupload/"
                        + str(self.ampliSeq_plan_metaData["upload_id"])
                        + "/",
                    }
                    if self.platform == "proton":
                        plugin_details["variantCaller"]["userInput"]["meta"][
                            "user_selections"
                        ]["chip"] = "proton_p1"
                    elif self.chipType in self.get_s5_chips():
                        plugin_details["variantCaller"]["userInput"]["meta"][
                            "user_selections"
                        ]["chip"] = self.chipType

                    if (
                        "tmapargs"
                        in plugin_details["variantCaller"]["userInput"]["meta"]
                    ):
                        alignmentargs_override = plugin_details["variantCaller"][
                            "userInput"
                        ]["meta"]["tmapargs"]
                except Exception:
                    self._set_error(traceback.print_exc())

                    print(self.get_error())
                    return None, None
        return plugin_details, alignmentargs_override

    @staticmethod
    def get_s5_chips():
        s5_chips = (
            models.Chip.objects.filter(isActive=True, instrumentType="S5")
            .values_list("name", flat=True)
            .order_by("name")
        )
        return s5_chips

    @staticmethod
    def decorate_S5_instruments(instrument_type):
        if instrument_type in processAmpliSeqPanel.get_s5_chips():
            instrument_type = "S5 Chip: " + instrument_type.upper()
        return instrument_type

    def get_applicationGroupDisplayedName(self):
        metaData = self.ampliSeq_plan_metaData
        app_group_name = ""
        run_type = metaData["run_type"]
        applicationGroupDescription = metaData["applicationGroupDescription"]
        if applicationGroupDescription:
            app_group = models.ApplicationGroup.objects.get(
                description=applicationGroupDescription
            )
            app_group_name = app_group.description
        else:
            run_type_model = models.RunType.objects.get(runType=run_type)
            app_group = run_type_model.applicationGroups.filter(isActive=True).order_by(
                "id"
            )[0]
            app_group_name = app_group.name
        return app_group, app_group_name

    def update_chip_inst_type(self, app=None):
        # Parse the meta data and set the default chip type and instrument type for the Panel which is being imported
        metaData = self.ampliSeq_plan_metaData
        # The plan.json has PGM, Proton (legacy), and S5 chip type as choice
        instrument_type = metaData["choice"]
        chip_type = "540"
        # "choice": "None" will be in the JSON from 3.6 schema imports
        decoratedInstType = None
        if instrument_type == "None":
            instrument_type = "s5"
        if instrument_type in self.get_s5_chips():
            chip_type = instrument_type
        elif app and app.applicationGroup: # TS-18344
            if app.defaultChipType in self.get_s5_chips():
                chip_type = app.defaultChipType
        self.chipType = chip_type

        return chip_type

    def set_decorated_inst_type(self):
        # Parse the meta data and set the default chip type and instrument type for the Panel which is being imported
        metaData = self.ampliSeq_plan_metaData
        instrument_type = metaData["choice"]
        decoratedInstType = None
        if instrument_type == "None":
            instrument_type = "s5"
        if instrument_type in self.get_s5_chips():
            decoratedInstType = self.decorate_S5_instruments(instrument_type)
            instrument_type = "s5"
        self.platform = instrument_type
        self.decoratedInstType = decoratedInstType

    def get_applProductObj(self):
        metaData = self.ampliSeq_plan_metaData
        run_type = metaData["run_type"]
        plan_name = metaData["plan_name"]
        application_product_id = metaData["application_product_id"]
        available_choice = metaData["available_choice"]

        app_group, app_group_name = self.get_applicationGroupDisplayedName()

        self.set_decorated_inst_type()
        instrument_type = self.platform
        decoratedInstType = self.decoratedInstType

        print(
            "plan_json processing plan_name=%s; run_type=%s; instrument_type=%s"
            % (plan_name, run_type, instrument_type)
        )

        # Access the appl product obj directly using the application_product_id in plan.json
        if application_product_id:
            app = models.ApplProduct.objects.get(id=application_product_id)
            if app.instrumentType != instrument_type:
                if decoratedInstType in available_choice:
                    print(
                        "Application Product ID does not match the Instrument Type. Please check"
                    )
                else:
                    if not decoratedInstType:
                        decoratedInstType = instrument_type.upper()
                    print(self.errMessage["E001"].format(decoratedInstType))
                    print(self.errMessage["E002"].format(available_choice))
                sys.exit(1)
        else:
            """
              This is for backward compatability
              construct the query to get the applProduct object as close as possible
              if we have application_product_id in plan.json the below code will not executed
              TS strongly recommends to specify app_product_id in plan.json
            """
            try:
                queryString = {
                    "applType__runType": run_type,
                    "isActive": True,
                    "instrumentType": instrument_type,
                }

                if instrument_type == "proton" and plan_name.endswith("_Hi-Q"):
                    queryString["productName__contains"] = "_Hi-Q"
                else:
                    query_app_group = queryString
                    query_app_group["applicationGroup"] = app_group
                    if models.ApplProduct.objects.filter(**query_app_group):
                        queryString = query_app_group
                    else:
                        queryString["isDefault"] = True
                app = models.ApplProduct.objects.filter(**queryString)[0]
            except Exception:
                # uncomment the below line during debugging for more detailed exception
                # print traceback.print_exc()
                print("Exception during application product parsing.")
                print()
                if available_choice and decoratedInstType in available_choice:
                    traceback.print_exc()
                else:
                    if not decoratedInstType:
                        decoratedInstType = instrument_type.upper()
                    print(self.errMessage["E001"].format(decoratedInstType))
                    print(self.errMessage["E002"].format(available_choice))
                sys.exit(1)

        return app, app_group_name, instrument_type

    def get_defaultTemplateKit(self, app):
        defaultTemplateKit = app.defaultTemplateKit and app.defaultTemplateKit.name
        if not defaultTemplateKit:
            defaultTemplateKit = (
                app.defaultIonChefPrepKit and app.defaultIonChefPrepKit.name
            )

        return defaultTemplateKit

    def get_flowOrder(self, app):
        defaultFlowOrder = app.defaultFlowOrder
        if not defaultFlowOrder:
            defaultFlowOrder = ""

        return defaultFlowOrder and defaultFlowOrder.flowOrder


def plan_json(
    meta, upload_id, target_regions_bed_path, hotspots_bed_path, sse_bed_path
):
    planName = str(meta["design"]["design_name"].encode("ascii", "ignore"))
    choice = meta.get("choice")

    if choice and "None" not in choice:
        planName += " - " + meta.get("choice")
    ampliSeq_plan_metaData = {
        "run_type": meta["design"]["plan"].get("runType", None),
        "plan_name": planName,
        "applicationGroupDescription": meta["design"]["plan"].get(
            "applicationGroup", None
        ),
        "application_product_id": meta["design"]["plan"].get(
            "application_product_id", None
        ),
        "available_choice": meta["design"]["plan"].get("available_choice", None),
        "choice": meta.get(
            "choice", "None"
        ),  # "None" this is default choice for legacy 3.6
        "design_name": meta["design"].get("design_name", None),
        "categories": meta["design"]["plan"].get("categories", None),
        "plugin_details": meta["design"]["plan"].get("selectedPlugins", {}),
        "upload_id": upload_id,
        "templatingKitName": meta["design"]["plan"].get("templatingKitName", None),
        "samplePrepKitName": meta["design"]["plan"].get("samplePrepKitName", None),
        "sequencekitname": meta["design"]["plan"].get("sequencekitname", None),
        "barcodeId": meta["design"]["plan"].get("barcodeId", None),
        "flows": meta["design"]["plan"].get("flows", None),
        "chipTecDfltAmbient": meta["design"]["plan"].get("chipTecDfltAmbient", None),
        "chipTecSlope": meta["design"]["plan"].get("chipTecSlope", None),
        "chipTecMinThreshold": meta["design"]["plan"].get("chipTecMinThreshold", None),
        "manTecDfltAmbient": meta["design"]["plan"].get("manTecDfltAmbient", None),
        "manTecSlope": meta["design"]["plan"].get("manTecSlope", None),
        "manTecMinThreshold": meta["design"]["plan"].get("manTecMinThreshold", None),
        "metaData": meta["design"]["plan"].get("metaData", {})
    }

    ampliSeqTemplate = processAmpliSeqPanel()
    ampliSeqTemplate.ampliSeq_plan_metaData = ampliSeq_plan_metaData

    app, app_group_name, instrument_type = (
        ampliSeqTemplate.get_applProductObj()
    )
    chip_type = ampliSeqTemplate.update_chip_inst_type(app=app)

    plugin_details, alignmentargs_override = ampliSeqTemplate.get_pluginDetails()

    isWarningExists = ampliSeqTemplate.get_error()
    if isWarningExists:
        print("WARNING while generating plan entry")
        print(ampliSeqTemplate.get_error())

    plan_stub = {
        "planDisplayedName": ampliSeq_plan_metaData["plan_name"],
        "planName": ampliSeq_plan_metaData["plan_name"],
        "runType": ampliSeq_plan_metaData["run_type"],
        "chipType": chip_type,
        "applicationGroupDisplayedName": app_group_name,
        "categories": ampliSeq_plan_metaData.get("categories", None),
        "barcodeId": ampliSeq_plan_metaData.get("barcodeId") or app.defaultBarcodeKitName,
        "bedfile": target_regions_bed_path,
        "regionfile": hotspots_bed_path,
        "sseBedFile": sse_bed_path,
        "flows": ampliSeq_plan_metaData.get("flows") or app.defaultFlowCount,
        "flowsInOrder": ampliSeqTemplate.get_flowOrder(app),
        "platform": instrument_type,
        "library": meta["reference"],
        "librarykitname": ampliSeq_plan_metaData.get("librarykitname") or app.defaultLibraryKit and app.defaultLibraryKit.name,
        "samplePrepKitName": ampliSeq_plan_metaData.get("samplePrepKitName") or app.defaultSamplePrepKit and app.defaultSamplePrepKit.name,
        "selectedPlugins": plugin_details,
        "sequencekitname": ampliSeq_plan_metaData.get("sequencekitname") or app.defaultSequencingKit and app.defaultSequencingKit.name,
        "templatingKitName": ampliSeq_plan_metaData.get("templatingKitName") or ampliSeqTemplate.get_defaultTemplateKit(app),
        "chipTecDfltAmbient": ampliSeq_plan_metaData.get("chipTecDfltAmbient", None),
        "chipTecSlope": ampliSeq_plan_metaData.get("chipTecSlope", None),
        "chipTecMinThreshold": ampliSeq_plan_metaData.get("chipTecMinThreshold", None),
        "manTecDfltAmbient": ampliSeq_plan_metaData.get("manTecDfltAmbient", None),
        "manTecSlope": ampliSeq_plan_metaData.get("manTecSlope", None),
        "manTecMinThreshold": ampliSeq_plan_metaData.get("manTecMinThreshold", None),
        "metaData": ampliSeq_plan_metaData.get("metaData", {})
    }
    # Update source template meta for Dynamic Tec Params
    templateMetadata = {
        "fromTemplate": plan_stub.get("planName"),
        "isAmpliseqTecParamEnabled": True if plan_stub.get("manTecDfltAmbient") or plan_stub.get("chipTecDfltAmbient") else False,
        "fromTemplateChipType": plan_stub.get("chipType"),
        "fromTemplateSequenceKitname": plan_stub.get("sequencekitname"),
        "fromAmpliseqTemplateSource": True
    }
    plan_stub.get("metaData").update(templateMetadata)

    ampliSeqTemplate.update(plan_stub)
    plan_stub = ampliSeqTemplate.get_plan_stub()

    return plan_stub, alignmentargs_override
