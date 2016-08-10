# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.models import Project, PlannedExperiment
from django.contrib.auth.models import User
from django.conf import settings

from iondb.rundb.models import PlannedExperiment, RunType, ApplProduct, \
    ReferenceGenome, Content, KitInfo, VariantFrequencies, dnaBarcode, \
    LibraryKey, ThreePrimeadapter, Chip, QCType, Project, Plugin, \
    PlannedExperimentQC

from iondb.rundb.plan.views_helper import getPlanDisplayedName

from traceback import format_exc

import logging
logger = logging.getLogger(__name__)


class PlanCSVcolumns():
    COLUMN_PLAN_CSV_VERSION = "CSV Version (required)"
    COLUMN_TEMPLATE_NAME = "Template name to plan from (required)"
    COLUMN_PLAN_NAME = "Plan name (required)"
    COLUMN_SAMPLE = "Sample (required)"

    COLUMN_SAMPLE_PREP_KIT = "Sample preparation kit name"
    COLUMN_LIBRARY_KIT = "Library kit name"
    COLUMN_TEMPLATING_KIT = "Templating kit name"
    COLUMN_TEMPLATING_SIZE = "Templating Size"
    COLUMN_CONTROL_SEQ_KIT = "Control sequence name"
    COLUMN_SEQ_KIT = "Sequence kit name"
    COLUMN_CHIP_TYPE = "Chip type"
    COLUMN_LIBRARY_READ_LENGTH = "Library Read Length"
    COLUMN_FLOW_COUNT = "Flows"
    COLUMN_SAMPLE_TUBE_LABEL = "Sample tube label"
    COLUMN_BEAD_LOAD_PCT = "Bead loading %"
    COLUMN_KEY_SIGNAL_PCT = "Key signal %"
    COLUMN_USABLE_SEQ_PCT = "Usable sequence %"

    COLUMN_REF = "Reference library"
    COLUMN_TARGET_BED = "Target regions BED file"
    COLUMN_HOTSPOT_BED = "Hotspot regions BED file"

    COLUMN_PLUGINS = "Plugins"
    COLUMN_PROJECTS = "Project names"
    COLUMN_EXPORT = "Export"
    COLUMN_NOTES = "Notes"
    COLUMN_LIMS_DATA = "LIMS Meta Data"
    COLUMN_CHIP_BARCODE = "Chip ID"
    COLUMN_BC_SAMPLE_KEY = ":Sample"

    # Samples
    COLUMN_BARCODE = "Barcode"
    COLUMN_SAMPLE_NAME = "Sample Name (required)"
    COLUMN_SAMPLE_ID = "Sample ID"
    COLUMN_NUCLEOTIDE_TYPE = "DNA/RNA/Fusions"
    COLUMN_SAMPLE_DESCRIPTION = "Sample Description"
    COLUMN_SAMPLE_FILE_HEADER = "Samples CSV file name (required)"

    # obsolete?
    COLUMN_IR_V1_0_WORKFLOW = "IR_v1_0_workflow"
    COLUMN_IR_V1_X_WORKFLOW = "IR_v1_x_workflow"

TOKEN_DELIMITER = ";"


def _get_kit_description(kitTypes, kitName):
    desc = ""

    if kitName:
        try:
            kits = KitInfo.objects.filter(kitType__in=kitTypes, name=kitName)
            desc = kits[0].description
        except:
            logger.exception(format_exc())
    return desc


def _get_sample_prep_kit_description(template):
    desc = ""
    if template:
        kitName = template.samplePrepKitName
        desc = _get_kit_description(["SamplePrepKit"], kitName)
    return desc


def _get_lib_kit_description(template):
    desc = ""
    if template:
        kitName = template.get_librarykitname()
        desc = _get_kit_description(["LibraryKit", "LibraryPrepKit"], kitName)
    return desc


def _get_template_kit_description(template):
    desc = ""
    if template:
        kitName = template.templatingKitName
        desc = _get_kit_description(["TemplatingKit", "IonChefPrepKit"], kitName)
    return desc


def _get_control_seq_kit_description(template):
    desc = ""
    if template:
        kitName = template.controlSequencekitname
        desc = _get_kit_description(["ControlSequenceKit"], kitName)
    return desc


def _get_seq_kit_description(template):
    desc = ""
    if template:
        kitName = template.get_sequencekitname()
        desc = _get_kit_description(["SequencingKit"], kitName)
    return desc


def _get_chip_type_description(template):
    desc = ""
    if template:
        chipName = template.get_chipType()
        if chipName:
            try:
                chip = Chip.objects.get(name=chipName)
                desc = chip.description
            except:
                logger.exception(format_exc())
    return desc


def _get_library_read_length(template):
    if template:
        readLength = template.libraryReadLength
        return readLength if readLength > 0 else ""
    return ""


def _get_templating_size(template):
    if template:
        templatingSize = template.templatingSize
        return templatingSize if templatingSize else ""
    return ""


def _get_flow_count(template):
    if template:
        return template.get_flows()
    return ""


def _get_qc(qcName, template):
    qc = ""

    if template and qcName:
        qcValues = template.plannedexperimentqc_set.all()
        for qcValue in qcValues:
            if (qcValue.qcType.qcName == qcName):
                qc = qcValue.threshold

    return qc


def _get_bead_loading_qc(template):
    return _get_qc("Bead Loading (%)", template)


def _get_key_signal_qc(template):
    return _get_qc("Key Signal (1-100)", template)


def _get_usable_seq_qc(template):
    return _get_qc("Usable Sequence (%)", template)


def _get_reference(template):
    ref = ""
    if template:
        ref = template.get_library()
    return ref


def _get_target_regions_bed_file(template):
    filePath = ""

    if template and template.get_bedfile():
        try:
            bed = Content.objects.get(file=template.get_bedfile())
            filePath = bed.path
        except:
            logger.exception(format_exc())
    return filePath


def _get_hotspot_regions_bed_file(template):
    filePath = ""

    if template and template.get_regionfile():
        try:
            bed = Content.objects.get(file=template.get_regionfile())
            filePath = bed.path
        except:
            logger.exception(format_exc())
    return filePath


def _get_plugins(template, delimiter):
    plugins = ''

    planPlugins = template.get_selectedPlugins()

    if planPlugins:
        for planPlugin in planPlugins.values():
            if 'export' in planPlugin.get('features', []):
                continue
            pluginName = planPlugin.get("name", '')
            if pluginName:
                plugins += pluginName
                plugins += delimiter

    if template.isSystem:
        default_selected_plugins = Plugin.objects.filter(active=True, selected=True, defaultSelected=True).order_by("name")

        if not default_selected_plugins:
            return plugins

        for default_plugin in default_selected_plugins:
            if planPlugins and default_plugin.name in planPlugins.keys():
                logger.debug("plan_csv_writer._get_plugins() SKIPPING default_selected_plugins=%s" % (default_plugin.name))
            else:
                pluginSettings = default_plugin.pluginsettings
                if 'export' not in pluginSettings.get('features', []):
                    if default_plugin.name:
                        plugins += default_plugin.name
                        plugins += delimiter

    # logger.info("EXIT plan_csv_writer._get_plugins() plugins=%s" %(plugins))
    return plugins


def _get_export(template, delimiter):
    uploaders = ''

    planPlugins = template.get_selectedPlugins()
    # logger.info("plan_csv_writer._get_export() planUploaders=%s" %(planUploaders))

    if planPlugins:
        for planPlugin in planPlugins.values():
            if 'export' not in planPlugin.get('features', []):
                continue
            uploaderName = planPlugin.get("name", '')
            if uploaderName:
                uploaders += uploaderName
                uploaders += delimiter

    if template.isSystem:
        default_selected_plugins = Plugin.objects.filter(active=True, selected=True, defaultSelected=True).order_by("name")

        if not default_selected_plugins:
            return uploaders

        for default_plugin in default_selected_plugins:
            if planPlugins and default_plugin.name in planPlugins.keys():
                logger.debug("plan_csv_writer._get_export() SKIPPING default_selected_plugins=%s" % (default_plugin.name))
            else:
                pluginSettings = default_plugin.pluginsettings
                if 'export' in pluginSettings.get('features', []):
                    if default_plugin.name:
                        uploaders += default_plugin.name
                        uploaders += delimiter

    # logger.info("EXIT plan_csv_writer._get_exports() uploaders=%s" %(uploaders))
    return uploaders


def _get_projects(template, delimiter):
    projectNames = ""
    selectedProjectNames = [selectedProject.name for selectedProject in list(template.projects.all())]

    index = 0
    if selectedProjectNames:
        for name in selectedProjectNames:
            projectNames += name
            index += 1

            if index < len(selectedProjectNames):
                projectNames += delimiter

    return projectNames


def _get_notes(template):
    return template.get_notes()


def _get_LIMS_data(template):
    return ""

#    metaData = template.metaData
#    if metaData:
#        return metaData.get("LIMS", "")
#    else:
#        return ""


def _has_ir(template):
    plugins = Plugin.objects.filter(name__icontains="IonReporter", selected=True, active=True)
    return plugins.count() > 0


def _has_ir_v1_0(template):
    if not _has_ir(template):
        return False

    plugins = Plugin.objects.filter(name__icontains="IonReporterUploader_V1_0", selected=True, active=True)

    return plugins.count() > 0


def _has_ir_beyond_v1_0(template):
    if not _has_ir(template):
        return False

    plugins = Plugin.objects.filter(selected=True, active=True).exclude(name__icontains="IonReporterUploader_V1_0")

    return plugins.count() > 0


def _get_sample_name(template):
    return ""


def _get_sample_id(template):
    return ""


def _get_sample_tube_label(template):
    return ""


def _is_barcoded(template):
    return True if template.get_barcodeId() else False


def _get_barcoded_sample_headers(template, prefix):
    hdrs = []
    if _is_barcoded(template):
        barcodes = dnaBarcode.objects.filter(name=template.get_barcodeId()).order_by("index")
        barcodeCount = barcodes.count()
        for barcode in barcodes:
            hdrs.append(barcode.id_str + prefix)

    return hdrs


def _get_barcoded_sample_names(template):
    cells = []
    if _is_barcoded(template):
        barcodes = dnaBarcode.objects.filter(name=template.get_barcodeId()).order_by("index")
        for barcode in barcodes:
            cells.append("")

    return cells


def _get_barcoded_sample_IR_beyond_v1_0_headers(template, prefix):
    hdrs = []
    if _is_barcoded(template):
        barcodes = dnaBarcode.objects.filter(name=template.get_barcodeId()).order_by("index")
        barcodeCount = barcodes.count()
        index = 0
        for barcode in barcodes:
            index += 1
            hdrs.append(PlanCSVcolumns.COLUMN_IR_V1_X_WORKFLOW + ": " + prefix + str(index))

    return hdrs

# currently no workflow config for template


def _get_barcoded_sample_IR_beyond_v1_0_workflows(template):
    return []


# currently no workflow config for template
def _get_sample_IR_beyond_v1_0_workflows(template):
    return ""


def get_plan_csv_version():
    systemCSV_version = settings.PLAN_CSV_VERSION
    return [PlanCSVcolumns.COLUMN_PLAN_CSV_VERSION, systemCSV_version]


def get_template_data_for_batch_planning(templateId, single_samples_file):
    try:
        template = PlannedExperiment.objects.get(pk=int(templateId))

        logger.info("plan_csv_writer.get_template_data_for_batch_planning() template retrieved. id=%d;" % (int(templateId)))

        hdr = [PlanCSVcolumns.COLUMN_TEMPLATE_NAME, PlanCSVcolumns.COLUMN_PLAN_NAME
               ]

        hdr2 = [PlanCSVcolumns.COLUMN_SAMPLE_PREP_KIT, PlanCSVcolumns.COLUMN_LIBRARY_KIT, PlanCSVcolumns.COLUMN_TEMPLATING_KIT, PlanCSVcolumns.COLUMN_TEMPLATING_SIZE, PlanCSVcolumns.COLUMN_CONTROL_SEQ_KIT, PlanCSVcolumns.COLUMN_SEQ_KIT, PlanCSVcolumns.COLUMN_CHIP_TYPE, PlanCSVcolumns.COLUMN_LIBRARY_READ_LENGTH, PlanCSVcolumns.COLUMN_FLOW_COUNT, PlanCSVcolumns.COLUMN_SAMPLE_TUBE_LABEL, PlanCSVcolumns.COLUMN_BEAD_LOAD_PCT, PlanCSVcolumns.COLUMN_KEY_SIGNAL_PCT, PlanCSVcolumns.COLUMN_USABLE_SEQ_PCT, PlanCSVcolumns.COLUMN_REF, PlanCSVcolumns.COLUMN_TARGET_BED, PlanCSVcolumns.COLUMN_HOTSPOT_BED, PlanCSVcolumns.COLUMN_PLUGINS, PlanCSVcolumns.COLUMN_PROJECTS, PlanCSVcolumns.COLUMN_EXPORT, PlanCSVcolumns.COLUMN_NOTES, PlanCSVcolumns.COLUMN_LIMS_DATA, PlanCSVcolumns.COLUMN_CHIP_BARCODE
                ]

        body = [getPlanDisplayedName(template), ""
                ]

        body2 = [_get_sample_prep_kit_description(template), _get_lib_kit_description(template), _get_template_kit_description(template), _get_templating_size(template), _get_control_seq_kit_description(template), _get_seq_kit_description(template), _get_chip_type_description(template), _get_library_read_length(template), _get_flow_count(template), _get_sample_tube_label(template), _get_bead_loading_qc(template), _get_key_signal_qc(template), _get_usable_seq_qc(template), _get_reference(template), _get_target_regions_bed_file(template), _get_hotspot_regions_bed_file(template), _get_plugins(template, ";"), _get_projects(template, ";"), _get_export(template, ";"), _get_notes(template), _get_LIMS_data(template)
                 ]

        # position of the fields below are based on the template selected and whether IR has been installed on the TS
        if _has_ir_v1_0(template):
            hdr2.append(PlanCSVcolumns.COLUMN_IR_V1_0_WORKFLOW)
            body2.append("")

        # has_ir_beyond_v1_0 = _has_ir_beyond_v1_0(template)
        if _is_barcoded(template):
            if single_samples_file:
                hdr.extend(_get_barcoded_sample_headers(template, PlanCSVcolumns.COLUMN_BC_SAMPLE_KEY))
                body.extend(_get_barcoded_sample_names(template))

                # if has_ir_beyond_v1_0:
                #    hdr.extend(_get_barcoded_sample_IR_beyond_v1_0_headers(template, PlanCSVcolumns.COLUMN_BC_SAMPLE_KEY, ))
                #    body.extend(_get_barcoded_sample_IR_beyond_v1_0_workflows(template))
            else:
                hdr.append(PlanCSVcolumns.COLUMN_SAMPLE_FILE_HEADER)
                body.append("")
        else:
            hdr.append(PlanCSVcolumns.COLUMN_SAMPLE)
            body.append(_get_sample_name(template))

            hdr.append(PlanCSVcolumns.COLUMN_SAMPLE_ID)
            body.append(_get_sample_id(template))
            # if has_ir_beyond_v1_0:
             #   hdr.append(PlanCSVcolumns.COLUMN_IR_V1_X_WORKFLOW)
             #   body.append(_get_sample_IR_beyond_v1_0_workflows(template))

        hdr.extend(hdr2)
        body.extend(body2)

        return hdr, body
    except:
        logger.exception(format_exc())
        return [], [], []


def get_samples_data_for_batch_planning(templateId):
    template = PlannedExperiment.objects.get(pk=int(templateId))
    reference = _get_reference(template)
    target_bed = _get_target_regions_bed_file(template)
    hotspot_bed = _get_hotspot_regions_bed_file(template)
    nucleotideType = template.get_default_nucleotideType()

    hdr = [
        PlanCSVcolumns.COLUMN_BARCODE,
        PlanCSVcolumns.COLUMN_SAMPLE_NAME,
        PlanCSVcolumns.COLUMN_SAMPLE_ID,
        PlanCSVcolumns.COLUMN_SAMPLE_DESCRIPTION,
        PlanCSVcolumns.COLUMN_NUCLEOTIDE_TYPE,
        PlanCSVcolumns.COLUMN_REF,
        PlanCSVcolumns.COLUMN_TARGET_BED,
        PlanCSVcolumns.COLUMN_HOTSPOT_BED,
    ]
    body = []
    barcodes = dnaBarcode.objects.filter(name=template.get_barcodeId()).order_by("index")

    for barcode in barcodes:
        row = []
        for column in hdr:
            if column == PlanCSVcolumns.COLUMN_BARCODE:
                row.append(barcode.id_str)
            elif column == PlanCSVcolumns.COLUMN_REF:
                row.append(reference)
            elif column == PlanCSVcolumns.COLUMN_TARGET_BED:
                row.append(target_bed)
            elif column == PlanCSVcolumns.COLUMN_HOTSPOT_BED:
                row.append(hotspot_bed)
            elif column == PlanCSVcolumns.COLUMN_NUCLEOTIDE_TYPE:
                row.append(nucleotideType)
            else:
                row.append("")
        body.append(row)
    return hdr, body
