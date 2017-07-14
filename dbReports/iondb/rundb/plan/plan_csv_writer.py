# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.models import Project, PlannedExperiment
from django.contrib.auth.models import User
from django.conf import settings

from iondb.rundb.models import PlannedExperiment, RunType, ApplProduct, \
    ReferenceGenome, Content, KitInfo, dnaBarcode, \
    LibraryKey, ThreePrimeadapter, Chip, QCType, Project, Plugin, \
    PlannedExperimentQC, AnalysisArgs

from iondb.rundb.plan.views_helper import getPlanDisplayedName
from iondb.rundb.plan.plan_validator import MAX_LENGTH_PLAN_NAME
from traceback import format_exc
import logging
logger = logging.getLogger(__name__)


class PlanCSVcolumns():
    COLUMN_PLAN_HEADING_KEY = "Plan Parameters"
    COLUMN_PLAN_HEADING_VALUE = "Plan "
    COLUMN_PLAN_CSV_VERSION = "CSV Version (required)"
    COLUMN_TEMPLATE_NAME = "Template name to plan from (required)"
    COLUMN_PLAN_NAME = "Plan name (required)"
    COLUMN_SAMPLE = "Sample (required)"

    COLUMN_SAMPLE_PREP_KIT = "Sample preparation kit name"
    COLUMN_LIBRARY_KIT = "Library kit name"
    COLUMN_TEMPLATING_KIT_V1 = "Templating kit name"
    COLUMN_TEMPLATING_KIT = "Templating kit name (required)"
    COLUMN_TEMPLATING_SIZE = "Templating Size"
    COLUMN_CONTROL_SEQ_KIT = "Control sequence name"
    COLUMN_SEQ_KIT = "Sequence kit name"
    COLUMN_CHIP_TYPE_V1 = "Chip type"
    COLUMN_CHIP_TYPE = "Chip type (required)"
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
    COLUMN_CHIP_BARCODE_V1 = "Chip ID"
    COLUMN_CHIP_BARCODE = "Chip Barcode"
    COLUMN_IR_ACCOUNT = "IR Account"
    COLUMN_BC_SAMPLE_KEY = ":Sample"

    # Samples
    COLUMN_BARCODE = "Barcode"
    COLUMN_SAMPLE_NAME = "Sample Name (required)"
    COLUMN_SAMPLE_ID = "Sample ID"
    COLUMN_NUCLEOTIDE_TYPE = "DNA/RNA/Fusions"
    COLUMN_SAMPLE_DESCRIPTION = "Sample Description"
    COLUMN_SAMPLE_FILE_HEADER = "Samples CSV file name (required)"
    COLUMN_SAMPLE_CONTROLTYPE = "Control Type"
    COLUMN_SAMPLE_CANCER_TYPE = "Cancer Type"
    COLUMN_SAMPLE_CELLULARITY = "Cellularity %"
    COLUMN_SAMPLE_BIOSPY_DAYS = "Biospy Days"
    COLUMN_SAMPLE_COUPLE_ID = "Coupld ID"
    COLUMN_SAMPLE_EMBRYO_ID = "Embryo ID"
    COLUMN_SAMPLE_IR_RELATION = "IR Relation"
    COLUMN_SAMPLE_IR_GENDER = "IR Gender"
    COLUMN_SAMPLE_IR_WORKFLOW = "IR Workflow"
    COLUMN_SAMPLE_IR_SET_ID = "IR Set ID"

    # obsolete?
    COLUMN_IR_V1_0_WORKFLOW = "IR_v1_0_workflow"
    COLUMN_IR_V1_X_WORKFLOW = "IR_v1_x_workflow"
    
    # for Template export
    TEMPLATE_NAME = "Template name (required)"
    APPLICATION = "Application"
    RUNTYPE = "Target Technique"
    LIBRARY_KEY = "Library Key"
    TF_KEY = "Test Fragment Key"
    BARCODE_SET = "Barcode Set"
    FAVORITE = "Set as Favorite"
    SAMPLE_GROUP = "Sample Grouping"
    THREEPRIME_ADAPTER = "Forward 3' Adapter"
    FLOW_ORDER = "Flow Order"
    CALIBRATION_MODE = "Base Calibration Mode"
    MARK_DUPLICATES = "Mark as Duplicate Reads"
    REALIGN = "Enable Realignment"
    CATEGORIES = "Categories"
    CUSTOM_ARGS = "Custom Args"

    FUSIONS_REF = "Fusions Reference library"
    FUSIONS_TARGET_BED = "Fusions Target regions BED file"

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

def _get_bed_file_path(bedfile):
    path = ""
    if bedfile:
        obj = Content.objects.filter(file=bedfile)
        path = obj[0].path if obj else ""
    return path

def _get_target_regions_bed_file(template):
    filePath = ""
    if template:
        filePath = _get_bed_file_path(template.get_bedfile())
    return filePath

def _get_hotspot_regions_bed_file(template):
    filePath = ""
    if template:
        filePath = _get_bed_file_path(template.get_regionfile())
    return filePath

def _get_fusions_target_regions_bed_file(template):
    filePath = ""
    if template:
        filePath = _get_bed_file_path(template.get_mixedType_rna_bedfile())
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

def _get_sample_description(template):
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

        hdr = [
            PlanCSVcolumns.COLUMN_TEMPLATE_NAME,
            PlanCSVcolumns.COLUMN_PLAN_NAME
        ]

        hdr2 = [
            PlanCSVcolumns.COLUMN_SAMPLE_PREP_KIT,
            PlanCSVcolumns.COLUMN_LIBRARY_KIT,
            PlanCSVcolumns.COLUMN_TEMPLATING_KIT,
            PlanCSVcolumns.COLUMN_TEMPLATING_SIZE,
            PlanCSVcolumns.COLUMN_CONTROL_SEQ_KIT,
            PlanCSVcolumns.COLUMN_SEQ_KIT,
            PlanCSVcolumns.COLUMN_CHIP_TYPE,
            PlanCSVcolumns.COLUMN_LIBRARY_READ_LENGTH,
            PlanCSVcolumns.COLUMN_FLOW_COUNT,
            PlanCSVcolumns.COLUMN_SAMPLE_TUBE_LABEL,
            PlanCSVcolumns.COLUMN_BEAD_LOAD_PCT,
            PlanCSVcolumns.COLUMN_KEY_SIGNAL_PCT,
            PlanCSVcolumns.COLUMN_USABLE_SEQ_PCT,
            PlanCSVcolumns.COLUMN_REF,
            PlanCSVcolumns.COLUMN_TARGET_BED,
            PlanCSVcolumns.COLUMN_HOTSPOT_BED,
            PlanCSVcolumns.COLUMN_PLUGINS,
            PlanCSVcolumns.COLUMN_PROJECTS,
            PlanCSVcolumns.COLUMN_EXPORT,
            PlanCSVcolumns.COLUMN_NOTES,
            PlanCSVcolumns.COLUMN_LIMS_DATA,
            PlanCSVcolumns.COLUMN_CHIP_BARCODE,
            PlanCSVcolumns.COLUMN_IR_ACCOUNT
        ]

        body = [
            getPlanDisplayedName(template),
            ""
        ]

        body2 = [
            _get_sample_prep_kit_description(template),
            _get_lib_kit_description(template),
            _get_template_kit_description(template),
            _get_templating_size(template),
            _get_control_seq_kit_description(template),
            _get_seq_kit_description(template),
            _get_chip_type_description(template),
            _get_library_read_length(template),
            _get_flow_count(template),
            _get_sample_tube_label(template),
            _get_bead_loading_qc(template),
            _get_key_signal_qc(template),
            _get_usable_seq_qc(template),
            _get_reference(template),
            _get_target_regions_bed_file(template),
            _get_hotspot_regions_bed_file(template),
            _get_plugins(template, ";"),
            _get_projects(template, ";"),
            _get_export(template, ";"),
            _get_notes(template),
            _get_LIMS_data(template),
            "",
            _get_template_IR_account(template),
            _get_template_IR_workflow(template)

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
            hdr.extend([PlanCSVcolumns.COLUMN_SAMPLE,
                        PlanCSVcolumns.COLUMN_SAMPLE_DESCRIPTION,
                        PlanCSVcolumns.COLUMN_SAMPLE_ID])

            body.extend([_get_sample_name(template),
                         _get_sample_description(template),
                         _get_sample_id(template)])

            # if has_ir_beyond_v1_0:
             #   hdr.append(PlanCSVcolumns.COLUMN_IR_V1_X_WORKFLOW)
             #   body.append(_get_sample_IR_beyond_v1_0_workflows(template))

        hdr.extend(hdr2)
        if single_samples_file:
            hdr.extend(get_irSettings())
        else:
            hdr.append(PlanCSVcolumns.COLUMN_SAMPLE_IR_WORKFLOW)

        body.extend(body2)

        return hdr, body
    except:
        logger.exception(format_exc())
        return [], [], []


def get_irSettings():
    irSettings = [PlanCSVcolumns.COLUMN_SAMPLE_IR_WORKFLOW,
                  PlanCSVcolumns.COLUMN_SAMPLE_IR_RELATION,
                  PlanCSVcolumns.COLUMN_SAMPLE_IR_GENDER,
                  PlanCSVcolumns.COLUMN_SAMPLE_IR_SET_ID]

    return irSettings

def _get_template_IR_account(template):
    planPlugins = template.get_selectedPlugins()
    existingIR = None
    if planPlugins:
        for planPlugin in planPlugins.values():
            if "IonReporterUploader" in planPlugin.get("name"):
                existingIR = planPlugin["userInput"]["accountName"]
                existingIR = existingIR.strip()
                return existingIR
    return existingIR

def _get_template_IR_workflow(template):
    return template.irworkflow

def get_samples_data_for_batch_planning(templateId):
    template = PlannedExperiment.objects.get(pk=int(templateId))
    reference = _get_reference(template)
    target_bed = _get_target_regions_bed_file(template)
    hotspot_bed = _get_hotspot_regions_bed_file(template)
    nucleotideType = template.get_default_nucleotideType()
    template_ir_workflow = _get_template_IR_workflow(template)

    hdr = [
        PlanCSVcolumns.COLUMN_BARCODE,
        PlanCSVcolumns.COLUMN_SAMPLE_CONTROLTYPE,
        PlanCSVcolumns.COLUMN_SAMPLE_NAME,
        PlanCSVcolumns.COLUMN_SAMPLE_ID,
        PlanCSVcolumns.COLUMN_SAMPLE_DESCRIPTION,
        PlanCSVcolumns.COLUMN_NUCLEOTIDE_TYPE,
        PlanCSVcolumns.COLUMN_REF,
        PlanCSVcolumns.COLUMN_TARGET_BED,
        PlanCSVcolumns.COLUMN_HOTSPOT_BED
    ]
    # include ir configuration settings
    hdr.extend(get_irSettings())

    # if selected template is for Oncology, include below properties
    isOnco = [cat for cat in ["Oncomine", "Onconet"] if cat in template.categories]
    if isOnco:
        hdr.extend([PlanCSVcolumns.COLUMN_SAMPLE_CANCER_TYPE,
                    PlanCSVcolumns.COLUMN_SAMPLE_CELLULARITY])
    else:
        hdr.extend([PlanCSVcolumns.COLUMN_SAMPLE_BIOSPY_DAYS,
                    PlanCSVcolumns.COLUMN_SAMPLE_COUPLE_ID,
                   PlanCSVcolumns.COLUMN_SAMPLE_EMBRYO_ID])

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
            elif column == PlanCSVcolumns.COLUMN_SAMPLE_IR_WORKFLOW:
                row.append(template_ir_workflow)
            else:
                row.append("")
        body.append(row)
    return hdr, body


def export_template_keys(custom_args):
    # map of keys from PlannedExperiment API fields to CSV columns
    keys = {
        'planDisplayedName':    PlanCSVcolumns.TEMPLATE_NAME,
        'applicationGroupDisplayedName': PlanCSVcolumns.APPLICATION,
        'barcodeId':            PlanCSVcolumns.BARCODE_SET,
        'base_recalibration_mode': PlanCSVcolumns.CALIBRATION_MODE,
        'bedfile':              PlanCSVcolumns.COLUMN_TARGET_BED,
        'categories':           PlanCSVcolumns.CATEGORIES,
        'chipType':             PlanCSVcolumns.COLUMN_CHIP_TYPE,
        'controlSequencekitname': PlanCSVcolumns.COLUMN_CONTROL_SEQ_KIT,
        'flows':                PlanCSVcolumns.COLUMN_FLOW_COUNT,
        'flowsInOrder':         PlanCSVcolumns.FLOW_ORDER,
        'forward3primeadapter': PlanCSVcolumns.THREEPRIME_ADAPTER,
        'isDuplicateReads':     PlanCSVcolumns.MARK_DUPLICATES,
        'isFavorite':           PlanCSVcolumns.FAVORITE,
        'library':              PlanCSVcolumns.COLUMN_REF,
        'libraryKey':           PlanCSVcolumns.LIBRARY_KEY,
        'librarykitname':       PlanCSVcolumns.COLUMN_LIBRARY_KIT,
        'libraryReadLength':    PlanCSVcolumns.COLUMN_LIBRARY_READ_LENGTH,
        'metaData':             PlanCSVcolumns.COLUMN_LIMS_DATA,
        'notes':                PlanCSVcolumns.COLUMN_NOTES,
        'runType':              PlanCSVcolumns.RUNTYPE,
        'sampleGroupingName':   PlanCSVcolumns.SAMPLE_GROUP,
        'samplePrepKitName':    PlanCSVcolumns.COLUMN_SAMPLE_PREP_KIT,
        'sequencekitname':      PlanCSVcolumns.COLUMN_SEQ_KIT,
        'tfKey':                PlanCSVcolumns.TF_KEY,
        'templatingKitName':    PlanCSVcolumns.COLUMN_TEMPLATING_KIT,
        'templatingSize':       PlanCSVcolumns.COLUMN_TEMPLATING_SIZE,
        'realign':              PlanCSVcolumns.REALIGN,
        'regionfile':           PlanCSVcolumns.COLUMN_HOTSPOT_BED,
        'selectedPlugins':      PlanCSVcolumns.COLUMN_PLUGINS,
        'projects':             PlanCSVcolumns.COLUMN_PROJECTS,
        'export':               PlanCSVcolumns.COLUMN_EXPORT,
        'custom_args':          PlanCSVcolumns.CUSTOM_ARGS,
        'mixedTypeRNA_reference': PlanCSVcolumns.FUSIONS_REF,
        'mixedTypeRNA_targetRegionBedFile': PlanCSVcolumns.FUSIONS_TARGET_BED
    }
    # QC values
    keys.update({
        'Bead Loading (%)':      PlanCSVcolumns.COLUMN_BEAD_LOAD_PCT,
        'Key Signal (1-100)':        PlanCSVcolumns.COLUMN_KEY_SIGNAL_PCT,
        'Usable Sequence (%)':        PlanCSVcolumns.COLUMN_USABLE_SEQ_PCT
    })
    # Analysis args, included only if custom
    if custom_args:
        args = AnalysisArgs().get_args()
        for key in args:
            keys[key] = key

    return keys


def get_template_data_for_export(templateId):
    ''' generates data for template export to CSV file
    '''
    template = PlannedExperiment.objects.get(pk=int(templateId))
    name = "exported " + getPlanDisplayedName(template).strip()
    runType = RunType.objects.get(runType=template.runType)

    data = [
        ( PlanCSVcolumns.TEMPLATE_NAME,  name[:MAX_LENGTH_PLAN_NAME]),
        ( PlanCSVcolumns.FAVORITE, template.isFavorite ),
        ( PlanCSVcolumns.APPLICATION, template.applicationGroup.description if template.applicationGroup else '' ),
        ( PlanCSVcolumns.RUNTYPE, runType.alternate_name ),
        ( PlanCSVcolumns.SAMPLE_GROUP, template.sampleGrouping.displayedName if template.sampleGrouping else '' ),
        ( PlanCSVcolumns.BARCODE_SET, template.get_barcodeId() ),
        ( PlanCSVcolumns.COLUMN_CHIP_TYPE, _get_chip_type_description(template) ),
        ( PlanCSVcolumns.COLUMN_SAMPLE_PREP_KIT, _get_sample_prep_kit_description(template) ),
        ( PlanCSVcolumns.COLUMN_LIBRARY_KIT, _get_lib_kit_description(template) ),
        ( PlanCSVcolumns.LIBRARY_KEY, template.get_libraryKey() ),
        ( PlanCSVcolumns.TF_KEY, template.get_tfKey() ),
        ( PlanCSVcolumns.THREEPRIME_ADAPTER, template.get_forward3primeadapter() ),
        ( PlanCSVcolumns.FLOW_ORDER, template.experiment.flowsInOrder or "default" ),
        ( PlanCSVcolumns.COLUMN_TEMPLATING_KIT, _get_template_kit_description(template) ),
        ( PlanCSVcolumns.COLUMN_TEMPLATING_SIZE, _get_templating_size(template) ),
        ( PlanCSVcolumns.COLUMN_SEQ_KIT, _get_seq_kit_description(template) ),
        ( PlanCSVcolumns.COLUMN_CONTROL_SEQ_KIT, _get_control_seq_kit_description(template) ),
        ( PlanCSVcolumns.COLUMN_LIBRARY_READ_LENGTH, _get_library_read_length(template) ),
        ( PlanCSVcolumns.CALIBRATION_MODE, template.latestEAS.base_recalibration_mode ),
        ( PlanCSVcolumns.MARK_DUPLICATES, template.latestEAS.isDuplicateReads ),
        ( PlanCSVcolumns.REALIGN, template.latestEAS.realign ),
        ( PlanCSVcolumns.COLUMN_FLOW_COUNT, template.get_flows() ),
        ( PlanCSVcolumns.COLUMN_REF, _get_reference(template) ),
        ( PlanCSVcolumns.COLUMN_TARGET_BED, _get_target_regions_bed_file(template) ),
        ( PlanCSVcolumns.COLUMN_HOTSPOT_BED,_get_hotspot_regions_bed_file(template) )
    ]

    # add fusions reference for DNA/Fusions application
    if runType.runType == 'AMPS_DNA_RNA':
        data.extend([
            ( PlanCSVcolumns.FUSIONS_REF, template.get_mixedType_rna_library() ),
            ( PlanCSVcolumns.FUSIONS_TARGET_BED, _get_fusions_target_regions_bed_file(template) )
        ])

    data.extend([
        ( PlanCSVcolumns.COLUMN_BEAD_LOAD_PCT, _get_bead_loading_qc(template) ),
        ( PlanCSVcolumns.COLUMN_KEY_SIGNAL_PCT, _get_key_signal_qc(template) ),
        ( PlanCSVcolumns.COLUMN_USABLE_SEQ_PCT, _get_usable_seq_qc(template) ),
        ( PlanCSVcolumns.COLUMN_PLUGINS, _get_plugins(template, TOKEN_DELIMITER) ),
        ( PlanCSVcolumns.COLUMN_PROJECTS, _get_projects(template, TOKEN_DELIMITER) ),
        ( PlanCSVcolumns.CATEGORIES, template.categories ),
        ( PlanCSVcolumns.COLUMN_NOTES, _get_notes(template) ),
        ( PlanCSVcolumns.COLUMN_LIMS_DATA, _get_LIMS_data(template) ),
    ])

    # add custom analysis args
    if template.latestEAS.custom_args:
        args = template.latestEAS.get_cmdline_args()
        data.append((PlanCSVcolumns.CUSTOM_ARGS, True))
        data.extend([(key, args[key]) for key in sorted(args)])

    return zip(*data)
