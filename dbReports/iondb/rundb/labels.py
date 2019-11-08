from collections import namedtuple, OrderedDict
from django.utils.translation import ugettext_lazy as _
import copy


def EntityFields(
    name, verbose_name, verbose_name_plural, fields=[], extends=None, debug=False
):
    """
    Creates a namedtuple that can be used for label lookup for translation.
    Intended to support a single language per instance constructed.

    Usage:

    EntityFields('MyType', 'My Type', 'My Types', (('name', ('Name', 'Names')),))


    :param name: name of entity or construct
    :param verbose_name: translated singular name
    :param verbose_name_plural: translated plural name
    :param fields: an optional parameter. tuple of (fieldname, (verbose_name, verbose_name_plural)) the verbose_name_plural is optional
    :param extends: an optional parameter. EntityFields instance (namedtuple) to inherits fields from

    :return: a namedtuple containing the information provided.
    """
    _mydict = OrderedDict()
    _mydict["verbose_name"] = verbose_name
    _mydict["verbose_name_plural"] = verbose_name_plural

    if extends and hasattr(extends, "_fields"):  # verify is a namedtuple
        for field_name in extends._fields:
            if field_name not in ["verbose_name", "verbose_name_plural"]:
                _mydict[field_name] = getattr(extends, field_name)

    for field_name, field_tuple in fields:
        _field_values = []
        if len(field_tuple) == 1 and hasattr(
            field_tuple[0], "_fields"
        ):  # verify is a namedtuple
            _mydict[field_name] = field_tuple[0]
        else:
            if len(field_tuple) > 0:
                _field_values.append("verbose_name")
            if len(field_tuple) > 1:
                _field_values.append("verbose_name_plural")

            _field_nt = namedtuple(name + "_" + field_name, _field_values, debug)
            _mydict[field_name] = _field_nt(*field_tuple)
    if debug:
        print(_mydict)

    class _Entity(namedtuple(name, list(_mydict.keys()), debug)):
        __slots__ = ()

    return _Entity._make(list(_mydict.values()))


PlanTemplate = EntityFields(
    "PlanTemplate",
    verbose_name=_("entity.PlanTemplate.verbose_name"),
    verbose_name_plural=_("entity.PlanTemplate.verbose_name_plural"),
)
Plan = EntityFields(
    "Plan",
    verbose_name=_("entity.Plan.verbose_name"),
    verbose_name_plural=_("entity.Plan.verbose_name_plural"),
)
ScientificApplication = EntityFields(
    "ScientificApplication",
    verbose_name=_("entity.ScientificApplication.verbose_name"),
    verbose_name_plural=_("entity.ScientificApplication.verbose_name_plural"),
)
BarcodedSample = EntityFields(
    "BarcodedSample",
    verbose_name=_("entity.BarcodedSample.verbose_name"),
    verbose_name_plural=_("entity.BarcodedSample.verbose_name_plural"),
    fields=(
        (
            "displayedName",
            (_("entity.BarcodedSample.field.displayedName.verbose_name"),),
        ),
        ("barcode", (_("entity.BarcodedSample.field.barcode.verbose_name"),)),
        ("description", (_("entity.BarcodedSample.field.description.verbose_name"),)),
        ("externalId", (_("entity.BarcodedSample.field.externalId.verbose_name"),)),
    ),
)
DNA_RNA_BarcodedSample = EntityFields(
    "DNA_RNA_BarcodedSample",
    verbose_name=_("entity.BarcodedSample.verbose_name"),
    verbose_name_plural=_("entity.BarcodedSample.verbose_name_plural"),
    fields=(
        (
            "controlSequenceType",
            (_("entity.BarcodedSample.field.controlSequenceType.verbose_name"),),
        ),
        (
            "hotSpotRegionBedFile",
            (_("entity.BarcodedSample.field.hotSpotRegionBedFile.verbose_name"),),
        ),
        (
            "nucleotideType",
            (_("entity.BarcodedSample.field.nucleotideType.verbose_name"),),
        ),
        ("reference", (_("entity.BarcodedSample.field.reference.verbose_name"),)),
        (
            "targetRegionBedFile",
            (_("entity.BarcodedSample.field.targetRegionBedFile.verbose_name"),),
        ),
    ),
)
NonBarcodedSample = EntityFields(
    "NonBarcodedSample",
    verbose_name=_("entity.NonBarcodedSample.verbose_name"),
    verbose_name_plural=_("entity.NonBarcodedSample.verbose_name_plural"),
)
NonBarcodedPlan = EntityFields(
    "NonBarcodedPlan",
    verbose_name=_("entity.NonBarcodedPlan.verbose_name"),
    verbose_name_plural=_("entity.NonBarcodedPlan.verbose_name_plural"),
)
BarcodedPlan = EntityFields(
    "BarcodedPlan",
    verbose_name=_("entity.BarcodedPlan.verbose_name"),
    verbose_name_plural=_("entity.BarcodedPlan.verbose_name_plural"),
)
Chip = EntityFields(
    "Chip",
    verbose_name=_("entity.Chip.verbose_name"),
    verbose_name_plural=_("entity.Chip.verbose_name_plural"),
)
Experiment = EntityFields(
    "Experiment",
    verbose_name="Experiment",
    verbose_name_plural="Experiments",
    fields=(("notes", (_("entity.Experiment.fields.notes.verbose_name"),)),),
)
Sample = EntityFields(
    "Sample",
    verbose_name=_("entity.Sample.verbose_name"),
    verbose_name_plural=_("entity.Sample.verbose_name_plural"),
    fields=(
        ("status", (_("entity.Sample.fields.status.verbose_name"),)),
        ("name", (_("entity.Sample.fields.name.verbose_name"),)),
        ("displayedName", (_("entity.Sample.fields.displayedName.verbose_name"),)),
        ("externalId", (_("entity.Sample.fields.externalId.verbose_name"),)),
        ("description", (_("entity.Sample.fields.description.verbose_name"),)),
    ),
)
SampleAttribute = EntityFields(
    "SampleAttribute",
    verbose_name=_("entity.SampleAttribute.verbose_name"),
    verbose_name_plural=_("entity.SampleAttribute.verbose_name_plural"),
    fields=(),
)
SampleSet = EntityFields(
    "SampleSet",
    verbose_name=_("entity.SampleSet.verbose_name"),
    verbose_name_plural=_("entity.SampleSet.verbose_name_plural"),
    fields=(),
)
SampleSetItem = EntityFields(
    "SampleSetItem",
    verbose_name=_("entity.SampleSetItem.verbose_name"),
    verbose_name_plural=_("entity.SampleSetItem.verbose_name_plural"),
    fields=(),
)
InstrumentTypes = namedtuple("InstrumentTypes", ["none", "pgm", "proton", "S5", "s5"])(
    none=_("entity.InstrumentTypes.none"),
    pgm=_("entity.InstrumentTypes.pgm"),
    proton=_("entity.InstrumentTypes.proton"),
    S5=_("entity.InstrumentTypes.S5"),
    s5=_("entity.InstrumentTypes.s5"),
)
ReferenceGenome = EntityFields(
    "ReferenceGenome",
    verbose_name=_("entity.ReferenceGenome.verbose_name"),
    verbose_name_plural=_("entity.ReferenceGenome.verbose_name_plural"),
    fields=(
        ("short_name", (_("entity.ReferenceGenome.fields.short_name.verbose_name"),)),
        ("name", (_("entity.ReferenceGenome.fields.name.verbose_name"),)),
        ("notes", (_("entity.ReferenceGenome.fields.notes.verbose_name"),)),
        ("date", (_("entity.ReferenceGenome.fields.date.verbose_name"),)),
        (
            "index_version",
            (_("entity.ReferenceGenome.fields.index_version.verbose_name"),),
        ),
        ("status", (_("entity.ReferenceGenome.fields.status.verbose_name"),)),
        ("enabled", (_("entity.ReferenceGenome.fields.enabled.verbose_name"),)),
        ("version", (_("entity.ReferenceGenome.fields.version.verbose_name"),)),
        ("genome_info", (_("entity.ReferenceGenome.fields.genome_info.verbose_name"),)),
        (
            "genome_fasta",
            (_("entity.ReferenceGenome.fields.genome_fasta.verbose_name"),),
        ),
        (
            "genome_fasta_orig",
            (_("entity.ReferenceGenome.fields.genome_fasta_orig.verbose_name"),),
        ),
    ),
)

ContentUpload = EntityFields(
    "ContentUpload",
    verbose_name=_("entity.ContentUpload.verbose_name"),
    verbose_name_plural=_("entity.ContentUpload.verbose_name_plural"),
    fields=(
        ("status", (_("entity.ContentUpload.fields.status.verbose_name"),)),
        ("upload_date", (_("entity.ContentUpload.fields.upload_date.verbose_name"),)),
        ("upload_type", (_("entity.ContentUpload.fields.upload_type.verbose_name"),)),
        ("file_path", (_("entity.ContentUpload.fields.file_path.verbose_name"),)),
        ("name", (_("entity.ContentUpload.fields.name.verbose_name"),)),
        ("logs", (_("entity.ContentUpload.fields.logs.verbose_name"),)),
    ),
)

Content = EntityFields(
    "Content",
    verbose_name=_("entity.Content.verbose_name"),
    verbose_name_plural=_("entity.Content.verbose_name_plural"),
    fields=(
        ("file_name", (_("entity.Content.fields.file_name.verbose_name"),)),
        ("reference", (_("entity.Content.fields.reference.verbose_name"),)),
        ("description", (_("entity.Content.fields.description.verbose_name"),)),
        ("notes", (_("entity.Content.fields.notes.verbose_name"),)),
        ("enabled", (_("entity.Content.fields.enabled.verbose_name"),)),
        ("upload_date", (_("entity.Content.fields.upload_date.verbose_name"),)),
    ),
)

HotspotsContent = EntityFields(
    "HotspotsContent",
    verbose_name=_("entity.HotspotsContent.verbose_name"),
    verbose_name_plural=_("entity.HotspotsContent.verbose_name_plural"),
    fields=(
        ("num_loci", (_("entity.HotspotsContent.fields.num_loci.verbose_name"),)),
        ("pickfile", (_("entity.HotspotsContent.fields.pickfile.verbose_name"),)),
    ),
    extends=Content,
)

TargetRegionsContent = EntityFields(
    "TargetRegionsContent",
    verbose_name=_("entity.TargetRegionsContent.verbose_name"),
    verbose_name_plural=_("entity.TargetRegionsContent.verbose_name_plural"),
    fields=(
        (
            "num_targets",
            (_("entity.TargetRegionsContent.fields.num_targets.verbose_name"),),
        ),
        (
            "num_genes",
            (_("entity.TargetRegionsContent.fields.num_genes.verbose_name"),),
        ),
        (
            "num_bases",
            (_("entity.TargetRegionsContent.fields.num_bases.verbose_name"),),
        ),
        ("pickfile", (_("entity.TargetRegionsContent.fields.pickfile.verbose_name"),)),
    ),
    extends=Content,
)

Publisher = EntityFields(
    "Publisher",
    verbose_name=_("entity.Publisher.verbose_name"),
    verbose_name_plural=_("entity.Publisher.verbose_name_plural"),
    fields=(
        ("name", (_("entity.Publisher.fields.name.verbose_name"),)),
        ("version", (_("entity.Publisher.fields.version.verbose_name"),)),
        ("date", (_("entity.Publisher.fields.date.verbose_name"),)),
        ("path", (_("entity.Publisher.fields.path.verbose_name"),)),
    ),
)

Barcode = EntityFields(
    "Barcode",
    verbose_name=_("entity.Barcode.verbose_name"),
    verbose_name_plural=_("entity.Barcode.verbose_name_plural"),
    fields=(
        ("name", (_("entity.Barcode.fields.index.verbose_name"),)),
        ("index", (_("entity.Barcode.fields.index.verbose_name"),)),
        ("id_str", (_("entity.Barcode.fields.id_str.verbose_name"),)),
        ("adapter", (_("entity.Barcode.fields.adapter.verbose_name"),)),
        ("annotation", (_("entity.Barcode.fields.annotation.verbose_name"),)),
        ("floworder", (_("entity.Barcode.fields.floworder.verbose_name"),)),
        ("sequence", (_("entity.Barcode.fields.sequence.verbose_name"),)),
        ("type", (_("entity.Barcode.fields.type.verbose_name"),)),
    ),
)
BarcodeSet = EntityFields(
    "BarcodeSet",
    verbose_name=_("entity.BarcodeSet.verbose_name"),
    verbose_name_plural=_("entity.BarcodeSet.verbose_name_plural"),
    fields=(("name", (_("entity.BarcodeSet.fields.name.verbose_name"),)),),
    extends=Barcode,
)

TestFragment = EntityFields(
    "TestFragment",
    verbose_name=_("entity.TestFragment.verbose_name"),
    verbose_name_plural=_("entity.TestFragment.verbose_name_plural"),
    fields=(
        ("name", (_("entity.TestFragment.fields.name.verbose_name"),)),
        ("key", (_("entity.TestFragment.fields.key.verbose_name"),)),
        ("sequence", (_("entity.TestFragment.fields.sequence.verbose_name"),)),
        ("isofficial", (_("entity.TestFragment.fields.isofficial.verbose_name"),)),
        ("comments", (_("entity.TestFragment.fields.comments.verbose_name"),)),
    ),
)

User = EntityFields(
    "User",
    verbose_name=_("entity.User.verbose_name"),
    verbose_name_plural=_("entity.User.verbose_name_plural"),
    fields=(
        ("username", (_("entity.User.fields.username.verbose_name"),)),
        ("email", (_("entity.User.fields.email.verbose_name"),)),
        ("password1", (_("entity.User.fields.password1.verbose_name"),)),
        ("password2", (_("entity.User.fields.password2.verbose_name"),)),
        ("get_full_name", (_("entity.User.fields.get_full_name.verbose_name"),)),
        ("date_joined", (_("entity.User.fields.date_joined.verbose_name"),)),
        ("api_key", (_("entity.User.fields.api_key.verbose_name"),)),
        ("accountlevel", (_("entity.User.fields.accountlevel.verbose_name"),)),
    ),
)
SuperUser = EntityFields(
    "SuperUser",
    verbose_name=_("entity.SuperUser.verbose_name"),
    verbose_name_plural=_("entity.SuperUser.verbose_name_plural"),
    fields=(),
    extends=User,
)

StaffUser = EntityFields(
    "StaffUser",
    verbose_name=_("entity.StaffUser.verbose_name"),
    verbose_name_plural=_("entity.StaffUser.verbose_name_plural"),
    fields=(),
    extends=User,
)

UserProfile = EntityFields(
    "UserProfile",
    verbose_name=_("entity.UserProfile.verbose_name"),
    verbose_name_plural=_("entity.UserProfile.verbose_name_plural"),
    fields=(
        ("name", (_("entity.UserProfile.fields.name.verbose_name"),)),
        ("phone_number", (_("entity.UserProfile.fields.phone_number.verbose_name"),)),
        ("user", (User,)),
    ),
)

Plugin = EntityFields(
    "Plugin",
    verbose_name=_("entity.Plugin.verbose_name"),
    verbose_name_plural=_("entity.Plugin.verbose_name_plural"),
    fields=(
        ("name", (_("entity.Plugin.fields.name.verbose_name"),)),
        ("description", (_("entity.Plugin.fields.description.verbose_name"),)),
        ("version", (_("entity.Plugin.fields.version.verbose_name"),)),
        ("CurrentVersion", (_("entity.Plugin.fields.CurrentVersion.verbose_name"),)),
        ("date", (_("entity.Plugin.fields.date.verbose_name"),)),
        (
            "active",
            (_("entity.Plugin.fields.active.verbose_name"),),
        ),  # True for installed, False for uninstalled ; Store and mask inactive (uninstalled) plugins
        (
            "isSupported",
            (_("entity.Plugin.fields.isSupported.verbose_name"),),
        ),  # is Plugin supported by Vendor (Ion)
        (
            "selected",
            (_("entity.Plugin.fields.selected.verbose_name"),),
        ),  # this toggles visibility on the interface
        (
            "defaultSelected",
            (_("entity.Plugin.fields.defaultSelected.verbose_name"),),
        ),  # this flag will indicate if the plugin will be included by default in new plans (not from template)
        (
            "isUpgradable",
            (_("entity.Plugin.fields.isUpgradable.verbose_name"),),
        ),  # True if the supported plugin has an aptitude version available, false otherwise
        (
            "script",
            (_("entity.Plugin.fields.script.verbose_name"),),
        ),  # True if the supported plugin has an aptitude version available, false otherwise
    ),
)

PluginResult = EntityFields(
    "PluginResult",
    verbose_name=_("entity.PluginResult.verbose_name"),
    verbose_name_plural=_("entity.PluginResult.verbose_name_plural"),
    fields=(
        ("result", (_("entity.PluginResult.fields.result.verbose_name"),)),
        ("starttime", (_("entity.PluginResult.fields.starttime.verbose_name"),)),
        ("endtime", (_("entity.PluginResult.fields.endtime.verbose_name"),)),
        ("resultName", (_("entity.PluginResult.fields.resultName.verbose_name"),)),
        ("pluginName", (_("entity.PluginResult.fields.pluginName.verbose_name"),)),
        ("state", (_("entity.PluginResult.fields.state.verbose_name"),)),
        ("size", (_("entity.PluginResult.fields.size.verbose_name"),)),
    ),
)

FileTypes = namedtuple("FileTypes", ["deb", "zip", "json"])(
    deb=_("entity.FileTypes.deb"),
    zip=_("entity.FileTypes.zip"),
    json=_("entity.FileTypes.json"),
)

IonMeshNode = EntityFields(
    "IonMeshNode",
    verbose_name=_("entity.IonMeshNode.verbose_name"),
    verbose_name_plural=_("entity.IonMeshNode.verbose_name_plural"),
    fields=(
        ("name", (_("entity.IonMeshNode.fields.name.verbose_name"),)),
        ("hostname", (_("entity.IonMeshNode.fields.hostname.verbose_name"),)),
        ("active", (_("entity.IonMeshNode.fields.active.verbose_name"),)),
        ("status", (_("entity.IonMeshNode.fields.status.verbose_name"),)),
        ("version", (_("entity.IonMeshNode.fields.version.verbose_name"),)),
        ("system_id", (_("entity.IonMeshNode.fields.system_id.verbose_name"),)),
        ("apikey_remote", (_("entity.IonMeshNode.fields.apikey_remote.verbose_name"),)),
        ("apikey_local", (_("entity.IonMeshNode.fields.apikey_local.verbose_name"),)),
    ),
)

IonMeshNodeStatus = namedtuple(
    "IonMeshNodeStatus",
    [
        "good",
        "connection_error",
        "timeout",
        "error",
        "incompatible",
        "unauthorized",
        "unknown",
    ],
)(
    good=_("entity.IonMeshNode.fields.status.choices.good"),
    connection_error=_("entity.IonMeshNode.fields.status.choices.connection_error"),
    timeout=_("entity.IonMeshNode.fields.status.choices.timeout"),
    error=_("entity.IonMeshNode.fields.status.choices.error"),
    incompatible=_("entity.IonMeshNode.fields.status.choices.incompatible"),
    unauthorized=_("entity.IonMeshNode.fields.status.choices.unauthorized"),
    unknown=_("entity.IonMeshNode.fields.status.choices.unknown"),
)

IonReporterUploader = EntityFields(
    "IonReporterUploader",
    verbose_name=_("entity.IonReporterUploader.verbose_name"),
    verbose_name_plural=_("entity.IonReporterUploader.verbose_name_plural"),
    fields=(),
)

EmailAddress = EntityFields(
    "EmailAddress",
    verbose_name=_("entity.EmailAddress.verbose_name"),
    verbose_name_plural=_("entity.EmailAddress.verbose_name_plural"),
    fields=(
        (
            "email",
            (_("entity.EmailAddress.fields.email.verbose_name"),),
        ),  # Email Address
        (
            "selected",
            (_("entity.EmailAddress.fields.selected.verbose_name"),),
        ),  # Enabled
    ),
)

GlobalConfig = EntityFields(
    "GlobalConfig",
    verbose_name=_("entity.GlobalConfig.verbose_name"),
    verbose_name_plural=_("entity.GlobalConfig.verbose_name_plural"),
    fields=(
        (
            "enable_nightly_email",
            (_("entity.GlobalConfig.fields.enable_nightly_email.verbose_name"),),
        ),
    ),
)

QcType = namedtuple(
    "QcType", ["Usable_Sequence_Percentage", "Key_Signal", "Bead_Loading_Percentage"]
)(
    Usable_Sequence_Percentage=_("entity.QcType.Usable_Sequence_Percentage"),
    Key_Signal=_("entity.QcType.Key_Signal"),
    Bead_Loading_Percentage=_("entity.QcType.Bead_Loading_Percentage"),
)


_ModelsQcTypeToLabelsQcType = {
    "Usable Sequence (%)": QcType.Usable_Sequence_Percentage,
    "Key Signal (1-100)": QcType.Key_Signal,
    "Bead Loading (%)": QcType.Bead_Loading_Percentage,
}


def ModelsQcTypeToLabelsQcTypeAsDict():
    return copy.deepcopy(_ModelsQcTypeToLabelsQcType)


def ModelsQcTypeToLabelsQcType(models_qctype_qcname):
    if _ModelsQcTypeToLabelsQcType.has_key(models_qctype_qcname):
        return _ModelsQcTypeToLabelsQcType[models_qctype_qcname]
    else:
        return False


SamplePrepInstrument = namedtuple("SamplePrepInstrument", ["OT", "IC", "IA"])(
    OT=_("entity.SamplePrepInstrument.OT"),
    IC=_("entity.SamplePrepInstrument.IC"),
    IA=_("entity.SamplePrepInstrument.IA"),
)
