{% spaceless %}
$.irConfigSelection_1 = jQuery.parseJSON('{{planTemplateData.irConfigSelection_1|escapejs}}');
$.irConfigSelection = jQuery.parseJSON('{{planTemplateData.irConfigSelection|escapejs}}');

var INTENT = '{{intent}}';
var submitUrl = null;
{% if selectedPlanTemplate and intent == "Edit" %}
submitUrl = "{% url save_plan_or_template selectedPlanTemplate.id %}";
{% endif %}
{% if selectedPlanTemplate and intent == "EditPlan" %}
submitUrl = "{% url save_plan_or_template selectedPlanTemplate.id %}";
{% endif %}

var PLANNED_URL = '{% url planned %}';

var selectedPlanTemplate = null;
{% if selectedPlanTemplate %}
selectedPlanTemplate = {};
selectedPlanTemplate.runMode = "{{selectedPlanTemplate.runMode}}";
selectedPlanTemplate.sampleDisplayedName = "{{selectedPlanTemplate.sampleDisplayedName}}";
selectedPlanTemplate.barcodedSamples = {{selectedPlanTemplate.barcodedSamples|safe}};
selectedPlanTemplate.barcodeId = "{{selectedPlanTemplate.barcodeId}}";
selectedPlanTemplate.notes = "{{selectedPlanTemplate.notes}}";
selectedPlanTemplate.isIonChef = "{{selectedPlanTemplate.isIonChef}}";
selectedPlanTemplate.reference = "{{selectedPlanTemplate.library}}";
selectedPlanTemplate.runType = "{{selectedPlanTemplate.runType}}";
{% endif %}

var selectedApplProductData = null;
{% if selectedApplProductData %}
var selectedApplProductData = {};
selectedApplProductData.isDefaultPairedEnd = "{{selectedApplProductData.isDefaultPairedEnd}}";
selectedApplProductData.isHotspotRegionBEDFileSupported = "{{selectedApplProductData.isHotspotRegionBEDFileSupported}}";
selectedApplProductData.isDefaultBarcoded = "{{selectedApplProductData.isDefaultBarcoded}}";
selectedApplProductData.defaultBarcodeKitName = "{{selectedApplProductData.defaultBarcodeKitName}}";
selectedApplProductData.defaultOneTouchTemplateKit = "{{selectedApplProductData.defaultOneTouchTemplateKit}}";
selectedApplProductData.defaultIonChefKit = "{{selectedApplProductData.defaultIonChefKit}}";
selectedApplProductData.reference = "{{selectedApplProductData.reference}}"
{% endif %}


var planTemplateData = null;
{% if planTemplateData %}
planTemplateData = {};
planTemplateData.irConfigSaved = '{{planTemplateData.irConfigSaved|escapejs}}';
planTemplateData.irConfigSaved_version = '{{planTemplateData.irConfigSaved_version|escapejs}}';
planTemplateData.peForwardLibKeys = "{{planTemplateData.peForwardLibKeys}}";

planTemplateData.selectedPlugins = [];
planTemplateData.selectedUploaders = [];
planTemplateData.pluginUserInput = {};
if (INTENT == "New" || INTENT == "Plan Run New") {
    {% for plugin in planTemplateData.plugins %}
    {% if plugin.autorun %}
    planTemplateData.selectedPlugins.push("{{plugin.name}}");
    {% endif %}
    {% endfor %}

    {% for plugin in planTemplateData.uploaders %}
    {% if plugin.autorun %}
    planTemplateData.selectedUploaders.push("{{plugin.name}}");
    {% endif %}
    {% endfor %}
} else {
    {% for plugin in planTemplateData.plugins %}
    {% if plugin.selected %}
    planTemplateData.selectedPlugins.push("{{plugin.name|escapejs}}");
    planTemplateData.pluginUserInput["{{plugin.pk}}"]="{{plugin.userInput|escapejs}}";
    {% endif %}
    {% endfor %}

    {% for plugin in planTemplateData.uploaders %}
    {% if plugin.selected %}
    planTemplateData.selectedUploaders.push("{{plugin.name|escapejs}}");
    {% endif %}
    {% endfor %}
}
{% endif %}
{% endspaceless %}

var isIR_v1_selected = false;
var isIR_beyond_v1_selected = false;

// add selected uploaders to review
{% spaceless %}
{% for uploader in planTemplateData.uploaders %}
{% if uploader.autorun %}
if (INTENT == "New" || INTENT == "Plan Run New") {
    if ("{{uploader.name|escapejs}}".toLowerCase().search('ionreporteruploader_v1_0') >= 0) {
        isIR_v1_selected = true;
    } else if ("{{uploader.name|escapejs}}".toLowerCase().search('ionreporteruploader') >= 0) {
        isIR_beyond_v1_selected = true;
    }
}
{% endif %}

{% if uploader.selected %}
if (INTENT != "New" && INTENT != "Plan Run New") {
    if ("{{uploader.name|escapejs}}".toLowerCase().search('ionreporteruploader_v1_0') >= 0) {
        isIR_v1_selected = true;
    } else if ("{{uploader.name|escapejs}}".toLowerCase().search('ionreporteruploader') >= 0) {
        isIR_beyond_v1_selected = true;
    }
}
{% endif %}
{% endfor %}
{% endspaceless %}

TB.namespace('TB.plan.wizard');

/**
 * TB.plan.wizard.getApplProduct - A method that will disappear once server returns JSON instead
 */
TB.plan.wizard.getApplProduct = function(runType) {
    var applProduct = {};
    applProduct.GENS = {};
    applProduct.GENS.runTypeDescription = '{{planTemplateData.GENS.runType.description}}';
    applProduct.GENS.variantFrequency = "{{planTemplateData.GENS.defaultVariantFrequency}}";
    applProduct.GENS.isDefaultPairedEnd = "{{planTemplateData.GENS.isDefaultPairedEnd}}";
    applProduct.GENS.reference = '{{planTemplateData.GENS.reference}}';
    applProduct.GENS.targetBedFile = '{{planTemplateData.GENS.targetBedFile}}';
    applProduct.GENS.hotSpotBedFile = '{{planTemplateData.GENS.hotSpotBedFile}}';
    if (applProduct.GENS.isDefaultPairedEnd == "True") {
        applProduct.GENS.seqKitName = "{{planTemplateData.GENS.peSeqKit.name}}";
        applProduct.GENS.libKitName = "{{planTemplateData.GENS.peLibKit.name}}";
    } else {
        applProduct.GENS.seqKitName = "{{planTemplateData.GENS.seqKit.name}}";
        applProduct.GENS.libKitName = "{{planTemplateData.GENS.libKit.name}}";
    }
    applProduct.GENS.chipType = '{{planTemplateData.GENS.chipType}}';
    if ("{{planTemplateData.GENS.defaultOneTouchTemplateKit}}") {
        applProduct.GENS.defaultOneTouchTemplateKitName = "{{planTemplateData.GENS.defaultOneTouchTemplateKit.name}}";
    }
    if ("{{planTemplateData.GENS.defaultIonChefKit}}") {
        applProduct.GENS.defaultIonChefKitName = "{{planTemplateData.GENS.defaultIonChefKit.name}}";
    }
    
    if ("{{planTemplateData.GENS.controlSeqKit}}") {
        applProduct.GENS.controlSeqName = "{{planTemplateData.GENS.controlSeqKit.name}}";
    }
    applProduct.GENS.flowCount = "{{planTemplateData.GENS.flowCount}}";
    applProduct.GENS.isHotspotRegionBEDFileSupported = "{{planTemplateData.GENS.isHotspotRegionBEDFileSupported}}";
    applProduct.GENS.reference = "{{planTemplateData.GENS.reference}}"
    	
    applProduct.AMPS = {};
    applProduct.AMPS.runTypeDescription = '{{planTemplateData.AMPS.runType.description}}';
    applProduct.AMPS.variantFrequency = "{{planTemplateData.AMPS.defaultVariantFrequency}}";
    applProduct.AMPS.isDefaultPairedEnd = "{{planTemplateData.AMPS.isDefaultPairedEnd}}";
    applProduct.AMPS.reference = '{{planTemplateData.AMPS.reference}}';
    applProduct.AMPS.targetBedFile = '{{planTemplateData.AMPS.targetBedFile}}';
    applProduct.AMPS.hotSpotBedFile = '{{planTemplateData.AMPS.hotSpotBedFile}}';
    if (applProduct.AMPS.isDefaultPairedEnd == "True") {
        applProduct.AMPS.libKitName = '{{planTemplateData.AMPS.peLibKit.name}}';
        applProduct.AMPS.seqKitName = "{{planTemplateData.AMPS.peSeqKit.name}}";
    } else {
        applProduct.AMPS.libKitName = '{{planTemplateData.AMPS.libKit.name}}';
        applProduct.AMPS.seqKitName = "{{planTemplateData.AMPS.seqKit.name}}";
    }
    applProduct.AMPS.chipType = '{{planTemplateData.AMPS.chipType}}';
    if ("{{planTemplateData.AMPS.defaultOneTouchTemplateKit}}") {
        applProduct.AMPS.defaultOneTouchTemplateKitName = "{{planTemplateData.AMPS.defaultOneTouchTemplateKit.name}}";
    }
    if ("{{planTemplateData.AMPS.defaultIonChefKit}}") {
        applProduct.AMPS.defaultIonChefKitName = "{{planTemplateData.AMPS.defaultIonChefKit.name}}";
    }    
    if ("{{planTemplateData.AMPS.controlSeqKit}}") {
        applProduct.AMPS.controlSeqName = "{{planTemplateData.AMPS.controlSeqKit.name}}";
    }
    applProduct.AMPS.flowCount = "{{planTemplateData.AMPS.flowCount}}";
    applProduct.AMPS.isHotspotRegionBEDFileSupported = "{{planTemplateData.AMPS.isHotspotRegionBEDFileSupported}}";
    applProduct.AMPS.reference = "{{planTemplateData.AMPS.reference}}"
    	
    applProduct.TARS = {};
    applProduct.TARS.runTypeDescription = '{{planTemplateData.TARS.runType.description}}';
    applProduct.TARS.variantFrequency = "{{planTemplateData.TARS.defaultVariantFrequency}}";
    applProduct.TARS.isDefaultPairedEnd = "{{planTemplateData.TARS.isDefaultPairedEnd}}";
    applProduct.TARS.reference = '{{planTemplateData.TARS.reference}}';
    applProduct.TARS.targetBedFile = '{{planTemplateData.TARS.targetBedFile}}';
    applProduct.TARS.hotSpotBedFile = '{{planTemplateData.TARS.hotSpotBedFile}}';
    if (applProduct.TARS.isDefaultPairedEnd == "True") {
        applProduct.TARS.libKitName = '{{planTemplateData.TARS.peLibKit.name}}';
        applProduct.TARS.seqKitName = '{{planTemplateData.TARS.peSeqKit.name}}';
    } else {
        applProduct.TARS.libKitName = '{{planTemplateData.TARS.libKit.name}}';
        applProduct.TARS.seqKitName = '{{planTemplateData.TARS.seqKit.name}}';
    }
    applProduct.TARS.chipType = '{{planTemplateData.TARS.chipType}}';
    if ("{{planTemplateData.TARS.defaultOneTouchTemplateKit}}") {
        applProduct.TARS.defaultOneTouchTemplateKitName = "{{planTemplateData.TARS.defaultOneTouchTemplateKit.name}}";
    }
    if ("{{planTemplateData.TARS.defaultIonChefKit}}") {
        applProduct.TARS.defaultIonChefKitName = "{{planTemplateData.TARS.defaultIonChefKit.name}}";
    }    
    if ("{{planTemplateData.TARS.controlSeqKit}}") {
        applProduct.TARS.controlSeqName = "{{planTemplateData.TARS.controlSeqKit.name}}";
    }

    applProduct.TARS.flowCount = "{{planTemplateData.TARS.flowCount}}";
    applProduct.TARS.isHotspotRegionBEDFileSupported = "{{planTemplateData.TARS.isHotspotRegionBEDFileSupported}}";
    applProduct.TARS.reference = "{{planTemplateData.TARS.reference}}"
    	
    applProduct.WGNM = {};
    applProduct.WGNM.runTypeDescription = '{{planTemplateData.WGNM.runType.description}}';
    applProduct.WGNM.variantFrequency = "{{planTemplateData.WGNM.defaultVariantFrequency}}";
    applProduct.WGNM.isDefaultPairedEnd = "{{planTemplateData.WGNM.isDefaultPairedEnd}}";
    applProduct.WGNM.reference = '{{planTemplateData.WGNM.reference}}';
    applProduct.WGNM.targetBedFile = '{{planTemplateData.WGNM.targetBedFile}}';
    applProduct.WGNM.hotSpotBedFile = '{{planTemplateData.WGNM.hotSpotBedFile}}';
    if (applProduct.WGNM.isDefaultPairedEnd == "True") {
        applProduct.WGNM.libKitName = '{{planTemplateData.WGNM.peLibKit.name}}';
        seqKitName = '{{planTemplateData.WGNM.peSeqKit.name}}';
    } else {
        applProduct.WGNM.libKitName = '{{planTemplateData.WGNM.libKit.name}}';
        applProduct.WGNM.seqKitName = '{{planTemplateData.WGNM.seqKit.name}}';
    }
    applProduct.WGNM.chipType = '{{planTemplateData.WGNM.chipType}}';
    if ("{{planTemplateData.WGNM.defaultOneTouchTemplateKit}}") {
        applProduct.WGNM.defaultOneTouchTemplateKitName = "{{planTemplateData.WGNM.defaultOneTouchTemplateKit.name}}";
    }
    if ("{{planTemplateData.WGNM.defaultIonChefKit}}") {
        applProduct.WGNM.defaultIonChefKitName = "{{planTemplateData.WGNM.defaultIonChefKit.name}}";
    }    
    if ("{{planTemplateData.WGNM.controlSeqKit}}") {
        applProduct.WGNM.controlSeqName = "{{planTemplateData.WGNM.controlSeqKit.name}}";
    }
    applProduct.WGNM.flowCount = "{{planTemplateData.WGNM.flowCount}}";
    applProduct.WGNM.isHotspotRegionBEDFileSupported = "{{planTemplateData.WGNM.isHotspotRegionBEDFileSupported}}";
    applProduct.WGNM.reference = "{{planTemplateData.WGNM.reference}}"
    	
    applProduct.RNA = {};
    applProduct.RNA.runTypeDescription = '{{planTemplateData.RNA.runType.description}}';
    applProduct.RNA.variantFrequency = "{{planTemplateData.RNA.defaultVariantFrequency}}";
    applProduct.RNA.isDefaultPairedEnd = "{{planTemplateData.RNA.isDefaultPairedEnd}}";
    applProduct.RNA.reference = '{{planTemplateData.RNA.reference}}';
    applProduct.RNA.targetBedFile = '{{planTemplateData.RNA.targetBedFile}}';
    applProduct.RNA.hotSpotBedFile = '{{planTemplateData.RNA.hotSpotBedFile}}';
    if (applProduct.RNA.isDefaultPairedEnd == "True") {
        applProduct.RNA.libKitName = '{{planTemplateData.RNA.peLibKit.name}}';
        applProduct.RNA.seqKitName = '{{planTemplateData.RNA.peSeqKit.name}}';
    } else {
        applProduct.RNA.libKitName = '{{planTemplateData.RNA.libKit.name}}';
        applProduct.RNA.seqKitName = '{{planTemplateData.RNA.seqKit.name}}';
    }
    applProduct.RNA.chipType = '{{planTemplateData.RNA.chipType}}';
    if ("{{planTemplateData.RNA.defaultOneTouchTemplateKit}}") {
        applProduct.RNA.defaultOneTouchTemplateKitName = "{{planTemplateData.RNA.defaultOneTouchTemplateKit.name}}";
    }
    if ("{{planTemplateData.RNA.defaultIonChefKit}}") {
        applProduct.RNA.defaultIonChefKitName = "{{planTemplateData.RNA.defaultIonChefKit.name}}";
    }    
    if ("{{planTemplateData.RNA.controlSeqKit}}") {
        applProduct.RNA.controlSeqName = "{{planTemplateData.RNA.controlSeqKit.name}}";
    }
    applProduct.RNA.flowCount = "{{planTemplateData.RNA.flowCount}}";
    applProduct.RNA.isHotspotRegionBEDFileSupported = "{{planTemplateData.RNA.isHotspotRegionBEDFileSupported}}";
    applProduct.RNA.reference = "{{planTemplateData.RNA.reference}}"
    	
    applProduct.AMPS_RNA = {};
    applProduct.AMPS_RNA.runTypeDescription = '{{planTemplateData.AMPS_RNA.runType.description}}';
    applProduct.AMPS_RNA.variantFrequency = "{{planTemplateData.AMPS_RNA.defaultVariantFrequency}}";
    applProduct.AMPS_RNA.isDefaultPairedEnd = "{{planTemplateData.AMPS_RNA.isDefaultPairedEnd}}";
    applProduct.AMPS_RNA.reference = '{{planTemplateData.AMPS_RNA.reference}}';
    applProduct.AMPS_RNA.targetBedFile = '{{planTemplateData.AMPS_RNA.targetBedFile}}';
    applProduct.AMPS_RNA.hotSpotBedFile = '{{planTemplateData.AMPS_RNA.hotSpotBedFile}}';
    if (applProduct.AMPS_RNA.isDefaultPairedEnd == "True") {
        applProduct.AMPS_RNA.libKitName = '{{planTemplateData.AMPS_RNA.peLibKit.name}}';
        applProduct.AMPS_RNA.seqKitName = "{{planTemplateData.AMPS_RNA.peSeqKit.name}}";
    } else {
        applProduct.AMPS_RNA.libKitName = '{{planTemplateData.AMPS_RNA.libKit.name}}';
        applProduct.AMPS_RNA.seqKitName = "{{planTemplateData.AMPS_RNA.seqKit.name}}";
    }
    applProduct.AMPS_RNA.chipType = '{{planTemplateData.AMPS_RNA.chipType}}';
    if ("{{planTemplateData.AMPS_RNA.defaultOneTouchTemplateKit}}") {
        applProduct.AMPS_RNA.defaultOneTouchTemplateKitName = "{{planTemplateData.AMPS_RNA.defaultOneTouchTemplateKit.name}}";
    }
    if ("{{planTemplateData.AMPS_RNA.defaultIonChefKit}}") {
        applProduct.AMPS_RNA.defaultIonChefKitName = "{{planTemplateData.AMPS_RNA.defaultIonChefKit.name}}";
    }    
    if ("{{planTemplateData.AMPS_RNA.controlSeqKit}}") {
        applProduct.AMPS_RNA.controlSeqName = "{{planTemplateData.AMPS_RNA.controlSeqKit.name}}";
    }
    applProduct.AMPS_RNA.flowCount = "{{planTemplateData.AMPS_RNA.flowCount}}";
    applProduct.AMPS_RNA.isHotspotRegionBEDFileSupported = "{{planTemplateData.AMPS_RNA.isHotspotRegionBEDFileSupported}}";
    applProduct.AMPS_RNA.reference = "{{planTemplateData.AMPS_RNA.reference}}"
    	
    return runType && applProduct[runType] || null;
};
