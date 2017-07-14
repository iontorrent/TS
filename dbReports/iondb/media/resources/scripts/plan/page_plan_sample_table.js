/**
 Update hidden samplesTable input whenever any samples/barcodes/IR params change
 */
updateSamplesTable = function () {
    var barcoded = $('input[id=chk_barcoded]').is(':checked');
    var samplesTableJSON = $("#grid").data("kendoGrid").dataSource.data().toJSON();

    if (samplesTableJSON.length > 0) {
        // update IR fields
        if (USERINPUT.is_ir_connected) {
            $.each(samplesTableJSON, function(i, row){
                var workflowObj = getWorkflowObj(row.irWorkflow, row.irtag_isFactoryProvidedWorkflow);
                row.ApplicationType = workflowObj['ApplicationType'];
                row.irRelationshipType = workflowObj["RelationshipType"];
                row.tag_isFactoryProvidedWorkflow = workflowObj['tag_isFactoryProvidedWorkflow'];
                // these values are saved as strings
                row.irSetID = row.irSetID ? row.irSetID.toString() : "";
                row.ircellularityPct = row.ircellularityPct ? row.ircellularityPct.toString() : "";
                row.irbiopsyDays = row.irbiopsyDays ? row.irbiopsyDays.toString() : "";
            });
        }
        //console.log("page_plan_sample_table.updateSamplesTable table=", JSON.stringify(samplesTable));
        $('#samplesTable').val(JSON.stringify(samplesTableJSON));
    }
}

var isEven = function(aNumber){
    return (aNumber % 2 == 0) ? true : false;
};


// ******************* Handle Dual Nucleotide type for DNA/RNA Applications ************************* //

// These fields will be disabled on the RNA sample row for dual nuc type if SameSample is checked
var fieldsToUpdateForRNASameSample = [ 'sampleName', 'sampleDescription', 'sampleExternalId', 'controlType',
        'irWorkflow', 'irRelationRole', 'irGender', 'irSetID', 'ircancerType', 'ircellularityPct'
    ];

function handleSameSampleForDualNucleotideType(){
    // If SameSample is checked:
    //  (1) for even row, set nucleotide type to DNA; set to RNA if odd row
    //  (2) if this results in nucleotide type change, process Reference+bedfiles selections
    //  (3) for odd row autopopulate dual fields to previous row's values
    //  (4) disable and disallow user input for RNA rows' dual fields
    
    // If not SameSample:
    //  (1) enable RNA rows' dual fields

    if (!planOpt.isDualNucleotideType)
        return;
    
    var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");
    
    if (isSameSampleForDual){
        var isSameRefInfoPerSample = $('input[id=isSameRefInfoPerSample]').is(":checked");
        var refInfo = getDefaultReferenceInfo();

        var samplesTableJSON = $("#grid").data("kendoGrid").dataSource.data().toJSON();
        $.each(samplesTableJSON, function(i, row){
            if (isEven(i)){
                // DNA row
                if (row.nucleotideType != "DNA"){
                    row.nucleotideType = "DNA";
                    if (refInfo.isSameRefInfoPerSample){
                        row.reference = refInfo.default_reference;
                        row.targetRegionBedFile = refInfo.default_targetBedFile;
                        row.hotSpotRegionBedFile = refInfo.default_hotSpotBedFile;
                    }
                }
                if (irWorkflowNotValid(row)){
                    row.irWorkflow = "";
                }
            } else {
                // RNA row
                row.nucleotideType = "RNA";
                row.reference = refInfo.mixedTypeRNA_reference;
                row.targetRegionBedFile = refInfo.mixedTypeRNA_targetBedFile;
                row.hotSpotRegionBedFile = "";
 
                var prevRow = samplesTableJSON[i-1];
                $.each(fieldsToUpdateForRNASameSample, function(i, field){
                    row[field] = prevRow[field];
                });
            }
        });
        $("#grid").data("kendoGrid").dataSource.data(samplesTableJSON);
    }

    toggleRNASampleForDualNucleotideTypeEnablements(isSameSampleForDual);
}

function toggleRNASampleForDualNucleotideTypeEnablements(isToDisable) {
    // Disable/enable RNA row for Dual Nucleotide type
    if (planOpt.isDualNucleotideType) {
        var $rnaRows = $('#grid').find('tbody>tr').filter(function(){
            return $('[name=nucleotideType]', this).text() == "Fusions";
        });

        $.each(fieldsToUpdateForRNASameSample, function(i, field){
            toggleDisableElements($rnaRows.find('[name='+field+']'), isToDisable);
        });

        // also disable the nucleotideType so rows remain consistent
        toggleDisableElements($('#grid').find('tbody>tr').find('[name=nucleotideType]'), isToDisable);
        
        // keep disabled sample parameters input for Plan-by-Sample
        if (planOpt.isPlanBySample && !isToDisable){
            toggleDisableElements($rnaRows.find('[name=sampleDescription]'), true);
            toggleDisableElements($rnaRows.find('[name=sampleExternalId]'), true);
        }
    }
}


// ******************* Reference and BED files ************************* //


function updateSampleReferenceColumnsWithDefaults() {
    var samplesTableJSON = $("#grid").data("kendoGrid").dataSource.data().toJSON();
    var refInfo = getDefaultReferenceInfo();

    $.each(samplesTableJSON, function(){
        if (planOpt.isDualNucleotideType && this.nucleotideType == "RNA"){
            this.reference = refInfo.mixedTypeRNA_reference;
            this.targetRegionBedFile = refInfo.mixedTypeRNA_targetBedFile;
            this.hotSpotRegionBedFile = "";
        } else {
            this.reference = refInfo.default_reference;
            this.targetRegionBedFile = refInfo.default_targetBedFile;
            this.hotSpotRegionBedFile = refInfo.default_hotSpotBedFile;
        }
        if (irWorkflowNotValid(this)){
            this.irWorkflow = "";
        }
    });
    $("#grid").data("kendoGrid").dataSource.data(samplesTableJSON);
}

function updateSamplesForReference(selectedReference) {
    if (planOpt.isReferenceSupported) {
        var samplesTableJSON = $("#grid").data("kendoGrid").dataSource.data().toJSON();
        $.each(samplesTableJSON, function(){
            if (planOpt.isDualNucleotideType && this.nucleotideType == "RNA"){
                // don't update the RNA sample for mixed type DNA/RNA application
            } else {
                this.reference = selectedReference;
                if (irWorkflowNotValid(this)){
                    this.irWorkflow = "";
                }
            }
        });
        $("#grid").data("kendoGrid").dataSource.data(samplesTableJSON);
    }
}

function updateMixedTypeRNASamplesForReference(selectedReference) {
    var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");
    
    if (planOpt.isReferenceSupported && planOpt.isDualNucleotideType) {
        var samplesTableJSON = $("#grid").data("kendoGrid").dataSource.data().toJSON();
        $.each(samplesTableJSON, function(){
            if (this.nucleotideType == "RNA"){
                this.reference = selectedReference;
                if (!isSameSampleForDual && irWorkflowNotValid(this)){
                    this.irWorkflow = "";
                }
            }
        });
        $("#grid").data("kendoGrid").dataSource.data(samplesTableJSON);
    }
}

function updateSamplesForTargetRegion(selectedTargetBedFile) {
    if (planOpt.isTargetBEDFileSupported) {
        var samplesTableJSON = $("#grid").data("kendoGrid").dataSource.data().toJSON();
        $.each(samplesTableJSON, function(){
            if (planOpt.isDualNucleotideType && this.nucleotideType == "RNA"){
                // don't update the RNA sample for mixed type DNA/RNA application
            } else {
                this.targetRegionBedFile = selectedTargetBedFile;
            }
        });
        $("#grid").data("kendoGrid").dataSource.data(samplesTableJSON);
    }
}

function updateMixedTypeRNASamplesForTargetRegion(selectedTargetBedFile) {
    if (planOpt.isTargetBEDFileSupported && planOpt.isDualNucleotideType) {
        var samplesTableJSON = $("#grid").data("kendoGrid").dataSource.data().toJSON();
        $.each(samplesTableJSON, function(){
            if (this.nucleotideType == "RNA"){
                this.targetRegionBedFile = selectedTargetBedFile;
            }
        });
        $("#grid").data("kendoGrid").dataSource.data(samplesTableJSON);
    }
}

function updateSamplesForHotSpot(selectedHotSpotBedFile) {
    if (planOpt.isHotspotBEDFileSupported) {
        var samplesTableJSON = $("#grid").data("kendoGrid").dataSource.data().toJSON();
        $.each(samplesTableJSON, function(){
            if (planOpt.isDualNucleotideType && this.nucleotideType == "RNA"){
                // don't update the RNA sample for mixed type DNA/RNA application
            } else {
                this.hotSpotRegionBedFile = selectedHotSpotBedFile;
            }
        });
        $("#grid").data("kendoGrid").dataSource.data(samplesTableJSON);
    }
}

function showHideSampleReferenceColumns(makeVisible) {
    var grid = $("#grid").data("kendoGrid");
    if (planOpt.isReferenceSupported && makeVisible) {
        grid.showColumn("reference");
    } else {
        grid.hideColumn("reference");
    };
    if (planOpt.isTargetBEDFileSupported && makeVisible) {
        grid.showColumn("targetRegionBedFile");
    } else {
        grid.hideColumn("targetRegionBedFile");
    }
    if (planOpt.isHotspotBEDFileSupported && makeVisible) {
        grid.showColumn("hotSpotRegionBedFile");
    } else {
        grid.hideColumn("hotSpotRegionBedFile");
    }
    $('#referenceSectionTab').toggleClass('k-state-active', makeVisible);
}

function toggleSampleReferenceColumnEnablements(isToDisable) {
    if (planOpt.isReferenceSupported) toggleDisableElements($('[name=reference]'), isToDisable);
    if (planOpt.isTargetBEDFileSupported) toggleDisableElements($('[name=targetRegionBedFile]'), isToDisable);
    if (planOpt.isHotspotBEDFileSupported) {
        toggleDisableElements($('[name=hotSpotRegionBedFile]'), isToDisable);

        // If dual nuc type disable dropdowns for all RNA rows
        if (planOpt.isDualNucleotideType) {
            var $rnaRows = $('#grid').find('tbody>tr').filter(function(){
                    return $('[name=nucleotideType]', this).text() == "Fusions";
                });
            toggleDisableElements($rnaRows.find('[name=hotSpotRegionBedFile]'), true);
        }
    }
}

function getDefaultReferenceInfo(){
    return {
        'isSameRefInfoPerSample': $('input[id=isSameRefInfoPerSample]').is(":checked"),
        'default_reference': $('#default_reference').val(),
        'default_targetBedFile': $('#default_targetBedFile').val(),
        'default_hotSpotBedFile': $('#default_hotSpotBedFile').val(),
        'mixedTypeRNA_reference': $('#mixedTypeRNA_reference').val(),
        'mixedTypeRNA_targetBedFile': $('#mixedTypeRNA_targetBedFile').val()
    }
}


toggleDisableElements = function($elements, disable){
    $elements.prop('disabled', disable);
    $elements.css('opacity', disable ? 0.5 : 1);
    $elements.css('cursor', disable ? 'not-allowed' : 'auto');
}

var dropDnTemplate = kendo.template(
    '${ html } <span class="k-icon k-i-arrow-s pull-right"></span>'
)
var columnSectionTemplate = kendo.template(
    '<span id=#=id# class="verticalTab k-button" title="Show/Hide #=text# Columns">#=text#</span>'
)

$(document).ready(function () {

    var dataSource = new kendo.data.DataSource({
        transport: {
            read: function (e) {
                e.success(samplesTableInit);
                //e.error("XHR response", "status code", "error message");
            },
        },
        schema: {
            model: {
                fields: $.extend({
                    row:                  { type: "number", editable: false },
                    barcodeId:            { type: "string",
                        defaultValue: function getdefaultbarcodeid(){var barcodeSet = $('#barcodeSet').val(); return barcodeSet? BARCODES[barcodeSet][0].id_str : "";},
                        isValueRequired: true
                    },
                    sampleName:           { type: "string", defaultValue: "" },
                    sampleDescription:    { type: "string", defaultValue: "" },
                    sampleExternalId:     { type: "string", defaultValue: "" },
                    controlType:          { type: "string", defaultValue: "" },
                    tubeLabel:            { type: "string", defaultValue: "", editable: !planOpt.isReserved },
                    chipBarcode:          { type: "string", defaultValue: "", editable: !planOpt.isReserved && !planOpt.isEditRun },
                    nucleotideType:       { type: "string",
                        defaultValue: planOpt.isDualNucleotideType ? "DNA" : $('[name=runType_nucleotideType]').val().toUpperCase(),
                        isValueRequired: true
                    },
                    reference:            { type: "string",
                        defaultValue: function getdefaultreference(){return $('#default_reference').val();},
                        notSupported: !planOpt.isReferenceSupported
                    },
                    targetRegionBedFile:  { type: "string",
                        defaultValue: function getdefaulttargetbedfile(){return $('#default_targetBedFile').val();},
                        notSupported: !planOpt.isTargetBEDFileSupported
                    },
                    hotSpotRegionBedFile: { type: "string",
                        defaultValue: function getdefaulthotspotbedfile(){ return $('#default_hotSpotBedFile').val();},
                        notSupported: !planOpt.isHotspotBEDFileSupported
                    },
                    controlSequenceType:  { type: "string", defaultValue: "" },

                }, getIonReporterFields()) // add IR fields
            },
            parse: function(data){
                // initialize data, this runs on dataSource.read()
                var model = this.model;
                $.each(data, function(i,item){
                    $.each(model.fields, function(key,opt){
                        // reset unsupported fields to blank
                        if (opt.notSupported) item[key] = "";
                        // fill in default values if key is undefined or value is required
                        if ((item[key] == undefined) || (opt.isValueRequired && !item[key])) {
                            item[key]= $.isFunction(opt.defaultValue) ? opt.defaultValue() : opt.defaultValue ;
                        }
                    });
                    // add row number
                    item["row"] = i;
                });
                return data;
            }
        }
    });

    var grid = $("#grid").kendoGrid({
        dataSource: dataSource,
        height: 350,
        scrollable: true,
        sortable: false,
        filterable: false,
        pageable: false,
        editable: true,
        navigatable: true,
        columns: [
            {
                field: "row", title: "#",
                width: '35px',
                template: '#=data.row+1#'
            },
            {
                field: "barcodeId", title: "Barcode",
                width: '220px',
                attributes: { "name": "barcodeId" },
                hidden: $('#chk_not_barcoded').is(':checked'),
                editor: barcodeEditor, 
                template: dropDnTemplate({'html': $('#barcodeColumnTemplate').html()})
            },
            {
                field: "sampleName", title: "Sample (required)",
                width: '200px',
                attributes: { "name": "sampleName" },
                editor: planOpt.isPlanBySample ? sampleForSamplesetEditor : "",
                template: planOpt.isPlanBySample? dropDnTemplate({'html':$('#sampleForSamplesetColumnTemplate').html()}) : "#=sampleName#",
            },
            {
                field: "_control_type", width: "22px",
                headerTemplate: columnSectionTemplate({'id':'controlTypeSectionTab','text':'Control Type'}),
                hidden: !$('#chk_barcoded').is(':checked'),
                editor: " ",
            },
            {
                field: "controlType", title: "Control Type",
                width: '150px',
                attributes: { "name": "controlType" },
                hidden: true,
                editor: controlTypeEditor,
                template: dropDnTemplate({'html': '#=controlType#'})
            },
            {
                field: "sampleExternalId", title: "Sample ID",
                width: '150px',
                attributes: { "name": "sampleExternalId" }
            },
            {
                field: "sampleDescription", title: "Sample Description",
                width: '200px',
                attributes: { "name": "sampleDescription" }
            },
            {
                field: "nucleotideType", title: planOpt.isDNAandFusions ? "DNA/Fusions" : "DNA/RNA",
                width: '100px',
                attributes: { "name": "nucleotideType" },
                hidden: !planOpt.isDualNucleotideType,
                editor: nucleotideTypeEditor,
                template: "#= (nucleotideType == 'RNA' && planOpt.isDNAandFusions) ? 'Fusions' : nucleotideType #"
            },
            {
                field: "_ref_details", width: "22px",
                headerTemplate: columnSectionTemplate({'id':'referenceSectionTab', 'text':'Reference'}),
                editor: " ",
            },
            {
                field: "reference", title: "Reference",
                width: '210px',
                attributes: { "name": "reference" },
                hidden: $('input[id=isSameRefInfoPerSample]').is(":checked"),
                editor: referenceEditor,
                template: dropDnTemplate({'html': $('#referenceColumnTemplate').html()})
            },
            {
                field: "targetRegionBedFile", title: "Target Regions",
                width: '210px',
                attributes: { "name": "targetRegionBedFile" },
                hidden: $('input[id=isSameRefInfoPerSample]').is(":checked"),
                editor: targetBEDfileEditor,
                template: dropDnTemplate({'html': '#=targetRegionBedFile.split("/").pop()#'})
            },
            {
                field: "hotSpotRegionBedFile", title: "Hotspot Regions",
                width: '210px',
                attributes: { "name": "hotSpotRegionBedFile" },
                hidden: $('input[id=isSameRefInfoPerSample]').is(":checked"),
                editor: hotspotBEDfileEditor,
                template: dropDnTemplate({'html': '#=hotSpotRegionBedFile.split("/").pop()#'})
            },
            {
                field: "tubeLabel", title: "Sample Tube Label",
                width: '150px',
                attributes: { "name": "tubeLabel" },
                hidden: $('#chk_barcoded').is(':checked')
            },
            {
                field: "chipBarcode", title: "Chip Barcode",
                width: '150px',
                attributes: { "name": "chipBarcode" },
                hidden: $('#chk_barcoded').is(':checked')
            },
            {
                field: "controlSequenceType", title: "Control Seq Type (optional)",
                width: '210px',
                attributes: { "name": "controlSequenceType" },
                hidden: !planOpt.isControlSeqTypeBySample,
                editor: controlSequenceTypeEditor,
                template: dropDnTemplate({'html': '#=controlSequenceType#'})
            },

        ].concat(getIonReporterColumns()), // add IR columns

        dataBound: function(e){
            //console.log('Grid dataBound', e);

            // disable/enable reference and bedfiles
            var isSameRefInfoPerSample = $("#isSameRefInfoPerSample").is(':checked');
            toggleSampleReferenceColumnEnablements(isSameRefInfoPerSample);

            // disable RNA rows for dual nuc type
            var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");
            if (planOpt.isDualNucleotideType && isSameSampleForDual)
                toggleRNASampleForDualNucleotideTypeEnablements(true);

            // disable sample parameters input for Plan-by-Sample
            if (planOpt.isPlanBySample){
                toggleDisableElements($('[name=sampleDescription]'), true);
                toggleDisableElements($('[name=sampleExternalId]'), true);
            }

            // display validation errors
            samplesTableValidationErrors = samplesTableValidationErrors.filter(function(obj){return obj.row < $('#grid').data('kendoGrid').dataSource.total() });
            $.each(samplesTableValidationErrors, function(i, obj){
                var tableCell = e.sender.tbody.find("tr:eq("+ obj.row +")>[name='"+ obj.field +"']");
                displayErrorInCell(tableCell, obj.error, obj.type);
            });

            $('.fillDown').hide(); //filldown btns refuse to disappear on Firefox
        },

        save:function(e){
            console.log('kendoGrid SAVE', e);
            // this event is fired if any data is changed via UI
            // e.values has the changed element
            var ds = this.dataSource;
            var field = Object.keys(e.values)[0];
            var rowIndex = e.container.parent().index();

            // copy fields for RNA row if same sample for dual nuc type
            var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");
            if (planOpt.isDualNucleotideType && isSameSampleForDual){
                if ((rowIndex < ds.total()-1) && (fieldsToUpdateForRNASameSample.indexOf(field) >= 0)){
                    ds.at(rowIndex+1).set(field, e.values[field]);
                    gridRefresh(this);
                }
            }

            // handle chip ID
            if (e.values.chipBarcode){
                e.model.set('chipBarcode', parse_chip_barcode(e.values.chipBarcode));
                e.preventDefault();
            }

            // refresh to display validation error
            var validationObj= $.grep(samplesTableValidationErrors, function(obj){return obj.row==rowIndex && obj.field==field })[0]
            if (validationObj){
                gridRefresh(this);
            }
        },

        edit:function(e){
            // activate dropdowns on single click
            var ddl = e.container.find('[data-role=dropdownlist]').data('kendoDropDownList');
            if(ddl){
                ddl.open();
                return;
            }
            // attach validation events as needed
            var field = e.container.attr('name');
            if (field in samplesTableValidators ){
                var input = e.container.find(".k-input:input");
                input.on("keyup", function(){
                    var error = samplesTableValidators[field](input.val());
                    displayErrorInCell(input.closest('td'),  error);
                    updateSamplesTableValidationErrors(e.model.row, field, input.val(), error);
                });
            }
        },
    });

    function barcodeEditor(container, options) {
        $('<input id="barcodeEditor" name="barcodeEditor" data-bind="value:' + options.field + '"/>')
            .appendTo(container)
            .kendoDropDownList({
                dataSource: BARCODES[$('#barcodeSet').val()] || [],
                dataTextField: "id_str",
                dataValueField: "id_str",
                template: '#=id_str# (#=sequence#)',
            });
    }

    function controlTypeEditor(container, options) {
        $('<input id="controlTypeEditor" name="controlTypeEditor" data-bind="value:' + options.field + '"/>')
            .appendTo(container)
            .kendoDropDownList({
                dataSource: controlTypes,
                dataTextField: "display",
                dataValueField: "value"
            });
    }
    
    function referenceEditor(container, options) {
        $('<input id="referenceEditor" name="referenceEditor" data-bind="value:' + options.field + '"/>')
            .appendTo(container)
            .kendoDropDownList({
                dataSource: references,
                dataTextField: "display",
                dataValueField: "short_name",
                optionLabel: "---",
                change: function(e){
                    options.model.set('targetRegionBedFile','');
                    options.model.set('hotSpotRegionBedFile','');
                    if (irWorkflowNotValid(options.model)){
                        options.model.set('irWorkflow', '');
                    }
                },
            });
    }

    function targetBEDfileEditor(container, options) {
        $('<input id="targetBEDfileEditor" name="targetBEDfileEditor" data-bind="value:' + options.field + '"/>')
            .appendTo(container)
            .kendoDropDownList({
                dataSource: targetBedFiles,
                dataTextField: "display",
                dataValueField: "file",
                optionLabel: "---",
                open: function(e) {
                    e.sender.dataSource.filter({
                        field: "reference", operator: "eq", value: options.model.reference
                    });
                },
            });
    }

    function hotspotBEDfileEditor(container, options) {
        $('<input id="hotspotBEDfileEditor" name="hotspotBEDfileEditor" data-bind="value:' + options.field + '"/>')
            .appendTo(container)
            .kendoDropDownList({
                dataSource: hotSpotBedFiles,
                dataTextField: "display",
                dataValueField: "file",
                optionLabel: "---",
                open: function(e) {
                    e.sender.dataSource.filter({
                        field: "reference", operator: "eq", value: options.model.reference
                    });
                },
            });
    }

    function nucleotideTypeEditor(container, options) {
        $('<input id="nucleotideTypeEditor" name="nucleotideTypeEditor" data-bind="value:' + options.field + '"/>')
            .appendTo(container)
            .kendoDropDownList({
                dataSource: [{"nuc":"DNA", "display":"DNA"}, {"nuc":"RNA", "display": planOpt.isDNAandFusions ? "Fusions" : "RNA"}],
                dataTextField: "display",
                dataValueField: "nuc",
                change: function(e){
                    var refInfo = getDefaultReferenceInfo();
                    if (options.model.nucleotideType == "DNA"){
                        options.model.set('reference', refInfo.default_reference);
                        options.model.set('targetRegionBedFile', refInfo.default_targetBedFile);
                        options.model.set('hotSpotRegionBedFile', refInfo.default_hotSpotBedFile);
                    } else {
                        options.model.set('reference', refInfo.mixedTypeRNA_reference);
                        options.model.set('targetRegionBedFile', refInfo.mixedTypeRNA_targetBedFile);
                        options.model.set('hotSpotRegionBedFile', "");
                    }
                    if (irWorkflowNotValid(options.model)){
                        options.model.set('irWorkflow', '');
                    }
                    if (irSetIdNotValid(options.model)){
                        options.model.set('irSetID', '');
                    }
                }
            });
    }

    function sampleForSamplesetEditor(container, options) {
        $('<input id="sampleNameEditor" name="sampleNameEditor" data-bind="value:' + options.field + '"/>')
            .appendTo(container)
            .kendoDropDownList({
                dataSource: SAMPLESETITEMS || [],
                dataTextField: "display",
                dataValueField: "sampleName",
                open: function(e){
                    // save next grid row to update for RNA/DNA plans
                    this.nextGridItem = $("#grid").data("kendoGrid").dataItem(this.element.closest("tr").next());
                },
                change: function(e){
                    var samplesetItem = this.dataItem();
                    options.model.set('sampleExternalId', samplesetItem.externalId);
                    options.model.set('sampleDescription', samplesetItem.description);
                    options.model.set('controlType', samplesetItem.controlType);

                    options.model.set('ircancerType', samplesetItem.ircancerType);
                    options.model.set('ircellularityPct', samplesetItem.ircellularityPct);
                    options.model.set('irbiopsyDays', samplesetItem.irbiopsyDays);
                    options.model.set('ircoupleID', samplesetItem.ircoupleID);
                    options.model.set('irembryoID', samplesetItem.irembryoID);

                    // update fields for RNA row if same sample for dual nuc type
                    var nextGridItem = this.nextGridItem;
                    var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");
                    if (planOpt.isDualNucleotideType && isSameSampleForDual && nextGridItem){
                        nextGridItem.set('sampleExternalId', samplesetItem.externalId);
                        nextGridItem.set('sampleDescription', samplesetItem.description);
                        nextGridItem.set('controlType', samplesetItem.controlType);

                        nextGridItem.set('ircancerType', samplesetItem.ircancerType);
                        nextGridItem.set('ircellularityPct', samplesetItem.ircellularityPct);
                        nextGridItem.set('irbiopsyDays', samplesetItem.irbiopsyDays);
                        nextGridItem.set('ircoupleID', samplesetItem.ircoupleID);
                        nextGridItem.set('irembryoID', samplesetItem.irembryoID);
                    }
                }
            });
    }
    
    function controlSequenceTypeEditor(container, options) {
        $('<input id="controlSequenceTypeEditor" name="controlSequenceTypeEditor" data-bind="value:' + options.field + '"/>')
            .appendTo(container)
            .kendoDropDownList({
                dataSource: controlSeqTypes,
                dataTextField: "display",
                dataValueField: "name",
                optionLabel: "---"
            });
    }
    
    function parse_chip_barcode(_chip_barcode){
            var chipID = _chip_barcode;
            if (_chip_barcode.substring(0,2) == "21")
            {
                 var sub_barcode = _chip_barcode.substring(2);
                 var reverse_subbarcode = sub_barcode.split("").reverse().join("");
                 var n1 = reverse_subbarcode.search("19");
                 var n2 = reverse_subbarcode.search("142");
                 if ((n1 != -1) && ((n2 == -1) || (n1 < n2)))
                 {
                     chipID = sub_barcode.substring(0,sub_barcode.length-n1-("91").length);
                 }
                 else if ((n2 != -1) && ((n1 == -1) || (n2 < n1)))
                 {
                     chipID = sub_barcode.substring(0,sub_barcode.length-n2-("241").length);
                 }
            }
            return chipID;
    }
    $("#chipBarcodeLabel").on('change', function(){
        $(this).val(parse_chip_barcode($(this).val()));
    });

    $("#barcodeSampleTubeLabel").on('keyup', function () {
        var error = validate_sampleTubeLabel($(this).val());
        $("#tubeerror").html(error);
    });
    
    /**
     Click event handler to fill the sample names
     */
    $('#fillSampleNames').click(function () {
        var samplesTableJSON = $("#grid").data("kendoGrid").dataSource.data().toJSON();
        var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");
        var sampleNameIndex = 0;
        $.each(samplesTableJSON, function(i){
            if (!planOpt.isDualNucleotideType || !isSameSampleForDual || isEven(i)) sampleNameIndex++;
            this.sampleName = 'Sample ' + sampleNameIndex;
        });
        $("#grid").data("kendoGrid").dataSource.data(samplesTableJSON);
    });

    /**
     Click event handler to clear the sample names
     */
    $('#clearSampleNames').click(function () {
        var samplesTableJSON = $("#grid").data("kendoGrid").dataSource.data().toJSON();
        $.each(samplesTableJSON, function(){
            this.sampleName = "";
        });
        $("#grid").data("kendoGrid").dataSource.data(samplesTableJSON);
    });

    /**
     *   Disallow non-integer user-input
     */
    $(".integer").live('keydown', function (event) {
        /* restrict user input for integer fields */
        if (event.shiftKey)
            if (event.keyCode != 9)
                event.preventDefault();
        if (event.keyCode == 46 || event.keyCode == 8 || event.keyCode == 9) {
        }
        else {
            if (event.keyCode < 95) {
                if (event.keyCode < 48 || event.keyCode > 57) {
                    event.preventDefault();
                }
            } else {
                if (event.keyCode < 96 || event.keyCode > 105) {
                    event.preventDefault();
                }
            }
        }
    });


    /**
     Click event handler for the refresh button which clears the IR fields,
     and re-calls the API to re-populate the IR fields
     */
    $(".refresh-uploader-information").on('click', function () {
        $("#loading").show();
        load_and_set_ir_fields();
        return false;
    });

    //making the refresh icon have a hand when the mouse hovers over it
    $(".refresh-uploader-information").css('cursor', 'pointer');


    /**
     Barcoded and Non-barcoded tables create/update/show
     */
    $("input[name=is_barcoded]").click(function () {
        if (this.id == 'chk_barcoded') {
            $('.barcoded').show();
            $('.nonbarcoded').hide();
            $('#barcodeSet').prop('selectedIndex', 1);
            $('#barcodeSet').change();
            $('#numRowsLabel').text("Number of barcodes");
        } else {
            $('.barcoded').hide();
            $('.nonbarcoded').show();
            $('#barcodeSet').val('');
            $('#numRowsLabel').text("Number of chips");
        }
        $('#numRows').change();
    });


    $('#barcodeSet').change(function () {
        var grid = $('#grid').data('kendoGrid');
        var barcodes = BARCODES[this.value];
        if (barcodes) {
            // barcoded
            $('input[id=chk_barcoded]').attr("checked", true);
            $('input[id=chk_not_barcoded]').attr("checked", false);

            $(".barcoded").show();
            $(".nonbarcoded").hide();
            $('#numRowsLabel').text("Number of barcodes");
            
            grid.showColumn('barcodeId');
            grid.hideColumn('tubeLabel');
            grid.hideColumn('chipBarcode');

            var num_barcodes = barcodes.length;
            // if more existing rows than number of barcodes need to update numRows and table
            if (grid.dataSource.total() > num_barcodes) {
                $('#numRows').val(num_barcodes);
                $('#numRows').change();
            }

            // update samples table
            var samplesTableJSON = grid.dataSource.data().toJSON();
            $.each(samplesTableJSON, function(i){
                this.barcodeId = barcodes[i].id_str;
            });
            grid.dataSource.data(samplesTableJSON);

        } else {
            // non-barcoded
            $('input[id=chk_barcoded]').attr("checked", false);
            $('input[id=chk_not_barcoded]').attr("checked", true);

            temp = $(this)
            $(".barcoded").hide();
            $(".nonbarcoded").show();
            $('#numRowsLabel').text("Number of chips");

            grid.hideColumn('barcodeId');
            grid.showColumn('tubeLabel');
            grid.showColumn('chipBarcode');
        }
    });


    $('#numRows').change(function () {
        var samplesTableJSON = $("#grid").data("kendoGrid").dataSource.data().toJSON();
        var nrows = samplesTableJSON.length;
        var nrows_new = parseInt(this.value);
        var barcoded = $('input[id=chk_barcoded]').is(':checked');

        if ((nrows_new == '0') || (isNaN(nrows_new)) || (typeof(nrows_new) === 'undefined')){
            //Do not allow to enter 0 samples - Value Error TS-12548
            $('#numRows').val(1);
            nrows_new = 1;
        }
        if (barcoded) {
            // limit to number of barcodes in set
            var selected_barcodes = BARCODES[$('#barcodeSet').val()];
            this.value = nrows_new = (nrows_new > selected_barcodes.length) ? selected_barcodes.length : nrows_new;
        }

        if (nrows_new > nrows) {
            var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");
            var sampleNameIndex = (planOpt.isDualNucleotideType && isSameSampleForDual) ? Math.ceil(nrows/2) : nrows;
            var refInfo = getDefaultReferenceInfo();
            
            for(var i=nrows; i<nrows_new; i++){
                if (!planOpt.isDualNucleotideType || isEven(i)){
                    sampleNameIndex++;
                    var row = {
                        'sampleName': "Sample " + sampleNameIndex,
                        'reference' : refInfo.default_reference,
                        'targetRegionBedFile' : refInfo.default_targetBedFile,
                        'hotSpotRegionBedFile': refInfo.default_hotSpotBedFile
                    };
                    if (planOpt.isDualNucleotideType) row['nucleotideType'] = "DNA";
                    
                } else {
                    // dual nuc type RNA row
                    if (!isSameSampleForDual) sampleNameIndex++;

                    var row = {
                        'sampleName': "Sample " + sampleNameIndex ,
                        'reference' : refInfo.mixedTypeRNA_reference,
                        'targetRegionBedFile' : refInfo.mixedTypeRNA_targetBedFile,
                        'hotSpotRegionBedFile': "",
                        'nucleotideType':       "RNA"
                    };
                }

                if (barcoded) row['barcodeId'] = selected_barcodes[i].id_str;

                // Ion Reporter fields
                if (USERINPUT.is_ir_connected) {
                    if (planOpt.isDualNucleotideType && isSameSampleForDual && row['nucleotideType'] == "RNA"){
                        var prevRow = samplesTableJSON[i-1];
                        row['irWorkflow'] = prevRow['irWorkflow']
                        row['irRelationRole'] = prevRow['irRelationRole']
                        row['irSetID'] = prevRow['irSetID'];
                    } else {
                        var workflowObj = getWorkflowObj(USERINPUT.workflow, USERINPUT.tag_isFactoryProvidedWorkflow);
                        row['irWorkflow'] = workflowObj.Workflow;
                        row['irRelationRole'] = workflowObj.relations_list.length == 1 ? workflowObj.relations_list[0] : "";
                        row['irSetID'] = generate_set_id(workflowObj, row, samplesTableJSON);
                    }
                }

                samplesTableJSON.push(row);
            }
        } else {
            samplesTableJSON = samplesTableJSON.slice(0, nrows_new);
        }

        // update via local data to run dataSource initialization
        samplesTableInit = samplesTableJSON;
        $("#grid").data("kendoGrid").dataSource.read();
    });

    
    /**
     Checkbox for same sample in a DNA + RNA plan is clicked
     */
    $("input[name=isOncoSameSample]").click(function () {
        handleSameSampleForDualNucleotideType();
    });


    /**
        Event handlers to show/hide grid columns
    */
    // show/hide control type column
    $('#controlTypeSectionTab').click(function (){
        var makeVisible = !$(this).hasClass('k-state-active');
        $(this).toggleClass('k-state-active', makeVisible);
        var grid = $("#grid").data("kendoGrid");
        makeVisible ? grid.showColumn('controlType') : grid.hideColumn('controlType');
    });
    
    
    $('#referenceSectionTab').click(function (){
        var makeVisible = !$(this).hasClass('k-state-active');
        showHideSampleReferenceColumns(makeVisible);
    });
    
    $("input[name=isOnco_Pgs]").click(function () {
        var grid = $("#grid").data("kendoGrid");
        var selectedAnnotation = $("input[name=isOnco_Pgs]:checked").val();
        if (selectedAnnotation == 'Oncology') {
            grid.showColumn('ircancerType');
            grid.showColumn('ircellularityPct');
            grid.hideColumn('irbiopsyDays');
            grid.hideColumn('ircoupleID');
            grid.hideColumn('irembryoID');
        } else if (selectedAnnotation == 'Pgs') {
            grid.hideColumn('ircancerType');
            grid.hideColumn('ircellularityPct');
            grid.showColumn('irbiopsyDays');
            grid.showColumn('ircoupleID');
            grid.showColumn('irembryoID');
        }
        grid.showColumn('_annotations');
        $('#annotationsSectionTab').toggleClass('k-state-active', true);
    });

    // show/hide the sample annotation columns
    $('#annotationsSectionTab').click(function (){
        var makeVisible = !$(this).hasClass('k-state-active');
        var selectedAnnotation = $("input[name=isOnco_Pgs]:checked").val();
        var annotation_columns = [];
        if (selectedAnnotation == 'Oncology') {
            annotation_columns = ['ircancerType','ircellularityPct'];
        } else if (selectedAnnotation == 'Pgs') {
            annotation_columns = ['irbiopsyDays','ircoupleID','irembryoID'];
        }
        var grid = $("#grid").data("kendoGrid");
        $.each(annotation_columns, function(i, column){
            makeVisible ? grid.showColumn(column) : grid.hideColumn(column);
        });
        $('#annotationsSectionTab').toggleClass('k-state-active', makeVisible);
    });

    /**
     *  Enable a fill down functionality on the sample/barcode grid.
     */
    var fillDownElements = [
        {name: "irWorkflow",      action: "copy",
            updateRelated: [{'field': 'irRelationRole', 'value': defaultRelation },
                            {'field': 'irSetID', 'value': ''} ]},
        {name: "irRelationRole",  action: "copy",
            updateRelated: [{'field': 'irGender', 'value': '' } ]},
        {name: "irGender",        action: "copy"},
        {name: "irSetID",         action: "increment"},
        {name: "ircancerType",    action: "copy"},
        {name: "ircellularityPct",action: "copy"},
        {name: "irbiopsyDays",    action: "copy"},
        {name: "ircoupleID",      action: "copy"},
        {name: "irembryoID",      action: "copy"},
    ];

    //Only enable certain fields if not OCP planning.
    if ($('input[id=isDualNucleotideTypeBySample]').val() != "True") {
        fillDownElements = fillDownElements.concat([
            {name: "reference",           action: "copy",
                updateRelated: [{'field': 'targetRegionBedFile', 'value': '' },
                                {'field': 'hotSpotRegionBedFile', 'value': '' } ]},
            {name: "targetRegionBedFile", action: "copy"},
            {name: "hotSpotRegionBedFile",action: "copy"}
        ]);
    }

    var tableContainer = $("#grid").css("position", "relative")

    $.each(fillDownElements, function (_, options) {
        var elementSelector = "[name=" + options.name + "]:first";
        var action = options.action;
        var name = options.name;

        var fillDownButton = $("<div class='fillDown btn btn-primary btn-mini'><i class='icon-circle-arrow-down icon-white'></i></div>")
            .css("border-radius", 20)
            .css("border-top-right-radius", 0)
            .css("border-bottom-right-radius", 0)
            .css("position", "absolute")
            .css("line-height", "1px")
            .appendTo("#grid");
        fillDownButton.tooltip({title: "Copy value to all rows."});
        fillDownButton.mousedown(function (e) {
            var grid = $("#grid").data("kendoGrid");
            var samplesTableJSON = grid.dataSource.data().toJSON();
            var val = grid.current().find("[name^="+name+"]").val();
            var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");

            $.each(samplesTableJSON, function(i, row){
                if (i==0){
                    if (val === undefined){
                        val = row[name];
                    }
                } else {
                    if (action == "increment" && !isNaN(val)) {
                        if (planOpt.isDualNucleotideType && isSameSampleForDual && row.nucleotideType == 'RNA'){
                            val = parseInt(val);
                        } else {
                            val = parseInt(val) + 1;
                        }
                    }
                    // update related fields
                    if (options.updateRelated){
                        $.each(options.updateRelated, function(i, update){
                            row[update.field]= $.isFunction(update.value) ? update.value(val) : update.value;
                        });
                    }
                }
                row[name] = val;
            });
            grid.dataSource.data(samplesTableJSON);
        });

        tableContainer.on('focus', elementSelector,
            function () {
                fillDownButton.css("top", $(elementSelector).position().top + 53);
                fillDownButton.css("left", $(elementSelector).position().left - 27);
                fillDownButton.show();
            }
        );

        tableContainer.on('blur', elementSelector,
            function () {
                fillDownButton.hide();
            }
        );
    });


    $('#saveTable').click(function(e){
        e.preventDefault();
        updateSamplesTable();
        var form = $('<form id="saveTable_form" method="POST" action="' + $(this).attr("href") + '">');
        $("#samplesTable").clone().appendTo(form);
        $(document.body).append(form); // need for IE9
        form.submit();
        form.remove();
    });
    
    $('#modal_load_samples_table .csv_load.btn').click(function(e){
        // load selected csv file and update samples table
        e.preventDefault();
        $('#modal_error').empty().hide();

        var filename = $('#modal_load_samples_table :file').val();
        if (!filename){
            $('#modal_error').empty().append('Please select a CSV file').show();
            return false;
        }

        var url = $(this).attr("href");
        var form = $('#step_form').attr('action', url);
        form.append(
            $('<input>', { name: 'irSelected', val: USERINPUT.is_ir_connected ? "1":"0" })
        );
        form.ajaxSubmit({
            dataType : "json",
            url: url,
            async: false,
            beforeSend: function() {
                console.log('submitting', form)
            },
            success: function(data) {
                console.log('success', data)
                //var table = data.samplesTable;
                //$('#numRows').val(table.length).change();
                if ($('#isSameRefInfoPerSample').is(':checked') && !data.same_ref_and_bedfiles){
                    $('#isSameRefInfoPerSample').prop('checked', false);
                    toggleSampleReferenceColumnEnablements(false);
                    showHideSampleReferenceColumns(true);
                }
                // update via local data to run dataSource initialization
                samplesTableInit = data.samplesTable;
                $("#grid").data("kendoGrid").dataSource.read();
                
                // validate IR values
                if (USERINPUT.is_ir_connected)
                    check_selected_values();

                $('#modal_load_samples_table').modal('hide');
                $('.modal-backdrop').remove();
            },
            error: function(data){
                $('#modal_error').empty().append(data.responseText).show();
            }
        });
        return false;
    });


    // ******************* Initializion on page load ************************* //

    // update reference info
    if ($('#isSameRefInfoPerSample').is(':checked')){
        updateSampleReferenceColumnsWithDefaults();
    }
    
    // new plans may initialize with different default number of rows
    var numRowValue = parseInt($('#numRows').val());
    var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");
    if (numRowValue != samplesTableInit.length){
        $('#numRows').change();
        if (planOpt.isDualNucleotideType && isSameSampleForDual){
            handleSameSampleForDualNucleotideType();
        }
    }
});


function gridRefresh(grid){
    // refreshes the grid and returns to scroll position
    var scroll = [grid.content.scrollTop(), grid.content.scrollLeft()];
    grid.refresh();
    setTimeout(function () {
        grid.content.scrollTop(scroll[0]);
        grid.content.scrollLeft(scroll[1]);
    }, 200);
}

// ********************** Validation functions **************************** //

var samplesTableValidators = {
    "sampleName": validate_sampleName,
    "tubeLabel":  validate_sampleTubeLabel,
    "ircoupleID": validate_ircoupleID,
    "irembryoID": validate_irembryoID
}

function updateSamplesTableValidationErrors(row, field, value, error){
    var validationObj = samplesTableValidationErrors.filter(function(obj){ return obj.row == row && obj.field == field } )[0];
    if (error){
        if (validationObj){
            validationObj['error'] = error;
            validationObj['value'] = value;
            validationObj['type'] = 'error';
        } else {
            samplesTableValidationErrors.push({
                'row': row,
                'field': field,
                'value': value,
                'error': error,
                'type': 'error',
            });
        }
    } else {
        if (validationObj)
            samplesTableValidationErrors.splice(samplesTableValidationErrors.indexOf(validationObj), 1);
    }
}

function displayErrorInCell(cell, error, type){
    cell.removeClass('alert alert-error alert-warning');
    cell.find('.field-validation-error').remove();
    cell.off('mouseenter mouseleave keyup');

    if (error){
        var validationErrorTemplate = kendo.template($('#validationErrorTemplate').html());
        $(validationErrorTemplate({ message: error })).appendTo(cell);
        if (type == 'warning'){
            cell.addClass('alert alert-warning');
        } else {
            cell.addClass('alert alert-error');
        }
        cell.on('mouseenter keyup', function(){
            var $el = $(this).find('.field-validation-error');
            // position element to make visible for 2 last columns and 2 last rows
            if (cell.is(':last-child') || cell.is(':nth-last-child(2)')){
                $el.css('margin-left', -$el.width());
            }
            if ($(this).closest('tr').position().top > 200){
                $el.css('margin-top', -($(this).height() + $el.height()));
            } else {
                $el.css('margin-top', '5.95px');
            }
            $el.show();
        });
        cell.on('mouseleave', function(){ $(this).find('.field-validation-error').hide() })
    }
}

function validate_sampleName(value) {
    var error = "";
    //call the Regex test function identified in validation.js file
    if (!is_valid_chars(value)) {
        error = 'Error, Sample name should contain only numbers, letters, spaces, and the following: . - _';
    }
    //call the check max length function that's in validation.js
    if (!is_valid_length(value, 127)) {
        error = 'Error, Sample name length should be 127 characters maximum';
    }
    if (!is_valid_leading_chars(value)) {
        error = 'Sample name cannot begin with (.) or (_) or (-)';
    }
    return error;
}

function validate_sampleTubeLabel(value){
    var error = "";
    //call the Regex test function identified in validation.js file
    if (!is_valid_chars(value)) {
        error = 'Error, Sample tube label should contain only numbers, letters, spaces, and the following: . - _';
    }
    //call the check max length function that's in validation.js
    if (!is_valid_length(value, 512)) {
        error = 'Error, Sample tube label length should be 512 characters maximum';
    }
    return error;
}

function validate_ircoupleID(value) {
    var error = "";
    //call the Regex test function identified in validation.js file
    if (!is_valid_chars(value)){
        error = 'Error, Couple ID should contain only numbers, letters, spaces, and the following: . - _';
    }
    //call the check max length function that's in validation.js
    if (!is_valid_length(value, 127)) {
        error = 'Error, Couple ID length should be 127 characters maximum';
    }
    return error;
}
    
function validate_irembryoID(value) {
    var error = "";
    //call the Regex test function identified in validation.js file
    if (!is_valid_chars(value)){
        error = 'Error, Embryo ID should contain only numbers, letters, spaces, and the following: . - _';
    }
    //call the check max length function that's in validation.js
    if (!is_valid_length(value, 127)) {
        error = 'Error, Embryo ID length should be 127 characters maximum';
    }
    return error;
}
