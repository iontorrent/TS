

/**
 NOTE:  IRU and TS terminology differs slightly.  In IRU, the column "Relation" is equivalent to TS' "RelationRole",
 and IRU's column RelationshipType is TS' "Relation" in the JSON blob that is saved to the selectedPlugins BLOB.
 */
var USERINPUT = USERINPUT || {};
USERINPUT.user_input_url = "/rundb/api/v1/plugin/IonReporterUploader/extend/userInput/";
USERINPUT.workflows = [];
USERINPUT.relations = [];
USERINPUT.genders = [];
USERINPUT.cancerTypes = [];
USERINPUT.populations = [];
USERINPUT.mouseStrains = [];
USERINPUT.relations_with_gender = {};
var SEPARATOR = " | ";
var ION_PRELOADED_LABEL = "Ion Torrent";

function getIonReporterFields(){
    if (!USERINPUT.account_id || USERINPUT.account_id == "0" || USERINPUT.account_id == "-1"){
        return [];
    }
    var iru_fields = {
        // Oncology
        ircancerType:   { type: "string", defaultValue: "",
            editable:function(){return USERINPUT.is_ir_connected}
        },
        ircellularityPct: { type: "number" },
        // PGx
        irbiopsyDays:   { type: "number" },
        ircellNum:     { type: "string", defaultValue: ""},
        ircoupleID:     { type: "string", defaultValue: ""},
        irembryoID:     { type: "string", defaultValue: ""},
        // IR Workflow
        irWorkflow:     { type: "array", defaultValue: USERINPUT.workflow,
            editable:function(){return USERINPUT.is_ir_connected}
        },
        irtag_isFactoryProvidedWorkflow: { type: "bool", defaultValue: USERINPUT.tag_isFactoryProvidedWorkflow },
        irRelationRole: { type: "string",
            defaultValue: function getdefaultrelation(){return defaultRelation(USERINPUT.workflow, USERINPUT.tag_isFactoryProvidedWorkflow)},
            isValueRequired: true,
            editable: function(){return USERINPUT.is_ir_connected}
        },
        irGender:       { type: "string", defaultValue: "",
            editable: function(){return USERINPUT.is_ir_connected}
        },
        irPopulation:       { type: "string", defaultValue: "",
            editable: function(){return USERINPUT.is_ir_connected}
        },
        irmouseStrains:       { type: "string", defaultValue: "",
            editable: function(){return USERINPUT.is_ir_connected}
        },
        irSampleCollectionDate:       { type: "date", defaultValue: "",
            editable: function(){return USERINPUT.is_ir_connected}
        },
        irSampleReceiptDate:       { type: "date", defaultValue: "",
            editable: function(){return USERINPUT.is_ir_connected}
        },
        // 16S only
        irBacterialMarkerType:      { type: "string", defaultValue: ""},
        // ReproSeq only
        irWitness:      { type: "string", defaultValue: ""},
        irSetID:        { type: "number" }
    }
    return iru_fields;
}

function getIonReporterColumns(){
    if (!USERINPUT.account_id || USERINPUT.account_id == "0" || USERINPUT.account_id == "-1"){
        return [];
    }
    var iru_columns = [
        {
            field: "_annotations", width: "22px",
            headerTemplate: columnSectionTemplate({'id':'annotationsSectionTab', 'text': 'Annotations'}),
            hidden: !$('#isOnco').is(':checked') && !$('#isPgs').is(':checked'),
            editor: " ",
        },
        {
            field: "irSampleCollectionDate", title: "Sample Collection Date",
            width: '150px',
            attributes: { "name": "irSampleCollectionDate", "class": "date" },
            editor: irSampleCollectionDateEditor,
            format: "{0:yyyy-MM-dd}",
        },
        {
            field: "irSampleReceiptDate", title: "Sample Receipt Date",
            width: '150px',
            attributes: { "name": "irSampleReceiptDate", "class": "date" },
            editor: irSampleReceiptDateEditor,
            format: "{0:yyyy-MM-dd}",
            headerAttributes: { "rel": "tooltip", "data-original-title": "Sample Receipt date must be later than or the same as the sample collection date"},
            headerTemplate: '<i class="icon-info-sign"></i> Sample Receipt Date'
        },
        // Oncology
        {
            field: "ircancerType", title: "Cancer Type",
            width: '150px',
            attributes: { "name": "ircancerType" },
            hidden: !$('#isOnco').is(':checked'),
            editor: irCancerTypeEditor,
            template: dropDnTemplate({'html': '#=ircancerType#'})
        },
        {
            field: "ircellularityPct", title: "Cellularity %",
            width: '150px',
            attributes: { "name": "ircellularityPct", "class": "integer" },
            hidden: !$('#isOnco').is(':checked'),
            editor: ircellularityPctEditor,
        },
        // PGx
        {
            field: "irbiopsyDays", title: "Biopsy Days",
            width: '100px',
            attributes: { "name": "irbiopsyDays", "class": "integer" },
            hidden: !$('#isPgs').is(':checked')
        },
        {
            field: "ircellNum", title: "Cell Number",
            width: '100px',
            attributes: { "name": "ircellNum" },
            hidden: !$('#isPgs').is(':checked')
        },
        {
            field: "ircoupleID", title: "Couple ID",
            width: '100px',
            attributes: { "name": "ircoupleID" },
            hidden: !$('#isPgs').is(':checked')
        },
        {
            field: "irembryoID", title: "Embryo ID",
            width: '100px',
            attributes: { "name": "irembryoID" },
            hidden: !$('#isPgs').is(':checked')
        },

        // IR Workflow
    ];
    var irWorkflowField;
    var headerText = 'Ion Reporter Workflow <label class="checkbox inline" style="font:inherit;" ' +
                'title="Show all available or show filtered IR workflows">' +
                '<input id="irWorkflowShowAll" type="checkbox" style="margin:0;">Show All Workflows</label>'
    if (!($('[name=planCategories]').val() === 'onco_immune;immunology')){
        irWorkflowField = {
            field: "irWorkflow", title: "Ion Reporter Workflow",
            width: '350px',
            attributes: {"name": "irWorkflow"},
            editor: irWorkflowEditor,
            headerTemplate: headerText,
            template: dropDnTemplate({'html': $('#irWorkflowColumnTemplate').html()}),
            hidden: ($('[name=planCategories]').val() === 'onco_immune;immunology')
        };
    } else {
        irWorkflowField = {
            field: "irWorkflow", title: "Ion Reporter Workflow",
            width: '350px',
            attributes: { "name": "irWorkflow" },
            editor: irMultiWorkflowEditor,
            headerTemplate: headerText,
            template: function (item) {
                 var irMultipleWorkflowSelected = [];
                 var selectedworkflowObj;
                 if ('irMultipleWorkflowSelected' in item) {
                     selectedworkflowObj = JSON.parse(JSON.stringify(item.irMultipleWorkflowSelected));
                 }
                 if(selectedworkflowObj) {
                     $.each(selectedworkflowObj, function (key, value) {
                         irMultipleWorkflowSelected.push(value['workflow']);
                     });
                 }
                 if(irMultipleWorkflowSelected && Array.isArray(irMultipleWorkflowSelected) && (irMultipleWorkflowSelected.length > 1)) {
                    return irMultipleWorkflowSelected.length + " IR workflows selected";
                 } else {
                    var workflowObj = getWorkflowObj(item.irWorkflow, item.irtag_isFactoryProvidedWorkflow);
                    var workflow = workflowObj ? workflowObj.display : "";
                    return workflow;
                }
            }
        };
    }
    iru_columns.push(irWorkflowField);
    iru_columns.push({
            field: "irRelationRole", title: "Relation",
            width: '100px',
            attributes: { "name": "irRelationRole" },
            editor: irRelationEditor,
            template: dropDnTemplate({'html': '#=irRelationRole#'})
        },
        {
            field: "irGender", title: "Gender",
            width: '100px',
            attributes: { "name": "irGender" },
            editor: irGenderEditor,
            template: dropDnTemplate({'html': '#=irGender#'})
        },
        {
            field: "irPopulation", title: "Population",
            width: '100px',
            attributes: { "name": "irPopulation" },
            editor: irPopulationEditor,
            template: dropDnTemplate({'html': '#=irPopulation#'})
        },
        {
            field: "irmouseStrains", title: "Mouse Strains",
            width: '100px',
            attributes: { "name": "irmouseStrains" },
            editor: irmouseStrainsEditor,
            template: dropDnTemplate({'html': '#=irmouseStrains#'})
        },
        // 16S
        {
            field: "irBacterialMarkerType", title: "Bacterial Marker Type",
            width: '100px',
            attributes: { "name": "irBacterialMarkerType" },
            hidden: planCategories.indexOf("16s") < 0,
            editor: irBacterialMarkerTypeEditor,
            template: dropDnTemplate({'html': '#=irBacterialMarkerType#'})
        },
        // ReproSeq
        {
            field: "irWitness", title: "Witness",
            width: '100px',
            attributes: { "name": "irWitness" },
            hidden: planCategories.indexOf("repro") < 0,
        },
        {
            field: "irSetID", title: "IR Set ID",
            width: '60px',
            attributes: { "name": "irSetID" },
            headerAttributes: { "rel": "tooltip", "data-original-title": "After file transfer, in Ion Reporter Software, samples with the same Set ID are considered related samples and are launched in the same analysis, such as a normal sample and its corresponding tumor samples. Do not give unrelated samples the same Set ID value (even if that value is zero or blank)."},
            headerTemplate: '<i class="icon-info-sign"></i> IR Set ID'
        });

    return iru_columns;
}


function getWorkflowObj(workflow, tag_isFactoryProvidedWorkflow) {
    if (!workflow || workflow == "Upload Only") workflow = "";
    var match = $.grep(USERINPUT.workflows, function (obj) {
        return obj.Workflow == workflow && obj.tag_isFactoryProvidedWorkflow == tag_isFactoryProvidedWorkflow
    });
    if (match.length == 0) {
        match = $.grep(USERINPUT.workflows, function (obj) {
            return obj.Workflow == workflow
        });
    }
    //console.log('getWorkflowObj', workflow, tag_isFactoryProvidedWorkflow, match);
    return (match.length > 0) ? match[0] : null;
}
function onSelect(e) {
    var dataItem = this.dataItem(e.node);
    console.log("Selected node with id=" + dataItem.id);
}
function irWorkflowEditor(container, options) {
    $('<input id="irWorkflowEditor" name="irWorkflowEditor" data-bind="value:' + options.field + '"/>')
        .appendTo(container)
        .kendoDropDownList({
            dataSource: USERINPUT.workflows,
            dataTextField: "display",
            dataValueField: "Workflow",
            open: function (e) {
                var filters = [];
                var reference = options.model.get('reference');
                if (reference) {
                    filters.push({
                        logic: "or",
                        filters: [
                            {field: "Reference", value: reference},
                            {field: "Reference", value: ""},
                        ]
                    });
                }
                var nucleotideType = options.model.get('nucleotideType');
                if (nucleotideType) {
                    filters.push({
                        logic: "or",
                        filters: [
                            {field: "nucleotideType", value: nucleotideType},
                            {field: "nucleotideType", value: ""},
                        ]
                    });
                }
                if (!$('#irWorkflowShowAll').is(':checked')) {
                    e.sender.dataSource.filter({
                        logic: "and",
                        filters: filters
                    });
                }
                // save next grid row to update for RNA/DNA plans
                this.nextGridItem = $("#grid").data("kendoGrid").dataItem(this.element.closest("tr").next());
            },
            change: function (e) {
                var workflowObj = this.dataItem();
                var relation = workflowObj.relations_list.length == 1 ? workflowObj.relations_list[0] : "";
                options.model.set('irtag_isFactoryProvidedWorkflow', workflowObj.tag_isFactoryProvidedWorkflow);
                options.model.set('irRelationRole', relation);
                if (hasGender(relation)) options.model.set('irGender', '');
                // fill in irSetID value
                var data = $("#grid").data("kendoGrid").dataSource.data().toJSON();
                data.splice(options.model.row, 1); // get rows other than this one
                var setid = generate_set_id(workflowObj, options.model, data);
                options.model.set('irSetID', setid);
                // update fields for RNA row if same sample for dual nuc type
                var nextGridItem = this.nextGridItem;
                var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");
                if (planOpt.isDualNucleotideType && isSameSampleForDual && nextGridItem) {
                    nextGridItem.set('irRelationRole', relation);
                    if (hasGender(relation)) nextGridItem.set('irGender', '');
                    nextGridItem.set('irSetID', setid);
                }
                // change IR validation errors to warnings
                updateIRvalidationErrors(options.model.row, ['irWorkflow', 'irRelationRole', 'irGender', 'irSetID']);
            },
        });
}

function irRelationEditor(container, options) {
    $('<input id="irRelationEditor" name="irRelationEditor" data-bind="value:' + options.field + '"/>')
        .appendTo(container)
        .kendoDropDownList({
            dataSource: USERINPUT.relations,
            dataTextField: "display",
            dataValueField: "Relation",
            optionLabel: "---",
            open: function (e) {
                var workflowObj = getWorkflowObj(options.model.irWorkflow, options.model.irtag_isFactoryProvidedWorkflow);
                var irRelationshipType = workflowObj ? workflowObj.RelationshipType : "";
                e.sender.dataSource.filter({
                    field: "RelationshipType", operator: "contains", value: irRelationshipType
                });
            },
            change: function (e) {
                if (hasGender(options.model.irRelationRole)) {
                    options.model.set('irGender', '');
                }
                if (irSetIdNotValid(options.model)) {
                    options.model.set('irSetID', '');
                }
                updateIRvalidationErrors(options.model.row, ['irRelationRole', 'irGender']);
            }
        });
}

function irGenderEditor(container, options) {
    $('<input id="irGenderEditor" name="irGenderEditor" data-bind="value:' + options.field + '"/>')
        .appendTo(container)
        .kendoDropDownList({
            dataSource: USERINPUT.genders,
            dataTextField: "display",
            dataValueField: "Gender",
            optionLabel: "---",
            open: function (e) {
                if (hasGender(options.model.irRelationRole)) {
                    e.sender.dataSource.filter({
                        field: "Relation", operator: "contains", value: options.model.irRelationRole
                    });
                }
            },
            change: function (e) {
                updateIRvalidationErrors(options.model.row, ['irGender']);
            }
        });
}

function irMultiWorkflowEditor(container, options) {
    var multiselectWorkflowInput = $('<input id="irWorkflowEditor" name="irWorkflowEditor"/>').appendTo(container);
    var selectedworkflowList = options.model.get('irMultipleWorkflowSelected');
    var selectedworkflow = (selectedworkflowList && selectedworkflowList.length > 1) ? selectedworkflowList : options.model.get('irWorkflow');

    if(!selectedworkflow) {
        selectedworkflow = [""];
    }
    var irWorkflows = USERINPUT.workflows;
    var filteredWorkflows;
    if (!$('#irWorkflowShowAll').is(':checked')) {
        filteredWorkflows = irWorkflows.filter(function (item) {
            var reference = options.model.get('reference');
            var nucleotideType = options.model.get('nucleotideType');
            var applicationType = item.ApplicationType;
            if(applicationType === "UploadOnly") {
                return item;
            }
            if (nucleotideType && reference) {
                if(item.nucleotideType === nucleotideType && item.Reference === reference || item.nucleotideType === nucleotideType){
                    return item;
                }
            } else if (nucleotideType && item.nucleotideType === nucleotideType) {
                return item;
            }

        });
    }
    multiselectWorkflowInput.kendoMultiSelectBox({
        dataSource: filteredWorkflows || irWorkflows,
        dataTextField: "display",
        dataValueField: "Workflow",
        emptySelectionLabel: "Select workflow",
        value: JSON.parse(JSON.stringify(selectedworkflow))
    });
    multiselectWorkflowInput.data("kendoMultiSelectBox").bind("selectionChanged", function (e) {
        // console.log("selected item count: " + e.newValue.length);
        options.model.set("irWorkflow", e.newValue[0]);
        options.model.set("irMultipleWorkflowSelected", e.newValue);

    });
}

function irPopulationEditor(container, options) {
    $('<input id="irPopulationEditor" name="irPopulationEditor" data-bind="value:' + options.field + '"/>')
        .appendTo(container)
        .kendoDropDownList({
            dataSource: populations,
            dataTextField: "display",
            dataValueField: "value",
            optionLabel: "---",
            change: function (e) {
                updateIRvalidationErrors(options.model.row, ['irPopulation']);
            }
        });
}
function irmouseStrainsEditor(container, options) {
    $('<input id="irmouseStrainsEditor" name="irmouseStrainsEditor" data-bind="value:' + options.field + '"/>')
        .appendTo(container)
        .kendoDropDownList({
            dataSource: mouseStrains,
            dataTextField: "display",
            dataValueField: "value",
            optionLabel: "---",
            change: function(e){
                updateIRvalidationErrors(options.model.row, ['irmouseStrains']);
            }
        });
}

function irCancerTypeEditor(container, options) {
    $('<input id="irCancerTypeEditor" name="irCancerTypeEditor" data-bind="value:' + options.field + '"/>')
        .appendTo(container)
        .kendoDropDownList({
            dataSource: cancerTypes,
            dataTextField: "display",
            dataValueField: "value",
            optionLabel: "---",
            change: function(e){
                updateIRvalidationErrors(options.model.row, ['ircancerType']);
            }
        });
}

function ircellularityPctEditor(container, options) {
    $('<input id="ircellularityPctEditor" name="ircellularityPctEditor" data-bind="value:' + options.field + '"/>')
        .appendTo(container)
        .kendoNumericTextBox({ min: 0, max: 100, step: 1 });
}

function irSampleCollectionDateEditor(container, options) {
    $('<input id="irSampleCollectionDateEditor" name="irSampleCollectionDateEditor" data-bind="value:' + options.field +'" />')
        .appendTo(container)
        .kendoDatePicker({format: "yyyy MMMM dd",
            change: function(e){
                /* validate: Sample Receipt date must be later than or the same as the sample collection date */
                var collectDate = this.value();
                var receiptDate = options.model.irSampleReceiptDate;
                if ((receiptDate) && (Date.parse(collectDate) > Date.parse(receiptDate))) {
                    options.model.set('irSampleReceiptDate', '')
                }
            }
        });
}

function irSampleReceiptDateEditor(container, options) {
    $('<input id="irSampleReceiptDateEditor" name="irSampleReceiptDateEditor" data-bind="value:' + options.field +'" />')
        .appendTo(container)
        .kendoDatePicker({format: "yyyy MMMM dd",
            min: new Date(options.model.irSampleCollectionDate)
        });
}

function irBacterialMarkerTypeEditor(container, options) {
    $('<input id="irBacterialMarkerTypeEditor" name="irBacterialMarkerTypeEditor" data-bind="value:' + options.field + '"/>')
        .appendTo(container)
        .kendoDropDownList({
            dataSource: markers,
            dataTextField: "display",
            dataValueField: "value",
            optionLabel: "---",
            change: function(e){
                updateIRvalidationErrors(options.model.row, ['irBacterialMarkerType']);
            }
        });
}

function hasGender(Relation){
    return (Relation in USERINPUT.relations_with_gender)
}

function defaultRelation(workflow, tag_isFactoryProvidedWorkflow){
    var workflowObj=getWorkflowObj(workflow, tag_isFactoryProvidedWorkflow)
    if (workflowObj && workflowObj.relations_list.length == 1){
        return workflowObj.relations_list[0];
    }
    return "";
}

function irWorkflowNotValid(row){
    var notValid = false;
    if (USERINPUT.is_ir_connected && !$('#irWorkflowShowAll').is(':checked')){
        var workflowObj = getWorkflowObj(row.irWorkflow, row.irtag_isFactoryProvidedWorkflow);
        if (row.reference && workflowObj && workflowObj.Reference){
            notValid = notValid || (row.reference != workflowObj.Reference);
        }
        if (row.nucleotideType && workflowObj && workflowObj.nucleotideType){
            notValid = notValid || (row.nucleotideType != workflowObj.nucleotideType);
        }
    }
    return notValid;
}

function irSetIdNotValid(row){
    var notValid = false;
    if (USERINPUT.is_ir_connected && row.irSetID){
        var workflowObj = getWorkflowObj(row.irWorkflow, row.irtag_isFactoryProvidedWorkflow);
        var relations_list_length = workflowObj.relations_list.length;
        if (workflowObj.RelationshipType == "DNA_RNA"){
            var relations_list_length = 2;
        }
        if (relations_list_length > 1){
            var otherRows = $("#grid").data("kendoGrid").dataSource.data().toJSON();
            otherRows.splice(row.row, 1); // get rows other than this one
            $.each(otherRows, function(i,obj){
                if (obj.irSetID == row.irSetID){
                    if (workflowObj.RelationshipType != "DNA_RNA" && row.irRelationRole == obj.irRelationRole){
                        notValid = true;
                        return false;
                    }
                    if (workflowObj.RelationshipType == "DNA_RNA" && row.nucleotideType == obj.nucleotideType){
                        notValid = true;
                        return false;
                    }
                }
            });
        }
    }
    return notValid;
}

function generate_set_id(workflowObj, thisRow, otherRows){
    var setid = thisRow.irSetID || '';
    var relation = thisRow.irRelationRole || '';
    var relations_list_length = workflowObj.relations_list.length;
    if (workflowObj.RelationshipType == "DNA_RNA"){
        var relations_list_length = 2;
    }
    var foundSetID = false;
    if (relations_list_length > 1){
        // count workflow groups by set id for this row's workflow
        var workflow_groups = [];
        $.each(otherRows, function(i,obj){
            if (obj.irWorkflow == workflowObj.Workflow && obj.irSetID) {
                workflow_groups[obj.irSetID] = workflow_groups[obj.irSetID] || [];
                workflow_groups[obj.irSetID].push(obj)
            }
        });
        $.each(workflow_groups, function(id,data){
            if (data && data.length < relations_list_length){
                if (workflowObj.RelationshipType == "DNA_RNA"){
                    // find if this matches for DNA/RNA nuc type
                    foundSetID = thisRow.nucleotideType != data[0].nucleotideType;
                } else {
                    // find if Relation Role is not already used
                    foundSetID = $.grep(data,function(obj){
                        return obj.irRelationRole && obj.irRelationRole==thisRow.irRelationRole
                    }).length == 0;
                }
                if (foundSetID){
                    setid = id;
                    return false;
                }
            }
        });
    }
    if (!foundSetID){
        var other_setids = otherRows.map(function(obj){ return obj.irSetID});
        if (!setid || ( $.inArray(setid, other_setids) >= 0) ){
            // assign unique set id for this workflow
            setid = Math.max.apply(Math, other_setids) + 1;
        }
    }
    return setid;
}

function updateIRvalidationErrors(row, fields){
    // change any IR validation errors to warnings when table values are updated
    var needs_refresh = false;
    $.each(samplesTableValidationErrors, function(i, error){
        if (error.row == row && fields.indexOf(error.field) > -1){
            error.type = 'warning';
            needs_refresh = true;
        }
    });
    if (needs_refresh) setTimeout(function(){ gridRefresh($('#grid').data('kendoGrid'))}, 200);
}

/**
 This function searches the API returned result for a relationship type
 for a given workflow
*/
function find_relationship_type(workflow, columns_map) {
    var relationshipType = "";
    $.each(columns_map, function(i){
        var column = columns_map[i];
        if (column["Workflow"] == workflow) {
            relationshipType = column["RelationshipType"];
        }
    });
    return relationshipType;
}

/**
* Validate selected values are compatible with data returned by IRU
*/
function check_selected_values(){
    var errors = [];
    var samplesTableJSON = $("#grid").data("kendoGrid").dataSource.data().toJSON();
    var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");
    $.each(samplesTableJSON, function(i,row){
        if( row.irWorkflow && ($.grep(USERINPUT.workflows, function(obj){ return obj.Workflow == row.irWorkflow } ).length == 0) ){
            errors.push("<br>Row "+ (row.row+1) + ": Previous Workflow for this plan is no longer available: " + row.irWorkflow)
            row.irWorkflow = "";
            row.irRelationRole = "Self";
        }
        if (irWorkflowNotValid(row)){
            $('#irWorkflowShowAll').prop('checked', true);
            /* allow all workflows selection
            errors.push("<br>Row "+ (row.row+1) + ": Selected Workflow not compatible: " + row.irWorkflow);
            row.irWorkflow = "";
            row.irRelationRole = "Self"; */
        }
        if (row.irRelationRole && ($.grep(USERINPUT.relations, function(obj){ return obj.Relation == row.irRelationRole } ).length == 0) ){
            errors.push("<br>Row "+ (row.row+1) + ": Selected Relation not found: " + row.irRelationRole);
            row.irRelationRole = "";
        }
        var workflowObj = getWorkflowObj(row.irWorkflow, row.irtag_isFactoryProvidedWorkflow);
        if (workflowObj && row.irRelationRole && (workflowObj.relations_list.indexOf(row.irRelationRole) == -1) ){
            errors.push("<br>Row "+ (row.row+1) + ": Selected Relation not compatible: " + row.irRelationRole);
            row.irRelationRole = "";
        }
        if (row.irGender && ($.grep(USERINPUT.genders, function(obj){ return obj.Gender == row.irGender } ).length == 0) ){
            errors.push("<br>Row "+ (row.row+1) + ": Selected Gender not found: " + row.irGender);
            row.irGender = "";
        }
        if (row.ircancerType && ($.grep(USERINPUT.cancerTypes, function(obj){ return obj.CancerType == row.ircancerType } ).length == 0) ){
            errors.push("<br>Row "+ (row.row+1) + ": Selected Cancer Type not found: " + row.ircancerType);
            //row.ircancerType = ""; // allow TS-based annotation even if not present on IR
        }
        if (row.irPopulation && ($.grep(USERINPUT.populations, function(obj){ return obj.Population == row.irPopulation } ).length == 0) ){
            errors.push("<br>Row "+ (row.row+1) + ": Selected Population not found: " + row.irPopulation);
            //console.log("UnComment the above line once IRU implements the validation for population");
            //row.irPopulation = ""; // allow TS-based annotation even if not present on IR
        }
        if (row.irmouseStrains && ($.grep(USERINPUT.mouseStrains, function(obj){ return obj.MouseStrains == row.irmouseStrains } ).length == 0) ){
            //errors.push("<br>Row "+ (row.row+1) + ": Selected Mouse Strains not found: " + row.irmouseStrains);
            console.log("UnComment the above line once IRU implements the validation for population");
            //row.irmouseStrains = ""; // allow TS-based annotation even if not present on IR
        }
        if (!row.irSetID){
            if( i > 0 && row.irWorkflow) {
                if (planOpt.isDualNucleotideType && isSameSampleForDual && !isEven(i)){
                    row.irSetID = samplesTableJSON[i-1].irSetID;
                } else {
                    row.irSetID = generate_set_id(workflowObj, row, samplesTableJSON.slice(0, i));
                }
            } else {
                row.irSetID = (i > 0)? 1 + samplesTableJSON[i-1].irSetID : 1;
            }
        }
    });
    // update via local data to run dataSource initialization
    samplesTableInit = samplesTableJSON;
    $("#grid").data("kendoGrid").dataSource.read();
    $("#error").html(errors.toString());
    if (errors.length > 0) $('html, body').animate({scrollTop : $('#error').prop("scrollHeight")},500);
}

/**
 Parse data returned from IRU and save in USERINPUT
**/
function populate_userinput_from_response(data){
    var columns = data["sampleRelationshipsTableInfo"]["columns"];
    var column_map = data["sampleRelationshipsTableInfo"]["column-map"];
    var restrictions = data["sampleRelationshipsTableInfo"]["restrictionRules"];
    //sorts the columns by the Order key
    columns.sort(function(columnA, columnB){
        return columnA.Order - columnB.Order;
    });
    // parse restriction rules
    var relation_to_relationshipType = {};
    var gender_to_relation = {};
    var relationshipType_to_relations = {};
    $.each(restrictions, function(i, restriction){
        if (typeof restriction["For"] != 'undefined') {
            if (restriction["For"]["Name"] == 'RelationshipType') {
                $.each(restriction["Valid"]["Values"], function(i, value){
                    relation_to_relationshipType[value] = relation_to_relationshipType[value] || [];
                    relation_to_relationshipType[value].push(restriction["For"]["Value"]);
                });
                if (!('AndFor' in restriction)){
                    relationshipType_to_relations[restriction["For"]["Value"]] = restriction["Valid"]["Values"];
                }
            } else if (restriction["For"]["Name"] == 'Relation') {
                $.each(restriction["Valid"]["Values"], function(i, value){
                    gender_to_relation[value] = gender_to_relation[value] || [];
                    gender_to_relation[value].push(restriction["For"]["Value"]);
                });
                // also save which relations have associated Gender
                USERINPUT.relations_with_gender[restriction["For"]["Value"]] = restriction["Valid"]["Values"];
            }
        }
    });
    // Workflow
    $.each(column_map, function(i, cm){
        var workflow = cm["Workflow"];
        var irReference = cm["irReference"] || "";
        // reference name conversion: TS "GRCh38.p2.mask1" = IR "GRCh38"
        var reference = (irReference == "GRCh38") ? "GRCh38.p2.mask1" : irReference
        var relationshipType = cm["RelationshipType"];
        var nucleotideType = cm["DNA_RNA_Workflow"] || "";
        var tagImmuneRepertoire = cm["tag_IMMUNE_REPERTOIRE_SHORT_ASSAY"] || "";
        var tagDNA = cm["tag_DNA"] || "";
        var tagRNA = cm["tag_RNA"] || "";
        var DNA_RNA = (tagImmuneRepertoire.toLowerCase() == "true" && tagDNA.toLowerCase() == "true") ? "DNA" : nucleotideType;
        DNA_RNA = (tagImmuneRepertoire.toLowerCase() == "true" && tagRNA.toLowerCase() == "true") ? "RNA" : DNA_RNA;
        nucleotideType = DNA_RNA;
        USERINPUT.workflows.push({
            "Workflow": workflow != "Upload Only" ? workflow : "",
            "display": get_decorated_workflow_name(cm, workflow, irReference, ION_PRELOADED_LABEL, SEPARATOR),
            "tag_isFactoryProvidedWorkflow": cm["tag_isFactoryProvidedWorkflow"],
            "ApplicationType": cm["ApplicationType"],
            "RelationshipType": relationshipType,
            "Reference": reference,
            "nucleotideType": (nucleotideType=="DNA" || nucleotideType=="RNA") ? nucleotideType : "",
            "relations_list": relationshipType_to_relations[relationshipType] || []
        });
    });
    // Relation and Gender
    $.each(columns, function(i, column){
        if (column["Name"] == "Relation"){
            USERINPUT.relations = $.map(column["Values"], function(value, key){
                var relationshipType = relation_to_relationshipType[value] || [];
                return {
                    "Relation": value,
                    "display": value,
                    "RelationshipType": relationshipType.toString()
                };
            });
        }
        if (column["Name"] == "Gender"){
            USERINPUT.genders = $.map(column["Values"], function(value, key){
                var relation =  gender_to_relation[value] || [];
                return {
                    "Gender": value,
                    "display": value,
                    "Relation": relation.toString()
                };
            });
        }
        if (column["Name"] == "CancerType"){
            USERINPUT.cancerTypes = $.map(column["Values"], function(value, key){
                return {
                    "CancerType": value,
                    "display": value
                };
            });
        }
        if (column["Name"] == "Population"){
            USERINPUT.populations = $.map(column["Values"], function(value, key){
                return {
                    "Population": value,
                    "display": value
                };
            });
        }
        if (column["Name"] == "MouseStrains"){
            USERINPUT.mouseStrains = $.map(column["Values"], function(value, key){
                return {
                    "MouseStrains": value,
                    "display": value
                };
            });
        }
    });
}

/**
 Main function to call IRU and process the response
**/
function load_and_set_ir_fields() {
    var myURL = get_workflow_url(USERINPUT.user_input_url, USERINPUT.account_id);
    $.blockUI();
    var jqhxhr = $.ajax({
        type : "get",
        url : myURL,
        timeout: 12000,//in milliseconds
        success: function(data){
            if (typeof data["sampleRelationshipsTableInfo"] == 'undefined') {
                // failed to retrieve info
                $("#loading").hide();
                USERINPUT.is_ir_connected = false;
                $("#error").text("Cannot contact Ion Reporter Server!");
                $("input[name=irDown]").val('1');
                return;
            } else {
                USERINPUT.is_ir_connected = true;
            }
            $("#error").text("");
            $("input[name=irDown]").val('0');
            // parse data and fill in USERINPUT fields
            populate_userinput_from_response(data);
            // set application type hidden input from selected workflow
            var default_workflow = getWorkflowObj(USERINPUT.workflow, USERINPUT.tag_isFactoryProvidedWorkflow);
            if (default_workflow)
                $("input[name=applicationType]").val(default_workflow.ApplicationType);
            // make sure existing selections are valid
            check_selected_values();
            hide_summary_view();
            $("#loading").hide();
        },
        error: function(jqXHR, textStatus, errorThrown){
            if(textStatus==="timeout") {
                //The API Server timedout
                $("#loading").hide();
                $("#error").text('The IR Server connection has timed out');
                $("input[name=irDown]").val('1');
                return;
            } else {
                console.log("iru_get_user_input - textStatus=", textStatus);
            }
        }
    });
}

$(document).ready(function(){
    //handler for asynchroneous Ajax calls to block and unblock the UI
    $(document).ajaxStop($.unblockUI);
    if (USERINPUT.account_id != "0" && USERINPUT.account_id != "-1"){
        load_and_set_ir_fields();
    }
});
