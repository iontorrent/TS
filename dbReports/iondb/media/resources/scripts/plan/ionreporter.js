var IONREPORTER = IONREPORTER || {};

//this is the REST API url for looking up all workflows for a given IR Account
IONREPORTER.workflow_url = "/rundb/api/v1/plugin/IonReporterUploader/extend/workflows/";
//this the REST API url for retrieving all IR Accounts and their configuration
IONREPORTER.ion_accounts_url = "/rundb/api/v1/plugin/IonReporterUploader/extend/configs/";

IONREPORTER.workflows = [];
IONREPORTER.ir_rel_type_to_sample_group_map = {};
IONREPORTER.sample_group_to_pk_map = {};
IONREPORTER.default_sample_grouping = [];
IONREPORTER.default_iruUploadMode = [];
IONREPORTER.$none_input;

var SEPARATOR = " | ";
var ION_PRELOADED_LABEL = "Ion Torrent";

/**
    This function creates the None Radio Option for the IR Accounts.
    If the None option is pressed, then the accountId is set to 0
    and the account name is set to None and the workflow is cleared
*/
function create_none_ir_account() {
    var $div = $("#ir_accounts");
    var $none_lbl = $("<label></label>");;
    $none_lbl.addClass('radio');
    $none_lbl.text("None");

    IONREPORTER.$none_input = $("<input type='radio'/>");

    IONREPORTER.$none_input.attr({'name' : 'irOptions', 'value' : '0'});
    IONREPORTER.$none_input.on('click', function(){
        $("#error").text("");
        $('input[name="irAccountId"]').val('0');
        $('input[name="irAccountName"]').val('None');
        $("#selectedIR").text('None');
        $("#selectedWorkflow").text('');
        $("#selectedGroup").text('');
        $("#selectedUploadMode").text('');
        $("input[name=irVersion]").val('0');
        $("#workflows").hide();
        $("#iruUploadMode").hide();
        var $div = $(".sampleGroupOptionsContent");
        $div.html('');
        $.each(IONREPORTER.default_sample_grouping, function(i){
            sample_group = IONREPORTER.default_sample_grouping[i];
            var $label = $("<label></label>", {'class' : 'radio', 'width' : '150px'});

            $label.text(sample_group);
            var $input = $("<input/>", {'type':'radio', 'name':'sampleGrouping', 'value':IONREPORTER.sample_group_to_pk_map[sample_group], 'data-label':sample_group});
            if (IONREPORTER.existing_sample_grouping == IONREPORTER.sample_group_to_pk_map[sample_group]) {
                $input.attr('checked', true);
                $("#selectedGroup").text($input.data('label'));
            }
            $input.on('click', function(){
                $("#selectedGroup").text($input.parent().text().trim());
            });
            $label.append($input);
            $div.append($label);
        });
    }).click();

    $none_lbl.append(IONREPORTER.$none_input);
    $div.append($none_lbl);
    return $div;
}

function getWorkflowObj(workflow, tag_isFactoryProvidedWorkflow){
    // find workflow by matching workflow name + tag_isFactoryProvidedWorkflow
    // if no match found then find by matching just the workflow name
    if (!workflow) return null;

    var match = $.grep(IONREPORTER.workflows, function(obj){ return obj.Workflow == workflow && obj.tag_isFactoryProvidedWorkflow == tag_isFactoryProvidedWorkflow });
    if (match.length == 0){
        match = $.grep(IONREPORTER.workflows, function(obj){ return obj.Workflow == workflow });
    }
    console.log('getWorkflowObj', workflow, tag_isFactoryProvidedWorkflow, match)
    return (match.length > 0) ? match[0] : null;
}

/**
    This function retrieves the workflows, relationship type, and application types for a given IR Account
    by its given ID.  It receives the account id and creates a workflow drop down list
    and creates radio buttons for the Sample Grouping (Relationship Types) based on a TSS to IR Mapping of names
*/
function get_workflow_and_meta_data(id, fullName) {
    $.blockUI();
    var myURL = get_workflow_url(IONREPORTER.workflow_url, id);
    var found_workflow = null;
    $('input[name="irAccountId"]').val(id);
    $('input[name="irAccountName"]').val(fullName);
    $("#selectedIR").text(fullName);
    $("#error").text("");

    //First we call the API to retrieve all workflows
    $.when($.ajax({
                url : myURL,
                timeout: 6000, //in milliseconds
                error: function(jqXHR, textStatus, errorThrown){
                    if(textStatus==="timeout") {
                        //The API Server timedout
                        $("#loading").hide();
                        $("#error").text('The IR Server connection has timed out');
                        return;
                    }
                }
            }).then(function(data, textStatus, jqXHR) {

                var workflows = data["userWorkflows"];
                var all_relationship_types = [];
                var all_ir_references = [];

                //clear the workflow select drop down list
                var $select = $("select[name=irworkflow]");
                $select.empty();
                $select.append($("<option>Upload Only</option>"));

                if (typeof workflows != 'undefined') {
                    IONREPORTER.workflows = workflows;
                    //loop through the workflows
                    $.each(workflows, function(i){
                        var workflowName = workflows[i]["Workflow"];
                        var ir_reference = workflows[i]["irReference"] || "";
                        var isfactory = workflows[i]["tag_isFactoryProvidedWorkflow"] || "";

                        //create a SET of all RelationshipType
                        if ($.inArray(workflows[i]["RelationshipType"], all_relationship_types) == -1) {
                            all_relationship_types.push(workflows[i]["RelationshipType"]);
                        }
                        //create a SET of all References
                        if ($.inArray(ir_reference, all_ir_references) == -1) {
                            all_ir_references.push(ir_reference);
                        }

                        // create workflow option
                        workflows[i]['display'] = get_decorated_workflow_name(workflows[i], workflowName, ir_reference, ION_PRELOADED_LABEL, SEPARATOR);
                        workflows[i]['option_id'] = 'workflow' + i;

                        var opt = $('<option/>', {
                            id: workflows[i]['option_id'],
                            value: workflowName,
                            text: workflows[i]['display']
                        });
                        $(opt).data('isfactory', isfactory);
                        $select.append(opt);
                    });
                } else {
                    //The API call errored out and we have to inform the user
                    var message = "Internal error at IonReporterUploader when fetching workflow information";
                    if (typeof data["error"] != 'undefined') {
                        console.log("ionreporter - data[status]=", data["status"]);
                        console.log("ionreporter - data[error]=", data["error"]);
                        message += ".   ";
                        message += data["error"];
                    }

                    $("#error").text(message);
                    $("#loading").hide();
                    $('#selectedWorkflow').text('');
                    //$("#sample_grouping").hide();
                    $("#iruUploadMode").hide();
                    $("#workflows").hide();
                    $("input[name=applicationType]").val('');
                    return;
                }

                // select existing workflow
                found_workflow = getWorkflowObj(IONREPORTER.existing_workflow, IONREPORTER.existing_isfactory);
                if (found_workflow){
                    $("select[name=irworkflow]").children('option[id="'+found_workflow.option_id+'"]').attr("selected", "selected");
                } else {
                    if (IONREPORTER.existing_workflow.length > 0 && IONREPORTER.prev_account_id != "-1" ) {
                        $("#error").text("The previous workflow for this plan is no longer available.  Please select another workflow");
                    }
                }

                $("#workflows").show();
                $('#selectedWorkflow').text($('select[name=irworkflow]').val());

                //Enable IRU QC/Upload Mode when IR is selected
                //IRU Upload Mode is displayed as QC mode in UI
                var iru_QC_JSONString = $('input[name="iru_QC_UploadModes"]').val();
                iru_QC_JSONString = iru_QC_JSONString.replace(/\'/g, "\"");
                iru_upload_modes = JSON.parse(iru_QC_JSONString);
                var $div = $(".iruUploadModeContent");
                $div.html('');
                $.each(iru_upload_modes, function(key, value){
                    iru_upload_mode = value;
                    var $label = $("<label></label>", {'class' : 'radio', 'width' : '410px'})
                    $div.append($label);
                    $label.text(key);
                    var $input = $("<input/>", {'type' : 'radio', 'name' : 'iru_UploadMode', 'value' : iru_upload_mode});
                    $label.append($input);

                    if ((IONREPORTER.existing_iruUploadMode == 'Automatically upload to Ion Reporter after run completion' || IONREPORTER.existing_iruUploadMode == '') && fullName) {
                        if (iru_upload_mode == 'no_check') {
                            $input.attr('checked', true);
                            $("#selectedUploadMode").text(key);
                            $input.click();
                        }
                    }
                    if (iru_upload_mode == IONREPORTER.existing_iruUploadMode) {
                        $input.attr('checked', true);
                        $("#selectedUploadMode").text(key);
                        $input.click();
                    }
                });
                
                //create radio buttons for sample groups
                var $div = $(".sampleGroupOptionsContent");
                $div.html('');

                $.each(all_relationship_types, function(i, term){
                    var sample_group = IONREPORTER.ir_rel_type_to_sample_group_map[term];
                    var $label = $("<label></label>", {'class' : 'radio', 'width' : '150px'});
                    $div.append($label);
                    $label.text(sample_group);
                    if (typeof IONREPORTER.sample_group_to_pk_map[sample_group] == 'undefined') {
                        $("#error").text("The sample group " + sample_group + " is not supported in TSS");
                    } else {
                        var $input = $("<input/>", {'type' : 'radio', 'name' : 'sampleGrouping', 'value' : IONREPORTER.sample_group_to_pk_map[sample_group]});
                        $label.append($input);
                        //now we create the OnClick event for the Sample Group radio button
                        //once a sample group is selected, we filter out the workflow drop down list
                        $input.on('click', function(){
                            $("#selectedGroup").text($input.parent().text().trim());
                            var self = $(this);
                            var $select = $("select[name=irworkflow]");
                            $select.empty();
                            $select.append($("<option>Upload Only</option>"));
                            $.each(IONREPORTER.workflows, function(i,workflowObj){
                                if (term == workflowObj['RelationshipType']){
                                    var opt = $('<option/>', {
                                        id: workflowObj['option_id'],
                                        value: workflowObj['Workflow'],
                                        text: workflowObj['display']
                                    });
                                    $(opt).data('isfactory', workflowObj['tag_isFactoryProvidedWorkflow']);
                                    if (workflowObj == found_workflow){
                                        opt.attr('selected', true);
                                    }
                                    $select.append(opt);
                                }
                            });
                            $select.change();
                        });
                    }
                    if (IONREPORTER.sample_group_to_pk_map[sample_group] == IONREPORTER.existing_sample_grouping) {
                        $input.attr('checked', true);
                        $input.click();
                    }
                });
                $("#sample_grouping").show();
                $("#loading").hide();
            })
        );
}

/**
    This function uses the REST API to retrieve all IR Accounts and their configurations
    and create Radio buttons for each account.  When an Account is selected, another REST API call is
    made to retrieve the workflows for that account
*/
function get_ir_accounts() {
    $div = create_none_ir_account();

    var jqhxhr = $.ajax({
        type : "get",
        url : IONREPORTER.ion_accounts_url+"?format=json",
        timout: 6000, //in milliseconds
        success : function(data){
            //The API call is successful
            if (data && data.length>0) {
                $("#error").text("");
                var accounts = data;
                //boolean to check if the IR account was previously selected, either in the template
                //or part of a previous step in the plan run phase    
                var matched_prev = false;

                //Sort IR accounts alphabetically
                accounts = accounts.sort(function(a, b) {
                    return a.name.localeCompare(b.name);
                });
                //loop through all accounts and create radio buttons for each
                $.each(accounts, function(i){
                    var account = accounts[i];
                    var id = account["id"];
                    var version = account["version"];
                    var checked = account["default"];
                    var name = account["name"];
                    if (IONREPORTER.prev_account_id == id){
                        matched_prev = true;
                    }

                    //we extract the version, i.e. 1.6 or 4.0
                    version = version.substring(2, version.length);
                    version = version.substring(0, 1) + "." + version.substring(1, version.length);

                    var fullName = name + " ";
                    //add orgname, firstname, lastname  and version if they are there.
                    var detailsStr = "";
                    if ('details' in account){
                        detailsStr += " (";
                        detailsStr += "Version: " + version;

                        if ('firstname' in account['details'] && 'lastname' in account['details']){
                            detailsStr += " | User: " + account['details']['firstname'] + " " + account['details']['lastname'] ;
                        } else {
                            detailsStr = detailsStr.substring(0, detailsStr.length-2);
                        }
                        if ('orgname' in account['details']){
                            detailsStr += " | Org: " + account['details']['orgname'];
                        }
                        detailsStr += ")";
                    }

                    //create a radio button for the account 
                    var $lbl = $("<label></label>");
                    $lbl.addClass('radio');
                    $lbl.text(fullName);
                    if (detailsStr != " ()") {
                        $lbl.append($('<span></span>')
                                .addClass('label_iru_details')
                                .text(detailsStr)
                        )
                    }

                    var $input = $("<input type='radio'/>");
                    $input.attr({'name' : 'irOptions', 'value' : version});
                    //attach OnClick event for the account
                    //the onclick event fires the REST API call to retrieve
                    //workflows and relationship types (sample groupings)
                    $input.on('click', function(){

                        $("input[name=irAccountId]").val(id);
                        $('input[name="irAccountName"]').val(fullName);
                        $('input[name=irVersion]').val(version);
                        //fire API CALL
                        get_workflow_and_meta_data(id, fullName);
                        if (version == "1.6") {
                            $("#new_workflow").hide();
                        } else {
                            $("#new_workflow").show();
                        }
                        if (version != '0') {
                            $("#iruUploadMode").show();
                        } else {
                            $("#iruUploadMode").hide();
                        }
                    });
                    $lbl.append($input);
                    $div.append($lbl);

                    //now we check if this account was selected in the template or a previous step
                    if(IONREPORTER.prev_account_id == id) {
                        $input.attr('checked', true);
                        $("input[name=irAccountId]").val(id);
                        $('input[name="irAccountName"]').val(fullName);
                        $("#selectedIR").text(fullName);
                        $input.click();
                    } else {
                        //now we know this is the default account
                        if (checked && !matched_prev) {
                            $input.attr('checked', true);
                            $("input[name=irAccountId]").val(id);
                            $('input[name="irAccountName"]').val(fullName);
                            $("#selectedIR").text(fullName);
                            $('input[name=irVersion]').val(version);
                            if (IONREPORTER.prev_account_id == "-1") {
                                $input.click();
                            }
                        } else {
                            $input.attr('checked', false);
                        }
                    }
                });

                //this checks if the previously selected IR account
                //is accessible to this user
                if (!matched_prev && IONREPORTER.prev_account_id != "-1") {
                    IONREPORTER.$none_input.attr('checked', true);
                    IONREPORTER.$none_input.click();

                    $("input[name=irAccountId]").val('0');
                    $('input[name="irAccountName"]').val('None');
                    $("#selectedIR").text('None');
                    if (IONREPORTER.prev_account_id != "0" && IONREPORTER.prev_account_id != "-1"){$("#irAccountDoesNotExistError").show();}
                }
            }
            $("#loading").hide();
        } ,
        error: function() {
            IONREPORTER.$none_input.attr('checked', true);
            $("input[name=irAccountId]").val('0');
            $('input[name="irAccountName"]').val('None');
            $("#selectedIR").text('None');
            $("#loading").hide();
            $("#iruUploadMode").hide();
        }
    });
}

/**
 This function is used to open a pop-up window to allow the user to create a new workflow
 */
function goToCreateWorkflowUrl() {
    var url = '/rundb/api/v1/plugin/IonReporterUploader/extend/newWorkflow/';
    $.ajax({
        async: false, //Bypass the popup blocker in chrome.
        url: url,
        data: {
            "format": "json",
            "id": $("input[name=irAccountId]").val()
        },
        success: function (data) {
            if (data["status"] != "false") {
                if (data["method"] == "get") {
                    window.open(data["workflowCreationLandingPageURL"], '_blank');
                } else {
                    var form = $("<form/>", {
                            action: data["workflowCreationLandingPageURL"],
                            method: "post",
                            target: "_blank"
                        }
                    );
                    form.append($("<input/>", {
                        name: "authToken",
                        value: data["token"]
                    }));
                    form.appendTo("body");
                    form.submit();
                }
            } else {
                alert("Error fetching IR workflow creation url.")
            }
        },
        error: function () {
            alert("Failed to retrieve the workflow creation url, make sure you are able to connect to IR.");
        }
    });
}

$(document).ready(function () {
    //we block the UR and show the spinner until Ajax calls are complete
    $(document).ajaxStop($.unblockUI);

    $("#loading").parent().parent().css('overflow', 'auto');
    //retrieve all IR accounts and their workflows/application types/relationship types    
    get_ir_accounts();

    //attach event to the iru QC/Upload Mode to show in the summary box
    $("#iruUploadMode").on('change', 'input:radio[name^="iru_UploadMode"]', function (event) {
        var none_option = "Automatically upload to Ion Reporter after run completion";
        var manual_option = "Review results after run completion, then upload to Ion Reporter"
        var iruUploadMode = $(this).val() ? $(this).val() : none_option;
        var qc_mode = iruUploadMode == "no_check" ? none_option : manual_option;
        $('#selectedUploadMode').text(qc_mode);
    });
    
    //refresh button which retrieves the workflows, etc for the selected account
    $("#ir_refresh").on('click', function(){
        get_workflow_and_meta_data($('input[name="irAccountId"]').val(), $('input[name="irAccountName"]').val());
    });

    //let stuff be configed
    $('#iruConfig').click(function(e){
        e.preventDefault();
        var url = $(this).attr('href');
        var modal = $("#modal-window");
        var content = $("#modal-content");
        $("h3", modal).text("Ion Reporter Uploader Configuration");
        content.attr('src', url);
        content.ready(function() {
            $('iframe.auto-height').iframeAutoHeight({minHeight: 400, heightOffset: 20});
            modal.modal("show");
        });
    });

    $('.closeIRU').on('click', function () {
        $("#ir_accounts").html("");
        $("#loading").parent().parent().css('overflow', 'auto');
        get_ir_accounts();
    });

    //attach change event to the workflow that auto selects the sample group
    //that corresponds to that workflow
    $("select[name=irworkflow]").on('change', function(e){
        var irworkflow = $(this).val();
        if (irworkflow && irworkflow != "Upload Only"){
            var workflowObj = getWorkflowObj(irworkflow, $(this).find(':selected').data('isfactory'));
            var pk = IONREPORTER.sample_group_to_pk_map[workflowObj['RelationshipType']];
            $.each($("input[name=sampleGrouping]"), function(){
                if ($(this).val() == pk) {
                    $(this).attr('checked', true);
                    $("#selectedGroup").text($(this).parent().text().trim());
                    return;
                }
            });

            $("input[name=applicationType]").val(workflowObj['ApplicationType']);
            $("input[name=tag_isFactoryProvidedWorkflow]").val(workflowObj['tag_isFactoryProvidedWorkflow']);
            $("#selectedWorkflow").text(irworkflow);
        } else {
            $("input[name=applicationType]").val('');
            $("input[name=tag_isFactoryProvidedWorkflow]").val('');
            $("#selectedWorkflow").text('Upload Only');
        }
    });

});
